"""Regression tests for the defects found in the v6.0 audit.

The v6.0 round introduced one method the earlier rounds had not used: an
independently written reference implementation of the decode step,
derived by accumulating per-layer matrix shapes rather than by aggregating
parameter counts. Two implementations that agree are not necessarily
right, but two that disagree cannot both be. The disagreement is what
found R6-1.
"""

import itertools
import math

import pytest

from caide import DeploymentState, ServingConfig, WorkloadClass
from caide.calibration import (
    FRAMEWORK_OVERHEAD_REFERENCE,
    Observation,
    READMITTED_OBSERVATIONS,
    REFERENCE_OBSERVATIONS,
    admissible_conventions,
    implied_mbu,
)
from caide.catalog import get_grid, get_hardware, get_model
from caide.roofline import _decode_mfu, decode_step_time, prefill_flops
from caide.specs import ModelSpec


# --------------------------------------------------------------------------
# The independent reference implementation. Kept in the test suite rather
# than in the package so that it cannot drift into being a wrapper around
# the code it is supposed to check.
# --------------------------------------------------------------------------

def reference_decode_step(model, hw, cfg, batch, context):
    """Decode step time derived from per-layer shapes.

    Deliberately structured unlike :func:`caide.roofline.decode_step_time`:
    attention and MLP weights are accumulated layer by layer from matrix
    dimensions, and the attention arithmetic is written out as two matmuls
    (QK^T then AV) rather than folded into a parameter count.
    """
    bandwidth = hw.memory_bandwidth * cfg.n_accelerators * cfg.mbu_decode
    flops = (hw.effective_flops(cfg.precision) * cfg.n_accelerators
             * _decode_mfu(cfg, batch))

    # Revised in v10.0: a decode step streams the head, not the input
    # embedding table. This reference had copied the production call for
    # the third time in this project's history -- see the v10 report on
    # references that share their subject's source document.
    weight_bytes = model.decode_weight_bytes(batch)
    kv_bytes = model.kv_bytes_per_token * context * batch
    memory_time = (weight_bytes + kv_bytes) / bandwidth

    gemm = 2.0 * model.active_params * batch
    qk = 2.0 * model.n_layers * model.d_model * context * batch
    av = 2.0 * model.n_layers * model.d_model * context * batch
    compute_time = (gemm + qk + av) / flops

    step = max(memory_time, compute_time)

    # Tensor-parallel synchronisation, derived here as "each of the two
    # collectives per layer moves one activation tensor, and a ring
    # all-reduce touches 2(N-1)/N of it".
    if cfg.n_accelerators > 1:
        per_collective = batch * model.d_model * 2.0
        payload = per_collective * 2.0 * model.n_layers
        ring = 2.0 * (cfg.n_accelerators - 1) / cfg.n_accelerators
        step += (payload * ring / hw.interconnect_bandwidth) \
            * (1.0 + cfg.tensor_parallel_penalty * 10.0)
    return step


SWEEP = list(itertools.product(
    ["dense-8b", "dense-70b", "dense-405b", "moe-8x7b", "moe-236b"],
    ["h100-sxm", "h200-sxm", "a100-80gb", "l40s"],
    ["bf16", "fp8", "int4"],
    [1, 8, 64, 256],
    [512, 4096, 32768, 131072],
))


def _spec(model_key, precision):
    model = get_model(model_key)
    bpp = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}[precision]
    return model.with_precision(bpp, bpp if precision != "bf16" else 2.0)


# -- R6-1: decode omitted the attention arithmetic --------------------------

def test_decode_matches_independent_reference_across_sweep():
    """960 configurations, two independently derived implementations.

    Before the fix, 155 of these disagreed by more than 1% and the worst
    disagreed by 4.83x.
    """
    worst, worst_at = 1.0, None
    for model_key, hw_key, precision, batch, context in SWEEP:
        model, hw = _spec(model_key, precision), get_hardware(hw_key)
        cfg = ServingConfig(n_accelerators=8, max_batch=batch,
                            precision=precision)
        mine, _ = decode_step_time(model, hw, cfg, batch, context)
        theirs = reference_decode_step(model, hw, cfg, batch, context)
        ratio = max(mine / theirs, theirs / mine)
        if ratio > worst:
            worst, worst_at = ratio, (model_key, hw_key, precision, batch, context)
    assert worst < 1.000001, f"diverges by {worst:.4f}x at {worst_at}"


def test_decode_attention_term_scales_with_context():
    """The omitted term is quadratic in the same sense prefill's is.

    Doubling the context doubles the per-step attention arithmetic, which
    is what makes it invisible at 512 tokens and dominant at 128k.
    """
    model, hw = get_model("dense-8b"), get_hardware("a100-80gb")
    cfg = ServingConfig(n_accelerators=1, max_batch=1)
    gemm = 2.0 * model.active_params
    for context in (32_768, 65_536, 131_072):
        attention = 4.0 * model.n_layers * model.d_model * context
        assert attention / gemm == pytest.approx(
            4.0 * model.n_layers * model.d_model * context / gemm)
    # at 128k the attention term exceeds the GEMM term for this model
    assert 4.0 * model.n_layers * model.d_model * 131_072 > 4 * gemm


def test_prefill_and_decode_model_the_same_attention_physics():
    """Both phases now account for scoring against the keys.

    Version 5.0 modelled the quadratic term in prefill and dropped it in
    decode. The two phases describe one mechanism; an implementation that
    includes it in one and not the other is inconsistent whatever the
    numerical effect.
    """
    model = get_model("dense-70b")
    tokens = 8192.0
    prefill_attention = prefill_flops(model, tokens) - 2.0 * model.active_params * tokens
    # prefill amortises 2*L*T^2*d over T tokens -> 2*L*T*d per token
    per_token_in_prefill = prefill_attention / tokens
    # decode at context T scores against T keys and weights T values
    per_token_in_decode = 4.0 * model.n_layers * model.d_model * tokens
    # the decode figure is twice the prefill figure because the causal
    # mask halves the prefill work and decode has no mask to exploit
    assert per_token_in_decode / per_token_in_prefill == pytest.approx(2.0)


def test_attention_term_can_flip_the_binding_resource():
    """At small batch the term decides which roofline binds.

    Decode FLOP utilisation at batch 1 is under one percent, which drops
    the machine's effective balance below attention's arithmetic
    intensity. This is the regime the omission was hiding in.
    """
    flipped = 0
    for model_key, hw_key, precision, batch, context in SWEEP:
        model, hw = _spec(model_key, precision), get_hardware(hw_key)
        cfg = ServingConfig(n_accelerators=8, max_batch=batch,
                            precision=precision)
        _, bound = decode_step_time(model, hw, cfg, batch, context)
        flops = (hw.effective_flops(precision) * cfg.n_accelerators
                 * _decode_mfu(cfg, batch))
        without_attention = 2.0 * model.active_params * batch / flops
        with_attention = (2.0 * model.active_params * batch
                          + 4.0 * model.n_layers * model.d_model
                          * context * batch) / flops
        memory = ((model.expert_bytes_touched(batch)
                   + model.kv_bytes_per_token * context * batch)
                  / (hw.memory_bandwidth * cfg.n_accelerators * cfg.mbu_decode))
        if without_attention <= memory < with_attention:
            flipped += 1
            assert bound in ("compute", "interconnect")
    assert flipped >= 40, f"only {flipped} configurations flip"


def test_published_findings_unchanged_by_the_attention_term():
    """The fix must not move any number the paper reports.

    Every analysis in the paper runs at batch 64 or above with contexts
    under 8.3k, where the memory term binds by a wide margin.
    """
    from caide.efficiency import TECHNIQUES
    hw, grid = get_hardware("h100-sxm"), get_grid("us-average")
    model = get_model("dense-70b")
    for context in (1700.0, 4750.0, 8300.0):
        for batch in (64, 256):
            cfg = ServingConfig(n_accelerators=4, max_batch=batch)
            step, bound = decode_step_time(model, hw, cfg, batch, context)
            gemm = 2.0 * model.active_params * batch
            attention = 4.0 * model.n_layers * model.d_model * context * batch
            # Revised in v16.0: ``parallel_efficiency`` was the pre-v6
            # multiplicative derate, kept as public API after the audit
            # replaced it with a per-layer all-reduce. This bound needs no
            # interconnect term at all -- it is a lower bound on compute
            # time, and dropping a derate below one only tightens it.
            flops = (hw.effective_flops("bf16") * cfg.n_accelerators
                     * _decode_mfu(cfg, batch))
            assert bound == "memory"
            assert (gemm + attention) / flops < step


# -- R6-2: framework overhead is not a constant -----------------------------

def test_constant_overhead_cannot_fit_both_published_points():
    """The residuals differ by a factor of two, which no constant absorbs."""
    model, hw = get_model("dense-8b"), get_hardware("a100-80gb")
    residuals = {}
    for batch, aggregate_tps in ((1, 38.0), (8, 187.0)):
        cfg = ServingConfig(n_accelerators=1, max_batch=batch)
        hardware_step, _ = decode_step_time(model, hw, cfg, batch, 1024)
        wall_step = batch / aggregate_tps
        residuals[batch] = wall_step - hardware_step
    assert all(r > 0 for r in residuals.values())
    assert residuals[8] / residuals[1] > 1.8


def test_linear_overhead_fits_both_points():
    """A per-step constant plus a per-sequence term reproduces both."""
    model, hw = get_model("dense-8b"), get_hardware("a100-80gb")
    per_step = FRAMEWORK_OVERHEAD_REFERENCE["per_step_seconds"]
    per_sequence = FRAMEWORK_OVERHEAD_REFERENCE["per_sequence_seconds"]
    for batch, aggregate_tps in ((1, 38.0), (8, 187.0)):
        cfg = ServingConfig(n_accelerators=1, max_batch=batch,
                            framework_overhead_per_step=per_step,
                            framework_overhead_per_sequence=per_sequence)
        step, _ = decode_step_time(model, hw, cfg, batch, 1024)
        assert batch / step == pytest.approx(aggregate_tps, rel=0.05)


def test_overhead_reference_declares_zero_degrees_of_freedom():
    """Two points fitting two parameters is consistency, not validation.

    The record has to say so, because a fit with no residual looks like
    the strongest possible evidence and is the weakest.
    """
    assert FRAMEWORK_OVERHEAD_REFERENCE["degrees_of_freedom"] == 0
    assert FRAMEWORK_OVERHEAD_REFERENCE["n_points"] == 2


def test_overhead_agrees_with_independent_profiling():
    """Two routes that share no data land 13% apart."""
    ours = FRAMEWORK_OVERHEAD_REFERENCE["gpu_share_of_wall_at_batch_1"]
    theirs = FRAMEWORK_OVERHEAD_REFERENCE["vllm_profiled_gpu_share"]
    assert abs(ours - theirs) / theirs < 0.20


def test_overhead_defaults_to_zero_and_stays_opt_in():
    """The bundled default is the pure hardware roofline."""
    cfg = ServingConfig()
    assert cfg.framework_overhead_per_step == 0.0
    assert cfg.framework_overhead_per_sequence == 0.0


def test_per_sequence_overhead_grows_with_batch():
    model, hw = get_model("dense-8b"), get_hardware("h100-sxm")
    steps = []
    for batch in (1, 8, 64):
        cfg = ServingConfig(n_accelerators=1, max_batch=batch,
                            framework_overhead_per_sequence=0.001)
        plain = ServingConfig(n_accelerators=1, max_batch=batch)
        with_oh, _ = decode_step_time(model, hw, cfg, batch, 1024)
        without, _ = decode_step_time(model, hw, plain, batch, 1024)
        steps.append(with_oh - without)
    assert steps[1] == pytest.approx(8 * steps[0])
    assert steps[2] == pytest.approx(64 * steps[0])


def test_negative_overhead_is_rejected():
    with pytest.raises(ValueError):
        ServingConfig(framework_overhead_per_sequence=-0.001)


# -- R6-3: the convention ambiguity is decidable ----------------------------

def _ambiguous_observation():
    return Observation(
        model=get_model("dense-8b"), hardware=get_hardware("a100-80gb"),
        n_accelerators=1, batch=8, tokens_in=1024, tokens_out=512,
        measured_output_tps=187.0, source="vLLM 8B A100 8-concurrent",
        convention="aggregate",
    )


def test_roofline_lower_bound_excludes_the_per_request_reading():
    """The per-request reading needs bandwidth utilisation above one."""
    obs = _ambiguous_observation()
    assert admissible_conventions(obs) == ["aggregate"]
    assert implied_mbu(obs, "per_request") > 1.0
    assert implied_mbu(obs, "aggregate") < 1.0


def test_disambiguation_is_one_sided():
    """A reading can be ruled out for being too fast, never for too slow.

    Framework overhead sits between the hardware bound and the wall clock
    and has no upper limit, so a slow figure is always admissible.
    """
    obs = Observation(
        model=get_model("dense-8b"), hardware=get_hardware("a100-80gb"),
        n_accelerators=1, batch=8, tokens_in=1024, tokens_out=512,
        measured_output_tps=0.01, source="absurdly slow", convention="aggregate",
    )
    assert set(admissible_conventions(obs)) == {"aggregate", "per_request"}


def test_genuinely_ambiguous_figures_stay_ambiguous():
    """The test must not resolve every case, or it is not a physical test."""
    obs = Observation(
        model=get_model("dense-8b"), hardware=get_hardware("h100-sxm"),
        n_accelerators=1, batch=2, tokens_in=512, tokens_out=128,
        measured_output_tps=30.0, source="mid-range figure",
        convention="aggregate",
    )
    assert len(admissible_conventions(obs)) == 2


def test_readmitted_observation_is_recorded_separately():
    """A validation set that grows must show why, or it is drifting."""
    assert len(READMITTED_OBSERVATIONS) >= 1
    for label, reason in READMITTED_OBSERVATIONS:
        assert label and len(reason) > 40
        assert "v5" in reason or "v6" in reason


def test_every_reference_observation_survives_its_own_test():
    """No observation in the validation set may be physically impossible."""
    for obs in REFERENCE_OBSERVATIONS():
        assert obs.convention in admissible_conventions(obs), obs.source


# -- R6-4: impossible mixture-of-experts geometry ---------------------------

def test_active_params_below_the_router_floor_is_rejected():
    """k of E experts cannot activate less than k/E of the parameters."""
    with pytest.raises(ValueError, match="arithmetic floor"):
        ModelSpec(name="impossible", n_params_total=100e9, n_params_active=8e9,
                  n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
                  n_experts=8, experts_per_token=1)


def test_geometry_exactly_at_the_floor_is_accepted():
    """The bound is attained when nothing is shared, so it must not be strict."""
    model = ModelSpec(name="all-expert", n_params_total=100e9,
                      n_params_active=12.5e9, n_layers=32, d_model=4096,
                      n_heads=32, n_kv_heads=8, n_experts=8,
                      experts_per_token=1)
    assert model.is_moe


def test_batch_one_reads_exactly_the_active_parameters():
    """The identity the clamp removed in v6.0 used to break.

    A batch of one routes to k experts, so the bytes it streams are the
    active parameter count by definition. v5.0's floor of 10% of active
    on the shared block broke this for geometries near the router floor.
    """
    catalogue = ["moe-8x7b", "moe-8x22b", "moe-236b"]
    for key in catalogue:
        model = get_model(key)
        touched = model.expert_bytes_touched(1) / model.bytes_per_param
        assert touched == pytest.approx(model.active_params, rel=1e-9)
    edge = ModelSpec(name="at-floor", n_params_total=100e9,
                     n_params_active=12.5e9, n_layers=32, d_model=4096,
                     n_heads=32, n_kv_heads=8, n_experts=8, experts_per_token=1)
    assert edge.expert_bytes_touched(1) / edge.bytes_per_param == \
        pytest.approx(edge.active_params, rel=1e-9)


def test_large_batch_converges_on_the_full_weight_footprint():
    for key in ("moe-8x7b", "moe-236b"):
        model = get_model(key)
        assert model.expert_bytes_touched(4096) == pytest.approx(
            model.weight_bytes, rel=1e-3)


# -- R6-5: expert parallelism is out of scope, and says so ------------------

def test_expert_imbalance_defaults_to_balanced():
    assert ServingConfig().expert_imbalance == 1.0


def test_expert_imbalance_only_affects_mixture_models():
    hw = get_hardware("h100-sxm")
    dense = get_model("dense-70b")
    balanced = ServingConfig(n_accelerators=8, max_batch=64)
    skewed = ServingConfig(n_accelerators=8, max_batch=64, expert_imbalance=0.5)
    assert (decode_step_time(dense, hw, balanced, 64, 4096)[0]
            == pytest.approx(decode_step_time(dense, hw, skewed, 64, 4096)[0]))


def test_expert_imbalance_lengthens_the_step_for_mixture_models():
    hw, moe = get_hardware("h100-sxm"), get_model("moe-236b")
    balanced = ServingConfig(n_accelerators=8, max_batch=64)
    skewed = ServingConfig(n_accelerators=8, max_batch=64, expert_imbalance=0.5)
    fast, _ = decode_step_time(moe, hw, balanced, 64, 4096)
    slow, _ = decode_step_time(moe, hw, skewed, 64, 4096)
    assert slow > fast


def test_imbalance_outside_the_unit_interval_is_rejected():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="expert_imbalance"):
            ServingConfig(expert_imbalance=bad)


def test_all_to_all_traffic_is_below_the_threshold_that_would_need_modelling():
    """The negative result from the v6.0 audit, kept as a test.

    v5.0 predicted expert routing would be the next structural bias. The
    communication half of that prediction did not hold: dispatch and
    combine move two hidden states per token per layer, which is under 6%
    of the step at the largest batch the catalogue supports. If a future
    model or interconnect pushes it past that, this test fails and the
    conclusion has to be revisited.
    """
    hw = get_hardware("h100-sxm")
    for key in ("moe-8x7b", "moe-8x22b", "moe-236b"):
        model = get_model(key)
        for batch in (8, 64, 256):
            cfg = ServingConfig(n_accelerators=8, max_batch=batch)
            step, _ = decode_step_time(model, hw, cfg, batch, 4096)
            traffic = (2.0 * model.experts_per_token * model.d_model
                       * model.bytes_per_param * batch * model.n_layers)
            share = (traffic / hw.interconnect_bandwidth) / step
            assert share < 0.10, f"{key} at batch {batch}: {share:.1%}"


# -- structural conclusions survive the new physics -------------------------

def test_regime_dependence_survives_the_attention_term():
    """The paper's first conclusion does not rest on the omission."""
    from caide.efficiency import get_technique
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    grid = get_grid("us-average")
    workload = WorkloadClass("q", 1.0, 1500, 400)
    multipliers = []
    for batch in (1, 256):
        cfg = ServingConfig(n_accelerators=4, max_batch=batch)
        state = DeploymentState(model, hw, cfg)
        after = get_technique("speculative_decoding").apply(state)
        from caide.costing import self_hosted_query_cost
        base = self_hosted_query_cost(state, workload, grid, respect_slo=False)
        new = self_hosted_query_cost(after, workload, grid, respect_slo=False)
        multipliers.append(new.compute_cost / base.compute_cost)
    assert multipliers[1] / multipliers[0] > 1.5
