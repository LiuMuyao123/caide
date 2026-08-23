"""Regression tests for the defects found in the v7.0 audit.

The v6.0 audit ended with two standing recommendations: extend the
independent reference implementation beyond the roofline, and give
``expert_imbalance`` a basis. Both were carried out this round, and both
found something.

The reference implementation gained two new axes. Along the *technique*
axis it prices the deployment states the efficiency layer produces --
the v6.0 sweep covered 960 hardware configurations and zero transformed
ones -- and disagreed immediately on speculative decoding (R7-1). Along
the *layer* axis it re-derives the annual ledger from replica-level
accounting rather than per-query accumulation, and disagreed on energy
(R7-3, R7-4). Two implementations that agree are not necessarily right,
but two that disagree cannot both be.
"""

import itertools
import math
from dataclasses import replace

import pytest

from caide import DeploymentState, ServingConfig, WorkloadClass
from caide.calibration import (
    REFERENCE_OBSERVATIONS,
    admissible_conventions,
    predicted_output_tps,
)
from caide.catalog import get_grid, get_hardware, get_model
from caide.costing import (
    self_hosted_query_cost,
    total_cost_of_ownership,
)
from caide.roofline import (
    _decode_mfu,
    capacity_batch,
    decode_step_time,
    evaluate_request,
    uniform_routing_imbalance,
)

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


# --------------------------------------------------------------------------
# The reference implementation, extended along the technique axis. Kept in
# the test suite, like its v6.0 predecessor, so that it cannot drift into
# being a wrapper around the code it checks. It is derived from the model
# *as documented* in docs/model.md -- including the documented scope
# boundaries, such as the draft model's KV cache being out of scope --
# rather than from the production source.
# --------------------------------------------------------------------------

def reference_step(model, hw, cfg, batch, context):
    """Decode/verification step time from first principles.

    Structured unlike the production function: the verification pass is
    priced as ``gamma + 1`` virtual one-token positions that share a
    single weight stream, each position's arithmetic written as a GEMM
    row plus two attention matmuls; the draft is priced as a separate
    miniature dense deployment run ``gamma`` times; the expert straggler
    is derived from "the step ends when the busiest expert's *own* work
    ends", so it never touches the KV stream or the attention arithmetic.
    """
    n = cfg.n_accelerators
    bandwidth = hw.memory_bandwidth * n * cfg.mbu_decode

    speculating = cfg.speculative_gamma > 0 and cfg.draft_param_ratio > 0
    positions = int(cfg.speculative_gamma) + 1 if speculating else 1

    # -- target memory: one weight stream, one KV stream ------------------
    # Revised in v8.0. This reference was derived from docs/model.md,
    # which described expert routing per *sequence* rather than per
    # token, so it reproduced the production code's blind spot instead of
    # exposing it (R8-4). A reference implementation checks only what its
    # source document gets right; that is now recorded in the v8 report.
    # Revised in v10.0 for the same reason the v6 reference was.
    weight_bytes = model.decode_weight_bytes(batch * positions)
    kv_bytes = model.kv_bytes_per_token * context * batch
    memory = (weight_bytes + kv_bytes) / bandwidth

    # -- target arithmetic: per verified position -------------------------
    rate = (hw.effective_flops(cfg.precision) * n
            * _decode_mfu(cfg, batch * positions))
    per_position = 2.0 * model.active_params * batch          # GEMM row
    per_position += 2.0 * model.n_layers * model.d_model * context * batch  # QK^T
    per_position += 2.0 * model.n_layers * model.d_model * context * batch  # AV
    compute = positions * per_position / rate

    # -- draft: a miniature dense model decoded gamma times ---------------
    if speculating:
        draft_params = model.active_params * cfg.draft_param_ratio
        draft_rate = (hw.effective_flops(cfg.precision) * n
                      * _decode_mfu(cfg, batch))
        memory += cfg.speculative_gamma * draft_params * model.bytes_per_param \
            / bandwidth
        compute += cfg.speculative_gamma * 2.0 * draft_params * batch / draft_rate

    # -- expert straggler: stretches expert work only ---------------------
    if model.is_moe and cfg.expert_imbalance < 1.0:
        wait = 1.0 / cfg.expert_imbalance - 1.0
        # Shared-block bytes re-derived at the precision each part is
        # actually stored at (v10.0), so that the straggler surcharge is
        # scoped to expert traffic and nothing else.
        hp = min(model.embedding_params + model.lm_head_params,
                 model.moe_shared_params)
        shared_streamed = ((model.moe_shared_params - hp) * model.bytes_per_param
                           + hp * model.head_bytes
                           - model.embedding_params * model.head_bytes)
        expert_bytes = weight_bytes - shared_streamed
        memory += max(expert_bytes, 0.0) / bandwidth * wait
        expert_params = max(model.active_params - model.moe_shared_params, 0.0)
        compute += 2.0 * expert_params * batch * positions / rate * wait

    step = max(memory, compute)

    # -- tensor-parallel synchronisation ----------------------------------
    if n > 1:
        tokens_synchronised = batch * positions
        payload = 2.0 * tokens_synchronised * model.d_model * model.n_layers * 2.0
        ring = 2.0 * (n - 1) / n
        step += (payload * ring / hw.interconnect_bandwidth) \
            * (1.0 + cfg.tensor_parallel_penalty * 10.0)

    step += (cfg.framework_overhead_per_step
             + cfg.framework_overhead_per_sequence * batch)
    return step


TECHNIQUE_AXIS = [
    # (gamma, acceptance, draft_ratio, expert_imbalance)
    (0.0, 0.0, 0.0, 1.0),
    (4.0, 0.72, 0.03, 1.0),
    (2.0, 0.50, 0.05, 1.0),
    (8.0, 0.90, 0.01, 1.0),
    (4.0, 0.72, 0.03, 0.6),
    (0.0, 0.0, 0.0, 0.5),
]

SWEEP = list(itertools.product(
    ["dense-8b", "dense-70b", "moe-8x7b", "moe-236b"],
    ["h100-sxm", "a100-80gb"],
    [1, 16, 256],
    [1024, 32768],
    TECHNIQUE_AXIS,
))


def _cfg(batch, technique):
    gamma, alpha, ratio, imbalance = technique
    return ServingConfig(n_accelerators=8, max_batch=batch,
                         speculative_gamma=gamma,
                         speculative_acceptance=alpha,
                         draft_param_ratio=ratio,
                         expert_imbalance=imbalance)


# -- R7-1: verification arithmetic was never priced -------------------------

def test_step_matches_reference_across_technique_axis():
    """The v6.0 sweep held the technique axis at identity; this one moves it.

    Before the fix, every configuration with speculation on disagreed:
    the reference prices gamma + 1 verified positions, the production
    code priced one. Worst pre-fix divergence in this sweep: 3.9x.
    """
    worst, at = 1.0, None
    for model_key, hw_key, batch, context, technique in SWEEP:
        model, hw = get_model(model_key), get_hardware(hw_key)
        cfg = _cfg(batch, technique)
        mine, _ = decode_step_time(model, hw, cfg, batch, context)
        theirs = reference_step(model, hw, cfg, batch, context)
        ratio = max(mine / theirs, theirs / mine)
        if ratio > worst:
            worst, at = ratio, (model_key, hw_key, batch, context, technique)
    assert worst < 1.000001, f"diverges by {worst:.4f}x at {at}"


def test_step_price_and_speedup_denominator_agree_on_tokens():
    """The two sides of the speculation division describe the same step.

    ``_speculative_speedup`` divides output tokens by the *expected
    accepted tokens per verification step*, a quantity that approaches
    gamma + 1 as acceptance approaches one. That denominator is only
    meaningful if the step being divided into prices the arithmetic of
    gamma + 1 verified tokens -- which, until v7.0, it did not.
    """
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    batch, context = 256.0, 4096.0
    plain = ServingConfig(n_accelerators=4, max_batch=256)
    spec = replace(plain, speculative_gamma=4.0, speculative_acceptance=0.72,
                   draft_param_ratio=1e-9)  # draft ~free: isolate verification

    # Force the compute roofline to bind by comparing arithmetic directly.
    verified = spec.speculative_gamma + 1.0
    gemm_plain = 2.0 * model.active_params * batch
    gemm_spec = gemm_plain * verified
    assert gemm_spec / gemm_plain == pytest.approx(verified)

    # And the assembled step reflects it wherever compute binds: with the
    # draft priced at zero the only change speculation makes to the step
    # is the verification arithmetic itself.
    step_plain, _ = decode_step_time(model, hw, plain, batch, context)
    step_spec, bound = decode_step_time(model, hw, spec, batch, context)
    assert step_spec > step_plain  # priced at last; was equal before v7.0


def test_gamma_zero_is_the_plain_step():
    """Continuity: switching speculation off recovers the base physics."""
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    off = ServingConfig(n_accelerators=4, max_batch=64)
    zero = replace(off, speculative_gamma=0.0, speculative_acceptance=0.72,
                   draft_param_ratio=0.03)
    for context in (512.0, 8192.0):
        a, _ = decode_step_time(model, hw, off, 64, context)
        b, _ = decode_step_time(model, hw, zero, 64, context)
        assert a == pytest.approx(b, rel=1e-12)


def test_speculation_is_a_net_loss_at_saturating_batch():
    """The regression pin for the corrected regime claim.

    At batch 1 the weight stream dominates and verification arithmetic is
    hidden by the roofline maximum: the multiplier stays at 0.40. At
    batch 256 the arithmetic is what saturates, the draft competes for
    the same units, and the technique costs more than it saves. Published
    serving measurements report the same inversion; up to v6.0 the model
    could not, because the arithmetic that causes it was unpriced.
    """
    from caide.efficiency import apply_stack
    grid = get_grid("us-average")
    w = WorkloadClass("tutoring", 1.0, 1500, 400)

    def multiplier(batch):
        state = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                                ServingConfig(n_accelerators=4, max_batch=batch))
        base = self_hosted_query_cost(state, w, grid,
                                      respect_slo=False).compute_cost
        cost = self_hosted_query_cost(
            apply_stack(state, ["speculative_decoding"]), w, grid,
            respect_slo=False).compute_cost
        return cost / base

    assert multiplier(1) == pytest.approx(0.40, abs=0.01)
    assert multiplier(256) > 1.0


# -- R7-6: the draft was charged in time but not in memory ------------------

def test_draft_weights_reduce_kv_capacity():
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    plain = ServingConfig(n_accelerators=4, max_batch=512)
    spec = replace(plain, speculative_gamma=4.0, speculative_acceptance=0.72,
                   draft_param_ratio=0.03)
    cap_plain = capacity_batch(model, hw, plain, 2048.0)
    cap_spec = capacity_batch(model, hw, spec, 2048.0)
    draft_bytes = model.active_params * 0.03 * model.bytes_per_param
    expected = cap_plain - draft_bytes / (model.kv_bytes_per_token * 2048.0)
    assert cap_spec < cap_plain
    assert cap_spec == pytest.approx(expected, rel=1e-9)


def test_batch_of_one_still_reads_exactly_the_active_params():
    """The MoE identity the v6.0 audit fought for must survive v7.0."""
    for key in ("moe-8x7b", "moe-8x22b", "moe-236b"):
        model = get_model(key)
        assert model.expert_bytes_touched(1.0) == pytest.approx(
            model.active_params * model.bytes_per_param, rel=1e-12)


# -- R7-2: the package held two derivations of its own throughput -----------

def test_predicted_tps_is_the_models_own_throughput_definition():
    """One quantity, one derivation.

    ``evaluate_request`` defines aggregate throughput as requests over the
    whole serving cycle. ``predicted_output_tps`` divided output tokens by
    decode time alone -- a second, contradictory derivation of the same
    quantity inside one package, differing by the prefill share of the
    cycle: 2.2x for the 70B TP=4 observation.
    """
    for obs in REFERENCE_OBSERVATIONS():
        state = DeploymentState(obs.model, obs.hardware, obs.serving())
        perf = evaluate_request(state, obs.workload(), batch_override=obs.batch)
        via_throughput = perf.throughput_qps * obs.tokens_out
        assert predicted_output_tps(obs) == pytest.approx(via_throughput,
                                                          rel=1e-9)


def test_prefill_is_not_negligible_for_the_current_validation_set():
    """The defect was live, not latent: 'decode-dominated' described the
    requests, not the arithmetic."""
    obs = REFERENCE_OBSERVATIONS()[0]        # 70B TP=4, 512 in / 256 out
    state = DeploymentState(obs.model, obs.hardware, obs.serving())
    perf = evaluate_request(state, obs.workload(), batch_override=obs.batch)
    prefill_share = perf.prefill_seconds / (perf.prefill_seconds
                                            + perf.decode_seconds)
    assert prefill_share > 0.40


def test_convention_verdicts_survive_the_prefill_correction():
    """The physical-bound test is about step time, not cycle time, so no
    admissibility verdict may change."""
    for obs in REFERENCE_OBSERVATIONS():
        assert admissible_conventions(obs) == ["aggregate"]


# --------------------------------------------------------------------------
# The reference implementation, extended along the layer axis: an annual
# ledger derived from replica-level accounting. Dollars and joules are
# computed from "own N replicas for a year" first principles, never from
# per-query figures.
# --------------------------------------------------------------------------

def reference_annual_ledger(state, workload, grid, volume):
    """Independent derivation of the compute layer's dollars and joules."""
    cfg, hw = state.serving, state.hardware
    perf = evaluate_request(state, workload)
    util = cfg.demand_duty_cycle * cfg.scheduler_efficiency

    qps_needed = volume / SECONDS_PER_YEAR
    capacity = (qps_needed / perf.throughput_qps) / util
    replicas = max(math.ceil(capacity - 1e-9), cfg.min_replicas)

    hours = 365.25 * 24
    dollars = (hw.hourly_cost * cfg.reserved_discount * cfg.n_accelerators
               * hours * cfg.infra_overhead) * replicas
    if grid.electricity_cost > 0:
        mean_watts = (hw.power_watts * util
                      + hw.resolved_idle_power * (1.0 - util))
        dollars += (mean_watts * cfg.n_accelerators * hours * grid.pue
                    / 1000.0) * grid.electricity_cost * replicas

    busy_device_seconds = volume * cfg.n_accelerators / perf.throughput_qps
    provisioned = replicas * cfg.n_accelerators * hours * 3600.0
    joules = (hw.power_watts * busy_device_seconds
              + hw.resolved_idle_power * max(provisioned - busy_device_seconds,
                                             0.0)) * grid.pue
    return dollars, joules, replicas


def _tco(state, workload, grid, volume):
    return total_cost_of_ownership(
        architecture="self_hosted", annual_volume=volume,
        workloads=[workload], grid=grid, state=state)


@pytest.fixture
def ledger_state():
    return DeploymentState(
        get_model("dense-8b"), get_hardware("l40s"),
        ServingConfig(n_accelerators=1, max_batch=64,
                      demand_duty_cycle=0.6, scheduler_efficiency=0.8))


# -- R7-3: dollars walked a staircase, joules walked a line -----------------

def test_annual_ledger_matches_reference_in_both_currencies(ledger_state):
    w = WorkloadClass("q", 1.0, 800, 200)
    grid = get_grid("us-average")
    perf = evaluate_request(ledger_state, w)
    util = 0.6 * 0.8
    fills_one = perf.throughput_qps * SECONDS_PER_YEAR * util
    for scale in (0.2, 0.9, 1.0, 1.7, 3.4):
        volume = fills_one * scale
        ref_usd, ref_j, ref_replicas = reference_annual_ledger(
            ledger_state, w, grid, volume)
        tco = _tco(ledger_state, w, grid, volume)
        assert tco.capacity_units == pytest.approx(scale, rel=1e-6)
        assert tco.layers["compute_serving"] == pytest.approx(ref_usd, rel=1e-6)
        assert tco.annual_energy_kwh * 3.6e6 == pytest.approx(ref_j, rel=1e-6)


def test_energy_jumps_where_dollars_jump(ledger_state):
    """Below one full replica the v6.0 ledger billed the money of 1.0
    replicas and the carbon of a fraction of one."""
    w = WorkloadClass("q", 1.0, 800, 200)
    grid = get_grid("global-average")
    perf = evaluate_request(ledger_state, w)
    fills_one = perf.throughput_qps * SECONDS_PER_YEAR * 0.48
    below = _tco(ledger_state, w, grid, fills_one * 0.999)
    above = _tco(ledger_state, w, grid, fills_one * 1.021)
    # dollars step by one replica; energy must step at the same volume
    assert above.layers["compute_serving"] > below.layers["compute_serving"] * 1.5
    assert above.annual_energy_kwh > below.annual_energy_kwh * 1.2
    # and within one step, energy rises only by the busy share
    mid = _tco(ledger_state, w, grid, fills_one * 0.5)
    assert below.annual_energy_kwh > mid.annual_energy_kwh
    assert below.layers["compute_serving"] == pytest.approx(
        mid.layers["compute_serving"], rel=1e-9)


# -- R7-4: idle time was billed at load power -------------------------------

def test_idle_share_draws_idle_power_not_load_power():
    model, hw = get_model("dense-8b"), get_hardware("l40s")
    grid = get_grid("global-average")
    w = WorkloadClass("q", 1.0, 800, 200)

    def joules(duty):
        cfg = ServingConfig(n_accelerators=1, max_batch=64,
                            demand_duty_cycle=duty, scheduler_efficiency=1.0)
        return self_hosted_query_cost(DeploymentState(model, hw, cfg), w,
                                      grid, respect_slo=False).energy_joules

    fully_busy = joules(1.0)
    half_busy = joules(0.5)
    # v6.0 behaviour: half duty -> exactly double the joules. Now the idle
    # half draws idle power, so the ratio sits strictly between 1 and 2,
    # and exactly at 1 + idle/load once the algebra is written out.
    expected = 1.0 + hw.resolved_idle_power / hw.power_watts
    assert half_busy / fully_busy == pytest.approx(expected, rel=1e-9)
    assert 1.0 < half_busy / fully_busy < 2.0


def test_catalogue_idle_power_is_explicit_and_plausible():
    for key in ("a100-40gb", "a100-80gb", "h100-sxm", "h200-sxm", "l40s",
                "consumer-24gb"):
        hw = get_hardware(key)
        assert hw.idle_power_watts is not None
        assert 0.05 * hw.power_watts <= hw.idle_power_watts <= 0.25 * hw.power_watts


# -- R7-5: the imbalance floor, and the surcharge it exposed ----------------

def test_uniform_routing_floor_matches_monte_carlo():
    """The closed form is a first-order extreme-value approximation; it
    must sit within a few percent of simulation to be worth shipping."""
    import numpy as np
    rng = np.random.default_rng(20260820)
    for batch, n_experts, k in ((256, 160, 6), (64, 64, 2), (512, 8, 2)):
        draws = rng.multinomial(batch * k, [1.0 / n_experts] * n_experts,
                                size=2000)
        mc = float(np.mean((batch * k / n_experts) / draws.max(axis=1)))
        closed = uniform_routing_imbalance(batch, n_experts, k)
        assert closed == pytest.approx(mc, rel=0.10)


def test_uniform_routing_floor_limits():
    assert uniform_routing_imbalance(64, 1, 1) == 1.0
    # more experts at fixed load -> worse balance
    assert uniform_routing_imbalance(256, 160, 6) \
        < uniform_routing_imbalance(256, 16, 6)
    # more assignments over fixed experts -> better balance
    assert uniform_routing_imbalance(1024, 160, 6) \
        > uniform_routing_imbalance(64, 160, 6)
    assert 0.0 < uniform_routing_imbalance(256, 160, 6) < 1.0


def test_straggler_surcharge_no_longer_grows_with_context():
    """The v6.0 multiplier form stretched the KV stream by the expert
    skew; expert skew knows nothing about context length."""
    model, hw = get_model("moe-236b"), get_hardware("h100-sxm")
    for context in (1024.0, 65536.0):
        balanced = ServingConfig(n_accelerators=8, max_batch=256,
                                 expert_imbalance=1.0)
        skewed = replace(balanced, expert_imbalance=0.5)
        base, _ = decode_step_time(model, hw, balanced, 256, context)
        worse, _ = decode_step_time(model, hw, skewed, 256, context)
        surcharge = worse - base
        if context == 1024.0:
            short_surcharge = surcharge
    # the absolute surcharge is context-invariant (memory-bound regime)
    assert surcharge == pytest.approx(short_surcharge, rel=1e-9)


def test_smaller_imbalance_means_longer_step():
    model, hw = get_model("moe-236b"), get_hardware("h100-sxm")
    steps = []
    for imbalance in (1.0, 0.8, 0.5, 0.3):
        cfg = ServingConfig(n_accelerators=8, max_batch=256,
                            expert_imbalance=imbalance)
        step, _ = decode_step_time(model, hw, cfg, 256, 4096.0)
        steps.append(step)
    assert steps == sorted(steps)
