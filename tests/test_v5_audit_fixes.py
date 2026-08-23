"""Tests added in v5.0 in response to the v4.0 audit.

Round five expanded the external validation to a second accelerator class,
as the v4 audit recommended, and it did expose the systematic bias that
audit predicted. It also introduced structural sensitivity analysis: not
"is the number right" but "does the paper's conclusion survive plausible
alternative modelling choices".
"""

from __future__ import annotations

import pytest

from caide import (
    CONVENTIONS,
    DeploymentState,
    EXCLUDED_OBSERVATIONS,
    Observation,
    REFERENCE_OBSERVATIONS,
    ServingConfig,
    WorkloadClass,
    apply_stack,
    evaluate_request,
    get_grid,
    get_hardware,
    get_model,
    self_hosted_query_cost,
)
from caide.roofline import decode_step_time

GRID = get_grid("us-average")


# ===========================================================================
# R5-1  Measurement convention
# ===========================================================================

def test_convention_must_be_stated_explicitly():
    """A benchmark reporting 'throughput' without saying which it means
    differs by the batch size -- 8x for a small concurrency test, 256x for
    a large one. There is no safe default to guess, so an unknown value is
    rejected rather than coerced."""
    with pytest.raises(ValueError, match="convention must be one of"):
        Observation(model=get_model("dense-8b"),
                    hardware=get_hardware("h100-sxm"), n_accelerators=1,
                    batch=64, tokens_in=512, tokens_out=256,
                    measured_output_tps=460, convention="throughput")


def test_per_request_convention_is_normalised_by_batch():
    obs = Observation(model=get_model("dense-8b"),
                      hardware=get_hardware("h100-sxm"), n_accelerators=1,
                      batch=64, tokens_in=512, tokens_out=256,
                      measured_output_tps=23.4, convention="per_request")
    assert obs.aggregate_output_tps == pytest.approx(23.4 * 64)


def test_aggregate_convention_passes_through():
    obs = Observation(model=get_model("dense-8b"),
                      hardware=get_hardware("h100-sxm"), n_accelerators=1,
                      batch=64, tokens_in=512, tokens_out=256,
                      measured_output_tps=1500.0, convention="aggregate")
    assert obs.aggregate_output_tps == pytest.approx(1500.0)


def test_the_two_conventions_differ_by_exactly_the_batch():
    """The size of the error this field prevents."""
    kwargs = dict(model=get_model("dense-8b"),
                  hardware=get_hardware("h100-sxm"), n_accelerators=1,
                  batch=128, tokens_in=512, tokens_out=256,
                  measured_output_tps=1000.0)
    per_request = Observation(**kwargs, convention="per_request")
    aggregate = Observation(**kwargs, convention="aggregate")
    assert (per_request.aggregate_output_tps
            / aggregate.aggregate_output_tps) == pytest.approx(128)


def test_every_reference_observation_states_its_convention():
    for obs in REFERENCE_OBSERVATIONS():
        assert obs.convention in CONVENTIONS
        assert obs.source, "an observation without a source cannot be audited"


def test_excluded_observations_are_recorded_with_reasons():
    """Exclusions are part of the evidence. Silently dropping the data that
    does not fit is how a validation set stops being one.

    v6.0 readmitted one of the three v5.0 exclusions after showing its
    ambiguity was decidable, so the count now spans both tuples: what the
    test protects is that every admission decision is on the record with a
    reason, not that the exclusion list only ever grows.
    """
    from caide.calibration import READMITTED_OBSERVATIONS
    decisions = tuple(EXCLUDED_OBSERVATIONS) + tuple(READMITTED_OBSERVATIONS)
    assert len(decisions) >= 3
    for label, reason in decisions:
        assert label and len(reason) > 40


def test_calibration_uses_normalised_throughput():
    """A per-request observation must not be fitted as if it were aggregate."""
    from caide.calibration import predicted_output_tps
    base = REFERENCE_OBSERVATIONS()[0]
    per_request = Observation(
        model=base.model, hardware=base.hardware,
        n_accelerators=base.n_accelerators, batch=base.batch,
        tokens_in=base.tokens_in, tokens_out=base.tokens_out,
        measured_output_tps=base.measured_output_tps / base.batch,
        precision=base.precision, convention="per_request")
    predicted = predicted_output_tps(base)
    assert (predicted / per_request.aggregate_output_tps) == pytest.approx(
        predicted / base.aggregate_output_tps)


# ===========================================================================
# R5-2  Framework overhead
# ===========================================================================

def test_framework_overhead_defaults_to_a_pure_hardware_roofline():
    assert ServingConfig().framework_overhead_per_step == 0.0


def test_framework_overhead_lowers_throughput():
    model, hw = get_model("dense-8b"), get_hardware("a100-80gb")
    w = WorkloadClass("q", 1.0, 512, 256)

    def tps(overhead):
        cfg = ServingConfig(n_accelerators=1, max_batch=8,
                            framework_overhead_per_step=overhead)
        perf = evaluate_request(DeploymentState(model, hw, cfg), w,
                                batch_override=8)
        return 8 * 256 / perf.decode_seconds

    assert tps(0.0) > tps(0.010) > tps(0.020)


def test_framework_overhead_can_become_the_binding_resource():
    model, hw = get_model("dense-1b"), get_hardware("h100-sxm")
    cfg = ServingConfig(n_accelerators=1, framework_overhead_per_step=0.5)
    _, bound = decode_step_time(model, hw, cfg, batch=1, context_length=128)
    assert bound == "framework"


def test_negative_framework_overhead_is_rejected():
    with pytest.raises(ValueError, match="framework_overhead_per_step"):
        ServingConfig(framework_overhead_per_step=-0.001)


def test_overhead_matters_more_at_small_batch():
    """Per-step CPU cost is amortised over the batch, so its relative
    weight falls as concurrency rises. This is why single-stream benchmarks
    test the framework and high-concurrency ones test the hardware."""
    model, hw = get_model("dense-8b"), get_hardware("a100-80gb")
    w = WorkloadClass("q", 1.0, 512, 256)

    def slowdown(batch):
        fast = ServingConfig(n_accelerators=1, max_batch=batch)
        slow = ServingConfig(n_accelerators=1, max_batch=batch,
                             framework_overhead_per_step=0.010)
        a = evaluate_request(DeploymentState(model, hw, fast), w,
                             batch_override=batch).decode_seconds
        b = evaluate_request(DeploymentState(model, hw, slow), w,
                             batch_override=batch).decode_seconds
        return b / a

    assert slowdown(1) > slowdown(64)


# ===========================================================================
# R5-3  Structural sensitivity of the published conclusions
# ===========================================================================

MODEL_VARIANTS = [
    ("baseline", {}),
    ("b_half_16", {"decode_mfu_half_batch": 16.0}),
    ("b_half_256", {"decode_mfu_half_batch": 256.0}),
    ("mfu_low", {"mfu_prefill": 0.30}),
    ("mfu_high", {"mfu_prefill": 0.60}),
    ("mbu_low", {"mbu_decode": 0.50}),
    ("mbu_high", {"mbu_decode": 0.90}),
    ("tp_ideal", {"tensor_parallel_penalty": 0.0}),
    ("tp_poor", {"tensor_parallel_penalty": 0.08}),
]


def _multiplier(technique: str, batch: int, **overrides) -> float:
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    w = WorkloadClass("t", 1.0, 1500, 400)
    cfg = ServingConfig(n_accelerators=4, max_batch=batch, **overrides)
    state = DeploymentState(model, hw, cfg)
    base = self_hosted_query_cost(state, w, GRID,
                                  respect_slo=False).compute_cost
    tuned = self_hosted_query_cost(apply_stack(state, [technique]), w, GRID,
                                   respect_slo=False).compute_cost
    return tuned / base


@pytest.mark.parametrize("name,overrides", MODEL_VARIANTS)
def test_regime_dependence_survives_alternative_modelling(name, overrides):
    """The paper's central claim must not depend on any one parameter
    choice. If it did, a reviewer changing a plausible default would
    overturn it."""
    low = _multiplier("speculative_decoding", 1, **overrides)
    high = _multiplier("speculative_decoding", 256, **overrides)
    assert high / low > 1.3, f"{name}: spread collapsed to {high/low:.2f}"


@pytest.mark.parametrize("name,overrides", MODEL_VARIANTS)
def test_published_int4_constant_stays_badly_wrong(name, overrides):
    """The 153% discrepancy against a published 0.65x constant must be
    robust to the modelling choices, or it is an artefact of them."""
    low = _multiplier("int4", 1, **overrides)
    high = _multiplier("int4", 256, **overrides)
    worst = max(abs(0.65 - low) / low, abs(0.65 - high) / high)
    assert worst > 0.5, f"{name}: worst-case error fell to {worst:.0%}"


def test_semantic_caching_stays_batch_invariant_under_every_variant():
    """The counterexample matters as much as the examples: a method that
    found every technique regime-dependent would be finding its own
    machinery, not the physics."""
    for _, overrides in MODEL_VARIANTS:
        low = _multiplier("semantic_caching", 1, **overrides)
        high = _multiplier("semantic_caching", 256, **overrides)
        assert high == pytest.approx(low, rel=1e-6)
