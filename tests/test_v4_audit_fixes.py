"""Tests added in v4.0 in response to the v3.0 audit.

Round four used metamorphic testing, numerical conditioning checks,
backward-compatibility probes, tool-performance measurement, and -- the
method that produced both findings -- validation of the model's predicted
throughput against published measurements from real serving frameworks.
"""

from __future__ import annotations

import time

import pytest

from caide import (
    DeploymentState,
    Observation,
    REFERENCE_OBSERVATIONS,
    ScenarioError,
    ServingConfig,
    WorkloadClass,
    api_query_cost,
    evaluate_request,
    fit_calibration,
    get_grid,
    get_hardware,
    get_model,
    get_pricing,
    load_scenario,
    predicted_output_tps,
    self_hosted_query_cost,
    total_cost_of_ownership,
)
from caide.costing import AssuranceProfile, replica_annual_cost
from caide.roofline import capacity_batch

GRID = get_grid("us-average")


# ===========================================================================
# R4-1  batch_override must respect KV-cache capacity
# ===========================================================================

def test_batch_override_is_clamped_to_capacity():
    """v3.0 returned a confident throughput figure for a batch the KV cache
    could not hold, which is the same failure mode the context-overflow
    check exists to prevent one level up."""
    model = get_model("dense-70b").with_precision(1.0)   # FP8, ~66 GiB
    hw = get_hardware("h100-sxm")                        # 80 GiB
    cfg = ServingConfig(n_accelerators=1, max_batch=256, precision="fp8")
    w = WorkloadClass("q", 1.0, 256, 512)

    cap = capacity_batch(model, hw, cfg, w.avg_sequence)
    assert cap < 64, "fixture no longer exercises the clamp"

    perf = evaluate_request(DeploymentState(model, hw, cfg), w,
                            batch_override=10_000)
    assert perf.batch <= cap + 1e-9
    assert perf.batch_truncated


def test_batch_override_within_capacity_is_untouched():
    model, hw = get_model("dense-8b"), get_hardware("h100-sxm")
    cfg = ServingConfig(n_accelerators=1, max_batch=256)
    perf = evaluate_request(DeploymentState(model, hw, cfg),
                            WorkloadClass("q", 1.0, 512, 128),
                            batch_override=32)
    assert perf.batch == pytest.approx(32)
    assert not perf.batch_truncated


def test_truncated_batch_throughput_never_exceeds_untruncated_capacity():
    """Asking for more than fits must not buy throughput."""
    model = get_model("dense-70b").with_precision(1.0)
    hw = get_hardware("h100-sxm")
    cfg = ServingConfig(n_accelerators=1, max_batch=256, precision="fp8")
    w = WorkloadClass("q", 1.0, 256, 512)
    cap = capacity_batch(model, hw, cfg, w.avg_sequence)

    at_cap = evaluate_request(DeploymentState(model, hw, cfg), w,
                              batch_override=cap)
    beyond = evaluate_request(DeploymentState(model, hw, cfg), w,
                              batch_override=cap * 10)
    assert beyond.throughput_qps == pytest.approx(at_cap.throughput_qps,
                                                  rel=1e-9)


# ===========================================================================
# R4-2  Calibration against measured throughput
# ===========================================================================

def test_reference_observations_are_all_feasible():
    for obs in REFERENCE_OBSERVATIONS():
        assert predicted_output_tps(obs) is not None, obs.source


def test_calibration_improves_the_fit_on_published_measurements():
    result = fit_calibration(REFERENCE_OBSERVATIONS())
    assert result.improved
    assert result.within_factor_after >= result.within_factor_before
    assert 0.2 <= result.mbu_scale <= 3.0


def test_uncalibrated_model_is_not_claimed_to_be_within_two_x():
    """The finding that motivated this module: only half of the published
    measurements fall inside the factor-of-two band that v3.0's
    documentation asserted. The test pins the honest number so that a
    future change cannot quietly restore the stronger claim."""
    result = fit_calibration(REFERENCE_OBSERVATIONS())
    assert result.within_factor_before < 1.0


def test_calibration_error_is_symmetric_in_log_space():
    """Kills: fitting on linear residuals instead of log residuals.

    Being 2x too fast and 2x too slow are equally wrong. A linear
    objective scores the first as an error of 1.0 and the second as 0.5,
    so it would systematically bias the fitted correction downward. The
    test asserts the objective itself, not merely that it improved.
    """
    from caide.calibration import _log_rmse
    assert _log_rmse([2.0]) == pytest.approx(_log_rmse([0.5]))
    assert _log_rmse([1.0]) == pytest.approx(0.0)
    # A linear objective would fail both: (2-1)^2 = 1 vs (0.5-1)^2 = 0.25.


def test_calibration_correction_is_scale_symmetric():
    """A model uniformly k-times too fast must be corrected by 1/k, and one
    uniformly k-times too slow by k -- with the same |log| magnitude."""
    from caide.calibration import _log_rmse
    fast = _log_rmse([3.0, 3.0])
    slow = _log_rmse([1 / 3, 1 / 3])
    assert fast == pytest.approx(slow)


def test_calibration_needs_more_than_one_observation():
    with pytest.raises(ValueError, match="at least 2 observations"):
        fit_calibration(REFERENCE_OBSERVATIONS()[:1])


def test_calibration_result_applies_to_a_serving_config():
    result = fit_calibration(REFERENCE_OBSERVATIONS())
    base = ServingConfig(mbu_decode=0.70)
    tuned = result.apply(base)
    assert tuned.mbu_decode == pytest.approx(0.70 * result.mbu_scale)
    assert 0.005 <= tuned.mbu_decode <= 0.98


def test_calibration_clamps_into_the_physical_range():
    """A pathological fit must not produce a utilisation above one."""
    from caide.calibration import CalibrationResult
    absurd = CalibrationResult(mfu_scale=100.0, mbu_scale=100.0,
                               n_observations=2, log_rmse_before=1.0,
                               log_rmse_after=0.5)
    tuned = absurd.apply(ServingConfig())
    assert tuned.mbu_decode <= 0.98
    assert tuned.mfu_prefill <= 0.95


def test_observation_rejects_impossible_measurements():
    model, hw = get_model("dense-8b"), get_hardware("h100-sxm")
    with pytest.raises(ValueError, match="measured_output_tps"):
        Observation(model=model, hardware=hw, n_accelerators=1, batch=8,
                    tokens_in=128, tokens_out=128, measured_output_tps=0.0)
    with pytest.raises(ValueError, match="batch"):
        Observation(model=model, hardware=hw, n_accelerators=1, batch=0,
                    tokens_in=128, tokens_out=128, measured_output_tps=100.0)


def test_predicted_throughput_is_none_when_infeasible():
    """Infeasible is not zero and must not be averaged into a fit."""
    obs = Observation(model=get_model("dense-405b"),
                      hardware=get_hardware("consumer-24gb"),
                      n_accelerators=1, batch=8, tokens_in=128,
                      tokens_out=128, measured_output_tps=100.0)
    assert predicted_output_tps(obs) is None


# ===========================================================================
# Metamorphic relations
# ===========================================================================

@pytest.fixture
def base_state():
    return DeploymentState(
        get_model("dense-8b"), get_hardware("h100-sxm"),
        ServingConfig(n_accelerators=1, max_batch=64,
                      demand_duty_cycle=0.8, scheduler_efficiency=0.7))


def test_doubling_self_consistency_doubles_cost(base_state):
    single = self_hosted_query_cost(
        base_state, WorkloadClass("q", 1.0, 1000, 300), GRID,
        respect_slo=False).compute_cost
    double = self_hosted_query_cost(
        base_state, WorkloadClass("q", 1.0, 1000, 300, self_consistency_k=2),
        GRID, respect_slo=False).compute_cost
    assert double == pytest.approx(2 * single, rel=1e-9)


def test_halving_duty_cycle_doubles_cost(base_state):
    # Revised in v7.0: the exact factor of two held only while idle hours
    # were charged electricity at load power. It still holds exactly where
    # the tariff is bundled into the hourly rate (electricity_cost == 0);
    # on a tariffed grid the idle share now draws idle power, so halving
    # the duty cycle costs strictly less than twice as much.
    from dataclasses import replace
    from caide import get_grid
    w = WorkloadClass("q", 1.0, 1000, 300)
    bundled = get_grid("global-average")
    idle_cfg = replace(base_state.serving, demand_duty_cycle=0.4)
    idle_state = DeploymentState(base_state.model, base_state.hardware,
                                 idle_cfg)

    busy0 = self_hosted_query_cost(base_state, w, bundled,
                                   respect_slo=False).compute_cost
    idle0 = self_hosted_query_cost(idle_state, w, bundled,
                                   respect_slo=False).compute_cost
    assert idle0 == pytest.approx(2 * busy0, rel=1e-9)

    busy = self_hosted_query_cost(base_state, w, GRID,
                                  respect_slo=False).compute_cost
    idle = self_hosted_query_cost(idle_state, w, GRID,
                                  respect_slo=False).compute_cost
    assert 1.8 * busy < idle < 2 * busy


def test_grid_intensity_moves_carbon_but_not_cost(base_state):
    from dataclasses import replace
    w = WorkloadClass("q", 1.0, 1000, 300)
    clean = self_hosted_query_cost(base_state, w, GRID, respect_slo=False)
    dirty_grid = replace(GRID, carbon_intensity=GRID.carbon_intensity * 2)
    dirty = self_hosted_query_cost(base_state, w, dirty_grid, respect_slo=False)
    assert dirty.carbon_kg == pytest.approx(2 * clean.carbon_kg, rel=1e-9)
    assert dirty.compute_cost == pytest.approx(clean.compute_cost, rel=1e-9)


def test_doubling_volume_doubles_linear_layers_only():
    w = WorkloadClass("q", 1.0, 1000, 300)
    assurance = AssuranceProfile(evaluation_annual=100_000, storage_per_query=0.0)

    def layers(volume):
        return total_cost_of_ownership(
            architecture="api", annual_volume=volume, workloads=[w], grid=GRID,
            pricing=get_pricing("api-frontier"), assurance=assurance).layers

    one, two = layers(1e6), layers(2e6)
    assert two["model_access"] == pytest.approx(2 * one["model_access"], rel=1e-9)
    assert two["assurance_governance"] == pytest.approx(
        one["assurance_governance"], rel=1e-9)


def test_doubling_accelerators_doubles_replica_cost(base_state):
    from dataclasses import replace
    one = replica_annual_cost(base_state, GRID)
    two_cfg = replace(base_state.serving, n_accelerators=2)
    two = replica_annual_cost(
        DeploymentState(base_state.model, base_state.hardware, two_cfg), GRID)
    assert two == pytest.approx(2 * one, rel=1e-9)


def test_doubling_api_tariff_doubles_api_cost():
    from dataclasses import replace
    w = WorkloadClass("q", 1.0, 1000, 300)
    cheap = get_pricing("api-frontier")
    dear = replace(cheap, input_per_mtok=cheap.input_per_mtok * 2,
                   output_per_mtok=cheap.output_per_mtok * 2)
    assert api_query_cost(dear, w, GRID).compute_cost == pytest.approx(
        2 * api_query_cost(cheap, w, GRID).compute_cost, rel=1e-9)


# ===========================================================================
# Numerical conditioning
# ===========================================================================

@pytest.mark.parametrize("volume", [1e3, 1e6, 1e9, 1e12, 1e15])
def test_api_unit_cost_is_stable_across_twelve_orders_of_magnitude(volume):
    w = WorkloadClass("q", 1.0, 1500, 400)
    result = total_cost_of_ownership(
        architecture="api", annual_volume=volume, workloads=[w], grid=GRID,
        pricing=get_pricing("api-frontier"))
    assert result.layers["model_access"] / volume == pytest.approx(0.0105,
                                                                   rel=1e-9)


def test_nearly_identical_curves_do_not_manufacture_crossings():
    from caide import find_break_even
    result = find_break_even(lambda v: 1e9 + v * 1e-9,
                             lambda v: 1e9 + v * 1e-9 * (1 + 1e-12),
                             volume_min=1e3, volume_max=1e12, samples=200)
    assert len(result.crossings) <= 2


# ===========================================================================
# Backward compatibility
# ===========================================================================

def test_v1_style_scenario_is_rejected_with_migration_guidance():
    doc = {
        "name": "legacy", "annual_volume": 1e6, "grid": "us-average",
        "workloads": [{"name": "a", "share": 1.0, "tokens_in": 1000,
                       "tokens_out": 300}],
        "architectures": [{"name": "self", "type": "self_hosted",
                           "model": "dense-8b", "hardware": "l40s",
                           "serving": {"n_accelerators": 1,
                                       "target_utilisation": 0.45}}],
    }
    with pytest.raises(ScenarioError, match="demand_duty_cycle"):
        load_scenario(doc)


def test_v2_style_scenario_still_loads():
    doc = {
        "name": "v2", "annual_volume": 1e6, "grid": "us-average",
        "workloads": [{"name": "a", "share": 1.0, "tokens_in": 1000,
                       "tokens_out": 300}],
        "architectures": [{"name": "self", "type": "self_hosted",
                           "model": "dense-8b", "hardware": "l40s",
                           "serving": {"n_accelerators": 1,
                                       "demand_duty_cycle": 0.45}}],
    }
    scenario = load_scenario(doc)
    assert scenario.architectures[0].state.serving.demand_duty_cycle == 0.45


# ===========================================================================
# Tool performance
# ===========================================================================

def test_a_large_scenario_evaluates_quickly():
    """Fifty workload classes against ten architectures is a plausible
    enterprise scenario and must not take minutes."""
    doc = {
        "name": "big", "annual_volume": 1e9, "grid": "us-average",
        "workloads": [{"name": f"w{i}", "share": 1 / 50,
                       "tokens_in": 1000 + i, "tokens_out": 200 + i}
                      for i in range(50)],
        "architectures": [{"name": f"a{i}", "type": "self_hosted",
                           "model": "dense-8b", "hardware": "l40s",
                           "serving": {"n_accelerators": 1}, "stack": "standard"}
                          for i in range(10)],
    }
    start = time.perf_counter()
    results = load_scenario(doc).evaluate_all()
    elapsed = time.perf_counter() - start
    assert len(results) == 10
    assert elapsed < 5.0, f"evaluation took {elapsed:.1f}s"
