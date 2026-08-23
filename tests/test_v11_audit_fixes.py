"""Regression tests for the defects found in the v11.0 audit.

Version 10.0 made feasibility the ranking criterion. This round audited
what feasibility now rests on, and found that both of its inputs were
unsound in the same way: a quantity computed on one path and consumed on
another.

The quality index was reduced by the scenario layer for thirteen
techniques and by the state transform for two, so the library path lost
the first thirteen and double-charged the last two (R11-1, R11-2). The
latency check was recorded as *passed* for architectures whose latency is
not modelled at all, so an API candidate satisfied by construction a
constraint every self-hosted candidate had to pass on evidence (R11-3).
And ``latency_sensitive`` -- named alongside ``quality_floor`` in the v9
report as the same family, and left behind when v10 fixed its sibling --
was still ignored (R11-4).
"""

import pytest

from caide import (
    DeploymentState,
    ServingConfig,
    WorkloadClass,
    apply_stack,
    get_grid,
    get_hardware,
    get_model,
    load_scenario,
)
from caide.calibration import (
    FRAMEWORK_OVERHEAD_HISTORY,
    FRAMEWORK_OVERHEAD_REFERENCE,
    FRAMEWORK_OVERHEAD_SENSITIVITY,
)
from caide.costing import total_cost_of_ownership
from caide.efficiency import TECHNIQUES, get_technique, stack_quality_delta
from caide.report import scenario_digest


def _state(model="dense-70b", batch=64, accelerators=4):
    return DeploymentState(get_model(model), get_hardware("h100-sxm"),
                           ServingConfig(n_accelerators=accelerators,
                                         max_batch=batch))


# ==========================================================================
# R11-1: the quality consequence is part of the state transformation
# ==========================================================================

def test_apply_stack_carries_the_quality_cost():
    """A technique is a function from state to state. Until v11.0 that
    held for every attribute except the one v10.0 had just made
    decisive."""
    base = _state()
    for keys in (["int4"], ["int8"], ["semantic_caching"],
                 ["int4", "semantic_caching", "kv_fp8"]):
        after = apply_stack(base, keys)
        expected = base.model.quality_index * (1.0 + stack_quality_delta(keys))
        assert after.model.quality_index == pytest.approx(expected, rel=1e-12)
        assert after.model.quality_index < base.model.quality_index


def test_lossless_techniques_leave_quality_alone():
    """The catalogue's split is the point: speculative decoding and paged
    attention are exact, so their delta is zero and must stay zero."""
    base = _state()
    for key in ("speculative_decoding", "paged_attention",
                "continuous_batching", "flash_attention", "chunked_prefill",
                "prefix_caching", "committed_use"):
        assert get_technique(key).quality_delta == 0.0, key
        after = apply_stack(base, [key])
        assert after.model.quality_index == base.model.quality_index


def test_every_technique_declares_a_quality_delta():
    for key in TECHNIQUES:
        delta = get_technique(key).quality_delta
        assert -1.0 < delta <= 0.0, key


def test_quality_composes_on_retention_not_on_loss():
    base = _state()
    stacked = apply_stack(base, ["int4", "semantic_caching"])
    retained = (1 - 0.010) * (1 - 0.012)
    assert stacked.model.quality_index == pytest.approx(
        base.model.quality_index * retained, rel=1e-12)
    # composing losses additively would give a strictly smaller number
    assert retained > 1 - (0.010 + 0.012)


# ==========================================================================
# R11-2: distillation was charged twice
# ==========================================================================

def test_distillation_quality_is_derived_once():
    """``_distil`` derives the student's quality from its geometry -- the
    derivation CAIDE prefers to a quoted constant. The catalogue also
    quoted a constant, and the scenario layer applied it on top: 0.880
    became 0.817 and then 0.776 for a half-size student."""
    base = _state()
    for key, ratio in (("distillation_50", 0.5), ("distillation_25", 0.25)):
        assert get_technique(key).quality_delta == 0.0
        after = apply_stack(base, [key])
        expected = base.model.quality_index * (0.55 + 0.45 * ratio ** 0.25)
        assert after.model.quality_index == pytest.approx(expected, rel=1e-9)


def test_distillation_through_a_scenario_matches_the_library_path():
    """The two paths agreed on every other attribute and disagreed on
    this one, which is the shape of the defect."""
    scenario = load_scenario({
        "name": "d", "annual_volume": 1e6, "grid": "us-average",
        "workloads": [{"name": "q", "share": 1.0,
                       "tokens_in": 800, "tokens_out": 200}],
        "architectures": [{
            "name": "student", "type": "self_hosted", "model": "dense-70b",
            "hardware": "h100-sxm",
            "serving": {"n_accelerators": 4, "max_batch": 64},
            "stack": ["distillation_50"]}],
    })
    via_scenario = scenario.evaluate_all()["student"].quality_index
    via_library = apply_stack(_state(), ["distillation_50"]).model.quality_index
    assert via_scenario == pytest.approx(via_library, rel=1e-12)
    assert via_scenario == pytest.approx(0.8170, abs=5e-4)


def test_a_scenario_may_still_declare_an_explicit_penalty():
    """The override survives: an organisation's own evaluation of a
    fine-tune is not something a technique catalogue can derive."""
    scenario = load_scenario({
        "name": "d", "annual_volume": 1e6, "grid": "us-average",
        "workloads": [{"name": "q", "share": 1.0,
                       "tokens_in": 800, "tokens_out": 200}],
        "architectures": [{
            "name": "a", "type": "self_hosted", "model": "dense-70b",
            "hardware": "h100-sxm", "quality_penalty": -0.2,
            "serving": {"n_accelerators": 4, "max_batch": 64}}],
    })
    result = scenario.evaluate_all()["a"]
    assert result.quality_index == pytest.approx(0.88 * 0.8, rel=1e-9)


# ==========================================================================
# R11-3: an unevaluated constraint is not a satisfied one
# ==========================================================================

def _two_architecture_scenario(ttft=0.001, tpot=0.0001, sensitive=True):
    return load_scenario({
        "name": "x", "annual_volume": 1e6, "grid": "us-average",
        "slo": {"ttft_seconds": ttft, "tpot_seconds": tpot, "enforce": True},
        "workloads": [{"name": "q", "share": 1.0, "tokens_in": 800,
                       "tokens_out": 200, "latency_sensitive": sensitive}],
        "architectures": [
            {"name": "api", "type": "api", "pricing": "api-frontier"},
            {"name": "sh", "type": "self_hosted", "model": "dense-70b",
             "hardware": "h100-sxm",
             "serving": {"n_accelerators": 4, "max_batch": 64}}],
    })


def test_api_latency_is_unevaluated_not_met():
    """A physically impossible objective. The self-hosted candidate
    misses it; the API candidate cannot be checked, and through v10.0
    reported a pass."""
    results = _two_architecture_scenario().evaluate_all()
    assert results["sh"].slo_violations == ["q"]
    assert results["sh"].slo_unevaluated == []
    assert results["api"].slo_violations == []
    assert results["api"].slo_unevaluated == ["q"]


def test_unevaluated_does_not_mean_infeasible():
    """Reported, not assumed either way -- the same treatment v10 gave a
    quality shortfall, which is named and never priced."""
    results = _two_architecture_scenario().evaluate_all()
    assert results["api"].feasible
    assert not results["api"].fully_evaluated
    assert results["sh"].fully_evaluated
    assert any("not evaluated" in n for n in results["api"].notes)


def test_no_slo_means_nothing_to_evaluate():
    scenario = load_scenario({
        "name": "x", "annual_volume": 1e6, "grid": "us-average",
        "workloads": [{"name": "q", "share": 1.0,
                       "tokens_in": 800, "tokens_out": 200}],
        "architectures": [{"name": "api", "type": "api",
                           "pricing": "api-frontier"}],
    })
    result = scenario.evaluate_all()["api"]
    assert result.slo_unevaluated == []
    assert result.fully_evaluated


@pytest.mark.parametrize("name", ["university_tutoring", "public_helpline"])
def test_shipped_api_architectures_are_not_fully_evaluated(name):
    from pathlib import Path
    import caide
    scenario = load_scenario(
        Path(caide.__file__).parent / "examples" / f"{name}.yaml")
    results = scenario.evaluate_all()
    api = [r for k, r in results.items() if "api" in k]
    assert api and all(not r.fully_evaluated for r in api)
    hosted = [r for k, r in results.items() if "api" not in k]
    assert all(r.fully_evaluated for r in hosted)


# ==========================================================================
# R11-4: the sibling constraint v10 left behind
# ==========================================================================

def test_a_latency_insensitive_class_does_not_disqualify():
    """v9 named ``quality_floor`` and ``latency_sensitive`` together as
    per-class constraints read only by the routing path. v10 wired up the
    first and left the second."""
    results = _two_architecture_scenario(sensitive=False).evaluate_all()
    assert results["sh"].slo_violations == []
    assert results["sh"].feasible
    assert any("latency-insensitive" in n for n in results["sh"].notes)


def test_the_miss_is_still_recorded_when_it_does_not_disqualify():
    results = _two_architecture_scenario(sensitive=False).evaluate_all()
    note = next(n for n in results["sh"].notes if "latency-insensitive" in n)
    assert "q" in note and "misses the latency objective" in note


def test_a_sensitive_class_still_disqualifies():
    results = _two_architecture_scenario(sensitive=True).evaluate_all()
    assert results["sh"].slo_violations == ["q"]
    assert not results["sh"].feasible


def test_mixed_sensitivity_disqualifies_only_on_the_sensitive_class():
    scenario = load_scenario({
        "name": "x", "annual_volume": 1e6, "grid": "us-average",
        "slo": {"ttft_seconds": 0.001, "tpot_seconds": 0.0001,
                "enforce": True},
        "workloads": [
            {"name": "fast", "share": 0.5, "tokens_in": 800,
             "tokens_out": 200, "latency_sensitive": True},
            {"name": "batchy", "share": 0.5, "tokens_in": 800,
             "tokens_out": 200, "latency_sensitive": False}],
        "architectures": [{"name": "sh", "type": "self_hosted",
                           "model": "dense-70b", "hardware": "h100-sxm",
                           "serving": {"n_accelerators": 4,
                                       "max_batch": 64}}],
    })
    result = scenario.evaluate_all()["sh"]
    assert result.slo_violations == ["fast"]
    assert any("batchy" in n for n in result.notes)


# ==========================================================================
# R11-5: a fit with zero degrees of freedom measures the residual
# ==========================================================================

def test_the_overhead_constants_declare_what_they_are():
    assert FRAMEWORK_OVERHEAD_REFERENCE["degrees_of_freedom"] == 0
    assert FRAMEWORK_OVERHEAD_REFERENCE["residual_absorber"] is True
    assert "modelling choice" in FRAMEWORK_OVERHEAD_REFERENCE["status"]


def test_the_drift_history_is_recorded():
    """Five physics corrections moved these constants and
    ``per_sequence_seconds`` returned to where it started: the v7 prefill
    correction pushed it down, the v10 weight-stream correction pushed it
    back. A residual that can absorb a correction and then absorb its
    reversal is measuring the model, not the framework."""
    versions = [row[0] for row in FRAMEWORK_OVERHEAD_HISTORY]
    assert versions == sorted(versions, key=lambda v: [int(x) for x in v.split(".")])
    per_sequence = [row[2] for row in FRAMEWORK_OVERHEAD_HISTORY]
    assert per_sequence[0] == per_sequence[-1]
    assert per_sequence[1] < per_sequence[0]
    assert FRAMEWORK_OVERHEAD_REFERENCE["per_step_seconds"] == \
        FRAMEWORK_OVERHEAD_HISTORY[-1][1]
    assert FRAMEWORK_OVERHEAD_REFERENCE["per_sequence_seconds"] == \
        FRAMEWORK_OVERHEAD_HISTORY[-1][2]


def test_the_sensitivity_interval_brackets_every_value_ever_used():
    for key, index in (("per_step_seconds", 1), ("per_sequence_seconds", 2)):
        low, high = FRAMEWORK_OVERHEAD_SENSITIVITY[key]
        assert low <= high
        for row in FRAMEWORK_OVERHEAD_HISTORY:
            assert low <= row[index] <= high


# ==========================================================================
# R11-6: the provider energy figure is an input
# ==========================================================================

def _api_scenario(energy=None):
    doc = {
        "name": "x", "annual_volume": 1e6, "grid": "us-average",
        "workloads": [{"name": "q", "share": 1.0,
                       "tokens_in": 800, "tokens_out": 200}],
        "architectures": [{"name": "api", "type": "api",
                           "pricing": "api-frontier"}],
    }
    if energy is not None:
        doc["provider_energy_wh_per_ktok"] = energy
    return load_scenario(doc)


def test_provider_energy_is_a_scenario_input():
    """Every API carbon and water figure came from one function default
    that no caller overrode and no scenario could set, so the provenance
    digest could not reach it and no distribution could move it."""
    base = _api_scenario().evaluate_all()["api"]
    tripled = _api_scenario(0.90).evaluate_all()["api"]
    assert tripled.annual_carbon_kg == pytest.approx(
        3.0 * base.annual_carbon_kg, rel=1e-9)
    assert tripled.annual_water_l == pytest.approx(
        3.0 * base.annual_water_l, rel=1e-9)


def test_provider_energy_reaches_the_digest():
    assert scenario_digest(_api_scenario(0.90)) != scenario_digest(_api_scenario())


def test_provider_energy_default_is_unchanged():
    assert _api_scenario().provider_energy_wh_per_ktok == 0.30


def test_provider_energy_does_not_touch_a_self_hosted_ledger():
    state = _state()
    workloads = [WorkloadClass("q", 1.0, 800, 200)]
    kw = dict(architecture="self_hosted", annual_volume=1e6,
              workloads=workloads, grid=get_grid("us-average"), state=state)
    a = total_cost_of_ownership(provider_energy_wh_per_ktok=0.3, **kw)
    b = total_cost_of_ownership(provider_energy_wh_per_ktok=9.9, **kw)
    assert a.annual_energy_kwh == pytest.approx(b.annual_energy_kwh, rel=1e-12)
