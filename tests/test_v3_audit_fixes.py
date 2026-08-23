"""Tests added in v3.0 in response to the v2.0 audit.

The v2 audit closed every v1 finding, so round three used methods the
earlier rounds had not: property-based search over the parameter space,
external validation against hardware bandwidth ceilings, adversarial
scenario construction, and a documentation-versus-code drift check. The
tests below lock in what those methods found.
"""

from __future__ import annotations

import math

import pytest

from caide import (
    DeploymentState,
    ServingConfig,
    WorkloadClass,
    apply_stack,
    evaluate_request,
    get_grid,
    get_hardware,
    get_model,
    load_scenario,
    self_hosted_query_cost,
)
from caide.roofline import _achievable_bandwidth, _decode_mfu, decode_step_time
from caide.scenario import example_scenario
from caide.specs import IMPLAUSIBLE_ABOVE, implausible

GRID = get_grid("us-average")


# ===========================================================================
# F3-1  Decode FLOP utilisation rises with batch
# ===========================================================================

def test_decode_mfu_rises_with_batch():
    """v2.0 used a constant, which put the memory-to-compute transition at
    an artificially low batch and understated quantisation's benefit there."""
    cfg = ServingConfig(mfu_prefill=0.45, decode_mfu_half_batch=64.0)
    values = [_decode_mfu(cfg, b) for b in (1, 8, 64, 256, 1024)]
    assert values == sorted(values)
    assert values[0] < 0.02                    # batch 1 is a GEMV
    assert values[2] == pytest.approx(0.225)   # half of prefill at B_half
    assert values[-1] < cfg.mfu_prefill        # never exceeds prefill


def test_decode_mfu_is_configurable():
    lazy = ServingConfig(decode_mfu_half_batch=256.0)
    eager = ServingConfig(decode_mfu_half_batch=16.0)
    assert _decode_mfu(eager, 64) > _decode_mfu(lazy, 64)


def test_non_positive_half_batch_is_rejected():
    with pytest.raises(ValueError, match="decode_mfu_half_batch"):
        ServingConfig(decode_mfu_half_batch=0.0)


def test_memory_to_compute_transition_moves_to_a_higher_batch():
    """The behavioural consequence of the fix: a 70B model at batch 256 with
    a retrieval-sized context stays memory bound rather than flipping to
    compute bound, which is what measured serving stacks report."""
    m, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    cfg = ServingConfig(n_accelerators=4, max_batch=256)
    _, bound = decode_step_time(m, hw, cfg, batch=256, context_length=1700)
    assert bound == "memory"


def test_quantisation_still_helps_at_production_batch():
    """With a constant decode MFU the model reported no benefit at all from
    INT4 at batch 256, because it had already declared the step compute
    bound. Guard against regressing to that."""
    m, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    cfg = ServingConfig(n_accelerators=4, max_batch=256)
    full, _ = decode_step_time(m, hw, cfg, 256, 1700)
    quant, _ = decode_step_time(m.with_precision(0.5), hw, cfg, 256, 1700)
    assert quant < full * 0.95


@pytest.mark.parametrize("model_key,hw_key,n_acc", [
    ("dense-8b", "h100-sxm", 1),
    ("dense-70b", "h100-sxm", 4),
    ("dense-70b", "a100-80gb", 4),
    ("dense-8b", "l40s", 1),
])
def test_predicted_throughput_respects_the_bandwidth_ceiling(model_key, hw_key,
                                                             n_acc):
    """External validation: decode throughput can never exceed what the
    memory system can stream. A model that violates this is not optimistic,
    it is wrong."""
    m, hw = get_model(model_key), get_hardware(hw_key)
    cfg = ServingConfig(n_accelerators=n_acc, max_batch=256)
    state = DeploymentState(m, hw, cfg)
    perf = evaluate_request(state, WorkloadClass("q", 1.0, 512, 256),
                            batch_override=256)
    predicted_tokens_per_second = 256 * 256 / perf.decode_seconds
    ceiling = _achievable_bandwidth(hw, cfg) / m.weight_bytes * 256
    assert predicted_tokens_per_second <= ceiling


# ===========================================================================
# F3-3  Magnitude plausibility
# ===========================================================================

def test_implausible_returns_none_for_ordinary_values():
    assert implausible("infra_overhead", 1.35) is None
    assert implausible("self_consistency_k", 3) is None
    assert implausible("not_a_tracked_field", 1e9) is None


@pytest.mark.parametrize("field,value", [
    ("self_consistency_k", 1_000_000),
    ("infra_overhead", 1000.0),
    ("carbon_intensity", 1e6),
    ("review_minutes", 10_000.0),
    ("pue", 12.0),
])
def test_implausible_flags_absurd_magnitudes(field, value):
    msg = implausible(field, value)
    assert msg is not None
    assert field in msg
    assert "accepted" in msg          # possible, not impossible


def test_scenario_warns_about_implausible_workload_values():
    doc = example_scenario()
    doc["workloads"][0]["self_consistency_k"] = 500
    warnings = load_scenario(doc).validate()
    assert any("self_consistency_k" in w for w in warnings)


def test_scenario_warns_about_implausible_serving_values():
    doc = example_scenario()
    doc["architectures"][1]["serving"]["infra_overhead"] = 50.0
    warnings = load_scenario(doc).validate()
    assert any("infra_overhead" in w for w in warnings)


def test_implausible_values_are_still_computed():
    """Warned about, not refused: a sensitivity sweep must be able to reach
    the tails, and refusing would make the tool less useful without making
    it more correct."""
    doc = example_scenario()
    doc["workloads"][0]["self_consistency_k"] = 500
    scenario = load_scenario(doc)
    results = scenario.evaluate_all()
    assert all(math.isfinite(r.total) and r.total > 0 for r in results.values())


def test_every_plausibility_ceiling_is_positive():
    assert all(v > 0 for v in IMPLAUSIBLE_ABOVE.values())


# ===========================================================================
# F3-4  Scenario round trip and portable reports
# ===========================================================================

@pytest.mark.parametrize("name", ["university_tutoring",
                                  "hospital_documentation",
                                  "public_helpline"])
def test_scenario_round_trip_preserves_every_result(name):
    from importlib.resources import files
    text = (files("caide") / "examples" / f"{name}.yaml").read_text(encoding="utf-8")
    original = load_scenario(text)
    restored = load_scenario(original.to_yaml())

    before = original.evaluate_all()
    after = restored.evaluate_all()
    assert set(before) == set(after)
    for key in before:
        assert after[key].total == pytest.approx(before[key].total, rel=1e-9)
        assert after[key].quality_index == pytest.approx(
            before[key].quality_index, rel=1e-9)


def test_round_trip_is_idempotent():
    scenario = load_scenario(example_scenario())
    once = load_scenario(scenario.to_yaml())
    twice = load_scenario(once.to_yaml())
    assert twice.to_yaml() == once.to_yaml()


def test_round_trip_preserves_stack_applied_state():
    """The stack is baked into the state and must not be re-applied on
    reload; its quality cost is carried explicitly instead."""
    doc = example_scenario()
    # "standard" costs no quality at all, so it cannot detect the bug this
    # test exists for. Use a stack that quantises and caches.
    doc["architectures"][1]["stack"] = "aggressive"
    scenario = load_scenario(doc)
    host = next(a for a in scenario.architectures if a.kind == "self_hosted")
    # Revised in v11.0: the stack's quality cost is carried in the state
    # rather than alongside it, so what must survive a round trip is the
    # model's quality index, not a separate penalty. The property this
    # test exists for is unchanged -- reload must not re-apply the stack.
    assert host.state.model.quality_index < get_model("dense-8b").quality_index

    restored = load_scenario(scenario.to_yaml())
    host2 = restored.architecture(host.name)
    assert host2.stack == ()                       # not re-applied
    assert host2.state.model.quality_index == pytest.approx(
        host.state.model.quality_index)
    assert host2.state.model.bytes_per_param == host.state.model.bytes_per_param
    assert host2.state.serving.semantic_cache_hit == pytest.approx(
        host.state.serving.semantic_cache_hit)


def test_markdown_report_embeds_a_runnable_scenario():
    """A digest proves the inputs did not change; only the inputs themselves
    let the recipient re-run the analysis."""
    import tempfile
    from pathlib import Path
    from caide.report import ReportBundle, write_markdown

    scenario = load_scenario(example_scenario())
    bundle = ReportBundle(scenario, seed=1)
    bundle.tco = scenario.evaluate_all()

    out = write_markdown(bundle, Path(tempfile.mkdtemp()) / "r.md")
    text = out.read_text(encoding="utf-8")
    assert "Reproducing this analysis" in text

    embedded = text.split("```yaml")[1].split("```")[0]
    rebuilt = load_scenario(embedded)
    before = scenario.evaluate_all()
    after = rebuilt.evaluate_all()
    for key in before:
        assert after[key].total == pytest.approx(before[key].total, rel=1e-9)


def test_html_report_embeds_the_scenario():
    import tempfile
    from pathlib import Path
    from caide.report import ReportBundle, write_html

    scenario = load_scenario(example_scenario())
    bundle = ReportBundle(scenario, seed=1)
    bundle.tco = scenario.evaluate_all()
    html = write_html(bundle, Path(tempfile.mkdtemp()) / "r.html").read_text(encoding="utf-8")
    assert "Reproducing this analysis" in html
    assert "annual_volume" in html


# ===========================================================================
# F3-5  Public API surface
# ===========================================================================

def test_every_public_callable_is_documented():
    import inspect
    import caide

    undocumented = [
        name for name in caide.__all__
        if not name.startswith("__")
        and (inspect.isfunction(getattr(caide, name))
             or inspect.isclass(getattr(caide, name)))
        and not (getattr(caide, name).__doc__ or "").strip()
    ]
    assert undocumented == []


# ===========================================================================
# Invariants promoted from the property-based audit
# ===========================================================================

@pytest.mark.parametrize("stack", ["none", "baseline_serving", "standard",
                                   "aggressive", "maximal"])
@pytest.mark.parametrize("duty", [0.05, 0.42, 0.99])
def test_effective_utilisation_never_exceeds_demand(stack, duty):
    cfg = ServingConfig(n_accelerators=2, demand_duty_cycle=duty,
                        scheduler_efficiency=0.9)
    state = DeploymentState(get_model("dense-8b"), get_hardware("h100-sxm"), cfg)
    after = apply_stack(state, stack).serving
    assert after.demand_duty_cycle == pytest.approx(duty)
    assert after.effective_utilisation <= duty + 1e-12


@pytest.mark.parametrize("model_key", ["dense-1b", "dense-8b", "dense-70b",
                                       "moe-8x7b", "moe-236b"])
@pytest.mark.parametrize("batch", [1, 16, 256])
def test_roofline_outputs_stay_physical(model_key, batch):
    state = DeploymentState(get_model(model_key), get_hardware("h100-sxm"),
                            ServingConfig(n_accelerators=8, max_batch=batch))
    perf = evaluate_request(state, WorkloadClass("q", 1.0, 1024, 256))
    for value in (perf.ttft, perf.tpot, perf.latency, perf.throughput_qps):
        assert not math.isnan(value)
        assert value >= 0
    if perf.feasible:
        assert perf.latency == pytest.approx(perf.ttft + perf.decode_seconds)


def test_cost_is_non_increasing_in_demand_duty_cycle():
    w = WorkloadClass("q", 1.0, 1500, 400)

    def cost(duty: float) -> float:
        cfg = ServingConfig(n_accelerators=4, max_batch=128,
                            demand_duty_cycle=duty, scheduler_efficiency=0.6)
        return self_hosted_query_cost(
            DeploymentState(get_model("dense-8b"), get_hardware("h100-sxm"), cfg),
            w, GRID, respect_slo=False).compute_cost

    values = [cost(d) for d in (0.1, 0.3, 0.6, 0.9)]
    assert values == sorted(values, reverse=True)
