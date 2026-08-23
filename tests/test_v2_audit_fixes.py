"""Tests added in v2.0 in response to the v1.0 audit.

The first three exist because mutation testing showed the v1.0 suite could
not detect three specific defect classes. Each names the mutant it kills,
so that a future simplification of the assertion is visibly a regression
in coverage rather than a tidy-up.
"""

from __future__ import annotations

import math

import pytest

from caide import (
    DeploymentState,
    ModelSpec,
    ServingConfig,
    ScenarioError,
    WorkloadClass,
    apply_stack,
    evaluate_request,
    example_scenario,
    get_grid,
    get_hardware,
    get_model,
    load_scenario,
    total_cost_of_ownership,
)
from caide.cli import main
from caide.costing import AssuranceProfile, replica_annual_cost
from caide.perturb import RECOGNISED_DRAW_KEYS, perturbed_cost, unrecognised_draw_keys


# ===========================================================================
# Mutation-killing tests (audit finding M-2)
# ===========================================================================

def test_kv_cache_counts_both_key_and_value_absolutely():
    """Kills: dropping the factor 2 from kv_bytes_per_token.

    The v1.0 test compared multi-head against grouped-query attention, so a
    common factor cancelled and halving the cache went undetected. The KV
    cache decides the feasible batch and therefore where the decode roofline
    turns over, so an absolute check is required, not a ratio.
    """
    m = ModelSpec("t", n_params_total=8e9, n_layers=32, d_model=4096,
                  n_heads=32, n_kv_heads=8, bytes_per_kv_element=2.0)
    # 2 (K and V) x 32 layers x 8 kv heads x 128 head dim x 2 bytes
    expected = 2 * 32 * 8 * 128 * 2
    assert m.kv_bytes_per_token == expected == 131_072


def test_kv_cache_scales_with_every_geometric_factor():
    """Each factor in the KV formula must actually be present."""
    base = ModelSpec("b", 8e9, n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8)
    deeper = ModelSpec("d", 8e9, n_layers=64, d_model=4096, n_heads=32, n_kv_heads=8)
    wider_kv = ModelSpec("k", 8e9, n_layers=32, d_model=4096, n_heads=32,
                         n_kv_heads=16)
    fp8_cache = ModelSpec("q", 8e9, n_layers=32, d_model=4096, n_heads=32,
                          n_kv_heads=8, bytes_per_kv_element=1.0)

    assert deeper.kv_bytes_per_token == pytest.approx(2 * base.kv_bytes_per_token)
    assert wider_kv.kv_bytes_per_token == pytest.approx(2 * base.kv_bytes_per_token)
    assert fp8_cache.kv_bytes_per_token == pytest.approx(base.kv_bytes_per_token / 2)


def test_partial_replica_demand_is_rounded_up_not_nearest(grid_fixture=None):
    """Kills: replicas = round(capacity) instead of ceil(capacity).

    Rounding to nearest would let a deployment needing 1.4 replicas be
    charged for one. Integral capacity is what makes the self-hosted cost
    curve a staircase, which is in turn what produces the multi-crossing
    break-even structure the paper reports.
    """
    grid = get_grid("us-average")
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                            ServingConfig(n_accelerators=1, max_batch=256,
                                          demand_duty_cycle=0.8,
                                          scheduler_efficiency=0.75))
    workloads = [WorkloadClass("q", 1.0, 1000, 300)]
    one = replica_annual_cost(state, grid)

    def compute(volume: float) -> float:
        return total_cost_of_ownership(
            architecture="self_hosted", annual_volume=volume,
            workloads=workloads, grid=grid, state=state,
        ).layers["compute_serving"]

    # Find the volume filling exactly one replica, then ask for 40% more.
    lo, hi = 1e3, 1e12
    for _ in range(90):
        mid = math.sqrt(lo * hi)
        if compute(mid) <= one * 1.0001:
            lo = mid
        else:
            hi = mid
    at_capacity = lo

    assert compute(at_capacity) == pytest.approx(one, rel=1e-3)
    # 1.4 replicas of demand must be charged as 2, not rounded down to 1.
    assert compute(at_capacity * 1.4) == pytest.approx(2 * one, rel=1e-3)
    assert compute(at_capacity * 2.05) == pytest.approx(3 * one, rel=1e-3)


def test_review_cost_scales_linearly_with_volume():
    """Kills: omitting the annual_volume factor from the review term.

    Human review dominates total cost in every shipped example, so a
    missing volume multiplier would understate the total by six orders of
    magnitude while still returning a plausible-looking number.
    """
    grid = get_grid("us-average")
    workloads = [WorkloadClass("w", 1.0, 1000, 300,
                               review_rate=0.5, review_minutes=6.0)]
    assurance = AssuranceProfile(reviewer_hourly_cost=60.0,
                                 storage_per_query=0.0)

    def assurance_layer(volume: float) -> float:
        from caide import get_pricing
        return total_cost_of_ownership(
            architecture="api", annual_volume=volume, workloads=workloads,
            grid=grid, pricing=get_pricing("api-economy"), assurance=assurance,
        ).layers["assurance_governance"]

    at_1m = assurance_layer(1_000_000)
    at_10m = assurance_layer(10_000_000)

    # 0.5 x 6 min x $60/h = $3.00 per query
    assert at_1m == pytest.approx(3_000_000, rel=1e-6)
    assert at_10m == pytest.approx(10 * at_1m, rel=1e-6)


# ===========================================================================
# Duty cycle split (audit finding M-1)
# ===========================================================================

def test_efficiency_stack_cannot_raise_demand_duty_cycle():
    """No serving optimisation creates traffic that was never sent."""
    cfg = ServingConfig(n_accelerators=4, max_batch=256,
                        demand_duty_cycle=0.42, scheduler_efficiency=0.45)
    state = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"), cfg)
    for stack in ("baseline_serving", "standard", "aggressive", "maximal"):
        after = apply_stack(state, stack).serving
        assert after.demand_duty_cycle == pytest.approx(0.42)
        assert after.effective_utilisation <= 0.42 + 1e-9


def test_scheduler_efficiency_improves_with_continuous_batching():
    cfg = ServingConfig(demand_duty_cycle=0.5, scheduler_efficiency=0.40)
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"), cfg)
    after = apply_stack(state, ["continuous_batching"]).serving
    assert after.scheduler_efficiency > cfg.scheduler_efficiency
    assert after.demand_duty_cycle == cfg.demand_duty_cycle


def test_declared_duty_cycle_changes_cost_even_with_a_full_stack():
    """The v1.0 defect: three different demand assumptions produced one cost."""
    grid = get_grid("us-average")
    w = WorkloadClass("q", 1.0, 1500, 400)
    costs = []
    for duty in (0.20, 0.42, 0.85):
        cfg = ServingConfig(n_accelerators=4, max_batch=256,
                            demand_duty_cycle=duty, scheduler_efficiency=0.45)
        st = apply_stack(
            DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"), cfg),
            "aggressive")
        from caide import self_hosted_query_cost
        costs.append(self_hosted_query_cost(st, w, grid,
                                            respect_slo=False).compute_cost)
    assert costs[0] > costs[1] > costs[2]
    assert costs[0] / costs[2] == pytest.approx(0.85 / 0.20, rel=0.02)


def test_target_utilisation_remains_readable_as_the_product():
    cfg = ServingConfig(demand_duty_cycle=0.6, scheduler_efficiency=0.5)
    assert cfg.effective_utilisation == pytest.approx(0.30)
    assert cfg.target_utilisation == pytest.approx(0.30)


def test_legacy_target_utilisation_in_a_scenario_is_rejected_with_guidance():
    doc = example_scenario()
    doc["architectures"][1]["serving"]["target_utilisation"] = 0.42
    with pytest.raises(ScenarioError, match="demand_duty_cycle"):
        load_scenario(doc)


# ===========================================================================
# Scenario robustness (audit findings H-2, M-3)
# ===========================================================================

def test_missing_scenario_file_says_so():
    """v1.0 reported 'scenario root must be a mapping' for a mistyped path."""
    with pytest.raises(ScenarioError, match="scenario file not found"):
        load_scenario("exmaples/university_tutoring.yaml")


def test_missing_absolute_path_says_so():
    with pytest.raises(ScenarioError, match="scenario file not found"):
        load_scenario("/nonexistent/path/scenario.yaml")


def test_yaml_document_string_still_loads():
    import yaml
    assert load_scenario(yaml.safe_dump(example_scenario())).name \
        == "minimal-example"


def test_unknown_root_key_is_reported():
    doc = example_scenario()
    doc["totally_unknown_section"] = {"foo": "bar"}
    warnings = load_scenario(doc).validate()
    assert any("totally_unknown_section" in w for w in warnings)


def test_misspelled_workload_field_suggests_the_correction():
    doc = example_scenario()
    doc["workloads"][0]["review_minuts"] = 5
    warnings = load_scenario(doc).validate()
    assert any("review_minuts" in w and "review_minutes" in w for w in warnings)


def test_unknown_architecture_key_is_reported():
    doc = example_scenario()
    doc["architectures"][0]["unknown_key"] = 1
    assert any("unknown_key" in w for w in load_scenario(doc).validate())


# ===========================================================================
# Context overflow (audit finding L-3)
# ===========================================================================

def test_context_overflow_is_flagged_not_silently_priced():
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                            ServingConfig())
    perf = evaluate_request(state, WorkloadClass("huge", 1.0, 10_000_000, 10))
    assert perf.context_overflow
    assert not perf.feasible
    assert not perf.slo_met


def test_normal_request_is_not_flagged():
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                            ServingConfig())
    perf = evaluate_request(state, WorkloadClass("ok", 1.0, 1000, 200))
    assert not perf.context_overflow
    assert perf.feasible


# ===========================================================================
# Public perturbation API (audit finding M-6)
# ===========================================================================

def test_perturbed_cost_is_importable_from_a_public_module():
    from caide import perturbed_cost as exported
    assert exported is perturbed_cost


def test_perturbation_scales_volume():
    scenario = load_scenario(example_scenario())
    arch = scenario.architectures[0]
    base = perturbed_cost(scenario, arch, {})
    doubled = perturbed_cost(scenario, arch, {"volume_scale": 2.0})
    assert doubled > base


def test_perturbation_of_accelerator_price_is_inert_on_an_api_architecture():
    """Documents the trap behind audit finding H-3: a near-zero sensitivity
    here is a property of the architecture, not a finding about prices."""
    scenario = load_scenario(example_scenario())
    api = next(a for a in scenario.architectures if a.kind == "api")
    cheap = perturbed_cost(scenario, api, {"accelerator_hourly": 0.01})
    dear = perturbed_cost(scenario, api, {"accelerator_hourly": 100.0})
    assert cheap == pytest.approx(dear)


def test_perturbation_of_accelerator_price_moves_a_self_hosted_architecture():
    scenario = load_scenario(example_scenario())
    host = next(a for a in scenario.architectures if a.kind == "self_hosted")
    cheap = perturbed_cost(scenario, host, {"accelerator_hourly": 0.10})
    dear = perturbed_cost(scenario, host, {"accelerator_hourly": 10.0})
    assert dear > cheap


def test_unrecognised_draw_keys_are_reported():
    assert unrecognised_draw_keys(["volume_scale", "made_up"]) == ["made_up"]
    assert "demand_duty_cycle" in RECOGNISED_DRAW_KEYS


# ===========================================================================
# Bundled examples (audit finding H-1)
# ===========================================================================

def test_examples_are_packaged_not_repository_only():
    from importlib.resources import files
    root = files("caide") / "examples"
    names = {p.name for p in root.iterdir()}
    assert "university_tutoring.yaml" in names
    assert "reproduce_paper.py" in names


@pytest.mark.parametrize("name", ["university_tutoring",
                                  "hospital_documentation",
                                  "public_helpline"])
def test_packaged_scenarios_load_from_package_data(name):
    from importlib.resources import files
    text = (files("caide") / "examples" / f"{name}.yaml").read_text(encoding="utf-8")
    scenario = load_scenario(text)
    assert scenario.evaluate_all()


def test_cli_examples_lists(capsys):
    assert main(["examples"]) == 0
    out = capsys.readouterr().out
    assert "university_tutoring.yaml" in out


def test_cli_examples_extracts_and_the_result_runs(tmp_path, capsys):
    assert main(["examples", "--extract", str(tmp_path)]) == 0
    target = tmp_path / "examples" / "university_tutoring.yaml"
    assert target.exists()
    capsys.readouterr()
    assert main(["run", str(target), "--samples", "0"]) == 0
    assert "cheapest" in capsys.readouterr().out


def test_cli_examples_does_not_clobber_without_force(tmp_path, capsys):
    main(["examples", "--extract", str(tmp_path)])
    capsys.readouterr()
    assert main(["examples", "--extract", str(tmp_path)]) == 0
    assert "skipped" in capsys.readouterr().out
