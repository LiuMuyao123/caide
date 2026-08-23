"""Tests for scenario parsing, routing, uncertainty and the CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from caide import (
    ScenarioError,
    Tier,
    WorkloadClass,
    example_scenario,
    load_scenario,
    lognormal,
    monte_carlo,
    optimise_routing,
    point,
    route_greedy,
    sensitivity,
    triangular,
    uniform,
)
from caide.cli import main

from importlib.resources import files as _files
EXAMPLES = Path(str(_files("caide") / "examples"))


# ===========================================================================
# scenario parsing
# ===========================================================================

def test_example_scenario_round_trips():
    scenario = load_scenario(example_scenario())
    assert scenario.architectures
    results = scenario.evaluate_all()
    assert all(math.isfinite(r.total) for r in results.values())


def test_scenario_loads_from_yaml_string():
    text = yaml.safe_dump(example_scenario())
    assert load_scenario(text).name == "minimal-example"


@pytest.mark.parametrize("name", ["university_tutoring",
                                  "hospital_documentation",
                                  "public_helpline"])
def test_shipped_examples_are_valid_and_runnable(name):
    scenario = load_scenario(EXAMPLES / f"{name}.yaml")
    scenario.validate()
    results = scenario.evaluate_all()
    assert results
    for r in results.values():
        assert math.isfinite(r.total) and r.total > 0


@pytest.mark.parametrize("name", ["university_tutoring",
                                  "hospital_documentation",
                                  "public_helpline"])
def test_shipped_examples_imply_plausible_review_workloads(name):
    """A scenario demanding more reviewer hours than an organisation could
    staff is arithmetic, not planning. The shipped examples must pass the
    check they exist to demonstrate."""
    scenario = load_scenario(EXAMPLES / f"{name}.yaml")
    result = next(iter(scenario.evaluate_all().values()))
    assert result.review_fte < 250


def test_shares_that_do_not_sum_are_rejected():
    doc = example_scenario()
    doc["workloads"][0]["share"] = 0.6
    with pytest.raises(ScenarioError, match="sum to"):
        load_scenario(doc)


def test_negative_token_count_is_rejected_with_path():
    doc = example_scenario()
    doc["workloads"][0]["tokens_in"] = -100
    with pytest.raises(ScenarioError, match=r"workloads\[0\]"):
        load_scenario(doc)


def test_unknown_model_preset_lists_the_alternatives():
    doc = example_scenario()
    doc["architectures"][1]["model"] = "dense-9000b"
    with pytest.raises(ScenarioError, match="dense-70b"):
        load_scenario(doc)


def test_unknown_architecture_type_is_rejected():
    doc = example_scenario()
    doc["architectures"][0]["type"] = "carrier-pigeon"
    with pytest.raises(ScenarioError, match="expected 'api' or 'self_hosted'"):
        load_scenario(doc)


def test_missing_required_field_names_the_field():
    doc = example_scenario()
    del doc["annual_volume"]
    with pytest.raises(ScenarioError, match="annual_volume"):
        load_scenario(doc)


def test_model_preset_can_be_overridden_inline():
    doc = example_scenario()
    doc["architectures"][1]["model"] = {"preset": "dense-8b", "n_layers": 48}
    scenario = load_scenario(doc)
    assert scenario.architectures[1].state.model.n_layers == 48


def test_oversized_model_produces_a_warning_not_a_crash():
    doc = example_scenario()
    doc["architectures"][1]["model"] = "dense-405b"
    doc["architectures"][1]["hardware"] = "l40s"
    warnings = load_scenario(doc).validate()
    assert any("do not fit" in w for w in warnings)


def test_malformed_yaml_is_reported_as_such():
    with pytest.raises(ScenarioError, match="invalid YAML"):
        load_scenario("name: [unclosed\n  bad: :")


def test_digest_is_stable_and_sensitive():
    from caide.report import scenario_digest
    a = load_scenario(example_scenario())
    b = load_scenario(example_scenario())
    assert scenario_digest(a) == scenario_digest(b)

    doc = example_scenario()
    doc["annual_volume"] *= 2
    assert scenario_digest(load_scenario(doc)) != scenario_digest(a)


# ===========================================================================
# routing
# ===========================================================================

@pytest.fixture
def ladder():
    return [
        Tier("small", quality_index=0.5, cost_fn=lambda w: 0.0001),
        Tier("medium", quality_index=0.8, cost_fn=lambda w: 0.0010),
        Tier("large", quality_index=1.0, cost_fn=lambda w: 0.0100),
    ]


def test_routing_respects_quality_floors(ladder):
    workloads = [WorkloadClass("easy", 0.5, 500, 100, quality_floor=0.3),
                 WorkloadClass("hard", 0.5, 500, 100, quality_floor=0.95)]
    plan = route_greedy(workloads, ladder, 1_000_000)
    assert plan.assignment["easy"] == "small"
    assert plan.assignment["hard"] == "large"


def test_routing_beats_single_tier_when_floors_differ(ladder):
    workloads = [WorkloadClass("easy", 0.9, 500, 100, quality_floor=0.3),
                 WorkloadClass("hard", 0.1, 500, 100, quality_floor=0.95)]
    plan = route_greedy(workloads, ladder, 1_000_000)
    all_large = 0.0100
    assert plan.per_query_cost < all_large


def test_unmeetable_quality_floor_is_reported_not_silently_downgraded(ladder):
    workloads = [WorkloadClass("impossible", 1.0, 500, 100, quality_floor=2.0)]
    plan = route_greedy(workloads, ladder, 1000)
    assert not plan.feasible
    assert "impossible" in plan.unroutable


def test_tier_fixed_costs_can_make_one_tier_optimal():
    """When opening a tier is expensive, using two cheap tiers can cost
    more than putting everything on one."""
    tiers = [
        Tier("cheap", 0.9, cost_fn=lambda w: 0.0001, annual_fixed_cost=900_000),
        Tier("dear", 1.0, cost_fn=lambda w: 0.0002, annual_fixed_cost=10_000),
    ]
    workloads = [WorkloadClass("a", 0.5, 500, 100),
                 WorkloadClass("b", 0.5, 500, 100, quality_floor=0.95)]
    plan = optimise_routing(workloads, tiers, annual_volume=1_000_000)
    assert plan.tiers_opened == ("dear",)


def test_optimiser_never_loses_to_greedy():
    tiers = [
        Tier("a", 0.9, cost_fn=lambda w: 0.0004, annual_fixed_cost=50_000),
        Tier("b", 1.0, cost_fn=lambda w: 0.0005, annual_fixed_cost=50_000),
    ]
    workloads = [WorkloadClass("x", 0.6, 500, 100),
                 WorkloadClass("y", 0.4, 500, 100, quality_floor=0.95)]
    greedy = route_greedy(workloads, tiers, 1_000_000)
    best = optimise_routing(workloads, tiers, 1_000_000)
    assert best.annual_cost <= greedy.annual_cost + 1e-6


def test_empty_ladder_is_rejected():
    with pytest.raises(ValueError, match="at least one tier"):
        optimise_routing([WorkloadClass("a", 1.0, 100, 100)], [], 1000)


# ===========================================================================
# uncertainty
# ===========================================================================

def test_monte_carlo_is_reproducible_under_a_seed():
    dists = [lognormal("x", 1.0, 0.3)]
    a = monte_carlo(lambda d: d["x"] * 100, dists, n_samples=500, seed=7)
    b = monte_carlo(lambda d: d["x"] * 100, dists, n_samples=500, seed=7)
    assert np.allclose(a.samples, b.samples)


def test_monte_carlo_recovers_a_known_distribution():
    result = monte_carlo(lambda d: d["x"], [lognormal("x", 10.0, 0.5)],
                         n_samples=20_000, seed=3)
    assert result.percentile(50) == pytest.approx(10.0, rel=0.05)


def test_failed_draws_are_counted_not_hidden():
    def flaky(draw):
        if draw["x"] > 1.0:
            raise RuntimeError("infeasible")
        return draw["x"]

    result = monte_carlo(flaky, [uniform("x", 0.0, 2.0)], n_samples=1000, seed=1)
    assert result.n_failed > 300
    assert result.valid.size < 1000


def test_point_distribution_is_excluded_from_sensitivity():
    result = monte_carlo(lambda d: d["a"] + d["b"],
                         [uniform("a", 0, 1), point("b", 5.0)],
                         n_samples=500, seed=2)
    names = [e.name for e in sensitivity(result)]
    assert names == ["a"]


def test_sensitivity_ranks_the_dominant_driver_first():
    result = monte_carlo(lambda d: 100 * d["big"] + 0.01 * d["small"],
                         [uniform("big", 0, 1), uniform("small", 0, 1)],
                         n_samples=3000, seed=5)
    entries = sensitivity(result)
    assert entries[0].name == "big"
    assert entries[0].contribution > 0.9


def test_sensitivity_detects_a_negative_relationship():
    result = monte_carlo(lambda d: -3.0 * d["x"], [uniform("x", 0, 1)],
                         n_samples=1500, seed=6)
    assert sensitivity(result)[0].spearman < -0.95


def test_probability_below_reads_as_a_comparison():
    result = monte_carlo(lambda d: d["x"], [uniform("x", 0.0, 10.0)],
                         n_samples=8000, seed=9)
    assert result.probability_below(2.5) == pytest.approx(0.25, abs=0.03)


def test_invalid_distribution_parameters_are_rejected():
    with pytest.raises(ValueError):
        triangular("t", low=5, mode=1, high=10)
    with pytest.raises(ValueError):
        lognormal("l", median=-1, sigma=0.2)
    with pytest.raises(ValueError):
        uniform("u", low=10, high=1)


# ===========================================================================
# CLI
# ===========================================================================

def test_cli_catalog_json_is_parseable(capsys):
    assert main(["catalog", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dense-70b" in payload["models"]


def test_cli_init_then_run(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    assert main(["init", str(path)]) == 0
    assert path.exists()
    assert main(["run", str(path), "--samples", "0"]) == 0
    assert "cheapest" in capsys.readouterr().out


def test_cli_init_refuses_to_clobber(tmp_path, capsys):
    path = tmp_path / "s.yaml"
    main(["init", str(path)])
    assert main(["init", str(path)]) == 2
    assert main(["init", str(path), "--force"]) == 0


def test_cli_validate_rejects_bad_scenario(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    doc = example_scenario()
    doc["workloads"][0]["share"] = 0.3
    path.write_text(yaml.safe_dump(doc))
    assert main(["validate", str(path)]) == 2


def test_cli_breakeven_runs_on_example(capsys):
    assert main(["breakeven", str(EXAMPLES / "university_tutoring.yaml"),
                 "-a", "api-frontier", "-b", "selfhost-70b"]) == 0
    assert "break-even" in capsys.readouterr().out


def test_cli_breakeven_rejects_identical_architectures(capsys):
    path = str(EXAMPLES / "university_tutoring.yaml")
    assert main(["breakeven", path, "-a", "api-frontier", "-b", "api-frontier"]) == 2


def test_cli_sweep_reports_a_spread(capsys):
    assert main(["sweep", str(EXAMPLES / "university_tutoring.yaml"),
                 "--architecture", "selfhost-70b",
                 "--technique", "speculative_decoding"]) == 0
    out = capsys.readouterr().out
    assert "spread across batch" in out


def test_cli_route_produces_an_assignment(capsys):
    assert main(["route", str(EXAMPLES / "university_tutoring.yaml")]) == 0
    assert "blended" in capsys.readouterr().out


def test_cli_run_writes_a_full_report(tmp_path, capsys):
    out = tmp_path / "report"
    assert main(["run", str(EXAMPLES / "university_tutoring.yaml"),
                 "--out", str(out), "--samples", "60", "--layers"]) == 0
    for name in ("report.md", "report.html", "results.csv"):
        assert (out / name).exists(), name
    html = (out / "report.html").read_text()
    assert "data:image/png;base64," in html      # figures must be embedded
    assert "<img" in html
