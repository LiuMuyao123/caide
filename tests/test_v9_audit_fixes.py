"""Regression tests for the defects found in the v9.0 audit.

Three of this round's findings share one shape: a summary that is only
true when the thing being summarised is contiguous, connected or
complete, applied to something that is none of those.

* the indistinguishable *band* assumed the tie set was an interval (R9-1)
* the sensitivity *ranking* assumed the declared inputs were the model's
  inputs (R9-2)
* the routing *plan* assumed a tier's cost was linear in what it carried,
  and that a declared capacity cap would be honoured (R9-3, R9-4)

The fourth is about the audit trail rather than the code: v8.0 recorded
that no shipped scenario contained a mixture-of-experts architecture.
One does, and has since v6.0. It was unaffected by the v8 routing fix
because it runs at saturating batch, not because it was absent -- and
the recommendation that scope claim generated pointed at a gap that was
already closed, while the real one went unnamed (R9-5).
"""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from caide import (
    DeploymentState,
    ServingConfig,
    WorkloadClass,
    apply_stack,
    find_break_even,
    get_grid,
    get_hardware,
    get_model,
    load_scenario,
    self_hosted_query_cost,
)
from caide.efficiency import PRESET_STACKS
from caide.perturb import (
    REVIEW_COST_FACTORS,
    RECOGNISED_DRAW_KEYS,
    perturbed_cost,
    uncovered_draw_keys,
)
from caide.routing import Tier, optimise_routing, route_greedy

import caide

EXAMPLES = Path(caide.__file__).parent / "examples"


# ==========================================================================
# R9-1: a tie set is a union of intervals, not an interval
# ==========================================================================

def _staircase_result():
    """A line against a staircase -- the shape the module exists for."""
    tuned = [k for k in PRESET_STACKS["aggressive"]
             if k != "speculative_decoding"]
    scenario = load_scenario({
        "name": "granularity", "annual_volume": 1e9, "grid": "us-average",
        "workloads": [{"name": "q", "share": 1.0,
                       "tokens_in": 800, "tokens_out": 200}],
        "architectures": [
            {"name": "api-economy", "type": "api", "pricing": "api-economy"},
            {"name": "selfhost-70b", "type": "self_hosted",
             "model": "dense-70b", "hardware": "h100-sxm",
             "serving": {"n_accelerators": 4, "max_batch": 256,
                         "demand_duty_cycle": 0.55,
                         "scheduler_efficiency": 0.80},
             "stack": tuned}],
    })
    return find_break_even(
        scenario.cost_curve("api-economy"),
        scenario.cost_curve("selfhost-70b"),
        label_a="api-economy", label_b="selfhost-70b",
        volume_min=1e7, volume_max=2e10, samples=900)


def test_tie_set_is_several_windows_not_one_band():
    """The published band spanned four separate windows."""
    r = _staircase_result()
    windows = r.tie_bands(0.05)
    assert len(windows) == 4
    # each is narrow: about a tenth of a decade, not a factor of four
    for lo, hi in windows:
        assert 1.0 < hi / lo < 1.2
    # and they are strictly ordered and disjoint
    for (_, hi), (lo, _) in zip(windows, windows[1:]):
        assert lo > hi


def test_every_reported_window_ties_throughout():
    """The property the v1-v8 summary claimed and did not have."""
    r = _staircase_result()
    points = sorted(r.curve_a)
    for lo, hi in r.tie_bands(0.05):
        inside = [v for v in points if lo <= v <= hi]
        assert len(inside) >= 2
        assert max(r.relative_gap(v) for v in inside) <= 0.05


def test_the_old_span_would_have_failed_that_property():
    """First-to-last of the qualifying points is not a tie region.

    Reconstructed here rather than described, so the regression is a
    fact about this data set and not a claim in a docstring: across the
    old span most scan points exceed the tolerance, some by an order of
    magnitude more.
    """
    r = _staircase_result()
    points = sorted(r.curve_a)
    qualifying = [v for v in points if r.relative_gap(v) <= 0.05]
    old_lo, old_hi = qualifying[0], qualifying[-1]
    spanned = [v for v in points if old_lo <= v <= old_hi]
    violating = [v for v in spanned if r.relative_gap(v) > 0.05]
    assert len(violating) / len(spanned) > 0.7
    assert max(r.relative_gap(v) for v in spanned) > 0.4


def test_tie_band_returns_the_widest_window():
    r = _staircase_result()
    windows = r.tie_bands(0.05)
    band = r.tie_band(0.05)
    assert band in windows
    assert band[1] / band[0] == max(hi / lo for lo, hi in windows)


def test_a_genuine_single_band_is_still_one_band():
    """Two smooth curves that converge and diverge once give one window."""
    result = find_break_even(lambda v: 1.0 + (v - 100.0) ** 2 / 1e4,
                             lambda v: 1.0,
                             volume_min=1.0, volume_max=1e4, samples=400)
    windows = result.tie_bands(0.20)
    assert len(windows) == 1


def test_margin_uses_the_same_denominator_as_relative_gap():
    """One quantity, one definition.

    ``margin_pct`` divided by the mean of the two costs until v9.0 while
    ``relative_gap`` divided by the cheaper one, so the module reported
    the same disagreement as two different numbers. Checked on a step
    where the two denominators are far apart.
    """
    def a(v):
        return 100.0

    def b(v):
        return 10.0 if v < 50.0 else 1000.0

    r = find_break_even(a, b, volume_min=1.0, volume_max=1e3, samples=200)
    assert r.crossings
    c = r.crossings[0]
    cheaper = min(a(c.volume), b(c.volume))
    mean = 0.5 * (a(c.volume) + b(c.volume))
    grid = sorted(r.curve_a)
    lo = max(v for v in grid if v <= c.volume)
    hi = min(v for v in grid if v >= c.volume)
    biggest = max(abs(r.curve_a[lo] - r.curve_b[lo]),
                  abs(r.curve_a[hi] - r.curve_b[hi]))
    assert c.margin_pct == pytest.approx(100.0 * biggest / cheaper, rel=1e-9)
    # the pre-v9 denominator would have reported a materially smaller gap
    assert cheaper < 0.5 * mean
    assert c.margin_pct > 1.5 * (100.0 * biggest / mean)


# ==========================================================================
# R9-2: a ranking over the inputs someone declared
# ==========================================================================

def test_all_three_review_factors_are_perturbable():
    assert set(REVIEW_COST_FACTORS) <= RECOGNISED_DRAW_KEYS


@pytest.fixture
def tutoring():
    return load_scenario(EXAMPLES / "university_tutoring.yaml")


def test_review_minutes_and_wage_move_the_total(tutoring):
    """Both were inert before v9.0: the draw key was ignored, so the
    input was a point mass no tornado chart could show."""
    arch = tutoring.architecture(
        [a.name for a in tutoring.architectures if a.kind == "self_hosted"][0])
    base = perturbed_cost(tutoring, arch, {})
    for key in ("review_minutes_scale", "reviewer_wage_scale"):
        doubled = perturbed_cost(tutoring, arch, {key: 2.0})
        assert doubled > base * 1.2, key
    # the three factors enter as a product, so doubling any one of them
    # moves review cost identically
    a = perturbed_cost(tutoring, arch, {"review_minutes_scale": 2.0})
    b = perturbed_cost(tutoring, arch, {"reviewer_wage_scale": 2.0})
    assert a == pytest.approx(b, rel=1e-9)


def test_review_factors_compose_multiplicatively(tutoring):
    arch = tutoring.architecture(
        [a.name for a in tutoring.architectures if a.kind == "self_hosted"][0])
    base = perturbed_cost(tutoring, arch, {})
    both = perturbed_cost(tutoring, arch, {"review_minutes_scale": 2.0,
                                           "reviewer_wage_scale": 2.0})
    one = perturbed_cost(tutoring, arch, {"review_minutes_scale": 2.0})
    # review cost quadruples; the rest of the ledger is untouched
    assert (both - base) == pytest.approx(3.0 * (one - base), rel=1e-9)


def test_held_fixed_inputs_are_reportable(tutoring):
    """The absence of an input from a tornado chart is invisible in the
    chart. It has to be reported separately, or not at all."""
    held = uncovered_draw_keys(tutoring.uncertainty)
    assert "review_minutes_scale" not in held      # now declared
    assert "reviewer_wage_scale" not in held
    assert "mbu" in held                           # genuinely held fixed
    assert set(held) <= RECOGNISED_DRAW_KEYS


def test_uncovered_is_the_complement_of_declared():
    assert uncovered_draw_keys(RECOGNISED_DRAW_KEYS) == []
    assert set(uncovered_draw_keys([])) == set(RECOGNISED_DRAW_KEYS)


# ==========================================================================
# R9-3 / R9-4: capacity caps and non-separable tier costs
# ==========================================================================

def _classes(*shares):
    return [WorkloadClass(f"w{i}", s, 800, 200)
            for i, s in enumerate(shares)]


def test_max_share_is_honoured():
    """Declared from the first release, read from none of them."""
    workloads = _classes(0.5, 0.3, 0.2)
    tiers = [Tier("cheap", 1.0, cost_fn=lambda w: 0.0001, max_share=0.55),
             Tier("dear", 1.0, cost_fn=lambda w: 0.0010)]
    plan = optimise_routing(workloads, tiers, 1_000_000)
    assert plan.tier_shares["cheap"] <= 0.55 + 1e-12
    assert plan.tier_shares.get("dear", 0.0) > 0.0


def test_uncapped_ladders_take_the_fast_exact_path():
    workloads = _classes(0.6, 0.4)
    tiers = [Tier("cheap", 1.0, cost_fn=lambda w: 0.0001),
             Tier("dear", 1.0, cost_fn=lambda w: 0.0010)]
    plan = optimise_routing(workloads, tiers, 1_000_000)
    assert plan.exact
    assert set(plan.assignment.values()) == {"cheap"}


def test_capped_routing_matches_brute_force():
    """With a cap the per-class greedy choice is no longer optimal, so
    the exact path has to be exact. Checked against full enumeration."""
    import itertools
    workloads = _classes(0.40, 0.25, 0.20, 0.15)
    prices = {"a": 0.0001, "b": 0.0004, "c": 0.0009}
    tiers = [Tier("a", 1.0, cost_fn=lambda w, p=prices["a"]: p, max_share=0.45),
             Tier("b", 1.0, cost_fn=lambda w, p=prices["b"]: p, max_share=0.45),
             Tier("c", 1.0, cost_fn=lambda w, p=prices["c"]: p)]
    volume = 1_000_000.0
    plan = optimise_routing(workloads, tiers, volume)
    assert plan.exact

    best = math.inf
    for combo in itertools.product(tiers, repeat=len(workloads)):
        used = {}
        ok = True
        for w, t in zip(workloads, combo):
            used[t.name] = used.get(t.name, 0.0) + w.share
            if used[t.name] > t.max_share + 1e-12:
                ok = False
                break
        if not ok:
            continue
        cost = sum(w.share * t.cost(w) for w, t in zip(workloads, combo)) * volume
        best = min(best, cost)
    assert plan.annual_cost == pytest.approx(best, rel=1e-9)


def test_class_that_cannot_be_placed_is_reported_not_dropped():
    workloads = _classes(0.7, 0.3)
    tiers = [Tier("only", 1.0, cost_fn=lambda w: 0.0001, max_share=0.5)]
    plan = route_greedy(workloads, tiers, 1_000_000)
    assert plan.unroutable
    assert any("max_share" in n for n in plan.notes)


def test_self_hosted_tier_is_charged_in_whole_replicas():
    """Routing priced a self-hosted tier marginally while the costing
    layer charged whole replicas -- one quantity, two derivations,
    disagreeing by 13x on the shipped public-service scenario."""
    from caide.costing import total_cost_of_ownership

    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    arch = scenario.architecture("selfhost-70b-only")
    small = min(scenario.workloads, key=lambda w: w.share)

    def annual(served, volume):
        shares = sum(w.share for w in served)
        return total_cost_of_ownership(
            architecture="self_hosted", annual_volume=volume * shares,
            workloads=[replace(w, share=w.share / shares) for w in served],
            grid=scenario.grid, state=arch.state, slo=scenario.slo,
        ).layers["compute_serving"]

    marginal = Tier("t", 1.0, cost_fn=lambda w: self_hosted_query_cost(
        arch.state, w, scenario.grid, scenario.slo).compute_cost)
    stepped = replace(marginal, annual_cost_fn=annual)

    volume = scenario.annual_volume
    a = route_greedy([small], [marginal], volume).annual_cost
    b = route_greedy([small], [stepped], volume).annual_cost
    assert b > 5.0 * a
    assert not marginal.is_separable or stepped.is_separable is False


def test_stepped_tier_cost_raises_a_note():
    from caide.costing import total_cost_of_ownership
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    arch = scenario.architecture("selfhost-70b-only")
    small = min(scenario.workloads, key=lambda w: w.share)

    def annual(served, volume):
        shares = sum(w.share for w in served)
        return total_cost_of_ownership(
            architecture="self_hosted", annual_volume=volume * shares,
            workloads=[replace(w, share=w.share / shares) for w in served],
            grid=scenario.grid, state=arch.state, slo=scenario.slo,
        ).layers["compute_serving"]

    tier = Tier("t", 1.0,
                cost_fn=lambda w: self_hosted_query_cost(
                    arch.state, w, scenario.grid, scenario.slo).compute_cost,
                annual_cost_fn=annual)
    plan = route_greedy([small], [tier], scenario.annual_volume)
    assert any("whole replicas" in n for n in plan.notes)


def test_separability_is_declared_not_guessed():
    plain = Tier("a", 1.0, cost_fn=lambda w: 1.0)
    capped = replace(plain, max_share=0.4)
    stepped = replace(plain, annual_cost_fn=lambda served, v: 1.0)
    assert plain.is_separable
    assert not capped.is_separable
    assert not stepped.is_separable


# ==========================================================================
# R9-5: the scope claim in the v8 report, checked
# ==========================================================================

def test_a_shipped_scenario_does_contain_a_moe_architecture():
    """v8.0 recorded that no published sweep used one. One has since
    v6.0, running a speculative stack."""
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    moe = [a for a in scenario.architectures
           if a.state is not None and a.state.model.is_moe]
    assert moe, "public_helpline no longer ships an MoE architecture"
    assert "speculative_decoding" in moe[0].stack


def test_the_shipped_moe_architecture_runs_at_saturating_batch():
    """Why the v8 fix left its numbers alone: at this batch the router
    already reaches every expert, so gamma + 1 tokens reach no more."""
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    arch = next(a for a in scenario.architectures
                if a.state is not None and a.state.model.is_moe)
    model = arch.state.model
    batch = arch.state.serving.max_batch
    assert model.expert_bytes_touched(batch * 5.0) == pytest.approx(
        model.expert_bytes_touched(batch), rel=1e-6)


def test_speculation_inverts_on_an_moe_target_at_small_batch():
    """The gap the v8 recommendation should have named: no published
    configuration ran an MoE model below saturating batch, which is the
    only regime where the routing defect was visible."""
    grid = get_grid("us-average")
    workload = WorkloadClass("tutoring", 1.0, 1500, 400)
    hw = get_hardware("h100-sxm")

    def curve(model_key, n_acc):
        out = []
        for batch in (1, 16, 256):
            state = DeploymentState(get_model(model_key), hw,
                                    ServingConfig(n_accelerators=n_acc,
                                                  max_batch=batch))
            base = self_hosted_query_cost(state, workload, grid,
                                          respect_slo=False).compute_cost
            spec = self_hosted_query_cost(
                apply_stack(state, ["speculative_decoding"]), workload, grid,
                respect_slo=False).compute_cost
            out.append(spec / base)
        return out

    dense = curve("dense-70b", 4)
    moe = curve("moe-8x7b", 2)
    # dense: best at batch 1, monotone worsening
    assert dense == sorted(dense)
    assert dense[0] < 0.45
    # MoE: worth nothing at batch 1, best in the middle
    assert moe[0] > 0.95
    assert moe[1] < moe[0] and moe[1] < moe[2]
