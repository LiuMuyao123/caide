"""Tests for costing, efficiency composition, break-even and scaling."""

from __future__ import annotations

import math

import pytest

from caide import (
    AssuranceProfile,
    CostLayer,
    DeploymentState,
    GridSpec,
    PricingSpec,
    ServingConfig,
    WorkloadClass,
    api_query_cost,
    apply_stack,
    find_break_even,
    get_grid,
    get_hardware,
    get_model,
    get_pricing,
    self_hosted_query_cost,
    stack_quality_delta,
    total_cost_of_ownership,
)
from caide.breakeven import dominance_intervals
from caide.costing import replica_annual_cost
from caide.scaling import ScalingAssumptions, estimate_elasticity, project


@pytest.fixture
def state():
    return DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                           ServingConfig(n_accelerators=1, max_batch=128))


@pytest.fixture
def grid():
    return get_grid("us-average")


@pytest.fixture
def workloads():
    return [WorkloadClass("chat", 0.7, 1000, 300),
            WorkloadClass("summary", 0.3, 3000, 500)]


# ===========================================================================
# per-query costing
# ===========================================================================

def test_api_cost_matches_tariff_arithmetic():
    p = PricingSpec("t", input_per_mtok=2.0, output_per_mtok=8.0)
    w = WorkloadClass("q", 1.0, 1_000_000, 1_000_000)
    cost = api_query_cost(p, w, GridSpec()).compute_cost
    assert cost == pytest.approx(10.0)


def test_cached_input_tier_reduces_cost():
    p = PricingSpec("t", 3.0, 15.0, cached_input_per_mtok=0.3)
    w = WorkloadClass("q", 1.0, 10_000, 1_000)
    full = p.query_cost(w.tokens_in, w.tokens_out, cached_fraction=0.0)
    cached = p.query_cost(w.tokens_in, w.tokens_out, cached_fraction=0.9)
    assert cached < full


def test_self_consistency_multiplies_cost(state, grid):
    single = self_hosted_query_cost(state, WorkloadClass("q", 1.0, 1000, 300),
                                    grid, respect_slo=False).compute_cost
    triple = self_hosted_query_cost(
        state, WorkloadClass("q", 1.0, 1000, 300, self_consistency_k=3),
        grid, respect_slo=False).compute_cost
    assert triple == pytest.approx(3 * single, rel=1e-6)


def test_semantic_cache_reduces_billable_fraction(state, grid):
    w = WorkloadClass("q", 1.0, 1000, 300, cacheable=True)
    cfg = ServingConfig(n_accelerators=1, max_batch=128, semantic_cache_hit=0.4)
    cached = DeploymentState(state.model, state.hardware, cfg)
    base = self_hosted_query_cost(state, w, grid, respect_slo=False).compute_cost
    hit = self_hosted_query_cost(cached, w, grid, respect_slo=False).compute_cost
    assert hit == pytest.approx(0.6 * base, rel=1e-6)


def test_uncacheable_workload_ignores_semantic_cache(state, grid):
    w = WorkloadClass("q", 1.0, 1000, 300, cacheable=False)
    cfg = ServingConfig(n_accelerators=1, max_batch=128, semantic_cache_hit=0.9)
    cached = DeploymentState(state.model, state.hardware, cfg)
    base = self_hosted_query_cost(state, w, grid, respect_slo=False).compute_cost
    hit = self_hosted_query_cost(cached, w, grid, respect_slo=False).compute_cost
    assert hit == pytest.approx(base, rel=1e-6)


def test_lower_duty_cycle_raises_unit_cost(state, grid):
    # Revised in v7.0. Until v6.0 a third of the duty meant exactly three
    # times the cost, because the electricity component charged idle hours
    # at load power. With idle draw priced at idle power the rental scales
    # exactly with 1/duty while electricity scales sub-linearly, so the
    # exact factor holds on a tariff-free grid and becomes a strict
    # inequality on a tariffed one.
    w = WorkloadClass("q", 1.0, 1000, 300)
    busy = ServingConfig(n_accelerators=1, max_batch=128,
                         demand_duty_cycle=0.9, scheduler_efficiency=1.0)
    idle = ServingConfig(n_accelerators=1, max_batch=128,
                         demand_duty_cycle=0.3, scheduler_efficiency=1.0)

    def cost(cfg, g):
        return self_hosted_query_cost(
            DeploymentState(state.model, state.hardware, cfg), w, g,
            respect_slo=False).compute_cost

    bundled = get_grid("global-average")          # electricity_cost == 0
    assert cost(idle, bundled) == pytest.approx(3 * cost(busy, bundled),
                                                rel=1e-6)

    c_busy, c_idle = cost(busy, grid), cost(idle, grid)
    assert 2.5 * c_busy < c_idle < 3 * c_busy


def test_carbon_scales_with_grid_intensity(state):
    w = WorkloadClass("q", 1.0, 1000, 300)
    dirty = self_hosted_query_cost(state, w, get_grid("coal-heavy"),
                                   respect_slo=False)
    clean = self_hosted_query_cost(state, w, get_grid("nordic-hydro"),
                                   respect_slo=False)
    assert dirty.carbon_kg > 10 * clean.carbon_kg


# ===========================================================================
# efficiency stack composition
# ===========================================================================

def test_stack_multiplier_is_measured_not_assumed(grid):
    """The emergent multiplier of a stack must differ from the product of
    the individual multipliers; if it did not, the physical-transform
    model would add nothing over a lookup table.

    In a memory-bound regime the interaction is *sub*-additive: INT4 has
    already freed the HBM that PagedAttention would have recovered, so
    stacking them delivers much less than their product predicts.
    Multiplying published constants understates cost here. Elsewhere the
    interaction runs the other way. That the error has no reliable sign
    is the argument for deriving the multiplier instead of quoting one.
    """
    w = WorkloadClass("q", 1.0, 8000, 600)
    state = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                            ServingConfig(n_accelerators=2, max_batch=512))

    def cost(st):
        return self_hosted_query_cost(st, w, grid, respect_slo=False).compute_cost

    base = cost(state)
    a = cost(apply_stack(state, ["int4"])) / base
    b = cost(apply_stack(state, ["paged_attention"])) / base
    both = cost(apply_stack(state, ["int4", "paged_attention"])) / base

    assert not math.isclose(both, a * b, rel_tol=0.05)
    assert both > a * b * 1.2      # the pair underperforms the naive product


def test_richer_stacks_do_not_cost_more(state, grid):
    w = WorkloadClass("q", 1.0, 1500, 400)
    costs = []
    for stack in ("none", "baseline_serving", "standard", "aggressive"):
        st = apply_stack(state, stack)
        costs.append(self_hosted_query_cost(st, w, grid,
                                            respect_slo=False).compute_cost)
    assert all(b <= a * 1.001 for a, b in zip(costs, costs[1:]))


def test_aggressive_stack_lands_in_published_range(grid):
    """Independent check against the 0.03-0.08x band that fixed-multiplier
    tables report for a fully optimised stack. CAIDE derives it from the
    transformed physics rather than assuming it, so agreement is evidence
    that the transforms are calibrated."""
    st = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                         ServingConfig(n_accelerators=4, max_batch=256))
    w = WorkloadClass("q", 1.0, 1500, 400)
    base = self_hosted_query_cost(st, w, grid, respect_slo=False).compute_cost
    full = self_hosted_query_cost(apply_stack(st, "maximal"), w, grid,
                                  respect_slo=False).compute_cost
    assert 0.02 <= full / base <= 0.10


def test_conflicting_quantisation_formats_are_rejected(state):
    with pytest.raises(ValueError, match="conflicts"):
        apply_stack(state, ["int4", "int8"])


def test_unknown_technique_names_the_alternatives(state):
    with pytest.raises(KeyError, match="unknown technique"):
        apply_stack(state, ["magic_speedup"])


def test_quality_degradation_composes_on_retention_not_on_loss():
    """Retention multiplies, so two 1% losses cost slightly less than 2%.
    The gap is immaterial for short stacks and grows with stack length;
    the test pins the direction so a future refactor cannot quietly
    switch to adding losses, which would overstate degradation."""
    delta = stack_quality_delta(["int4", "semantic_caching"])
    naive_sum = -0.010 + -0.012
    expected = (1 - 0.010) * (1 - 0.012) - 1
    assert delta == pytest.approx(expected)
    assert delta > naive_sum
    assert delta < 0


def test_distillation_shrinks_model_and_quality(state):
    small = apply_stack(state, ["distillation_25"])
    assert small.model.n_params_total < state.model.n_params_total
    assert small.model.quality_index < state.model.quality_index


# ===========================================================================
# total cost of ownership
# ===========================================================================

def test_workload_shares_must_sum_to_one(grid):
    with pytest.raises(ValueError, match="sum to 1.0"):
        total_cost_of_ownership(
            architecture="api", annual_volume=1e6,
            workloads=[WorkloadClass("a", 0.4, 100, 100)],
            grid=grid, pricing=get_pricing("api-economy"))


def test_missing_state_for_self_hosted_is_rejected(grid, workloads):
    with pytest.raises(ValueError, match="requires a DeploymentState"):
        total_cost_of_ownership(architecture="self_hosted", annual_volume=1e6,
                                workloads=workloads, grid=grid)


def test_unknown_architecture_is_rejected(grid, workloads):
    with pytest.raises(ValueError, match="unknown architecture"):
        total_cost_of_ownership(architecture="quantum", annual_volume=1e6,
                                workloads=workloads, grid=grid)


def test_fixed_layers_dominate_at_low_volume(grid, workloads):
    assurance = AssuranceProfile(evaluation_annual=200_000)
    result = total_cost_of_ownership(
        architecture="api", annual_volume=1000, workloads=workloads,
        grid=grid, pricing=get_pricing("api-frontier"), assurance=assurance)
    assert result.layers["assurance_governance"] > 100 * result.layers["model_access"]


def test_retrieval_layer_scales_sublinearly():
    layer = CostLayer("retrieval_data", fixed_annual=10_000,
                      sublinear_coefficient=50.0, sublinear_exponent=0.35)
    small = layer.annual_cost(1e6) - 10_000
    large = layer.annual_cost(1e8) - 10_000
    assert large < 100 * small          # 100x volume, far less than 100x cost


def test_workforce_layer_decays_after_year_one():
    layer = CostLayer("workforce_redesign", front_load_year1=500_000, decay=0.3)
    assert layer.annual_cost(1e6, year=1) == pytest.approx(500_000)
    assert layer.annual_cost(1e6, year=2) == pytest.approx(150_000)


def test_capacity_is_charged_in_whole_replicas(grid, workloads):
    """You cannot rent a fraction of a GPU node. Cost must not fall below
    one replica however small the volume."""
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                            ServingConfig(n_accelerators=1, max_batch=128))
    floor = replica_annual_cost(state, grid)
    result = total_cost_of_ownership(
        architecture="self_hosted", annual_volume=100, workloads=workloads,
        grid=grid, state=state)
    assert result.layers["compute_serving"] >= floor * 0.999


def test_review_hours_are_reported_for_sanity_checking(grid):
    workloads = [WorkloadClass("w", 1.0, 1000, 300,
                               review_rate=0.5, review_minutes=6.0)]
    result = total_cost_of_ownership(
        architecture="api", annual_volume=1_000_000, workloads=workloads,
        grid=grid, pricing=get_pricing("api-economy"))
    assert result.review_hours_annual == pytest.approx(50_000)
    assert result.review_fte == pytest.approx(50_000 / 1700)


def test_displaced_labour_is_reported_but_not_netted(grid):
    workloads = [WorkloadClass("w", 1.0, 1000, 300, review_rate=1.0,
                               review_minutes=2.0, baseline_minutes=10.0)]
    result = total_cost_of_ownership(
        architecture="api", annual_volume=100_000, workloads=workloads,
        grid=grid, pricing=get_pricing("api-economy"),
        assurance=AssuranceProfile(reviewer_hourly_cost=60.0))
    assert result.displaced_labour_annual == pytest.approx(100_000 * 10 / 60 * 60)
    assert result.total > 0                       # not silently reduced
    assert result.net_of_displaced_labour < result.total


# ===========================================================================
# break-even
# ===========================================================================

def test_linear_curves_cross_once_at_the_analytic_point():
    result = find_break_even(lambda v: 1000 + 0.001 * v, lambda v: 5000 + 0.0005 * v,
                             label_a="a", label_b="b",
                             volume_min=1e3, volume_max=1e9)
    assert len(result.crossings) == 1
    assert result.crossings[0].volume == pytest.approx(8e6, rel=1e-3)


def test_no_crossing_when_one_option_always_wins():
    result = find_break_even(lambda v: 1.0 + 0.001 * v, lambda v: 100 + 0.002 * v,
                             volume_min=1e3, volume_max=1e9)
    assert result.crossings == []


def test_step_costs_produce_multiple_crossings():
    """Granular capacity is what makes a single break-even volume wrong."""
    def stepped(v):
        return 10_000 * math.ceil(v / 1e6)

    result = find_break_even(lambda v: 0.011 * v, stepped,
                             label_a="linear", label_b="stepped",
                             volume_min=1e5, volume_max=1e8, samples=800)
    assert len(result.crossings) > 2


def test_tie_band_summarises_indistinguishable_region():
    def stepped(v):
        return 10_000 * math.ceil(v / 1e6)

    result = find_break_even(lambda v: 0.011 * v, stepped,
                             label_a="linear", label_b="stepped",
                             volume_min=1e5, volume_max=1e8, samples=800)
    band = result.tie_band(0.10)
    assert band is not None and band[1] > band[0]


def test_dominance_intervals_partition_the_scan():
    result = find_break_even(lambda v: 1000 + 0.001 * v,
                             lambda v: 5000 + 0.0005 * v,
                             label_a="a", label_b="b",
                             volume_min=1e3, volume_max=1e9)
    intervals = dominance_intervals(result)
    assert intervals[0][0] == pytest.approx(1e3)
    assert intervals[-1][1] == pytest.approx(1e9)
    assert all(hi > lo for lo, hi, _ in intervals)


def test_invalid_bracket_is_rejected():
    with pytest.raises(ValueError):
        find_break_even(lambda v: v, lambda v: v, volume_min=10, volume_max=5)


# ===========================================================================
# scaling dynamics
# ===========================================================================

def test_elastic_demand_raises_spend_as_price_falls():
    p = project(0.01, 1e6, ScalingAssumptions(annual_price_decline=0.4,
                                              price_elasticity=1.5,
                                              autonomous_growth=0.0,
                                              horizon_years=5))
    assert p.regime == "jevons"
    assert p.spend_ratio > 1.0


def test_inelastic_demand_lowers_spend_as_price_falls():
    p = project(0.01, 1e6, ScalingAssumptions(annual_price_decline=0.4,
                                              price_elasticity=0.5,
                                              autonomous_growth=0.0,
                                              horizon_years=5))
    assert p.regime == "inelastic"
    assert p.spend_ratio < 1.0


def test_unit_elasticity_holds_spend_constant():
    p = project(0.01, 1e6, ScalingAssumptions(annual_price_decline=0.4,
                                              price_elasticity=1.0,
                                              autonomous_growth=0.0,
                                              horizon_years=4))
    spends = [y.variable_spend for y in p.years]
    assert max(spends) == pytest.approx(min(spends), rel=1e-9)


def test_capacity_ceiling_truncates_growth():
    p = project(0.01, 1e6, ScalingAssumptions(annual_price_decline=0.5,
                                              price_elasticity=2.0,
                                              horizon_years=6,
                                              capacity_ceiling=5e6))
    assert max(y.volume for y in p.years) == pytest.approx(5e6)
    assert p.saturated_from is not None


def test_elasticity_recovered_from_synthetic_history():
    true_eps = 1.4
    costs = [0.010, 0.008, 0.0064, 0.0051, 0.0041]
    volumes = [1e6 * (c / costs[0]) ** -true_eps for c in costs]
    fit = estimate_elasticity(costs, volumes)
    assert fit["elasticity"] == pytest.approx(true_eps, rel=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_elasticity_needs_price_variation():
    with pytest.raises(ValueError, match="no variation"):
        estimate_elasticity([0.01] * 4, [1e6, 2e6, 3e6, 4e6])


def test_elasticity_needs_enough_observations():
    with pytest.raises(ValueError, match="at least 3"):
        estimate_elasticity([0.01, 0.02], [1e6, 2e6])
