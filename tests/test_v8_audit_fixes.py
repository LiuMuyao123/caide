"""Regression tests for the defects found in the v8.0 audit.

The v7.0 audit left three standing recommendations, and this round
executed all three. Two of them found something.

Extending the independent reference ledger from ``compute_serving`` to
all six layers (v7 recommendation 3) meant writing down, independently,
what each layer is supposed to do with volume -- and the taxonomy the
package has shipped since v1.0 turned out to describe the largest layer
backwards (R8-1). Following that thread into the scaling model showed
the published five-year projection declining reviewer wages at the speed
of GPU prices (R8-2, R8-3).

Re-deriving the mixture-of-experts shared/expert split independently
(v7 recommendation 5) found that the router's token count was never
updated when v7.0 corrected the verified-token count everywhere else
(R8-4) -- and that the v7 reference implementation had reproduced the
same omission, because it was derived from the same document. A
reference implementation checks only what its source gets right.
"""

import math
from dataclasses import replace

import pytest

from caide import DeploymentState, ServingConfig, WorkloadClass
from caide.catalog import get_grid, get_hardware, get_model
from caide.costing import (
    AssuranceProfile,
    CostLayer,
    layer_volume_elasticity,
    total_cost_of_ownership,
)
from caide.perturb import saturated_draw_keys
from caide.roofline import decode_step_time
from caide.scaling import ScalingAssumptions, project
from caide.uncertainty import monte_carlo, triangular

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


# ==========================================================================
# R8-4: the router sees tokens, not sequences
# ==========================================================================

def _reference_expert_bytes(model, tokens):
    """Expert weight traffic re-derived from the routing event itself.

    Written from "each token independently picks k of E experts, and an
    expert's weights are read if at least one token picked it", with no
    reference to batch size at all -- which is the property the
    production call site got wrong. The shared/expert split is
    re-derived here from the batch-of-one identity rather than read from
    ``ModelSpec.moe_shared_params``, so the two derivations can disagree.
    """
    if not model.is_moe:
        return model.weight_bytes
    E, k = model.n_experts, model.experts_per_token
    # shared solved from: shared + (N - shared) * (k/E) == active
    shared = (model.active_params * E - model.n_params_total * k) / (E - k)
    pool = model.n_params_total - shared
    p_touch = 1.0 - (1.0 - k / E) ** max(tokens, 1.0)
    return (shared + pool * p_touch) * model.bytes_per_param


def test_shared_expert_split_matches_independent_derivation():
    for key in ("moe-8x7b", "moe-8x22b", "moe-236b"):
        model = get_model(key)
        E, k = model.n_experts, model.experts_per_token
        mine = model.moe_shared_params
        theirs = (model.active_params * E - model.n_params_total * k) / (E - k)
        assert mine == pytest.approx(theirs, rel=1e-12)
        assert 0.0 <= mine <= model.n_params_total


def test_expert_traffic_counts_verified_tokens_not_sequences():
    """192 MoE configurations; before the fix 96 disagreed, worst 3.47x.

    A verification step submits ``batch * (gamma + 1)`` tokens to the
    router. Pricing the weight stream at ``batch`` tokens gave a
    mixture-of-experts model the dense model's amortisation for free --
    but for MoE more tokens reach more experts, so the weight stream
    grows where a dense model's stands still.
    """
    worst_traffic, at = 1.0, None
    disagreed = 0
    for key in ("moe-8x7b", "moe-8x22b", "moe-236b"):
        for batch in (1, 8, 64, 256):
            for gamma, ratio in ((0.0, 0.0), (2.0, 0.05),
                                 (4.0, 0.03), (8.0, 0.01)):
                model = get_model(key)
                positions = gamma + 1.0 if (gamma > 0 and ratio > 0) else 1.0
                want = _reference_expert_bytes(model, batch * positions)
                # what the package now charges, and what it charged before
                assert model.expert_bytes_touched(batch * positions) == \
                    pytest.approx(want, rel=1e-12)
                pre_fix = model.expert_bytes_touched(batch)
                if want / pre_fix > 1.000001:
                    disagreed += 1
                if want / pre_fix > worst_traffic:
                    worst_traffic, at = want / pre_fix, (key, batch, gamma)
    # every speculating configuration was mispriced, worst by 2.8x
    assert disagreed == 24, disagreed
    assert worst_traffic > 2.7, f"{worst_traffic:.2f} at {at}"

    # and the assembled step now carries it: switching speculation on at
    # fixed batch must lengthen an MoE step by more than the draft alone
    hw = get_hardware("a100-80gb")
    model = get_model("moe-236b")
    plain = ServingConfig(n_accelerators=8, max_batch=1)
    spec = replace(plain, speculative_gamma=8.0, speculative_acceptance=0.9,
                   draft_param_ratio=0.01)
    a, _ = decode_step_time(model, hw, plain, 1, 1024.0)
    b, _ = decode_step_time(model, hw, spec, 1, 1024.0)
    draft_only = (model.active_params * 0.01 * model.bytes_per_param * 8.0
                  / (hw.memory_bandwidth * 8 * plain.mbu_decode))
    assert b - a > 5.0 * draft_only


def test_speculation_does_not_amortise_an_moe_weight_stream():
    """The mechanism, stated as a property rather than a number.

    For a dense model, verifying gamma + 1 tokens streams the weights
    once: traffic per verified token falls. For MoE it rises, because
    more tokens reach more experts. A model that reports the dense
    behaviour for both has not modelled the router.
    """
    hw = get_hardware("h100-sxm")
    plain = ServingConfig(n_accelerators=8, max_batch=1)
    spec = replace(plain, speculative_gamma=4.0, speculative_acceptance=0.72,
                   draft_param_ratio=0.03)

    dense = get_model("dense-70b")
    assert dense.expert_bytes_touched(1.0) == dense.expert_bytes_touched(5.0)

    moe = get_model("moe-8x7b")
    assert moe.expert_bytes_touched(5.0) > 2.0 * moe.expert_bytes_touched(1.0)

    # and the step reflects it: at batch 1 the MoE step grows more than
    # the dense one when speculation is switched on
    d_plain, _ = decode_step_time(dense, hw, plain, 1, 2048.0)
    d_spec, _ = decode_step_time(dense, hw, spec, 1, 2048.0)
    m_plain, _ = decode_step_time(moe, hw, plain, 1, 2048.0)
    m_spec, _ = decode_step_time(moe, hw, spec, 1, 2048.0)
    assert m_spec / m_plain > d_spec / d_plain


def test_dense_models_are_untouched_by_the_router_fix():
    hw = get_hardware("h100-sxm")
    for key in ("dense-8b", "dense-70b", "dense-405b"):
        model = get_model(key)
        for batch in (1, 64, 256):
            cfg = ServingConfig(n_accelerators=4, max_batch=batch,
                                speculative_gamma=4.0,
                                speculative_acceptance=0.72,
                                draft_param_ratio=0.03)
            step, _ = decode_step_time(model, hw, cfg, batch, 4096.0)
            assert math.isfinite(step) and step > 0
            assert model.expert_bytes_touched(batch) == model.weight_bytes


# ==========================================================================
# R8-6: a scope declaration needs a checked bound, not a stated one
# ==========================================================================

def test_draft_kv_is_modelled_and_its_omission_is_bounded():
    """v7.0 declared the draft's KV out of scope and asserted "under 2%
    of the step in every regime the paper reports". The claim was never
    evaluated. At the paper's own operating points it is 3.3%, and
    across the catalogue at the same contexts it reaches 10.8%."""
    hw = get_hardware("h100-sxm")
    model = get_model("dense-70b")
    worst = 0.0
    for batch in (64, 256):
        for context in (1700.0, 4750.0, 8300.0):
            base = ServingConfig(n_accelerators=4, max_batch=batch,
                                 speculative_gamma=4.0,
                                 speculative_acceptance=0.72,
                                 draft_param_ratio=0.03)
            a, _ = decode_step_time(model, hw, base, batch, context)
            b, _ = decode_step_time(model, hw,
                                    replace(base, draft_kv_ratio=0.03),
                                    batch, context)
            worst = max(worst, b / a - 1.0)
    # the bound is now measured, and it is not 2%
    assert 0.02 < worst < 0.05


def test_draft_kv_defaults_to_the_declared_scope_boundary():
    hw, model = get_hardware("h100-sxm"), get_model("dense-8b")
    cfg = ServingConfig(n_accelerators=1, max_batch=32,
                        speculative_gamma=4.0, speculative_acceptance=0.72,
                        draft_param_ratio=0.03)
    assert cfg.draft_kv_ratio == 0.0
    off, _ = decode_step_time(model, hw, cfg, 32, 4096.0)
    on, _ = decode_step_time(model, hw, replace(cfg, draft_kv_ratio=0.05),
                             32, 4096.0)
    assert on > off


# ==========================================================================
# R8-1: the six-layer taxonomy, measured rather than asserted
# ==========================================================================

@pytest.fixture
def ledger():
    state = DeploymentState(
        get_model("dense-8b"), get_hardware("l40s"),
        ServingConfig(n_accelerators=1, max_batch=64,
                      demand_duty_cycle=0.6, scheduler_efficiency=0.8))
    workloads = [WorkloadClass("q", 1.0, 800, 200, review_rate=0.2,
                               review_minutes=4.0)]
    assurance = AssuranceProfile(evaluation_annual=200_000.0,
                                 red_team_annual=50_000.0)
    retrieval = CostLayer("retrieval", fixed_annual=0.0,
                          sublinear_coefficient=900.0, sublinear_exponent=0.35)
    integration = CostLayer("integration", fixed_annual=300_000.0)
    workforce = CostLayer("workforce", front_load_year1=400_000.0, decay=0.3)

    def evaluate(volume, year=1):
        return total_cost_of_ownership(
            architecture="self_hosted", annual_volume=volume,
            workloads=workloads, grid=get_grid("us-average"), state=state,
            assurance=assurance, retrieval=retrieval, integration=integration,
            workforce=workforce, year=year)

    return evaluate


def test_measured_layer_elasticities_match_their_declared_laws(ledger):
    """One layer at a time, against the law the module docstring states."""
    base = ledger(4_000_000.0)
    e = layer_volume_elasticity(base, ledger)
    assert e["integration_sre"] == pytest.approx(0.0, abs=1e-9)
    assert e["workforce_redesign"] == pytest.approx(0.0, abs=1e-9)
    assert e["retrieval_data"] == pytest.approx(0.35, rel=1e-6)
    # and the row that was wrong for six releases: not volume-free
    assert e["assurance_governance"] > 0.5


def test_assurance_is_the_layer_the_taxonomy_described_backwards(ledger):
    """Per-query review is most of the assurance layer, and the assurance
    layer is the largest one. The v1-v7 table called it volume-free."""
    base = ledger(4_000_000.0)
    linear = base.price_inelastic_per_query * base.annual_volume
    assert linear / base.layers["assurance_governance"] > 0.6
    assert base.layers["assurance_governance"] == max(base.layers.values())


def test_stepped_layer_elasticity_is_reported_not_smoothed(ledger):
    """A staircase has no single exponent, and the measurement says so
    rather than averaging one into existence."""
    base = ledger(4_000_000.0)
    e = layer_volume_elasticity(base, ledger)
    assert e["compute_serving"] >= 0.0
    assert math.isfinite(e["compute_serving"])


def test_scaling_inputs_partition_the_total(ledger):
    for volume in (500_000.0, 4_000_000.0, 90_000_000.0):
        r = ledger(volume)
        s = r.scaling_inputs()
        rebuilt = (s["declining_per_query"] * volume
                   + s["price_inelastic_per_query"] * volume
                   + s["fixed_annual"])
        assert rebuilt == pytest.approx(r.total, rel=1e-9)
        assert 0.0 <= s["declining_share"] <= 1.0


# ==========================================================================
# R8-2 / R8-3: what declines, and what demand responds to
# ==========================================================================

def test_pre_v8_behaviour_is_reproduced_when_the_split_is_degenerate():
    """With no inelastic and no fixed component the closed form returns,
    exactly -- so every scenario written before v8.0 still reproduces."""
    a = ScalingAssumptions(annual_price_decline=0.38, price_elasticity=1.35,
                           autonomous_growth=0.12, horizon_years=5)
    p = project(0.257776, 9_000_000.0, a)
    r, eps, g = 0.38, 1.35, 0.12
    closed = ((1 - r) ** 4) ** (1 - eps) * (1 + g) ** 4
    assert p.spend_ratio == pytest.approx(closed, rel=1e-9)
    assert all(y.converged for y in p.years)


def test_crossover_at_unit_elasticity_survives_the_correction():
    """The structural result. Spend is ``c_eff^(1-eps)`` whatever c_eff is
    made of, so eps = 1 is still exactly neutral once autonomous growth
    is removed -- the v8 correction moves magnitudes, not the sign."""
    common = dict(annual_price_decline=0.38, autonomous_growth=0.0,
                  horizon_years=5, price_inelastic_per_query=0.1305,
                  fixed_annual_cost=904_240.0)
    at_one = project(0.0268, 9_000_000.0,
                     ScalingAssumptions(price_elasticity=1.0, **common))
    assert at_one.spend_ratio == pytest.approx(1.0, rel=1e-6)
    below = project(0.0268, 9_000_000.0,
                    ScalingAssumptions(price_elasticity=0.6, **common))
    above = project(0.0268, 9_000_000.0,
                    ScalingAssumptions(price_elasticity=1.8, **common))
    assert below.spend_ratio < 1.0 < above.spend_ratio


def test_composition_governs_the_magnitude_of_the_jevons_effect():
    """Same elasticity, same tariff decline, different cost composition.

    Declining a blended per-query figure at the tariff rate overstates
    the effect by roughly a factor of three in the shipped education
    scenario, in both directions.
    """
    common = dict(annual_price_decline=0.38, price_elasticity=1.35,
                  autonomous_growth=0.0, horizon_years=5)
    all_declining = project(0.2578, 9_000_000.0,
                            ScalingAssumptions(**common))
    realistic = project(
        0.0268, 9_000_000.0,
        ScalingAssumptions(price_inelastic_per_query=0.1305,
                           fixed_annual_cost=904_240.0, **common))
    assert all_declining.spend_ratio > 1.8
    assert 1.0 < realistic.spend_ratio < 1.2
    # the effective price the buyer faces barely moves
    eff = (realistic.years[-1].effective_unit_cost
           / realistic.years[0].effective_unit_cost)
    tariff = (1 - 0.38) ** 4
    assert eff > 4.0 * tariff


def test_effective_price_is_what_demand_responds_to():
    """Halving the tariff cannot double demand when the tariff is a tenth
    of the price."""
    common = dict(annual_price_decline=0.38, price_elasticity=1.8,
                  autonomous_growth=0.0, horizon_years=5)
    bare = project(0.2578, 9_000_000.0, ScalingAssumptions(**common))
    loaded = project(
        0.0268, 9_000_000.0,
        ScalingAssumptions(price_inelastic_per_query=0.1305,
                           fixed_annual_cost=904_240.0, **common))
    assert bare.volume_ratio > 4.0 * loaded.volume_ratio


def test_fixed_cost_amortisation_is_solved_not_assumed():
    """The fixed layer divided by the volume it helps determine. The
    fixed point converges, and it moves demand in the right direction:
    more volume spreads the fixed layer thinner, which lowers the
    effective price, which raises volume."""
    common = dict(annual_price_decline=0.0, price_elasticity=1.2,
                  autonomous_growth=0.25, horizon_years=4,
                  price_inelastic_per_query=0.05)
    with_fixed = project(0.02, 5_000_000.0,
                         ScalingAssumptions(fixed_annual_cost=500_000.0,
                                            **common))
    assert all(y.converged for y in with_fixed.years)
    # amortisation makes the effective price fall even with a flat tariff
    assert (with_fixed.years[-1].effective_unit_cost
            < with_fixed.years[0].effective_unit_cost)
    assert with_fixed.volume_ratio > (1.25 ** 3)


def test_year_ledger_partitions_spend_three_ways():
    p = project(0.03, 1_000_000.0,
                ScalingAssumptions(annual_price_decline=0.2,
                                   price_elasticity=1.0,
                                   autonomous_growth=0.0, horizon_years=3,
                                   price_inelastic_per_query=0.07,
                                   fixed_annual_cost=250_000.0))
    for y in p.years:
        assert y.total_spend == pytest.approx(
            y.variable_spend + y.inelastic_spend + y.fixed_spend, rel=1e-12)
        assert y.variable_spend == pytest.approx(y.unit_cost * y.volume,
                                                 rel=1e-12)
        assert y.fixed_spend == pytest.approx(250_000.0)


def test_negative_components_are_rejected():
    with pytest.raises(ValueError):
        ScalingAssumptions(price_inelastic_per_query=-0.01)
    with pytest.raises(ValueError):
        ScalingAssumptions(fixed_annual_cost=-1.0)


# ==========================================================================
# R8-5: a clamped draw is a finding, not a nuisance
# ==========================================================================

class _Scen:
    """Minimal stand-in carrying just what saturated_draw_keys reads."""

    def __init__(self, workloads):
        self.workloads = workloads


def test_review_rate_clamp_is_reported():
    workloads = [WorkloadClass("bulk", 0.9, 800, 200, review_rate=0.1),
                 WorkloadClass("authored", 0.1, 800, 200, review_rate=1.0)]
    scen = _Scen(workloads)
    assert saturated_draw_keys(scen, None, {"review_rate_scale": 0.8}) == []
    assert saturated_draw_keys(scen, None,
                               {"review_rate_scale": 1.5}) == ["review_rate_scale"]


def test_bounded_substitutions_are_reported():
    scen = _Scen([WorkloadClass("q", 1.0, 800, 200, review_rate=0.1)])
    assert "mbu" in saturated_draw_keys(scen, None, {"mbu": 1.4})
    assert "scheduler_efficiency" in saturated_draw_keys(
        scen, None, {"scheduler_efficiency": 1.2})
    assert "utilisation" in saturated_draw_keys(scen, None,
                                                {"utilisation": 1.3})
    assert saturated_draw_keys(scen, None, {"mbu": 0.6}) == []


def test_monte_carlo_counts_saturated_draws():
    """The declared distribution and the propagated one differ, and the
    result says by how much instead of leaving it to be discovered."""
    scen = _Scen([WorkloadClass("q", 1.0, 800, 200, review_rate=1.0)])
    dist = [triangular("review_rate_scale", 0.5, 1.0, 2.2)]

    def model(draw):
        return min(draw["review_rate_scale"], 1.0)

    mc = monte_carlo(model, dist, n_samples=2000, seed=8,
                     saturation=lambda d: saturated_draw_keys(scen, None, d))
    share = mc.saturation_share()["review_rate_scale"]
    # triangular(0.5, 1.0, 2.2): P(X > 1.0) = (2.2 - 1.0) / (2.2 - 0.5)
    assert share == pytest.approx(1.2 / 1.7, abs=0.03)
    assert mc.summary()["n_saturated"] > 0


def test_saturation_is_absent_when_nothing_binds():
    scen = _Scen([WorkloadClass("q", 1.0, 800, 200, review_rate=0.2)])
    dist = [triangular("review_rate_scale", 0.5, 1.0, 2.0)]
    mc = monte_carlo(lambda d: d["review_rate_scale"], dist, n_samples=500,
                     seed=3,
                     saturation=lambda d: saturated_draw_keys(scen, None, d))
    assert mc.saturation_share() == {}
    assert mc.summary()["n_saturated"] == 0.0
