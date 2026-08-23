"""Regression tests for the defects found in the v12.0 audit.

Two rounds ago the ranking criterion became feasibility; one round ago
the quality index behind it was made to travel with the state. This round
looked at the two places where the package issues a *verdict* rather than
a number -- the sensitivity ranking and the elasticity regime -- and found
both issuing them from a filtered or a stripped view of their own
evidence.

The sensitivity ranking discarded the draws in which the configuration
failed, so an input whose entire effect is to break the deployment had no
variation left to correlate with and scored zero (R12-1). The percentiles
beneath it were conditional on feasibility and did not say so (R12-2). The
elasticity regime named a side of the crossover from a point estimate
while its own standard error sat two lines away in the same dictionary
(R12-3).

The remaining three are about the quality axis, which v10.0 made
load-bearing and nobody has audited since: quoted constants with no stated
basis (R12-4), a documented scale that contradicts the arithmetic
performed on it (R12-5), and shares that always sum to one however little
is explained (R12-6).
"""

import math

import numpy as np
import pytest

from caide import (
    DeploymentState,
    ServingConfig,
    apply_stack,
    get_hardware,
    get_model,
)
from caide.efficiency import TECHNIQUES, get_technique
from caide.scaling import estimate_elasticity
from caide.uncertainty import (
    lognormal,
    monte_carlo,
    sensitivity,
    triangular,
    uniform,
)


# ==========================================================================
# R12-1: failure is an outcome, not an absence of one
# ==========================================================================

def _cliff_model():
    """An input whose only effect is to make the configuration infeasible.

    Below the cliff the deployment does not run at all; above it the
    input barely moves cost. This is the shape of an SLO cliff or a KV
    capacity limit, both of which the roofline produces.
    """
    def model(draw):
        if draw["duty"] < 0.35:
            return float("inf")
        return 100.0 + 0.5 * draw["duty"] + 40.0 * draw["other"]
    return model


def _cliff_result(seed=7):
    return monte_carlo(_cliff_model(),
                       [triangular("duty", 0.2, 0.5, 0.9),
                        lognormal("other", 1.0, 0.3)],
                       n_samples=4000, seed=seed)


def test_an_input_that_only_breaks_things_scores_zero_on_cost():
    """The defect, reproduced. Not a bug in the correlation -- among
    surviving draws ``duty`` really is irrelevant, because the draws
    where it mattered are the ones that were removed."""
    result = _cliff_result()
    assert 0.05 < 1.0 - result.feasible_fraction < 0.2
    by_name = {e.name: e for e in sensitivity(result)}
    assert by_name["duty"].contribution < 0.01
    assert by_name["other"].contribution > 0.98


def test_the_association_with_failure_is_reported():
    """So the ranking is readable rather than misleading: ``duty``
    contributes nothing to cost among survivors and everything to whether
    there are survivors."""
    by_name = {e.name: e for e in sensitivity(_cliff_result())}
    assert by_name["duty"].failure_spearman < -0.4
    assert abs(by_name["other"].failure_spearman) < 0.1
    assert "failure_spearman" in by_name["duty"].as_dict()


def test_failure_association_is_absent_when_nothing_failed():
    result = monte_carlo(lambda d: d["a"] + d["b"],
                         [uniform("a", 1.0, 2.0), uniform("b", 1.0, 2.0)],
                         n_samples=500, seed=3)
    assert result.feasible_fraction == 1.0
    assert all(math.isnan(e.failure_spearman) for e in sensitivity(result))


def test_the_sign_of_the_failure_association_is_readable():
    """A high draw breaking the deployment must read as positive."""
    def model(draw):
        return float("inf") if draw["x"] > 1.5 else 10.0 + draw["y"]

    result = monte_carlo(model,
                         [uniform("x", 0.0, 2.0), uniform("y", 0.0, 1.0)],
                         n_samples=3000, seed=11)
    by_name = {e.name: e for e in sensitivity(result)}
    assert by_name["x"].failure_spearman > 0.5


# ==========================================================================
# R12-2: the percentiles are conditional
# ==========================================================================

def test_feasible_fraction_is_reported():
    result = _cliff_result()
    assert result.summary()["feasible_fraction"] == pytest.approx(
        result.feasible_fraction, rel=1e-12)
    assert 0.8 < result.feasible_fraction < 0.95


def test_percentiles_describe_the_surviving_draws():
    """Stated, not fixed: a percentile over infeasible outcomes has no
    meaning, so the conditioning stays and is now declared."""
    result = _cliff_result()
    assert math.isfinite(result.percentile(95))
    assert result.valid.size == int(round(result.feasible_fraction
                                          * result.samples.size))


def test_a_fully_feasible_run_is_unconditional():
    result = monte_carlo(lambda d: d["a"], [uniform("a", 1.0, 2.0)],
                         n_samples=400, seed=5)
    assert result.feasible_fraction == 1.0


# ==========================================================================
# R12-6: shares of what?
# ==========================================================================

def test_explained_rank_variance_is_available_before_normalisation():
    """Contributions total one however little the inputs account for.
    The total is the thing that tells a reader which of those two worlds
    they are in."""
    result = monte_carlo(lambda d: d["a"] + d["b"],
                         [uniform("a", 0.0, 1.0), uniform("b", 0.0, 1.0)],
                         n_samples=2000, seed=2)
    entries = sensitivity(result)
    assert sum(e.contribution for e in entries) == pytest.approx(1.0, rel=1e-9)
    total = result.explained_rank_variance()
    assert total == pytest.approx(sum(e.spearman ** 2 for e in entries),
                                  rel=1e-12)
    assert 0.0 < total <= 2.0


def test_shares_still_sum_to_one_when_little_is_explained():
    """The case the total exists for: an output driven by an interaction
    no single input tracks."""
    def model(draw):
        return draw["a"] * draw["b"]        # sign flips with either input

    result = monte_carlo(model,
                         [uniform("a", -1.0, 1.0), uniform("b", -1.0, 1.0)],
                         n_samples=4000, seed=4)
    entries = sensitivity(result)
    assert sum(e.contribution for e in entries) == pytest.approx(1.0, rel=1e-6)
    # the output is entirely determined by the two inputs and neither has
    # any monotone association with it: shares of nothing, still summing
    # to one
    assert result.explained_rank_variance() < 0.05


# ==========================================================================
# R12-3: a verdict that ignores its own standard error
# ==========================================================================

def test_regime_is_undetermined_when_the_interval_straddles_one():
    """Three points is the minimum this function accepts, and on three
    points the interval usually contains the crossover the whole scaling
    model turns on."""
    result = estimate_elasticity([1.0, 0.7, 0.45], [100, 150, 140])
    assert result["ci_low"] < 1.0 < result["ci_high"]
    assert result["regime"] == "undetermined"
    assert result["point_regime"] == "inelastic"


def test_a_tight_estimate_still_gets_a_verdict():
    result = estimate_elasticity([1.0, 0.8, 0.62, 0.45, 0.3],
                                 [100, 118, 150, 168, 240])
    assert result["ci_high"] < 1.0
    assert result["regime"] == "inelastic"
    assert result["regime"] == result["point_regime"]


def test_a_tight_jevons_estimate_is_named():
    costs = [1.0, 0.8, 0.6, 0.4, 0.25]
    volumes = [100 * (1.0 / c) ** 1.6 for c in costs]
    result = estimate_elasticity(costs, volumes)
    assert result["elasticity"] == pytest.approx(1.6, rel=1e-6)
    assert result["ci_low"] > 1.0
    assert result["regime"] == "jevons"


def test_the_interval_brackets_the_point_estimate():
    result = estimate_elasticity([1.0, 0.7, 0.45, 0.3], [100, 150, 200, 260])
    assert result["ci_low"] <= result["elasticity"] <= result["ci_high"]
    half = 1.96 * result["std_error"]
    assert result["ci_high"] - result["elasticity"] == pytest.approx(half,
                                                                     rel=1e-9)


# ==========================================================================
# R12-4: a quoted constant should say so
# ==========================================================================

def test_every_quality_cost_states_its_basis():
    """A non-zero delta with no basis is a quoted constant wearing the
    clothes of a derived one -- the practice this package exists to argue
    against, applied to its own numbers now that v10.0 made the quality
    axis decide admissibility."""
    for key in TECHNIQUES:
        technique = get_technique(key)
        if technique.quality_delta < 0:
            assert technique.quality_basis, key
            assert "Quoted constant" in technique.quality_basis, key


def test_lossless_techniques_need_no_basis():
    for key in ("speculative_decoding", "paged_attention",
                "continuous_batching"):
        technique = get_technique(key)
        assert technique.quality_delta == 0.0
        assert technique.quality_basis == ""


def test_the_quantisation_basis_names_the_size_dependence():
    """The limitation is stated where the number is, not only in a
    document nobody opens next to the code."""
    basis = get_technique("int4").quality_basis
    assert "size dependent" in basis
    assert "8B" in basis and "405B" in basis


def test_the_constant_is_still_applied_uniformly():
    """Recorded so the declaration is not mistaken for a fix: the delta
    is the same for every model until someone supplies a size-aware
    form."""
    hardware = get_hardware("h100-sxm")
    ratios = []
    for key in ("dense-8b", "dense-405b"):
        model = get_model(key)
        state = DeploymentState(model, hardware,
                                ServingConfig(n_accelerators=8, max_batch=64))
        after = apply_stack(state, ["int4"]).model
        ratios.append(after.quality_index / model.quality_index)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)


# ==========================================================================
# R12-5: the scale the arithmetic assumes
# ==========================================================================

def test_quality_index_is_used_as_a_ratio_scale():
    """Multiplied by ``1 + delta`` and composed on retention. That is
    ratio-scale arithmetic; the v10 limits table called the index ordinal,
    which would have made all of it meaningless. The arithmetic is the
    older claim, so the documentation was the one that was wrong."""
    state = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                            ServingConfig(n_accelerators=4, max_batch=64))
    once = apply_stack(state, ["int4"]).model.quality_index
    twice = apply_stack(state, ["int4", "semantic_caching"]).model.quality_index
    base = state.model.quality_index
    assert once / base == pytest.approx(1 - 0.010, rel=1e-12)
    assert twice / base == pytest.approx((1 - 0.010) * (1 - 0.012), rel=1e-12)


def test_the_docstring_declares_the_scale():
    from caide.specs import ModelSpec
    source_doc = ModelSpec.__dataclass_fields__["quality_index"]
    assert source_doc is not None
    # the declaration lives in the module text, checked directly so that a
    # future edit that quietly reverts it fails here
    import inspect
    import caide.specs
    text = inspect.getsource(caide.specs)
    assert "ratio scale" in text
    assert "ordinal" in text          # the correction is recorded, not erased


def test_quality_never_goes_negative_or_above_the_frontier():
    state = DeploymentState(get_model("dense-405b"), get_hardware("h100-sxm"),
                            ServingConfig(n_accelerators=8, max_batch=64))
    stacked = apply_stack(state, ["int4", "semantic_caching", "gqa", "kv_fp8"])
    assert 0.0 < stacked.model.quality_index <= state.model.quality_index


def test_ranking_by_quality_survives_a_uniform_delta():
    """A delta applied uniformly cannot reorder two models, which is why
    the size independence in R12-4 matters: the ordering it preserves is
    exactly the one a size-aware form would disturb."""
    hardware = get_hardware("h100-sxm")
    before, after = [], []
    for key in ("dense-8b", "dense-32b", "dense-70b", "dense-405b"):
        model = get_model(key)
        state = DeploymentState(model, hardware,
                                ServingConfig(n_accelerators=8, max_batch=64))
        before.append(model.quality_index)
        after.append(apply_stack(state, ["int4"]).model.quality_index)
    assert np.argsort(before).tolist() == np.argsort(after).tolist()
