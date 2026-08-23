"""Regression tests for the defects found in the v13.0 audit.

The v12 audit ended by observing that two consecutive rounds had moved no
published number because the audit frontier had left the paper's coverage
behind, and recommended making that coverage a measured artefact. It was
measured: the reproduction script, run alone, exercised 59% of statements,
with ``report.py`` at 0% and ``routing.py`` at 25% -- the two modules in
which the v9 and v10 audits had found four defects between them.

The measurement also found something larger. The manuscript's validation
paragraph -- the one place a reader looks to decide whether to trust
anything else -- was not produced by the script at all, while the script's
preamble said every result was regenerated under a fixed seed. Putting
those figures under the script immediately showed that two of them had
drifted since v10.0, and that one published claim had become false:
calibration no longer lifts the within-a-factor-of-two fraction from 60%
to 80%. It leaves it at 60%.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import caide
from caide.calibration import REFERENCE_OBSERVATIONS, fit, predicted_output_tps
from caide.costing import CostLayer
from caide.scaling import ScalingAssumptions, project

EXAMPLES = Path(caide.__file__).parent / "examples"
REPRODUCE = EXAMPLES / "reproduce_paper.py"


# ==========================================================================
# R13-1: every published result is produced by the script that claims to
# produce every published result
# ==========================================================================

REQUIRED_RESULTS = (
    "regime_dependence", "interaction", "duty_cycle", "break_even",
    "cross_domain", "uncertainty", "jevons",
    "validation", "structural_sensitivity", "provenance_and_routing",
)


@pytest.fixture(scope="module")
def findings(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    proc = subprocess.run([sys.executable, str(REPRODUCE), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads((out / "findings.json").read_text(encoding="utf-8"))


def test_the_script_produces_every_published_result(findings):
    """Through v12.0 it produced seven of ten. The validation paragraph
    and the nine-variant sweep were carried into the manuscript by hand,
    under a preamble stating that everything below was regenerated."""
    assert set(REQUIRED_RESULTS) <= set(findings)


def test_validation_figures_are_regenerated(findings):
    """The figures the manuscript quotes, computed rather than typed."""
    validation = findings["validation"]
    assert validation["n_observations"] == 5.0
    assert validation["n_within_2x"] == 3.0
    assert len(validation["observations"]) == 5
    for row in validation["observations"]:
        assert row["predicted"] > 0 and row["measured"] > 0
        assert row["ratio"] == pytest.approx(row["predicted"] / row["measured"],
                                             rel=1e-9)


def test_the_ratios_match_a_direct_computation(findings):
    """The script and the library must agree; a hand-carried number
    cannot be checked this way, which is the point."""
    direct = [predicted_output_tps(o) / o.aggregate_output_tps
              for o in REFERENCE_OBSERVATIONS()]
    from_script = [row["ratio"] for row in findings["validation"]["observations"]]
    assert from_script == pytest.approx(direct, rel=1e-9)


def test_structural_sensitivity_is_regenerated(findings):
    sweep = findings["structural_sensitivity"]
    assert sweep["n_variants"] == 9.0
    assert 2.3 < sweep["dense_spread_min"] <= sweep["dense_spread_max"] < 2.8
    assert sweep["n_past_parity"] == 8.0
    # the MoE multiplier at batch one is set by counting statistics in the
    # router, which none of the nine variants touches
    assert sweep["moe_at_1_min"] == pytest.approx(sweep["moe_at_1_max"],
                                                  rel=1e-3)
    assert sweep["moe_at_1_min"] > 1.0


def test_provenance_and_routing_are_exercised(findings):
    """Two capabilities the manuscript describes and no published result
    entered: ``report.py`` was at 0% of statements and ``routing.py`` at
    25% when the script was measured on its own."""
    record = findings["provenance_and_routing"]
    assert len(record["digest"]) == 12
    assert record["n_feasible"] == 0.0          # the helpline scenario
    assert record["n_tiers_opened"] >= 1
    assert 0.0 < record["blended_quality"] <= 1.05


# ==========================================================================
# R13-5: a headline claim that turned on a threshold nobody measured the
# distance to
# ==========================================================================

def test_calibration_no_longer_lifts_the_within_two_fraction():
    """Published in v10.0, v11.0 and v12.0 as "60% to 80%". The v10.0
    weight-stream correction pushed the worst observation to a calibrated
    ratio of 2.01 -- just outside a boundary at 2.0 -- and the claim was
    not re-checked because it was not computed anywhere."""
    summary = fit(REFERENCE_OBSERVATIONS()).summary()
    assert summary["within_2x_before"] == pytest.approx(0.6)
    assert summary["within_2x_after"] == pytest.approx(0.6)


def test_calibration_does_improve_the_continuous_measure():
    """Which is why the discrete one was misleading rather than merely
    stale: calibration halves the log error while moving no observation
    across the boundary."""
    summary = fit(REFERENCE_OBSERVATIONS()).summary()
    assert summary["log_rmse_after"] < 0.8 * summary["log_rmse_before"]


def test_the_worst_observation_sits_just_outside_the_band():
    """The distance to the threshold, which is the thing a threshold
    claim has to report and this one never did."""
    result = fit(REFERENCE_OBSERVATIONS())
    worst = max(result.ratios_after)
    assert 2.0 < worst < 2.1


# ==========================================================================
# R13-3: the one spec dataclass with no validation
# ==========================================================================

def test_step_cost_without_step_size_is_rejected():
    """It contributed exactly zero: a layer the author wrote down, priced,
    and never saw again."""
    with pytest.raises(ValueError, match="together"):
        CostLayer("blocks", step_cost=1000.0)
    with pytest.raises(ValueError, match="together"):
        CostLayer("blocks", step_size=1e6)


def test_a_complete_step_layer_still_works():
    layer = CostLayer("blocks", step_size=1e6, step_cost=1000.0)
    assert layer.annual_cost(2.5e6) == pytest.approx(3000.0)
    assert layer.annual_cost(0.0) == pytest.approx(0.0)


def test_negative_components_are_rejected():
    for field, value in (("fixed_annual", -1.0), ("per_query", -1e-6),
                         ("sublinear_coefficient", -5.0),
                         ("front_load_year1", -1.0)):
        with pytest.raises(ValueError, match="non-negative"):
            CostLayer("bad", **{field: value})


def test_decay_and_exponent_are_range_checked():
    with pytest.raises(ValueError, match="decay"):
        CostLayer("bad", decay=1.4)
    with pytest.raises(ValueError, match="not sublinear"):
        CostLayer("bad", sublinear_coefficient=900.0, sublinear_exponent=1.0)


def test_the_shipped_layers_all_validate():
    from caide import load_scenario
    for name in ("university_tutoring", "hospital_documentation",
                 "public_helpline"):
        scenario = load_scenario(EXAMPLES / f"{name}.yaml")
        for layer in (scenario.retrieval, scenario.integration,
                      scenario.workforce):
            if layer is not None:
                assert layer.annual_cost(1e6) >= 0.0


# ==========================================================================
# R13-4: the safety net nothing had ever tripped
# ==========================================================================

def test_the_fixed_point_converges_across_the_parameter_space():
    """``converged`` was never False in any shipped scenario or test, so
    the flag and the narrative behind it were an untested safety net.
    Swept here instead of assumed."""
    for elasticity in (0.0, 0.6, 1.0, 1.8, 4.0, 12.0):
        for fixed in (0.0, 1e5, 9e5, 5e6):
            projection = project(
                0.02, 1e6,
                ScalingAssumptions(annual_price_decline=0.38,
                                   price_elasticity=elasticity,
                                   autonomous_growth=0.1, horizon_years=5,
                                   fixed_annual_cost=fixed,
                                   price_inelastic_per_query=0.05))
            assert all(y.converged for y in projection.years), (elasticity, fixed)
            assert all(math.isfinite(y.volume) for y in projection.years)


def test_non_convergence_would_be_reported_if_it_happened():
    """The narrative path exists and is reachable; exercised directly so
    that a future parameterisation which does diverge produces a stated
    caveat rather than a silent last iterate."""
    from dataclasses import replace
    projection = project(0.02, 1e6,
                         ScalingAssumptions(annual_price_decline=0.2,
                                            price_elasticity=1.0,
                                            autonomous_growth=0.0,
                                            horizon_years=2))
    forced = replace(projection.years[0], converged=False)
    projection.years[0] = forced
    assert "did not converge" in projection.narrative()


# ==========================================================================
# R13-2: the coverage of the published results is a measured artefact
# ==========================================================================

def test_the_published_coverage_script_ships():
    script = Path(caide.__file__).resolve().parent.parent.parent / "audit"
    # the audit directory sits beside the package in the source tree and is
    # absent from an installed wheel; only assert when it is present
    if not script.exists():
        pytest.skip("audit tooling not present in this layout")
    assert (script / "published_coverage.py").exists()


def test_every_result_function_is_registered():
    """A result that exists and is not called is the same failure as a
    figure carried by hand: present in the source, absent from the
    output."""
    source = REPRODUCE.read_text(encoding="utf-8")
    defined = {line.split("(")[0].removeprefix("def result_")
               for line in source.splitlines() if line.startswith("def result_")}
    for name in defined:
        assert f"result_{name}(out)" in source, name
    assert len(defined) == len(REQUIRED_RESULTS)
