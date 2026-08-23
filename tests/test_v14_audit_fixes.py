"""Regression tests for the defects found in the v14.0 audit.

The v13 audit measured how much of the library the published results
exercise and found 59%, with ``cli.py`` at zero. It also established that
hand-carried numbers are a defect class rather than an oversight, and
recommended looking for them wherever the project states a figure outside
the reproduction script.

Both threads led to the same place. The command line is a second assembly
of the analysis pipeline, and because no published result and few tests
enter it, four rounds of headline fixes never arrived: ``caide run``
reported an architecture that missed every declared quality floor as the
cheapest (v10), and fed the blended per-query cost into the elasticity
projection with the other two components at zero (v8). The README, the
first artefact any reader meets, quoted transcripts frozen at several past
versions, four of its figures now wrong in direction and its test badge
five releases stale.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import caide
from caide import load_scenario

EXAMPLES = Path(caide.__file__).parent / "examples"
ROOT = Path(caide.__file__).resolve().parent.parent.parent
README = ROOT / "README.md"


def _run_cli(*args, cwd=None):
    proc = subprocess.run([sys.executable, "-m", "caide.cli", *args],
                          capture_output=True, text=True, cwd=cwd)
    return proc


# ==========================================================================
# R14-1: the command line is the same analysis, not a second one
# ==========================================================================

@pytest.fixture(scope="module")
def hospital_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli")
    proc = _run_cli("run", str(EXAMPLES / "hospital_documentation.yaml"),
                    "--out", str(out))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_the_command_line_reports_the_cheapest_admissible_architecture(hospital_run):
    """It reported ``onprem-8b-int4`` -- which misses all four declared
    quality floors -- for four releases after v10.0 made feasibility the
    ranking criterion, because the fix landed in ReportBundle.cheapest()
    and this command computed its own minimum."""
    assert "cheapest: onprem-70b" in hospital_run
    assert "cheapest: onprem-8b-int4" not in hospital_run


def test_the_command_line_names_the_quality_violations(hospital_run):
    assert "onprem-8b-int4: quality floor not met" in hospital_run
    assert "onprem-32b: quality floor not met" in hospital_run


def test_the_violations_carry_their_distance(hospital_run):
    """"Inadmissible" is a label pressed onto a continuous quantity. The
    shipped scenarios contain a candidate that misses by 1.2% and one
    that misses by 22%, and through v13.0 both were simply infeasible."""
    assert re.search(r"discharge_summary \(short by 2[0-9]\.\d%\)", hospital_run)
    assert re.search(r"discharge_summary \(short by 5\.\d%\)", hospital_run)


def test_the_scaling_projection_uses_the_three_way_split():
    """Passing the blended per-query figure declines reviewer wages at
    the speed of GPU prices. That was v8.0's finding; the fix reached the
    reproduction script and not this command, which reported spend
    *falling* where the corrected split has it rising."""
    scenario = load_scenario(EXAMPLES / "hospital_documentation.yaml")
    from caide.cli import _scaling_inputs
    results = scenario.evaluate_all()
    best = min((r for r in results.values() if r.feasible),
               key=lambda r: r.total)
    declining, volume, assumptions = _scaling_inputs(best, scenario.scaling)

    split = best.scaling_inputs()
    assert declining == pytest.approx(split["declining_per_query"], rel=1e-12)
    assert declining < best.effective_per_query
    assert assumptions.price_inelastic_per_query > 0
    assert assumptions.fixed_annual_cost > 0
    assert volume == pytest.approx(scenario.annual_volume)


def test_the_projection_direction_changed(hospital_run):
    assert "total spend rises" in hospital_run


def test_a_scenario_with_no_admissible_architecture_says_so(tmp_path):
    proc = _run_cli("run", str(EXAMPLES / "public_helpline.yaml"),
                    "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "no architecture meets every declared constraint" in proc.stdout


def test_every_verb_runs(tmp_path):
    """``cli.py`` was at 0% of statements under the published results and
    93 lines uncovered by the tests; the two gaps overlapping is why the
    divergences above survived four rounds."""
    extracted = tmp_path / "examples"
    assert _run_cli("examples", "--extract", str(tmp_path)).returncode == 0
    scenario = extracted / "university_tutoring.yaml"
    assert scenario.exists()

    for args in (("catalog",),
                 ("validate", str(scenario)),
                 ("breakeven", str(EXAMPLES / "public_helpline.yaml")),
                 ("sweep", str(scenario), "--technique", "int4"),
                 ("route", str(EXAMPLES / "public_helpline.yaml"))):
        proc = _run_cli(*args)
        assert proc.returncode == 0, (args, proc.stderr[-1500:])


def test_init_produces_a_scenario_that_validates(tmp_path):
    target = tmp_path / "new.yaml"
    assert _run_cli("init", str(target)).returncode == 0
    assert _run_cli("validate", str(target)).returncode == 0
    load_scenario(target)


# ==========================================================================
# R14-2: the numbers in the README are the numbers the software produces
# ==========================================================================

@pytest.fixture(scope="module")
def readme():
    if not README.exists():
        pytest.skip("README not present in this layout")
    return README.read_text()


def test_the_test_badge_matches_the_suite(readme):
    """It said 310 while the suite ran 420: five releases of a badge
    nobody checked, in the most visible file in the repository."""
    match = re.search(r"tests-(\d+)%20passing", readme)
    assert match, "no test badge found"
    claimed = int(match.group(1))
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only",
                           "-q", str(Path(__file__).parent)],
                          capture_output=True, text=True, cwd=ROOT)
    counts = re.findall(r"^\S+\.py: (\d+)$", proc.stdout, re.MULTILINE)
    assert counts, proc.stdout[-800:]
    collected = sum(int(c) for c in counts)
    assert claimed == collected, f"badge says {claimed}, suite has {collected}"


def test_the_readme_states_the_current_speculative_multiplier(readme):
    """It said 0.81× at batch 256 -- the v6.0 value, from before the
    verification arithmetic was priced. The corrected figure crosses
    parity, which reverses the recommendation the sentence carries."""
    import json
    findings = json.loads((ROOT / "paper_figures" / "findings.json").read_text())
    at_256 = findings["regime_dependence"]["speculative_decoding"]["max"]
    assert f"{at_256:.2f}×" in readme
    assert "0.81×" not in readme


def test_the_readme_states_the_current_break_even_shape(readme):
    import json
    findings = json.loads((ROOT / "paper_figures" / "findings.json").read_text())
    assert f"{findings['break_even']['n_crossings']} crossings" in readme
    assert f"{findings['break_even']['n_tie_windows']} windows" in readme
    assert "54 crossings" not in readme


def test_the_readme_states_the_current_uncertainty_verdict(readme):
    """It said self-hosting was cheaper in 78% of draws and named
    utilisation as the dominant driver. The endpoint is cheaper in all of
    them and utilisation carries 0.1%."""
    import json
    findings = json.loads((ROOT / "paper_figures" / "findings.json").read_text())
    share = findings["uncertainty"]["probability_api_cheaper"]
    assert f"cheaper in {share:.0%} of 4,000 draws" in readme
    assert "utilisation (54% of explained variance)" not in readme
    assert findings["uncertainty"]["top_driver"].replace("_scale", "") in readme


def test_no_stale_directional_claims_survive(readme):
    """The four figures that had become wrong in direction, pinned by
    absence so that reintroducing one fails here."""
    for stale in ("0.81×", "54 crossings", "615.8M", "78% of 4,000 draws",
                  "utilisation (54%"):
        assert stale not in readme, stale


# ==========================================================================
# R14-3: the shortfall is part of the verdict
# ==========================================================================

@pytest.mark.parametrize("name", ["university_tutoring",
                                  "hospital_documentation",
                                  "public_helpline"])
def test_shortfalls_are_recorded_for_every_violation(name):
    scenario = load_scenario(EXAMPLES / f"{name}.yaml")
    for result in scenario.evaluate_all().values():
        assert set(result.quality_shortfall) == set(result.quality_violations)
        for cls, shortfall in result.quality_shortfall.items():
            assert 0.0 < shortfall < 1.0
            floor = next(w.quality_floor for w in scenario.workloads
                         if w.name == cls)
            assert shortfall == pytest.approx(
                (floor - result.quality_index) / floor, rel=1e-9)


def test_the_smallest_shipped_shortfall_is_a_hair():
    """The case the distance exists for: an architecture ruled out by
    1.2% reads identically to one ruled out by 22% without it."""
    scenario = load_scenario(EXAMPLES / "university_tutoring.yaml")
    results = scenario.evaluate_all()
    shortfalls = [s for r in results.values()
                  for s in r.quality_shortfall.values()]
    assert min(shortfalls) < 0.02
    assert max(shortfalls) > 0.15


def test_a_feasible_architecture_has_no_shortfall():
    scenario = load_scenario(EXAMPLES / "hospital_documentation.yaml")
    feasible = [r for r in scenario.evaluate_all().values() if r.feasible]
    assert feasible
    assert all(r.quality_shortfall == {} for r in feasible)
