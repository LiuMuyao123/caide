"""Regression tests for the v17.0 round.

This round opened by settling the carry-forward ledger's two overdue
items rather than by looking for something new, which is the practice the
ledger exists to enforce.

The older of the two -- reviewing the quality floors, raised in v11 and
outstanding for five rounds -- turned out to matter more than its age
suggested. Three of the fourteen floors the shipped scenarios declare sit
within 1.2% of an architecture's quality index, and each of the three
decides a published answer. One of them, 0.5%, is the sole reason an
entire scenario has no admissible architecture. The index is a declared
scale whose catalogue values are round numbers to two decimals; nothing
in the package distinguishes 0.856 from 0.860.

The second -- a check for mutually cancelling errors, raised in v10 --
produced a tool whose first version returned "ok" for both of the
historical pairs it was calibrated on.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import caide
from caide import load_scenario
from caide.specs import QUALITY_INDEX_RESOLUTION

EXAMPLES = Path(caide.__file__).parent / "examples"
ROOT = Path(caide.__file__).resolve().parent.parent.parent
AUDIT = ROOT / "audit"
SCENARIOS = ["university_tutoring", "hospital_documentation", "public_helpline"]


# ==========================================================================
# Ledger item: quality_floor provenance (raised v11, outstanding 5 rounds)
# ==========================================================================

@pytest.mark.parametrize("name", SCENARIOS)
def test_every_declared_floor_states_its_basis(name):
    """A floor that decides an answer and carries no reason is a number
    nobody can argue with, which is not the same as a number nobody
    disagrees with."""
    text = (EXAMPLES / f"{name}.yaml").read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "quality_floor:" in line and not line.strip().startswith("#"):
            preceding = "\n".join(lines[max(0, i - 3):i])
            assert "floor basis:" in preceding, f"{name}:{i+1} {line.strip()}"


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_provenance_header_declares_the_scale(name):
    text = (EXAMPLES / f"{name}.yaml").read_text()
    assert "DECLARED THRESHOLDS" in text
    assert "not measurements" in text


def test_margins_are_reported_for_every_declared_floor():
    for name in SCENARIOS:
        scenario = load_scenario(EXAMPLES / f"{name}.yaml")
        floors = {w.name for w in scenario.workloads if w.quality_floor > 0}
        for result in scenario.evaluate_all().values():
            assert set(result.quality_margin) == floors
            for cls, margin in result.quality_margin.items():
                floor = next(w.quality_floor for w in scenario.workloads
                             if w.name == cls)
                assert margin == pytest.approx(
                    (result.quality_index - floor) / floor, rel=1e-9)


def test_verdicts_below_the_index_resolution_are_flagged():
    """The three margins that decide the published answers are 1.2%, 1.1%
    and 0.5% of their floors -- 0.010, 0.009 and 0.004 in index units,
    against a declared resolution of 0.05."""
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    results = scenario.evaluate_all()
    hosted = results["selfhost-70b-only"]
    assert "appeal_drafting" in hosted.marginal_verdicts
    assert not hosted.feasible
    margin = abs(hosted.quality_margin["appeal_drafting"]) * 0.86
    assert margin < QUALITY_INDEX_RESOLUTION / 10


def test_a_comfortable_verdict_is_not_flagged():
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    result = scenario.evaluate_all()["api-economy"]
    assert "faq_lookup" not in result.marginal_verdicts
    assert result.quality_margin["faq_lookup"] > 0.5


def test_the_resolution_is_declared_not_inferred():
    assert 0.0 < QUALITY_INDEX_RESOLUTION < 0.2
    source = (Path(caide.__file__).parent / "specs.py").read_text()
    assert "no procedure" in source and "0.856" in source


@pytest.mark.parametrize("channel", ["terminal", "markdown", "html", "csv"])
def test_the_flag_reaches_every_channel(channel, tmp_path):
    """v15 established that a correction stopping at the terminal is a
    correction three quarters undone."""
    from caide.report import ReportBundle, write_csv, write_html, write_markdown
    scenario = load_scenario(EXAMPLES / "hospital_documentation.yaml")
    bundle = ReportBundle(scenario)
    bundle.tco = scenario.evaluate_all()
    if channel == "terminal":
        proc = subprocess.run(
            [sys.executable, "-m", "caide.cli", "run",
             str(EXAMPLES / "hospital_documentation.yaml"),
             "--out", str(tmp_path)], capture_output=True, text=True)
        assert "cannot resolve" in proc.stdout
    elif channel == "markdown":
        assert "below it" in write_markdown(bundle, tmp_path / "r.md").read_text()
    elif channel == "html":
        assert "index resolution" in write_html(bundle,
                                                tmp_path / "r.html").read_text()
    else:
        text = write_csv(bundle, tmp_path / "r.csv").read_text()
        assert "marginal_verdicts" in text
        assert "discharge_summary" in text


# ==========================================================================
# Ledger item: opposing-error check (raised v10, outstanding 6 rounds)
# ==========================================================================

pytestmark_audit = pytest.mark.skipif(
    not (AUDIT / "opposing_errors.py").exists(),
    reason="audit tooling absent from an installed wheel")


@pytest.fixture(scope="module")
def scanner():
    if not (AUDIT / "opposing_errors.py").exists():
        pytest.skip("audit tooling absent")
    sys.path.insert(0, str(AUDIT))
    try:
        import opposing_errors
    finally:
        sys.path.pop(0)
    return opposing_errors


def test_the_scan_rediscovers_both_historical_pairs(scanner):
    """The calibration that the first version of this check failed. A
    detector that cannot find the cases it was built from is measuring
    nothing, and it says "ok" while doing it."""
    quantities = {hit["quantity"] for hit in scanner.scan()}
    assert any("decode weight stream" in q for q in quantities)
    assert any("annual facility energy" in q for q in quantities)


def test_the_criterion_is_relative_not_absolute(scanner):
    """The first version compared the joint move against a fixed 2%. The
    decode pair moves 3.5% jointly against a 10.5% single, which is the
    signature and was below the old bar."""
    assert hasattr(scanner, "CANCELLATION_RATIO")
    assert not hasattr(scanner, "SMALL")
    for hit in scanner.scan():
        assert hit["ratio"] < scanner.CANCELLATION_RATIO
        assert max(abs(hit["single_a"]), abs(hit["single_b"])) > scanner.LARGE


def test_current_surfaces_are_clean(scanner):
    """Neither live composite is held up by a cancelling pair. Recorded so
    that a future change which introduces one fails here rather than
    producing a plausible total for six releases, as both historical
    pairs did."""
    live = {hit["quantity"] for hit in scanner.scan()
            if "decode weight stream" not in hit["quantity"]
            and "annual facility energy" not in hit["quantity"]}
    assert not live, live


def test_the_scanner_runs_as_a_script():
    proc = subprocess.run([sys.executable, str(AUDIT / "opposing_errors.py")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0
    assert "2 cancelling pair(s)" in proc.stdout
