"""Regression tests for the defects found in the v15.0 audit.

The v14 audit found the terminal reporting one answer while the Markdown
report written by the same invocation reported another, and fixed the
terminal. This round opened the other things that invocation writes.

``caide run`` produces four artefacts: a terminal summary, a Markdown
report, an HTML dashboard and a CSV. Every constraint-aware correction
since v9 had reached the first and stopped there. The CSV, the only one
meant for a machine, had never been opened by any audit round at all: its
``architecture`` column carried the architecture *kind*, so a clinical
scenario produced three rows all reading ``self_hosted`` and the file
could not distinguish the candidates it was comparing.
"""

import csv
import re
from pathlib import Path

import pytest

import caide
from caide import load_scenario
from caide.report import ReportBundle, write_csv, write_html, write_markdown

EXAMPLES = Path(caide.__file__).parent / "examples"
CLINICAL = EXAMPLES / "hospital_documentation.yaml"


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("v15")
    scenario = load_scenario(CLINICAL)
    bundle = ReportBundle(scenario, seed=1)
    bundle.tco = scenario.evaluate_all()
    return {
        "bundle": bundle,
        "csv": write_csv(bundle, out / "results.csv"),
        "md": write_markdown(bundle, out / "report.md"),
        "html": write_html(bundle, out / "report.html"),
    }


# ==========================================================================
# R15-1: the machine-readable output identifies what it is comparing
# ==========================================================================

def test_the_csv_names_the_architectures(artefacts):
    rows = list(csv.DictReader(artefacts["csv"].open()))
    names = [r["name"] for r in rows]
    assert sorted(names) == ["onprem-32b", "onprem-70b", "onprem-8b-int4"]
    assert len(set(names)) == len(rows)


def test_the_csv_kind_column_is_still_the_kind(artefacts):
    """The old column was not wrong about its own meaning, only about the
    job it was doing. It stays, and the name joins it."""
    rows = list(csv.DictReader(artefacts["csv"].open()))
    assert {r["architecture"] for r in rows} == {"self_hosted"}


def test_the_csv_carries_the_admissibility_verdict(artefacts):
    """Sorting this file by total reproduced the pre-v10 answer, with no
    column that could have warned the reader."""
    rows = list(csv.DictReader(artefacts["csv"].open()))
    by_name = {r["name"]: r for r in rows}
    assert by_name["onprem-8b-int4"]["feasible"] == "False"
    assert by_name["onprem-70b"]["feasible"] == "True"
    assert "discharge_summary" in by_name["onprem-32b"]["quality_violations"]
    assert by_name["onprem-70b"]["quality_violations"] == ""


def test_sorting_the_csv_by_cost_no_longer_hides_the_constraint(artefacts):
    rows = sorted(csv.DictReader(artefacts["csv"].open()),
                  key=lambda r: float(r["total_usd"]))
    assert rows[0]["feasible"] == "False"        # cheapest overall
    admissible = [r for r in rows if r["feasible"] == "True"]
    assert admissible[0]["name"] == "onprem-70b"


# ==========================================================================
# R15-2: every written artefact states what the constraints ruled out
# ==========================================================================

def test_the_markdown_report_lists_what_was_ruled_out(artefacts):
    text = artefacts["md"].read_text()
    assert "not admissible" in text
    assert "onprem-8b-int4" in text and "discharge_summary" in text


def test_the_html_dashboard_lists_what_was_ruled_out(artefacts):
    text = artefacts["html"].read_text()
    assert "Not admissible" in text
    assert "onprem-8b-int4" in text


def test_the_written_artefacts_name_the_same_architecture(artefacts):
    """One invocation, one answer -- extended from the terminal and the
    Markdown report to all four."""
    best = artefacts["bundle"].cheapest()
    md = re.search(r"Cheapest architecture is \*\*([\w-]+)\*\*",
                   artefacts["md"].read_text())
    html = re.search(r"<strong>([\w-]+)</strong> is cheapest",
                     artefacts["html"].read_text())
    assert md and html
    assert md.group(1) == html.group(1) == best == "onprem-70b"

    rows = list(csv.DictReader(artefacts["csv"].open()))
    admissible = sorted((r for r in rows if r["feasible"] == "True"),
                        key=lambda r: float(r["total_usd"]))
    assert admissible[0]["name"] == best


def test_unevaluated_constraints_reach_the_written_artefacts(tmp_path):
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    bundle = ReportBundle(scenario)
    bundle.tco = scenario.evaluate_all()
    md = write_markdown(bundle, tmp_path / "r.md").read_text()
    html = write_html(bundle, tmp_path / "r.html").read_text()
    assert "not evaluated" in md.lower()
    assert "not evaluated" in html.lower()


# ==========================================================================
# R15-3: the tie set is a union of windows in every consumer
# ==========================================================================

def _staircase_bundle(tmp_path):
    from caide import find_break_even
    from caide.efficiency import PRESET_STACKS
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
    result = find_break_even(
        scenario.cost_curve("api-economy"),
        scenario.cost_curve("selfhost-70b"),
        label_a="api-economy", label_b="selfhost-70b",
        volume_min=1e7, volume_max=2e10, samples=900)
    bundle = ReportBundle(scenario)
    bundle.tco = scenario.evaluate_all()
    bundle.break_evens.append(result)
    return bundle, result


def test_the_reports_describe_windows_not_a_band(tmp_path):
    """The v9 audit gave ``tie_bands`` for this and every consumer kept
    calling ``tie_band``: three artefacts describing four narrow windows,
    and the 47% gaps between them, as one region where cost does not
    decide."""
    bundle, result = _staircase_bundle(tmp_path)
    assert len(result.tie_bands(0.05)) == 4
    md = write_markdown(bundle, tmp_path / "b.md").read_text()
    html = write_html(bundle, tmp_path / "b.html").read_text()
    assert "4 window" in md
    assert "indistinguishable band" not in md
    # the HTML dashboard renders the same result through the same helper
    assert "4 narrow window" in html or "4 window" in html
    assert "inside that band" not in html


def test_the_reports_state_the_gap_between_the_windows(tmp_path):
    """A threshold summary has to report its distance to the threshold --
    the lesson of v9, v12 and v13, applied to the consumers this time."""
    bundle, _ = _staircase_bundle(tmp_path)
    md = write_markdown(bundle, tmp_path / "b.md").read_text()
    assert re.search(r"differ by up to 4\d%", md)


def test_a_single_window_still_reads_naturally(tmp_path):
    from caide import find_break_even
    result = find_break_even(lambda v: 1.0 + (v - 100.0) ** 2 / 1e4,
                             lambda v: 1.0,
                             volume_min=1.0, volume_max=1e4, samples=400)
    assert len(result.tie_bands(0.20)) == 1
