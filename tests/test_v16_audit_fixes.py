"""Regression tests for the defects found in the v16.0 audit.

The v15 audit found ``ReportBundle.infeasible()`` sitting unused for five
releases and recommended extending the v10 dangling-field check to public
*methods*. Ten came back with no call site anywhere in the package. A
method nothing calls is a method nothing tests, and three of the ten were
models the project had already rejected, still exported.

The worst was ``ServingConfig.parallel_efficiency``: the multiplicative
tensor-parallel derate the v6.0 audit replaced with a per-layer
all-reduce. It survived ten releases as public API, claiming a 21% loss at
eight accelerators, batch-independent, where the model that replaced it
measures 0.02% at batch one and 1.43% at batch 256. A batch-independent
constant standing in for a regime-dependent cost is the thing this
package was written to argue against.
"""

import ast
import pathlib

import pytest

import caide
from caide import (
    ServingConfig,
    WorkloadClass,
    find_break_even,
    get_hardware,
    get_model,
    load_scenario,
)
from caide.report import ReportBundle, write_html, write_markdown
from caide.roofline import decode_step_time

SRC = pathlib.Path(caide.__file__).parent
EXAMPLES = SRC / "examples"


# ==========================================================================
# R16-1: one model of one physical effect
# ==========================================================================

def test_the_superseded_parallel_derate_is_gone():
    cfg = ServingConfig(n_accelerators=8, max_batch=64)
    assert not hasattr(cfg, "parallel_efficiency")


def test_the_interconnect_cost_is_regime_dependent():
    """Which is why a single derate could not express it. The share of a
    decode step spent synchronising rises with batch; the old property
    returned the same 21% at every batch."""
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")
    shares = []
    for batch in (1, 256):
        cfg = ServingConfig(n_accelerators=8, max_batch=batch)
        bare = ServingConfig(n_accelerators=8, max_batch=batch,
                             tensor_parallel_penalty=0.0)
        step, _ = decode_step_time(model, hw, cfg, batch, 4096.0)
        floor, _ = decode_step_time(model, hw, bare, batch, 4096.0)
        shares.append(1.0 - floor / step)
    assert shares[1] > 10 * shares[0]
    assert max(shares) < 0.05          # far below the 21% the derate claimed


# ==========================================================================
# R16-2: there is no primary crossing when there are several
# ==========================================================================

def _staircase():
    from caide.efficiency import PRESET_STACKS
    tuned = [k for k in PRESET_STACKS["aggressive"]
             if k != "speculative_decoding"]
    scenario = load_scenario({
        "name": "g", "annual_volume": 1e9, "grid": "us-average",
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
    return find_break_even(scenario.cost_curve("api-economy"),
                           scenario.cost_curve("selfhost-70b"),
                           label_a="api-economy", label_b="selfhost-70b",
                           volume_min=1e7, volume_max=2e10, samples=900)


def test_primary_refuses_to_summarise_a_staircase():
    result = _staircase()
    assert len(result.crossings) > 1
    with pytest.raises(ValueError, match="no primary one"):
        _ = result.primary


def test_primary_still_answers_when_there_is_one_crossing():
    result = find_break_even(lambda v: v, lambda v: 500.0,
                             volume_min=1.0, volume_max=1e4, samples=200)
    assert len(result.crossings) == 1
    assert result.primary is result.crossings[0]


def test_primary_is_none_when_nothing_crosses():
    result = find_break_even(lambda v: v, lambda v: 10 * v + 1.0,
                             volume_min=1.0, volume_max=1e3, samples=100)
    assert not result.crossings
    assert result.primary is None


# ==========================================================================
# R16-3: one formula, written once
# ==========================================================================

def test_the_roofline_uses_the_declared_average_sequence():
    workload = WorkloadClass("q", 1.0, 1500, 400)
    assert workload.avg_sequence == pytest.approx(1500 + 400 / 2.0)
    source = (SRC / "roofline.py").read_text()
    assert "workload.avg_sequence" in source
    assert "tokens_in + tokens_out / 2.0" not in source
    assert "workload.tokens_in + workload.tokens_out / 2.0" not in source


# ==========================================================================
# R16-4: a public method with no caller is a method with no test
# ==========================================================================

def _public_methods():
    out = []
    for path in sorted(SRC.glob("*.py")) + sorted(SRC.glob("examples/*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if (isinstance(stmt, ast.FunctionDef)
                            and not stmt.name.startswith("_")):
                        out.append((path, node.name, stmt.name, stmt.lineno))
    return out


#: Public API that a *user* calls and the package legitimately does not.
#: Every entry needs a reason, so that "unused" cannot quietly become the
#: default explanation for "never checked".
CALLER_IS_THE_USER = {
    "MonteCarloResult.probability_below",
    "MonteCarloResult.saturation_share",
    "CalibrationResult.improved",
    "BreakEvenResult.is_indistinguishable",
    "Scenario.blended_tokens",
    "ReportBundle.add_table",
    # Kept as user-facing API and now refusing to summarise a staircase;
    # exercised by the three tests above rather than by a call site.
    "BreakEvenResult.primary",
}


def test_no_unexplained_public_method_lacks_a_caller():
    """The v10 audit added this check for dataclass fields after four
    dangling parameters. v15 found the same failure in a method, so it
    extends here. ``parallel_efficiency`` and ``any_feasible`` were on
    this list until v16.0 -- one deleted, one wired up."""
    texts = {p: p.read_text().splitlines()
             for p in sorted(SRC.glob("*.py")) + sorted(SRC.glob("examples/*.py"))}
    dangling = []
    for path, cls, name, lineno in _public_methods():
        uses = 0
        for p, lines in texts.items():
            for i, line in enumerate(lines, 1):
                if p == path and i == lineno:
                    continue
                if f"{name}(" in line or f".{name}" in line:
                    uses += 1
        if uses == 0 and f"{cls}.{name}" not in CALLER_IS_THE_USER:
            dangling.append(f"{path.name}:{cls}.{name}")
    assert not dangling, f"public and never called: {dangling}"


def test_any_feasible_reaches_both_writers(tmp_path):
    """Delivered in v11 beside ``infeasible()``; v15 wired up the first
    and left this one. Third time a named pair has been half-cleared."""
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    bundle = ReportBundle(scenario)
    bundle.tco = scenario.evaluate_all()
    assert not bundle.any_feasible
    md = write_markdown(bundle, tmp_path / "r.md").read_text()
    html = write_html(bundle, tmp_path / "r.html").read_text()
    assert "No architecture here meets" in md
    assert "No architecture here meets" in html


def test_a_scenario_with_an_admissible_option_says_nothing(tmp_path):
    scenario = load_scenario(EXAMPLES / "hospital_documentation.yaml")
    bundle = ReportBundle(scenario)
    bundle.tco = scenario.evaluate_all()
    assert bundle.any_feasible
    md = write_markdown(bundle, tmp_path / "r.md").read_text()
    assert "No architecture here meets" not in md
