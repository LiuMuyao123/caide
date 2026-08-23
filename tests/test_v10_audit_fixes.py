"""Regression tests for the defects found in the v10.0 audit.

This round began by shipping the check the v9 report asked for: walk
every public dataclass field and assert something reads it. It found two
fields that nothing read, and one of them -- ``vocab_size`` -- turned out
to be the field needed to price an effect the model had been getting
wrong in both directions at once.

It also found the check's own limit. ``quality_floor`` passes it: one
module reads it. That module is :mod:`caide.routing`, and the *other*
path -- the architecture comparison behind every published verdict --
ignored it, so in all three shipped scenarios the architecture reported
cheapest failed a floor the scenario itself declared. A field being read
somewhere is not the property worth checking; the property is that every
path which should honour a constraint does.
"""

import ast
import math
import pathlib
from dataclasses import replace

import pytest

import caide
from caide import (
    DeploymentState,
    ServingConfig,
    WorkloadClass,
    apply_stack,
    get_grid,
    get_hardware,
    get_model,
    load_scenario,
    self_hosted_query_cost,
)
from caide.costing import total_cost_of_ownership
from caide.efficiency import QUANTISATION_HEAD_BYTES
from caide.report import ReportBundle, scenario_digest
from caide.roofline import decode_step_time
from caide.scaling import ScalingAssumptions, project
from caide.uncertainty import lognormal, point, triangular, uniform

EXAMPLES = pathlib.Path(caide.__file__).parent / "examples"
SRC = pathlib.Path(caide.__file__).parent


# ==========================================================================
# The check the v9 report asked for, shipped
# ==========================================================================

def _public_dataclass_fields():
    out = []
    for path in sorted(SRC.glob("*.py")) + sorted(SRC.glob("examples/*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            decorated = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Call)
                    and getattr(d.func, "id", "") == "dataclass")
                or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                for d in node.decorator_list)
            if not decorated:
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and not stmt.target.id.startswith("_")):
                    out.append((path, node.name, stmt.target.id, stmt.lineno))
    return out


def test_every_public_dataclass_field_is_read_somewhere():
    """Four dangling parameters in nine rounds is a pattern, not a series
    of accidents: ``framework_overhead_per_step`` (v5),
    ``expert_imbalance`` (v6), the draft KV term (v8), ``max_share`` (v9),
    and this round ``vocab_size``. A field that is declared, documented
    and never read is invisible to a test suite, because every test that
    could have caught it would have had to be written by someone who had
    already noticed."""
    texts = {p: p.read_text().splitlines()
             for p in sorted(SRC.glob("*.py")) + sorted(SRC.glob("examples/*.py"))}
    dangling = []
    for path, cls, name, lineno in _public_dataclass_fields():
        uses = 0
        for p, lines in texts.items():
            for i, line in enumerate(lines, 1):
                if p == path and i == lineno:
                    continue
                if name in line:
                    uses += 1
        if uses == 0:
            dangling.append(f"{path.name}:{cls}.{name}")
    assert not dangling, f"declared and never read: {dangling}"


def test_the_field_check_would_not_have_caught_the_worst_case():
    """Recorded so the check is not mistaken for more than it is.

    ``quality_floor`` is read -- by one module. The check counts
    references, and one reference is enough to pass it, which is exactly
    the shape of R10-1: a constraint honoured on one path and ignored on
    the one that mattered.
    """
    hits = sum(
        line.count("quality_floor")
        for p in SRC.glob("*.py")
        for line in p.read_text().splitlines()
    )
    assert hits > 1


# ==========================================================================
# R10-1: a declared constraint, honoured on every path
# ==========================================================================

@pytest.mark.parametrize("name", ["university_tutoring",
                                  "hospital_documentation",
                                  "public_helpline"])
def test_quality_floor_violations_are_recorded(name):
    scenario = load_scenario(EXAMPLES / f"{name}.yaml")
    results = scenario.evaluate_all()
    for arch, result in results.items():
        expected = [w.name for w in scenario.workloads
                    if w.quality_floor > result.quality_index + 1e-12]
        assert result.quality_violations == expected, arch
        assert result.feasible == (not expected and not result.slo_violations)


def test_the_cheapest_architecture_is_the_cheapest_admissible_one():
    """In every shipped scenario the pre-v10 answer failed a floor."""
    for name, expected in (("university_tutoring", "api-frontier"),
                           ("hospital_documentation", "onprem-70b")):
        scenario = load_scenario(EXAMPLES / f"{name}.yaml")
        results = scenario.evaluate_all()
        bundle = ReportBundle(scenario)
        bundle.tco = results
        assert bundle.cheapest() == expected
        # and the old answer is present but ruled out
        cheapest_overall = min(results, key=lambda k: results[k].total)
        assert cheapest_overall != expected
        assert not results[cheapest_overall].feasible


def test_a_scenario_can_have_no_admissible_architecture():
    """The public-service scenario has none, and the routing command has
    been saying so about the same file since v6.0 -- two commands in one
    package, contradicting each other about one input."""
    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    results = scenario.evaluate_all()
    assert not any(r.feasible for r in results.values())
    bundle = ReportBundle(scenario)
    bundle.tco = results
    assert not bundle.any_feasible
    assert set(bundle.infeasible()) == set(results)


def test_a_floor_of_zero_admits_everything():
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                            ServingConfig(n_accelerators=1, max_batch=64))
    workloads = [WorkloadClass("q", 1.0, 800, 200, quality_floor=0.0)]
    result = total_cost_of_ownership(
        architecture="self_hosted", annual_volume=1e6, workloads=workloads,
        grid=get_grid("us-average"), state=state)
    assert result.quality_violations == []
    assert result.feasible


def test_the_violation_is_named_not_priced():
    """CAIDE does not know what a capability shortfall costs; inventing a
    penalty would be worse than naming the classes."""
    state = DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                            ServingConfig(n_accelerators=1, max_batch=64))
    low = [WorkloadClass("q", 1.0, 800, 200, quality_floor=0.0)]
    high = [WorkloadClass("q", 1.0, 800, 200, quality_floor=0.99)]
    kw = dict(architecture="self_hosted", annual_volume=1e6,
              grid=get_grid("us-average"), state=state)
    a = total_cost_of_ownership(workloads=low, **kw)
    b = total_cost_of_ownership(workloads=high, **kw)
    assert b.total == pytest.approx(a.total, rel=1e-12)
    assert b.quality_violations == ["q"]
    assert any("quality floor" in n for n in b.notes)


# ==========================================================================
# R10-2 / R10-5: a digest that moves when the inputs move
# ==========================================================================

def _tutoring_raw():
    import yaml
    return yaml.safe_load((EXAMPLES / "university_tutoring.yaml").read_text())


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda r: r["assurance"].__setitem__("reviewer_hourly_cost", 95.0),
                 id="reviewer_wage"),
    pytest.param(lambda r: r["assurance"].__setitem__("evaluation_annual", 9_999_999),
                 id="evaluation_programme"),
    pytest.param(lambda r: [w.__setitem__("quality_floor", 0.99)
                            for w in r["workloads"]], id="quality_floor"),
    pytest.param(lambda r: r["uncertainty"]["volume_scale"].__setitem__("sigma", 0.9),
                 id="distribution_spread"),
    pytest.param(lambda r: r.__setitem__(
        "grid", {"name": "custom", "carbon_intensity": 0.8, "pue": 1.2,
                 "wue": 1.8, "electricity_cost": 0.08}), id="grid_intensity"),
])
def test_digest_moves_when_an_input_moves(mutate):
    """The report tells the reader a digest proves the inputs did not
    change. Three of these edits move total cost by 32-43% and left the
    hash byte-identical through v9.0."""
    import copy
    raw = _tutoring_raw()
    base = scenario_digest(load_scenario(raw))
    mutated = copy.deepcopy(raw)
    mutate(mutated)
    assert scenario_digest(load_scenario(mutated)) != base


def test_digest_is_stable_under_reserialisation():
    scenario = load_scenario(EXAMPLES / "university_tutoring.yaml")
    assert scenario_digest(scenario) == scenario_digest(scenario)


def test_distributions_retain_their_parameters():
    """A sampler is a closure; without the parameters the assumption
    exists nowhere a report or a digest can reach it."""
    assert uniform("u", 1.0, 3.0).params == {"low": 1.0, "high": 3.0}
    assert triangular("t", 0.5, 1.0, 2.2).params == {
        "low": 0.5, "mode": 1.0, "high": 2.2}
    assert lognormal("l", 1.0, 0.3).params == {"median": 1.0, "sigma": 0.3}
    assert point("p", 4.0).params == {"value": 4.0}
    described = lognormal("l", 1.0, 0.3).describe()
    assert described["kind"] == "lognormal" and described["sigma"] == 0.3


# ==========================================================================
# R10-3 / R10-4: what a decode step streams, and at what precision
# ==========================================================================

def test_quantisation_leaves_the_head_alone():
    """GPTQ, AWQ and the k-quants all skip the embedding and the head.
    Quantising them uniformly overstated int4's saving by 1.39x on the
    smallest bundled archetype."""
    for key in ("dense-8b", "dense-70b", "moe-8x7b"):
        model = get_model(key)
        state = DeploymentState(model, get_hardware("h100-sxm"),
                                ServingConfig(n_accelerators=8, max_batch=64))
        quantised = apply_stack(state, ["int4"]).model
        assert quantised.bytes_per_param == 0.5
        assert quantised.head_bytes == QUANTISATION_HEAD_BYTES

        hp = model.embedding_params + model.lm_head_params
        expected = (model.n_params_total - hp) * 0.5 + hp * 2.0
        assert quantised.weight_bytes == pytest.approx(expected, rel=1e-9)


def test_parameter_count_is_preserved_by_the_precision_split():
    """Splitting precision moves bytes between blocks; it must never
    invent or lose a parameter."""
    for key in ("dense-8b", "dense-405b", "moe-8x7b", "moe-236b"):
        model = get_model(key)
        assert model.weight_bytes == pytest.approx(
            model.n_params_total * model.bytes_per_param, rel=1e-12)
        split = model.with_precision(0.5, head_bytes_per_param=2.0)
        hp = model.embedding_params + model.lm_head_params
        implied = ((split.weight_bytes - hp * 2.0) / 0.5) + hp
        assert implied == pytest.approx(model.n_params_total, rel=1e-9)


def test_the_input_embedding_is_gathered_not_streamed():
    """A decode step reads ``batch`` rows of the table, not the table."""
    for key in ("dense-8b", "dense-70b"):
        model = get_model(key)
        streamed = model.decode_weight_bytes(1.0)
        resident = model.weight_bytes
        assert streamed < resident
        assert resident - streamed == pytest.approx(
            model.embedding_params * model.head_bytes, rel=1e-9)


def test_the_two_errors_pointed_in_opposite_directions():
    """Which is why the total looked reasonable for nine releases: the
    head was over-quantised (understating traffic) while the embedding
    was streamed (overstating it)."""
    model = get_model("dense-8b")
    old_bf16 = model.n_params_total * model.bytes_per_param
    new_bf16 = model.decode_weight_bytes(1.0)
    assert new_bf16 < old_bf16                       # embedding removed

    int4 = model.with_precision(0.5, head_bytes_per_param=2.0)
    old_int4 = model.n_params_total * 0.5
    new_int4 = int4.decode_weight_bytes(1.0)
    assert new_int4 > old_int4                       # head kept at bf16


def test_int4_multiplier_moved_but_kept_its_shape():
    """The published constant is still wrong by well over half its value,
    and still wrong in the same direction, at every batch."""
    grid = get_grid("us-average")
    workload = WorkloadClass("tutoring", 1.0, 1500, 400)
    hw = get_hardware("h100-sxm")
    curve = []
    for batch in (1, 16, 64, 256):
        state = DeploymentState(get_model("dense-70b"), hw,
                                ServingConfig(n_accelerators=4, max_batch=batch))
        base = self_hosted_query_cost(state, workload, grid,
                                      respect_slo=False).compute_cost
        cost = self_hosted_query_cost(apply_stack(state, ["int4"]), workload,
                                      grid, respect_slo=False).compute_cost
        curve.append(cost / base)
    assert curve == sorted(curve)
    assert 0.26 < curve[0] < 0.28
    assert 0.52 < curve[-1] < 0.55
    assert max(abs(0.65 - c) / c for c in curve) > 1.3


def test_moe_batch_one_identity_survives_the_precision_split():
    for key in ("moe-8x7b", "moe-8x22b", "moe-236b"):
        model = get_model(key)
        assert model.expert_bytes_touched(1.0) == pytest.approx(
            model.active_params * model.bytes_per_param, rel=1e-12)


def test_vocabulary_size_now_changes_something():
    """The test the field never had. A model with twice the vocabulary
    streams a larger head and quantises a smaller share of itself."""
    model = get_model("dense-8b")
    # Grow the vocabulary and the parameter count together, so the body
    # is held fixed and only the two vocabulary matrices change.
    extra = 2.0 * model.vocab_size * model.d_model
    wide = replace(model, vocab_size=model.vocab_size * 2,
                   n_params_total=model.n_params_total + extra)
    assert wide.decode_weight_bytes(1.0) > model.decode_weight_bytes(1.0)
    q_narrow = model.with_precision(0.5, head_bytes_per_param=2.0)
    q_wide = wide.with_precision(0.5, head_bytes_per_param=2.0)
    narrow_ratio = q_narrow.weight_bytes / (model.n_params_total * 0.5)
    wide_ratio = q_wide.weight_bytes / (wide.n_params_total * 0.5)
    assert wide_ratio > narrow_ratio


def test_tied_embeddings_count_one_matrix():
    model = get_model("dense-8b")
    tied = replace(model, tied_embeddings=True)
    assert tied.lm_head_params == 0.0
    assert tied.weight_bytes < model.weight_bytes or \
        tied.bytes_per_param == tied.head_bytes


# ==========================================================================
# R10-6: a projection that hits its ceiling says so
# ==========================================================================

def test_saturation_is_reported_in_the_narrative():
    """Computed since v1.0, read by nothing until now: a projection that
    hit its declared ceiling reported a flattened curve and no reason."""
    assumptions = ScalingAssumptions(annual_price_decline=0.38,
                                     price_elasticity=1.8,
                                     autonomous_growth=0.2, horizon_years=5,
                                     capacity_ceiling=2.0e7)
    projection = project(0.01, 1.0e7, assumptions)
    assert projection.saturated_from is not None
    assert "ceiling" in projection.narrative()
    rows = projection.table()
    assert any(r["saturated"] for r in rows)
    assert not rows[0]["saturated"]


def test_an_unsaturated_projection_says_nothing_about_ceilings():
    projection = project(0.01, 1.0e7,
                         ScalingAssumptions(annual_price_decline=0.1,
                                            price_elasticity=0.5,
                                            autonomous_growth=0.0,
                                            horizon_years=3))
    assert projection.saturated_from is None
    assert "ceiling" not in projection.narrative()
    assert not any(r["saturated"] for r in projection.table())


def test_non_convergence_is_reported_too():
    projection = project(0.02, 1.0e6,
                         ScalingAssumptions(annual_price_decline=0.2,
                                            price_elasticity=1.0,
                                            autonomous_growth=0.0,
                                            horizon_years=3))
    assert all(y.converged for y in projection.years)
    assert "did not converge" not in projection.narrative()


def test_step_time_stays_finite_and_ordered_after_the_stream_change():
    """Guard rail: the precision split touches every model in the
    catalogue, so check nothing became infinite or non-monotone."""
    hw = get_hardware("h100-sxm")
    for key in ("dense-8b", "dense-70b", "dense-405b", "moe-8x7b", "moe-236b"):
        model = get_model(key)
        previous = 0.0
        for batch in (1, 16, 256):
            cfg = ServingConfig(n_accelerators=8, max_batch=batch)
            step, bound = decode_step_time(model, hw, cfg, batch, 4096.0)
            assert math.isfinite(step) and step > 0
            assert step >= previous
            previous = step
            assert bound in ("memory", "compute", "interconnect")
