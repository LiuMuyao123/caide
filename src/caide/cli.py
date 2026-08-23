"""Command-line interface.

Six verbs, each answering one planning question:

``run``        what will this cost, where does the money go, how uncertain is it
``breakeven``  at what volume does the cheaper architecture change
``sweep``      how does a technique's benefit vary across the operating range
``route``      which model tier should serve which class of traffic
``catalog``    what presets are available
``init``       give me a scenario file to edit

Exit codes follow the usual convention: 0 success, 1 analysis-level
failure (infeasible configuration, SLO violation with ``--strict``),
2 malformed input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from . import __version__
from .breakeven import dominance_intervals, find_break_even
from .catalog import catalogue_summary, get_grid, get_hardware, get_model
from .costing import SIX_LAYERS
from .efficiency import PRESET_STACKS, TECHNIQUES, apply_stack
from .routing import Tier, optimise_routing
from .perturb import perturbed_cost, saturated_draw_keys
from .scaling import project
from .scenario import ScenarioError, Scenario, example_scenario, load_scenario
from .specs import DeploymentState
from .uncertainty import monte_carlo, sensitivity

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _money(v: float) -> str:
    if not math.isfinite(v):
        return "     n/a"
    a = abs(v)
    if a >= 1e9:
        return f"${v/1e9:>8,.2f}B"
    if a >= 1e6:
        return f"${v/1e6:>8,.2f}M"
    if a >= 1e3:
        return f"${v/1e3:>8,.1f}k"
    return f"${v:>9,.2f}"


def _count(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.2f}B"
    if a >= 1e6:
        return f"{v/1e6:.1f}M"
    if a >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def _scaling_inputs(result, assumptions):
    """Split the ownership total the way :mod:`caide.scaling` expects.

    The v8.0 audit established that declining a blended per-query figure
    at the token-tariff rate makes reviewer wages fall at the speed of GPU
    prices and a fixed audit programme scale with query volume. That fix
    landed in the reproduction script; this command kept passing
    ``effective_per_query`` with the other two components at zero, which
    is the defect verbatim, for five releases.
    """
    from dataclasses import replace as _replace
    split = result.scaling_inputs()
    return (split["declining_per_query"], result.annual_volume,
            _replace(assumptions,
                     price_inelastic_per_query=split["price_inelastic_per_query"],
                     fixed_annual_cost=split["fixed_annual"]))


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _echo(msg: str = "") -> None:
    print(msg, flush=True)


def _load(path: str) -> Scenario:
    try:
        return load_scenario(path)
    except ScenarioError as exc:
        _echo(f"error: {exc}")
        raise SystemExit(EXIT_USAGE)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    scenario = _load(args.scenario)
    warnings = scenario.validate()

    _echo(_rule("="))
    _echo(f"CAIDE {__version__}  |  {scenario.name}")
    _echo(f"{_count(scenario.annual_volume)} queries/yr  |  grid {scenario.grid.name}"
          f"  |  {len(scenario.workloads)} workload classes")
    _echo(_rule("="))

    for w in warnings:
        _echo(f"  ! {w}")
    if warnings:
        _echo()

    results = scenario.evaluate_all()

    _echo(f"{'architecture':<24}{'annual TCO':>13}{'$/query':>12}"
          f"{'quality':>9}{'replicas':>10}{'tCO2e':>9}")
    _echo(_rule())
    ordered = sorted(results.items(), key=lambda kv: kv[1].total)
    for name, r in ordered:
        rep = f"{r.capacity_units:.2f}" if math.isfinite(r.capacity_units) else "n/a"
        _echo(f"{name:<24}{_money(r.total):>13}"
              f"{r.effective_per_query:>12.6f}{r.quality_index:>9.3f}"
              f"{rep:>10}{r.annual_carbon_kg/1000:>9.1f}")
    _echo()

    # Cheapest *admissible*, not cheapest. The v10.0 audit made
    # feasibility the ranking criterion and the fix landed in
    # ReportBundle.cheapest(); this command computed its own minimum and
    # kept reporting an architecture that missed every declared quality
    # floor. Four rounds of headline fixes never reached here because the
    # command line is a second assembly of the same pipeline.
    admissible = [(n, r) for n, r in ordered if r.feasible]
    best_name, best = (admissible or ordered)[0]
    if not admissible:
        _echo("  ! no architecture meets every declared constraint; the "
              "cheapest overall is shown, and a mixed workload of this "
              "shape needs routing rather than one architecture")
    share = best.share()
    infra = share.get("compute_serving", 0) + share.get("model_access", 0)
    _echo(f"cheapest: {best_name}  ({_money(best.total)}/yr)")
    _echo(f"  inference is {infra:.0%} of TCO; "
          f"{1-infra:.0%} sits in layers that do not shrink with token prices")
    if best.review_hours_annual > 0:
        _echo(f"  human review: {best.review_hours_annual:,.0f} h/yr "
              f"(~{best.review_fte:.1f} FTE)")
    if best.displaced_labour_annual > 0:
        _echo(f"  displaced human effort: {_money(best.displaced_labour_annual)}/yr "
              f"-> net {_money(best.net_of_displaced_labour)}/yr")

    if args.layers:
        _echo()
        _echo("layer decomposition")
        _echo(_rule())
        names = [n for n, _ in ordered]
        header = f"{'layer':<26}" + "".join(f"{n[:13]:>14}" for n in names)
        _echo(header)
        for layer in SIX_LAYERS:
            vals = [results[n].layers.get(layer, 0.0) for n in names]
            if all(abs(v) < 1e-9 for v in vals):
                continue
            _echo(f"{layer:<26}" + "".join(f"{_money(v):>14}" for v in vals))

    violated = {n: r.slo_violations for n, r in results.items() if r.slo_violations}
    quality_bad = {n: r for n, r in results.items() if r.quality_violations}
    unchecked = {n: r.slo_unevaluated for n, r in results.items()
                 if r.slo_unevaluated}
    if violated or quality_bad or unchecked:
        _echo()
    for n, classes in violated.items():
        _echo(f"  ! {n}: SLO not met for {', '.join(classes)}")
    for n, r in quality_bad.items():
        detail = ", ".join(f"{c} (short by {r.quality_shortfall[c]:.1%})"
                           for c in r.quality_violations)
        _echo(f"  ! {n}: quality floor not met for {detail}")
    for n, classes in unchecked.items():
        _echo(f"  ? {n}: latency not evaluated for {', '.join(classes)}")
    for n, r in results.items():
        marginal = r.marginal_verdicts
        if marginal:
            _echo(f"  ~ {n}: admissibility rests on margins the quality "
                  f"index cannot resolve for {', '.join(marginal)}")
    if (violated or quality_bad) and args.strict:
        return EXIT_FAIL

    if scenario.scaling is not None:
        proj = project(*_scaling_inputs(best, scenario.scaling))
        _echo()
        _echo("scaling projection")
        _echo(_rule())
        _echo(f"  {proj.narrative()}")

    if args.out:
        _write_reports(scenario, results, args)

    return EXIT_OK


def _write_reports(scenario: Scenario, results: Dict, args: argparse.Namespace
                   ) -> None:
    from . import plotting
    from .report import ReportBundle, write_csv, write_html, write_markdown

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = ReportBundle(scenario, seed=args.seed)
    bundle.tco = results
    bundle.warnings = scenario.validate()

    fig = plotting.plot_tco_breakdown(results, out / "tco_breakdown.png")
    bundle.add_figure(fig)

    names = list(results)
    if len(names) >= 2:
        be = find_break_even(
            scenario.cost_curve(names[0]), scenario.cost_curve(names[1]),
            label_a=names[0], label_b=names[1],
            volume_min=max(scenario.annual_volume / 1e3, 1e3),
            volume_max=scenario.annual_volume * 1e3)
        bundle.break_evens.append(be)
        bundle.add_figure(plotting.plot_break_even(be, out / "break_even.png"))

    if scenario.uncertainty and args.samples > 0:
        mc_results = {}
        for name in names[:2]:
            arch = scenario.architecture(name)

            def _model(draw: Dict[str, float], _arch=arch) -> float:
                return perturbed_cost(scenario, _arch, draw)

            def _sat(draw: Dict[str, float], _arch=arch):
                return saturated_draw_keys(scenario, _arch, draw)

            mc = monte_carlo(_model, list(scenario.uncertainty.values()),
                             n_samples=args.samples, seed=args.seed, label=name,
                             saturation=_sat)
            mc_results[name] = mc
        bundle.monte_carlo = mc_results
        if mc_results:
            first = next(iter(mc_results.values()))
            bundle.sensitivity = sensitivity(first)
            bundle.add_figure(
                plotting.plot_uncertainty(mc_results, out / "uncertainty.png"))
            if bundle.sensitivity:
                bundle.add_figure(
                    plotting.plot_tornado(bundle.sensitivity, out / "tornado.png"))

    if scenario.scaling is not None:
        feasible = {k: v for k, v in results.items() if v.feasible}
        pool = feasible or results
        best = min(pool, key=lambda k: pool[k].total)
        proj = project(*_scaling_inputs(results[best], scenario.scaling))
        bundle.projection = proj
        bundle.add_figure(plotting.plot_scaling(proj, out / "scaling.png"))

    md = write_markdown(bundle, out / "report.md")
    csv_path = write_csv(bundle, out / "results.csv")
    html = write_html(bundle, out / "report.html")
    _echo()
    _echo(f"wrote {md}")
    _echo(f"wrote {csv_path}")
    _echo(f"wrote {html}")
    for f in bundle.figures:
        _echo(f"wrote {f}")


# ---------------------------------------------------------------------------
# breakeven
# ---------------------------------------------------------------------------

def cmd_breakeven(args: argparse.Namespace) -> int:
    scenario = _load(args.scenario)
    names = [a.name for a in scenario.architectures]
    a = args.a or names[0]
    b = args.b or (names[1] if len(names) > 1 else names[0])
    if a == b:
        _echo("error: need two distinct architectures to compare")
        return EXIT_USAGE
    for n in (a, b):
        if n not in names:
            _echo(f"error: unknown architecture {n!r}; defined: {names}")
            return EXIT_USAGE

    result = find_break_even(
        scenario.cost_curve(a), scenario.cost_curve(b),
        label_a=a, label_b=b,
        volume_min=args.min_volume, volume_max=args.max_volume)

    _echo(_rule("="))
    _echo(f"break-even: {a} vs {b}")
    _echo(f"scan {_count(args.min_volume)} .. {_count(args.max_volume)} queries/yr")
    _echo(_rule("="))

    if not result.crossings:
        winner = result.winner_at(
            math.sqrt(args.min_volume * args.max_volume))
        _echo(f"no crossing in range; {winner} is cheaper throughout")
    else:
        windows = result.tie_bands(args.tolerance)
        if len(result.crossings) > 4 and windows:
            _echo(f"{len(result.crossings)} crossings between "
                  f"{_count(result.crossings[0].volume)} and "
                  f"{_count(result.crossings[-1].volume)} queries/yr")
            _echo()
            _echo(f"{len(windows)} window"
                  f"{'s' if len(windows) > 1 else ''} in which cost does not "
                  f"decide (within {args.tolerance:.0%}), one per riser:")
            for lo, hi in windows:
                _echo(f"  {_count(lo)} .. {_count(hi)} queries/yr "
                      f"({hi / lo:.2f}x wide)")
            outside = max((result.relative_gap(v)
                           for v in sorted(result.curve_a)
                           if windows[0][0] <= v <= windows[-1][1]),
                          default=0.0)
            _echo(f"  Between the windows the options differ by up to "
                  f"{outside:.0%}; the whole span is not a tie region.")
            _echo("  Inside a window choose on latency, data residency,")
            _echo("  control, staffing or exit risk instead.")
            _echo()
            _echo("  The crossings are real but not actionable: self-hosted")
            _echo("  capacity arrives in whole replicas, so the advantage")
            _echo("  alternates each time a replica is added and then fills.")
        else:
            for c in result.crossings[:8]:
                _echo(f"  crossing at {_count(c.volume):>8} queries/yr "
                      f"({_money(c.cost_at_crossing)}/yr): "
                      f"{c.winner_below} -> {c.winner_above}")
            if len(result.crossings) > 8:
                _echo(f"  ... and {len(result.crossings) - 8} more")
            _echo()
            _echo("planning intervals")
            for lo, hi, who in dominance_intervals(result)[:10]:
                _echo(f"  {_count(lo):>8} .. {_count(hi):>8}  ->  {who}")

    if args.json:
        _echo()
        _echo(json.dumps(result.as_dict(), indent=2))
    return EXIT_OK


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def cmd_sweep(args: argparse.Namespace) -> int:
    scenario = _load(args.scenario)
    arch = scenario.architecture(args.architecture) if args.architecture \
        else next((a for a in scenario.architectures if a.kind == "self_hosted"), None)
    if arch is None or arch.state is None:
        _echo("error: sweep requires a self_hosted architecture")
        return EXIT_USAGE

    techniques = args.technique or ["speculative_decoding", "int4",
                                    "semantic_caching"]
    for t in techniques:
        if t not in TECHNIQUES:
            _echo(f"error: unknown technique {t!r}; "
                  f"available: {sorted(TECHNIQUES)}")
            return EXIT_USAGE

    from .costing import self_hosted_query_cost
    from dataclasses import replace as _replace

    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    workload = scenario.workloads[0]
    base_state = arch.state

    _echo(_rule("="))
    _echo(f"technique regime sweep  |  {arch.name}  |  workload {workload.name!r}")
    _echo(_rule("="))
    header = f"{'batch':>7}{'baseline $/q':>15}" + "".join(
        f"{t[:13]:>15}" for t in techniques)
    _echo(header)
    _echo(_rule())

    series: Dict[str, List[float]] = {t: [] for t in techniques}
    for bsz in batches:
        cfg = _replace(base_state.serving, max_batch=bsz)
        st = DeploymentState(base_state.model, base_state.hardware, cfg,
                             base_state.notes)
        base = self_hosted_query_cost(st, workload, scenario.grid,
                                      respect_slo=False).compute_cost
        row = f"{bsz:>7}{base:>15.6f}"
        for t in techniques:
            try:
                mod = apply_stack(st, [t])
                cost = self_hosted_query_cost(mod, workload, scenario.grid,
                                              respect_slo=False).compute_cost
                mult = cost / base if base > 0 else math.nan
            except (ValueError, KeyError):
                mult = math.nan
            series[t].append(mult)
            row += f"{mult:>14.3f}x"
        _echo(row)

    _echo()
    _echo("interpretation")
    _echo(_rule())
    for t, values in series.items():
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            continue
        lo, hi = min(finite), max(finite)
        spread = hi / lo if lo > 0 else math.inf
        _echo(f"  {t:<22} ranges {lo:.3f}x .. {hi:.3f}x  "
              f"({spread:.1f}x spread across batch)")
    _echo()
    _echo("  A single published multiplier can only be correct at one batch")
    _echo("  size. Where the spread above is large, quoting a constant")
    _echo("  misstates the benefit by that factor.")

    if args.out:
        from . import plotting
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = plotting.plot_technique_regimes(
            batches, series, out / "technique_regimes.png")
        _echo()
        _echo(f"wrote {path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

def cmd_route(args: argparse.Namespace) -> int:
    scenario = _load(args.scenario)
    from dataclasses import replace
    from .costing import (api_query_cost, self_hosted_query_cost,
                          total_cost_of_ownership)

    tiers: List[Tier] = []
    for arch in scenario.architectures:
        if arch.kind == "api" and arch.pricing is not None:
            tiers.append(Tier(
                name=arch.name, quality_index=arch.pricing.quality_index,
                cost_fn=lambda w, p=arch.pricing: api_query_cost(
                    p, w, scenario.grid).compute_cost,
            ))
        elif arch.state is not None:
            quality = arch.state.model.quality_index * (1 + arch.quality_penalty)

            # A self-hosted tier is billed in whole replicas, so its
            # annual cost is a staircase in what is routed to it, not a
            # per-query price times a share. Until v9.0 this tier was
            # priced marginally with a fixed cost that defaulted to zero,
            # which made standing up a replica look free and understated
            # a lightly loaded tier by 13x on the shipped public-service
            # scenario.
            def _annual(served, volume, s=arch.state, a=arch):
                shares = sum(w.share for w in served) or 1.0
                normalised = [replace(w, share=w.share / shares)
                              for w in served]
                return total_cost_of_ownership(
                    architecture="self_hosted",
                    annual_volume=volume * shares,
                    workloads=normalised, grid=scenario.grid, state=s,
                    slo=scenario.slo,
                    platform_engineering_annual=a.platform_engineering_annual,
                ).layers["compute_serving"] + args.tier_fixed_cost

            tiers.append(Tier(
                name=arch.name, quality_index=quality,
                cost_fn=lambda w, s=arch.state: self_hosted_query_cost(
                    s, w, scenario.grid, scenario.slo).compute_cost,
                annual_fixed_cost=args.tier_fixed_cost,
                annual_cost_fn=_annual,
            ))

    plan = optimise_routing(scenario.workloads, tiers, scenario.annual_volume)

    _echo(_rule("="))
    _echo(f"optimal routing  |  {scenario.name}")
    _echo(_rule("="))
    if not plan.feasible:
        _echo(f"  ! unroutable classes (no tier meets quality floor): "
              f"{', '.join(plan.unroutable)}")
    _echo(f"{'workload':<24}{'tier':<24}{'share':>8}{'$/query':>12}{'annual':>13}")
    _echo(_rule())
    for row in plan.summary_rows():
        _echo(f"{row['workload']:<24}{row['tier']:<24}"
              f"{row['share']:>8.1%}{row['usd_per_query']:>12.6f}"
              f"{_money(row['annual_usd']):>13}")
    _echo(_rule())
    _echo(f"{'blended':<48}{plan.per_query_cost:>12.6f}"
          f"{_money(plan.annual_cost):>13}")
    _echo()
    _echo(f"tiers opened: {', '.join(plan.tiers_opened)}")
    _echo(f"blended quality index: {plan.blended_quality:.3f}")
    if plan.tier_shares:
        caps = {t.name: t.max_share for t in tiers if t.max_share < 1.0}
        for name, share in sorted(plan.tier_shares.items()):
            if name in caps:
                _echo(f"  {name}: {share:.1%} of traffic "
                      f"(cap {caps[name]:.0%})")
    if not plan.exact:
        _echo("  ! this plan is not proven optimal")
    for note in plan.notes:
        _echo(f"  note: {note}")

    single = {}
    for t in tiers:
        try:
            cost = sum(w.share * t.cost(w) for w in scenario.workloads)
            if math.isfinite(cost):
                single[t.name] = cost * scenario.annual_volume + t.annual_fixed_cost
        except Exception:
            continue
    if single:
        cheapest_single = min(single, key=lambda k: single[k])
        saving = single[cheapest_single] - plan.annual_cost
        if saving > 0:
            _echo(f"routing saves {_money(saving)}/yr versus sending all traffic "
                  f"to {cheapest_single}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# examples / catalog / init / validate
# ---------------------------------------------------------------------------

def cmd_examples(args: argparse.Namespace) -> int:
    """List or extract the example scenarios bundled inside the package.

    Shipping the examples as package data rather than as repository-only
    files is what makes ``pip install caide`` followed by the commands in
    the documentation actually work. Before v2.0 they did not.
    """
    try:
        from importlib.resources import files as _res_files
        root = _res_files("caide") / "examples"
        entries = sorted(p.name for p in root.iterdir() if p.is_file())
    except Exception as exc:                             # pragma: no cover
        _echo(f"error: bundled examples are unavailable ({exc})")
        return EXIT_FAIL

    scenarios = [e for e in entries if e.endswith((".yaml", ".yml"))]
    scripts = [e for e in entries if e.endswith(".py")]
    docs = [e for e in entries if e.endswith(".md")]

    if not args.extract:
        _echo(_rule("="))
        _echo(f"bundled examples ({len(entries)} files)")
        _echo(_rule("="))
        _echo("\nscenarios")
        for name in scenarios:
            _echo(f"  {name}")
        if scripts:
            _echo("\nscripts")
            for name in scripts:
                _echo(f"  {name}")
        if docs:
            _echo("\ndocumentation")
            for name in docs:
                _echo(f"  {name}")
        _echo("\nextract them with:")
        _echo("  caide examples --extract .")
        return EXIT_OK

    target = Path(args.extract) / "examples"
    target.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for name in entries:
        dest = target / name
        if dest.exists() and not args.force:
            skipped.append(name)
            continue
        dest.write_bytes((root / name).read_bytes())
        written.append(name)

    for name in written:
        _echo(f"wrote {target / name}")
    if skipped:
        _echo(f"skipped {len(skipped)} existing file(s); pass --force to overwrite")
    if written:
        first = next((n for n in written if n.endswith(".yaml")), None)
        if first:
            _echo()
            _echo("try:")
            _echo(f"  caide run {target / first} --layers --out report/")
    return EXIT_OK


def cmd_catalog(args: argparse.Namespace) -> int:
    summary = catalogue_summary()
    if args.json:
        _echo(json.dumps(summary, indent=2))
        return EXIT_OK

    _echo(_rule("="))
    _echo(f"CAIDE {__version__} built-in catalogue")
    _echo(_rule("="))

    _echo("\nmodels")
    _echo(f"  {'key':<16}{'total':>10}{'active':>10}{'layers':>8}"
          f"{'d_model':>9}{'kv/tok':>10}{'quality':>9}")
    for key in summary["models"]:
        m = get_model(key)
        _echo(f"  {key:<16}{m.n_params_total/1e9:>9.1f}B"
              f"{m.active_params/1e9:>9.1f}B{m.n_layers:>8}"
              f"{m.d_model:>9}{m.kv_bytes_per_token/1024:>9.0f}K"
              f"{m.quality_index:>9.2f}")

    _echo("\naccelerators")
    _echo(f"  {'key':<16}{'TFLOP/s':>10}{'HBM':>8}{'BW TB/s':>10}"
          f"{'watts':>8}{'$/hr':>8}")
    for key in summary["hardware"]:
        h = get_hardware(key)
        _echo(f"  {key:<16}{h.peak_flops/1e12:>10.0f}"
              f"{h.memory_bytes/2**30:>7.0f}G{h.memory_bandwidth/1e12:>10.2f}"
              f"{h.power_watts:>8.0f}{h.hourly_cost:>8.2f}")

    _echo("\napi tiers ($/Mtok, illustrative)")
    from .catalog import get_pricing
    _echo(f"  {'key':<24}{'input':>9}{'output':>9}{'cached':>9}{'quality':>9}")
    for key in summary["pricing"]:
        p = get_pricing(key)
        cached = (f"{p.cached_input_per_mtok:.4f}"
                  if p.cached_input_per_mtok is not None else "-")
        _echo(f"  {key:<24}{p.input_per_mtok:>9.2f}{p.output_per_mtok:>9.2f}"
              f"{cached:>9}{p.quality_index:>9.2f}")

    _echo("\ngrids")
    _echo(f"  {'key':<18}{'kgCO2e/kWh':>12}{'PUE':>7}{'$/kWh':>8}{'L/kWh':>8}")
    for key in summary["grids"]:
        g = get_grid(key)
        _echo(f"  {key:<18}{g.carbon_intensity:>12.3f}{g.pue:>7.2f}"
              f"{g.electricity_cost:>8.2f}{g.wue:>8.1f}")

    _echo("\nefficiency techniques")
    _echo(f"  {'key':<24}{'quality':>9}{'hours':>8}  maturity")
    for key, t in TECHNIQUES.items():
        _echo(f"  {key:<24}{t.quality_delta:>+9.3f}{t.engineering_hours:>8.0f}"
              f"  {t.maturity}")

    _echo("\npreset stacks")
    for key, keys in PRESET_STACKS.items():
        _echo(f"  {key:<20}{', '.join(keys) if keys else '(empty)'}")
    return EXIT_OK


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        _echo(f"error: {path} exists; pass --force to overwrite")
        return EXIT_USAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(example_scenario(), sort_keys=False,
                          default_flow_style=False, allow_unicode=True)
    header = (
        "# CAIDE scenario. Edit freely, then run:\n"
        f"#   caide run {path.name} --out report/\n"
        "# Prices in the built-in catalogue are illustrative anchors --\n"
        "# replace them with your own quotations before trusting absolutes.\n\n"
    )
    path.write_text(header + text, encoding="utf-8")
    _echo(f"wrote {path}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except ScenarioError as exc:
        _echo(f"invalid: {exc}")
        return EXIT_USAGE
    warnings = scenario.validate()
    _echo(f"valid: {scenario.name}")
    _echo(f"  {len(scenario.workloads)} workload classes, "
          f"{len(scenario.architectures)} architectures")
    for w in warnings:
        _echo(f"  ! {w}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caide",
        description="Cost-Aware Inference Deployment Evaluator",
        epilog="Documentation: https://github.com/caide-tools/caide",
    )
    p.add_argument("--version", action="version",
                   version=f"caide {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="full analysis of a scenario")
    r.add_argument("scenario")
    r.add_argument("--out", help="directory for report, figures and CSV")
    r.add_argument("--samples", type=int, default=2000,
                   help="Monte Carlo draws (0 disables)")
    r.add_argument("--seed", type=int, default=20260101)
    r.add_argument("--layers", action="store_true",
                   help="print the six-layer decomposition")
    r.add_argument("--strict", action="store_true",
                   help="exit non-zero when an SLO is violated")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("breakeven", help="volume at which the cheaper option changes")
    b.add_argument("scenario")
    b.add_argument("-a", help="first architecture name")
    b.add_argument("-b", help="second architecture name")
    b.add_argument("--min-volume", type=float, default=1e4)
    b.add_argument("--max-volume", type=float, default=1e10)
    b.add_argument("--tolerance", type=float, default=0.05,
                   help="relative gap below which the options are called a tie")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_breakeven)

    s = sub.add_parser("sweep", help="technique benefit across the operating range")
    s.add_argument("scenario")
    s.add_argument("--technique", action="append",
                   help="technique key (repeatable)")
    s.add_argument("--architecture", help="which self-hosted architecture to sweep")
    s.add_argument("--out", help="directory for the regime figure")
    s.set_defaults(func=cmd_sweep)

    rt = sub.add_parser("route", help="optimal assignment of classes to tiers")
    rt.add_argument("scenario")
    rt.add_argument("--tier-fixed-cost", type=float, default=0.0,
                    help="annual cost of standing up one self-hosted tier")
    rt.set_defaults(func=cmd_route)

    ex = sub.add_parser("examples",
                        help="list or extract the bundled example scenarios")
    ex.add_argument("--extract", metavar="DIR",
                    help="write the examples into DIR/examples/")
    ex.add_argument("--force", action="store_true",
                    help="overwrite files that already exist")
    ex.set_defaults(func=cmd_examples)

    c = sub.add_parser("catalog", help="list built-in presets")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_catalog)

    i = sub.add_parser("init", help="write a starter scenario file")
    i.add_argument("path", nargs="?", default="scenario.yaml")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)

    v = sub.add_parser("validate", help="check a scenario without running it")
    v.add_argument("scenario")
    v.set_defaults(func=cmd_validate)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ScenarioError as exc:
        _echo(f"error: {exc}")
        return EXIT_USAGE
    except KeyboardInterrupt:                            # pragma: no cover
        _echo("\ninterrupted")
        return EXIT_FAIL


if __name__ == "__main__":                               # pragma: no cover
    sys.exit(main())
