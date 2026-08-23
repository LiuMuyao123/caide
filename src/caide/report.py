"""Report generation: Markdown, CSV and a self-contained HTML dashboard.

The HTML output embeds its figures as base64 data URIs and carries no
external assets, so a single file can be mailed to a finance office or
attached to a procurement record and still render in five years. That
durability is the point: a deployment decision outlives the analysis
environment that produced it.

Every report records the CAIDE version, the scenario digest and the seed,
because a cost figure without its provenance cannot be defended in a
budget review.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import __version__
from .breakeven import BreakEvenResult, dominance_intervals
from .costing import SIX_LAYERS, TCOResult
from .scaling import ScalingProjection
from .scenario import Scenario
from .uncertainty import MonteCarloResult, SensitivityEntry

__all__ = [
    "ReportBundle",
    "write_markdown",
    "write_csv",
    "write_html",
    "scenario_digest",
]

LAYER_LABELS = {
    "model_access": "Model access",
    "compute_serving": "Compute & serving",
    "retrieval_data": "Retrieval & data",
    "integration_sre": "Integration & SRE",
    "assurance_governance": "Assurance & governance",
    "workforce_redesign": "Workforce & redesign",
}


def scenario_digest(scenario: Scenario) -> str:
    """Stable short hash of the analysis inputs, for provenance."""
    # A digest that does not move when an input moves is worse than no
    # digest, because the report says it proves the inputs did not change.
    # Until v10.0 this payload omitted the assurance profile, every cost
    # layer, the uncertainty distributions, the scaling assumptions, four
    # of the workload fields and everything about a grid except its name.
    # Three edits worth 32-43% of total cost left the hash byte-identical.
    def _layer(layer):
        if layer is None:
            return None
        return [layer.name, layer.fixed_annual, layer.per_query,
                layer.sublinear_coefficient, layer.sublinear_exponent,
                layer.step_size, layer.step_cost, layer.front_load_year1,
                layer.decay]

    a = scenario.assurance
    g = scenario.grid
    payload = {
        "name": scenario.name,
        "annual_volume": scenario.annual_volume,
        "workloads": [
            {"n": w.name, "s": w.share, "i": w.tokens_in, "o": w.tokens_out,
             "k": w.self_consistency_k, "r": w.review_rate,
             "m": w.review_minutes, "b": w.baseline_minutes,
             "q": w.quality_floor, "c": w.cacheable,
             "l": w.latency_sensitive}
            for w in scenario.workloads
        ],
        "architectures": [a_.describe() for a_ in scenario.architectures],
        "grid": [g.name, g.carbon_intensity, g.pue, g.wue,
                 g.electricity_cost],
        "slo": [scenario.slo.ttft_seconds, scenario.slo.tpot_seconds,
                scenario.slo.enforce] if scenario.slo else None,
        "assurance": [a.audit_logging_annual, a.evaluation_annual,
                      a.red_team_annual, a.privacy_review_annual,
                      a.incident_response_annual, a.reviewer_hourly_cost,
                      a.storage_per_query] if a is not None else None,
        "layers": [_layer(scenario.retrieval), _layer(scenario.integration),
                   _layer(scenario.workforce)],
        "uncertainty": sorted(
            (name, d.kind, d.nominal, sorted(d.params.items()))
            for name, d in (scenario.uncertainty or {}).items()),
        "provider_energy_wh_per_ktok": scenario.provider_energy_wh_per_ktok,
        "scaling": [scenario.scaling.annual_price_decline,
                    scenario.scaling.price_elasticity,
                    scenario.scaling.autonomous_growth,
                    scenario.scaling.horizon_years,
                    scenario.scaling.capacity_ceiling,
                    scenario.scaling.fixed_annual_cost,
                    scenario.scaling.price_inelastic_per_query]
        if scenario.scaling is not None else None,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


class ReportBundle:
    """Collects everything an analysis produced, then serialises it."""

    def __init__(self, scenario: Scenario, seed: Optional[int] = None):
        self.scenario = scenario
        self.seed = seed
        self.generated = _dt.datetime.now(_dt.timezone.utc)
        self.tco: Dict[str, TCOResult] = {}
        self.break_evens: List[BreakEvenResult] = []
        self.monte_carlo: Dict[str, MonteCarloResult] = {}
        self.sensitivity: List[SensitivityEntry] = []
        self.projection: Optional[ScalingProjection] = None
        self.figures: List[Path] = []
        self.warnings: List[str] = []
        self.extra_tables: Dict[str, List[Dict[str, Any]]] = {}

    # -- assembly -------------------------------------------------------

    def add_figure(self, path: Path) -> "ReportBundle":
        self.figures.append(Path(path))
        return self

    def add_table(self, name: str, rows: List[Dict[str, Any]]) -> "ReportBundle":
        self.extra_tables[name] = rows
        return self

    @property
    def digest(self) -> str:
        return scenario_digest(self.scenario)

    def cheapest(self) -> Optional[str]:
        """Cheapest architecture that meets every declared constraint.

        Falls back to the cheapest overall only when nothing is feasible,
        and :meth:`infeasible` says so. Ranking an architecture that
        cannot serve the workload above one that can is not a cost
        comparison; it is a comparison of two different services.
        """
        if not self.tco:
            return None
        feasible = {k: v for k, v in self.tco.items() if v.feasible}
        pool = feasible or self.tco
        return min(pool, key=lambda k: pool[k].total)

    def infeasible(self) -> Dict[str, List[str]]:
        """Architectures ruled out, with the classes that ruled them out."""
        return {k: sorted(set(v.quality_violations) | set(v.slo_violations))
                for k, v in self.tco.items() if not v.feasible}

    @property
    def any_feasible(self) -> bool:
        return any(v.feasible for v in self.tco.values())


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _money(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"${v/1e9:,.2f}B"
    if a >= 1e6:
        return f"${v/1e6:,.2f}M"
    if a >= 1e3:
        return f"${v/1e3:,.1f}k"
    return f"${v:,.2f}"


def _count(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:,.2f}B"
    if a >= 1e6:
        return f"{v/1e6:,.1f}M"
    if a >= 1e3:
        return f"{v/1e3:,.0f}k"
    return f"{v:,.0f}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def write_markdown(bundle: ReportBundle, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = bundle.scenario
    parts: List[str] = []

    parts.append(f"# CAIDE deployment analysis: {s.name}\n")
    if s.description:
        parts.append(f"{s.description}\n")
    parts.append(_md_table(
        ["Field", "Value"],
        [["Generated (UTC)", bundle.generated.strftime("%Y-%m-%d %H:%M:%S")],
         ["CAIDE version", __version__],
         ["Scenario digest", f"`{bundle.digest}`"],
         ["Random seed", bundle.seed if bundle.seed is not None else "n/a"],
         ["Annual volume", _count(s.annual_volume)],
         ["Grid", f"{s.grid.name} ({s.grid.carbon_intensity:.3f} kg CO2e/kWh, "
                  f"PUE {s.grid.pue:.2f})"],
         ["SLO", f"TTFT <= {s.slo.ttft_seconds:.2f}s, "
                 f"TPOT <= {s.slo.tpot_seconds*1000:.0f}ms" if s.slo else "none"]],
    ))
    parts.append("")

    if bundle.warnings:
        parts.append("## Warnings\n")
        for w in bundle.warnings:
            parts.append(f"- {w}")
        parts.append("")

    parts.append("## Workload mix\n")
    parts.append(_md_table(
        ["Class", "Share", "Tokens in", "Tokens out", "k", "Review rate"],
        [[w.name, f"{w.share:.1%}", f"{w.tokens_in:,.0f}",
          f"{w.tokens_out:,.0f}", w.self_consistency_k, f"{w.review_rate:.0%}"]
         for w in s.workloads],
    ))
    parts.append("")

    if bundle.tco:
        parts.append("## Total cost of ownership\n")
        rows = []
        for name, r in sorted(bundle.tco.items(), key=lambda kv: kv[1].total):
            rows.append([
                name, _money(r.total), f"${r.effective_per_query:.6f}",
                f"${r.compute_per_query:.6f}", f"{r.quality_index:.3f}",
                f"{r.capacity_units:.2f}" if math.isfinite(r.capacity_units) else "n/a",
                f"{r.annual_carbon_kg/1000:,.1f}",
            ])
        parts.append(_md_table(
            ["Architecture", "Annual TCO", "$/query (all-in)", "$/query (compute)",
             "Quality", "Replicas", "tCO2e/yr"], rows))
        parts.append("")

        parts.append("### Layer decomposition\n")
        arch_names = list(bundle.tco)
        layer_rows = []
        for layer in SIX_LAYERS:
            values = [bundle.tco[a].layers.get(layer, 0.0) for a in arch_names]
            if all(abs(v) < 1e-9 for v in values):
                continue
            layer_rows.append([LAYER_LABELS[layer]] + [_money(v) for v in values])
        layer_rows.append(["**Total**"] +
                          [f"**{_money(bundle.tco[a].total)}**" for a in arch_names])
        parts.append(_md_table(["Layer"] + arch_names, layer_rows))
        parts.append("")

        best = bundle.cheapest()
        if best:
            r = bundle.tco[best]
            share = r.share()
            compute_share = share.get("compute_serving", 0) + share.get(
                "model_access", 0)
            parts.append(
                f"Cheapest architecture is **{best}** at {_money(r.total)}/yr. "
                f"Inference itself accounts for {compute_share:.0%} of that; "
                f"the remaining {1-compute_share:.0%} sits in retrieval, "
                f"integration, assurance and workforce layers that do not "
                f"shrink when the model gets cheaper.\n"
            )
            if not bundle.any_feasible:
                parts.append(
                    "> **No architecture here meets every declared "
                    "constraint.** The figure above is the cheapest overall, "
                    "not the cheapest admissible one, and a mixed workload of "
                    "this shape needs routing rather than a single "
                    "architecture.\n")
            ruled_out = bundle.infeasible()
            if ruled_out:
                parts.append(
                    "The following architectures are **not admissible** and "
                    "are excluded from that comparison; a cheaper number "
                    "against a constraint the scenario declares is not a "
                    "cheaper option:\n")
                parts.append(_md_table(
                    ["Architecture", "Constraint not met"],
                    [[a, ", ".join(cs)] for a, cs in sorted(ruled_out.items())]))
                parts.append("")
            marginal = {a: t.marginal_verdicts for a, t in bundle.tco.items()
                        if t.marginal_verdicts}
            if marginal:
                parts.append(
                    "The quality index is a declared scale with a stated "
                    "resolution, and the following verdicts rest on margins "
                    "below it. They are reported as verdicts because the "
                    "scenario declares the floors, and flagged because the "
                    "index cannot order numbers this close:\n")
                parts.append(_md_table(
                    ["Architecture", "Classes decided within the resolution"],
                    [[a, ", ".join(cs)] for a, cs in sorted(marginal.items())]))
                parts.append("")
            unchecked = {a: t.slo_unevaluated for a, t in bundle.tco.items()
                         if t.slo_unevaluated}
            if unchecked:
                parts.append(
                    "Latency was not evaluated for "
                    + "; ".join(f"{a} ({', '.join(cs)})"
                               for a, cs in sorted(unchecked.items()))
                    + ". An unevaluated constraint is not a satisfied one.\n")

    for be in bundle.break_evens:
        parts.append(f"## Break-even: {be.label_a} vs {be.label_b}\n")
        if not be.crossings:
            winner = be.winner_at(math.sqrt(be.scanned_min * be.scanned_max))
            parts.append(
                f"No crossing between {_count(be.scanned_min)} and "
                f"{_count(be.scanned_max)} queries/yr: **{winner}** is cheaper "
                f"across the whole scanned range.\n")
        else:
            # The v9 audit established that the tie set is a union of
            # intervals, not an interval, and gave ``tie_bands`` for it.
            # Every consumer kept calling ``tie_band``, so all three
            # written artefacts described several narrow windows -- and
            # the stretches between them, where the gap reaches 47% -- as
            # one region in which cost does not decide.
            windows = be.tie_bands(0.05)
            if len(be.crossings) > 4 and windows:
                parts.append(
                    f"{len(be.crossings)} crossings were found between "
                    f"{_count(be.crossings[0].volume)} and "
                    f"{_count(be.crossings[-1].volume)} queries/yr. "
                    "Enumerating them would imply a precision the model does "
                    "not have.\n")
                parts.append(
                    f"**{len(windows)} window"
                    f"{'s' if len(windows) > 1 else ''} in which cost does "
                    "not decide** (the two options within 5%), one around "
                    "each replica riser:\n")
                parts.append(_md_table(
                    ["From", "To", "Width"],
                    [[_count(lo), _count(hi), f"{hi / lo:.2f}x"]
                     for lo, hi in windows]))
                outside = max(
                    (be.relative_gap(v) for v in sorted(be.curve_a)
                     if windows[0][0] <= v <= windows[-1][1]), default=0.0)
                parts.append(
                    f"\nBetween the windows the options differ by up to "
                    f"{outside:.0%}, so the span from the first to the last "
                    "is not itself a tie region. Inside a window the "
                    "decision should rest on latency, data residency, "
                    "control, staffing or exit risk rather than on cost.\n")
                parts.append(
                    "> The crossings are real. Self-hosted capacity arrives in "
                    "whole replicas, so the advantage alternates each time a "
                    "replica is added and then fills. A single break-even "
                    "volume would misrepresent this as a clean threshold.\n")
            else:
                parts.append(_md_table(
                    ["Crossing volume", "Annual cost", "Cheaper below",
                     "Cheaper above"],
                    [[_count(c.volume), _money(c.cost_at_crossing),
                      c.winner_below, c.winner_above]
                     for c in be.crossings[:8]]))
                parts.append("")
                parts.append("Planning intervals:\n")
                for lo, hi, who in dominance_intervals(be)[:10]:
                    parts.append(
                        f"- {_count(lo)} to {_count(hi)} queries/yr -> **{who}**")
                parts.append("")

    if bundle.monte_carlo:
        parts.append("## Uncertainty\n")
        rows = []
        for name, mc in bundle.monte_carlo.items():
            st = mc.summary()
            if not st.get("n"):
                continue
            rows.append([name, _money(st["p05"]), _money(st["p50"]),
                         _money(st["p95"]),
                         f"{st.get('cv', float('nan')):.2f}",
                         f"{int(st.get('n_failed', 0))}"])
        parts.append(_md_table(
            ["Architecture", "P05", "P50", "P95", "CV", "Infeasible draws"], rows))
        parts.append("")

        names = list(bundle.monte_carlo)
        if len(names) == 2:
            a, b = (bundle.monte_carlo[n].valid for n in names)
            k = min(a.size, b.size)
            if k:
                p = float((a[:k] < b[:k]).mean())
                parts.append(
                    f"**{names[0]}** is cheaper than **{names[1]}** in "
                    f"{p:.0%} of {k:,} draws. Read the comparison as a "
                    f"probability, not a verdict.\n")

    if bundle.sensitivity:
        parts.append("## Sensitivity\n")
        parts.append(_md_table(
            ["Input", "Spearman rho", "Variance share", "At P10", "At P90", "Swing"],
            [[e.name, f"{e.spearman:+.3f}", f"{e.contribution:.1%}",
              _money(e.low_output), _money(e.high_output), _money(abs(e.swing))]
             for e in bundle.sensitivity[:12]]))
        parts.append("")
        top = bundle.sensitivity[0]
        parts.append(
            f"The dominant driver is **{top.name}** "
            f"({top.contribution:.0%} of explained variance). Narrowing this "
            f"single input is worth more than refining every other estimate.\n")

    if bundle.projection:
        p = bundle.projection
        parts.append("## Scaling projection\n")
        parts.append(p.narrative() + "\n")
        parts.append(_md_table(
            ["Year", "Unit cost", "Volume", "Variable", "Fixed", "Total"],
            [[y.year, f"${y.unit_cost:.6f}", _count(y.volume),
              _money(y.variable_spend), _money(y.fixed_spend),
              _money(y.total_spend)] for y in p.years]))
        parts.append("")
        if p.regime == "jevons":
            parts.append(
                "> Elasticity exceeds 1, so falling unit cost raises total "
                "spend. A budget built on the unit-cost trend alone will be "
                "wrong in direction, not merely in magnitude.\n")

    for name, rows in bundle.extra_tables.items():
        if not rows:
            continue
        parts.append(f"## {name}\n")
        headers = list(rows[0])
        parts.append(_md_table(
            headers,
            [[f"{r.get(h):.4g}" if isinstance(r.get(h), float) else r.get(h)
              for h in headers] for r in rows]))
        parts.append("")

    if bundle.figures:
        parts.append("## Figures\n")
        for f in bundle.figures:
            parts.append(f"![{f.stem}]({f.name})\n")

    parts.append("## Reproducing this analysis\n")
    parts.append(
        "The scenario below regenerates every figure above. Save it and run "
        "`caide run <file>`.\n")
    parts.append("```yaml")
    parts.append(bundle.scenario.to_yaml().rstrip())
    parts.append("```\n")

    parts.append("---\n")
    parts.append(
        "Generated by CAIDE. Bundled prices are illustrative anchors; "
        "absolute figures should be re-run against current quotations. "
        "Relative structure -- which layer dominates, where curves cross, "
        "which input drives variance -- is robust to price level.\n")

    out.write_text("\n".join(parts), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------------

def write_csv(bundle: ReportBundle, out: Path) -> Path:
    import csv

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The dictionary keys are the architecture *names*; dropping them left
    # every row carrying ``TCOResult.architecture``, which is the kind, so
    # a clinical scenario produced three rows all reading "self_hosted".
    # The machine-readable output could not distinguish the candidates it
    # was comparing, and sorting it by total reproduced the pre-v10 answer
    # -- the architecture no declared quality floor admits. This is the one
    # artefact no audit round had ever opened.
    rows = []
    for name, r in bundle.tco.items():
        row = {"name": name,
               "feasible": r.feasible,
               "fully_evaluated": r.fully_evaluated,
               "quality_violations": ";".join(r.quality_violations),
               "slo_violations": ";".join(r.slo_violations),
               "slo_unevaluated": ";".join(r.slo_unevaluated),
               "marginal_verdicts": ";".join(r.marginal_verdicts)}
        row.update(r.as_dict())
        rows.append(row)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------

_HTML_SHELL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAIDE - {name}</title>
<style>
 :root {{ --ink:#16202c; --mute:#5d6b7a; --line:#dfe5ec; --accent:#2f4b7c;
          --warn:#8a5a00; --warnbg:#fff8e6; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:2.2rem 1.4rem 4rem; color:var(--ink);
   font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
   background:#fbfcfd; }}
 main {{ max-width:1080px; margin:0 auto; }}
 h1 {{ font-size:1.65rem; margin:0 0 .3rem; letter-spacing:-.015em; }}
 h2 {{ font-size:1.12rem; margin:2.4rem 0 .8rem; padding-bottom:.4rem;
   border-bottom:1px solid var(--line); }}
 .sub {{ color:var(--mute); margin:0 0 1.6rem; font-size:.94rem; }}
 .meta {{ display:flex; flex-wrap:wrap; gap:.5rem 1.6rem; font-size:.82rem;
   color:var(--mute); margin-bottom:1.8rem; }}
 .meta code {{ background:#eef2f7; padding:.1rem .35rem; border-radius:3px; }}
 .cards {{ display:grid; gap:.9rem;
   grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); margin:1rem 0 .4rem; }}
 .card {{ border:1px solid var(--line); border-radius:9px; padding:.9rem 1rem;
   background:#fff; }}
 .card .k {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em;
   color:var(--mute); }}
 .card .v {{ font-size:1.4rem; font-weight:650; margin-top:.25rem;
   letter-spacing:-.02em; }}
 .card .n {{ font-size:.78rem; color:var(--mute); margin-top:.2rem; }}
 .best {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(47,75,124,.09); }}
 table {{ border-collapse:collapse; width:100%; font-size:.87rem;
   background:#fff; border:1px solid var(--line); border-radius:8px;
   overflow:hidden; }}
 th,td {{ padding:.5rem .7rem; text-align:right; border-bottom:1px solid var(--line); }}
 th:first-child,td:first-child {{ text-align:left; }}
 thead th {{ background:#f4f7fa; font-weight:600; font-size:.78rem;
   text-transform:uppercase; letter-spacing:.04em; color:var(--mute); }}
 tbody tr:last-child td {{ border-bottom:none; }}
 figure {{ margin:1.2rem 0; }}
 figure img {{ width:100%; border:1px solid var(--line); border-radius:8px;
   background:#fff; }}
 .warn {{ background:var(--warnbg); border:1px solid #f0dcb0; color:var(--warn);
   padding:.7rem .95rem; border-radius:7px; font-size:.86rem; margin:.5rem 0; }}
 .note {{ border-left:3px solid var(--accent); background:#f4f7fb;
   padding:.7rem .95rem; margin:1rem 0; font-size:.9rem; border-radius:0 6px 6px 0; }}
 pre.scenario {{ background:#f6f8fb; border:1px solid var(--line);
   border-radius:8px; padding:.9rem 1.1rem; overflow-x:auto; font-size:.76rem;
   line-height:1.5; max-height:26rem; }}
 footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
   font-size:.8rem; color:var(--mute); }}
</style></head><body><main>
{body}
</main></body></html>"""


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


def _img_tag(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (f'<figure><img alt="{path.stem}" '
            f'src="data:image/png;base64,{data}"></figure>')


def _html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(bundle: ReportBundle, out: Path) -> Path:
    """Single-file dashboard with figures embedded as data URIs."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = bundle.scenario
    b: List[str] = []

    b.append(f"<h1>{s.name}</h1>")
    b.append(f'<p class="sub">{s.description or "CAIDE deployment analysis"}</p>')
    b.append(
        f'<div class="meta">'
        f"<span>CAIDE {__version__}</span>"
        f'<span>digest <code>{bundle.digest}</code></span>'
        f"<span>{bundle.generated.strftime('%Y-%m-%d %H:%M UTC')}</span>"
        f"<span>{_count(s.annual_volume)} queries/yr</span>"
        f"<span>grid: {s.grid.name}</span>"
        f"<span>seed: {bundle.seed if bundle.seed is not None else 'n/a'}</span>"
        f"</div>")

    for w in bundle.warnings:
        b.append(f'<div class="warn">{w}</div>')

    best = bundle.cheapest()
    if bundle.tco:
        b.append("<h2>Annual total cost of ownership</h2>")
        b.append('<div class="cards">')
        for name, r in sorted(bundle.tco.items(), key=lambda kv: kv[1].total):
            cls = "card best" if name == best else "card"
            share = r.share()
            infra = share.get("compute_serving", 0) + share.get("model_access", 0)
            b.append(
                f'<div class="{cls}"><div class="k">{name}</div>'
                f'<div class="v">{_money(r.total)}</div>'
                f'<div class="n">${r.effective_per_query:.6f}/query &middot; '
                f"{infra:.0%} inference</div></div>")
        b.append("</div>")

        arch_names = list(bundle.tco)
        rows = []
        for layer in SIX_LAYERS:
            values = [bundle.tco[a].layers.get(layer, 0.0) for a in arch_names]
            if all(abs(v) < 1e-9 for v in values):
                continue
            rows.append([LAYER_LABELS[layer]] + [_money(v) for v in values])
        rows.append(["<strong>Total</strong>"] +
                    [f"<strong>{_money(bundle.tco[a].total)}</strong>"
                     for a in arch_names])
        b.append(_html_table(["Layer"] + arch_names, rows))

        if best:
            r = bundle.tco[best]
            sh = r.share()
            infra = sh.get("compute_serving", 0) + sh.get("model_access", 0)
            b.append(
                f'<div class="note"><strong>{best}</strong> is cheapest at '
                f"{_money(r.total)}/yr. Inference is {infra:.0%} of it. The "
                f"other {1-infra:.0%} does not fall when token prices fall."
                f"</div>")
            if not bundle.any_feasible:
                b.append(
                    '<div class="note"><strong>No architecture here meets '
                    "every declared constraint.</strong> The figure above is "
                    "the cheapest overall, not the cheapest admissible one."
                    "</div>")
            ruled_out = bundle.infeasible()
            if ruled_out:
                b.append(_html_table(
                    ["Not admissible", "Constraint not met"],
                    [[a, ", ".join(cs)] for a, cs in sorted(ruled_out.items())]))
            marginal = {a: t.marginal_verdicts for a, t in bundle.tco.items()
                        if t.marginal_verdicts}
            if marginal:
                b.append(_html_table(
                    ["Decided within the index resolution", "Classes"],
                    [[a, ", ".join(cs)] for a, cs in sorted(marginal.items())]))
            unchecked = {a: t.slo_unevaluated for a, t in bundle.tco.items()
                         if t.slo_unevaluated}
            if unchecked:
                b.append(
                    '<div class="sub">Latency not evaluated for '
                    + _escape("; ".join(f"{a} ({', '.join(cs)})"
                                        for a, cs in sorted(unchecked.items())))
                    + ". An unevaluated constraint is not a satisfied one."
                    "</div>")

    for be in bundle.break_evens:
        b.append(f"<h2>Break-even: {be.label_a} vs {be.label_b}</h2>")
        if be.crossings:
            windows = be.tie_bands(0.05)
            if len(be.crossings) > 4 and windows:
                spans = "; ".join(f"{_count(lo)}&ndash;{_count(hi)}"
                                  for lo, hi in windows)
                outside = max(
                    (be.relative_gap(v) for v in sorted(be.curve_a)
                     if windows[0][0] <= v <= windows[-1][1]), default=0.0)
                b.append(
                    f'<div class="note"><strong>{len(be.crossings)} crossings'
                    f"</strong> between {_count(be.crossings[0].volume)} and "
                    f"{_count(be.crossings[-1].volume)} queries/yr. Cost does "
                    f"not decide inside {len(windows)} narrow window"
                    f"{'s' if len(windows) > 1 else ''}, one around each "
                    f"replica riser: {spans} queries/yr. Between them the "
                    f"options differ by up to {outside:.0%}, so the whole "
                    f"span is not a tie region. The alternation is real: "
                    f"self-hosted capacity arrives in whole replicas.</div>")
            else:
                b.append(_html_table(
                    ["Crossing", "Annual cost", "Cheaper below", "Cheaper above"],
                    [[_count(c.volume), _money(c.cost_at_crossing),
                      c.winner_below, c.winner_above]
                     for c in be.crossings[:8]]))
        else:
            winner = be.winner_at(math.sqrt(be.scanned_min * be.scanned_max))
            b.append(f'<div class="note">No crossing in range; '
                     f"<strong>{winner}</strong> is cheaper throughout.</div>")

    if bundle.monte_carlo:
        b.append("<h2>Uncertainty</h2>")
        rows = []
        for name, mc in bundle.monte_carlo.items():
            st = mc.summary()
            if st.get("n"):
                rows.append([name, _money(st["p05"]), _money(st["p50"]),
                             _money(st["p95"]), f"{st.get('cv', 0):.2f}"])
        b.append(_html_table(["Architecture", "P05", "P50", "P95", "CV"], rows))

    if bundle.sensitivity:
        b.append("<h2>Sensitivity</h2>")
        b.append(_html_table(
            ["Input", "rho", "Variance share", "P10", "P90"],
            [[e.name, f"{e.spearman:+.3f}", f"{e.contribution:.1%}",
              _money(e.low_output), _money(e.high_output)]
             for e in bundle.sensitivity[:12]]))

    if bundle.projection:
        p = bundle.projection
        b.append("<h2>Scaling projection</h2>")
        b.append(f'<div class="note">{p.narrative()}</div>')
        b.append(_html_table(
            ["Year", "Unit cost", "Volume", "Total spend"],
            [[y.year, f"${y.unit_cost:.6f}", _count(y.volume),
              _money(y.total_spend)] for y in p.years]))

    if bundle.figures:
        b.append("<h2>Figures</h2>")
        for f in bundle.figures:
            if Path(f).exists():
                b.append(_img_tag(Path(f)))

    b.append("<h2>Reproducing this analysis</h2>")
    b.append('<p class="sub">The scenario below regenerates every figure on '
             'this page. Save it and run <code>caide run &lt;file&gt;</code>. '
             'A digest proves the inputs did not change; the inputs '
             'themselves are what let you re-run it.</p>')
    b.append(f"<pre class=\"scenario\">{_escape(bundle.scenario.to_yaml())}</pre>")

    b.append(
        "<footer>Generated by CAIDE. Bundled prices are illustrative "
        "anchors and should be replaced with current quotations before "
        "absolute figures are used in a budget. Relative structure is "
        "robust to price level.</footer>")

    out.write_text(_HTML_SHELL.format(name=s.name, body="\n".join(b)),
                   encoding="utf-8")
    return out
