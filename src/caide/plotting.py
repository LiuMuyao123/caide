"""Figure generation.

Matplotlib is imported through the Agg backend so that figures render on
headless machines, which is where batch analyses actually run. Every
function takes an explicit output path and returns it, so a report
generator can collect paths without knowing how the figures were made.

The house style is deliberately plain: no gridlines competing with data,
no colour where shape will do, and units in the axis label rather than
in a caption the reader may not have.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator   # noqa: E402

from .breakeven import BreakEvenResult, dominance_intervals   # noqa: E402
from .costing import SIX_LAYERS, TCOResult                    # noqa: E402
from .scaling import ScalingProjection                        # noqa: E402
from .uncertainty import MonteCarloResult, SensitivityEntry   # noqa: E402

__all__ = [
    "plot_tco_breakdown",
    "plot_break_even",
    "plot_tornado",
    "plot_uncertainty",
    "plot_scaling",
    "plot_technique_regimes",
    "STYLE",
]

STYLE = {
    "figure.figsize": (7.2, 4.4),
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
}

PALETTE = ["#2f4b7c", "#f95d6a", "#ffa600", "#665191", "#a05195",
           "#d45087", "#003f5c", "#7a9e9f"]

LAYER_LABELS = {
    "model_access": "Model access",
    "compute_serving": "Compute & serving",
    "retrieval_data": "Retrieval & data",
    "integration_sre": "Integration & SRE",
    "assurance_governance": "Assurance & governance",
    "workforce_redesign": "Workforce & redesign",
}


def _money(value: float, _pos: int = 0) -> str:
    a = abs(value)
    if a >= 1e9:
        return f"${value/1e9:.1f}B"
    if a >= 1e6:
        return f"${value/1e6:.1f}M"
    if a >= 1e3:
        return f"${value/1e3:.0f}k"
    return f"${value:.0f}"


def _count(value: float, _pos: int = 0) -> str:
    a = abs(value)
    if a >= 1e9:
        return f"{value/1e9:.0f}B"
    if a >= 1e6:
        return f"{value/1e6:.0f}M"
    if a >= 1e3:
        return f"{value/1e3:.0f}k"
    return f"{value:.0f}"


def _prepare(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_tco_breakdown(results: Dict[str, TCOResult], out: Path,
                       title: str = "Total cost of ownership by layer") -> Path:
    """Stacked bars: which layer actually holds the money.

    The point of the figure is usually that ``compute_serving`` is not the
    tall bar, which contradicts how most deployment debates are framed.
    """
    out = _prepare(out)
    names = list(results)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        bottoms = [0.0] * len(names)
        for i, layer in enumerate(SIX_LAYERS):
            values = [results[n].layers.get(layer, 0.0) for n in names]
            if all(abs(v) < 1e-9 for v in values):
                continue
            ax.bar(names, values, bottom=bottoms,
                   color=PALETTE[i % len(PALETTE)],
                   label=LAYER_LABELS.get(layer, layer), width=0.62,
                   edgecolor="white", linewidth=0.7)
            bottoms = [b + v for b, v in zip(bottoms, values)]

        for x, total in enumerate(bottoms):
            ax.text(x, total * 1.02, _money(total), ha="center",
                    va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_ylabel("Annual cost (USD)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(FuncFormatter(_money))
        ax.set_ylim(0, max(bottoms) * 1.16 if bottoms else 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.grid(axis="x", visible=False)
        fig.savefig(out)
        plt.close(fig)
    return out


def plot_break_even(result: BreakEvenResult, out: Path,
                    title: Optional[str] = None) -> Path:
    """Two cost curves on log-log axes with every crossing marked."""
    out = _prepare(out)
    volumes = sorted(result.curve_a)
    ya = [result.curve_a[v] for v in volumes]
    yb = [result.curve_b[v] for v in volumes]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(volumes, ya, color=PALETTE[0], label=result.label_a)
        ax.plot(volumes, yb, color=PALETTE[1], label=result.label_b)

        for iv_lo, iv_hi, winner in dominance_intervals(result):
            colour = PALETTE[0] if winner == result.label_a else PALETTE[1]
            ax.axvspan(iv_lo, iv_hi, color=colour, alpha=0.06, linewidth=0)

        # Alternate the label offset so that closely spaced crossings --
        # which is exactly what granular capacity produces -- stay legible.
        for i, c in enumerate(result.crossings[:12]):
            ax.axvline(c.volume, color="0.35", linestyle=":", linewidth=1.0)
            dy = 8 if i % 2 == 0 else -14
            ax.annotate(_count(c.volume), xy=(c.volume, c.cost_at_crossing),
                        xytext=(5, dy), textcoords="offset points",
                        fontsize=7.5, color="0.25")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Annual query volume")
        ax.set_ylabel("Annual total cost (USD)")
        n = len(result.crossings)
        ax.set_title(title or
                     f"Break-even: {result.label_a} vs {result.label_b} "
                     f"({n} crossing{'s' if n != 1 else ''})")
        ax.xaxis.set_major_formatter(FuncFormatter(_count))
        ax.yaxis.set_major_formatter(FuncFormatter(_money))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.legend(loc="upper left")
        fig.savefig(out)
        plt.close(fig)
    return out


def plot_tornado(entries: Sequence[SensitivityEntry], out: Path,
                 baseline: Optional[float] = None,
                 title: str = "Sensitivity of annual cost to inputs",
                 top_n: int = 10) -> Path:
    """Horizontal bars from the 10th- to the 90th-percentile output."""
    out = _prepare(out)
    entries = [e for e in entries
               if math.isfinite(e.low_output) and math.isfinite(e.high_output)]
    entries = entries[:top_n][::-1]
    if not entries:
        raise ValueError("no finite sensitivity entries to plot")

    if baseline is None:
        baseline = sum(0.5 * (e.low_output + e.high_output)
                       for e in entries) / len(entries)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(entries) + 1.5))
        for i, e in enumerate(entries):
            lo, hi = sorted((e.low_output, e.high_output))
            ax.barh(i, hi - lo, left=lo, height=0.6,
                    color=PALETTE[0] if e.spearman >= 0 else PALETTE[1],
                    alpha=0.85, edgecolor="white", linewidth=0.6)
            ax.text(hi, i, f"  rho={e.spearman:+.2f}", va="center",
                    fontsize=7.5, color="0.3")

        ax.axvline(baseline, color="0.25", linestyle="--", linewidth=1.1)
        lo_all = min(min(e.low_output, e.high_output) for e in entries)
        hi_all = max(max(e.low_output, e.high_output) for e in entries)
        span = max(hi_all - lo_all, abs(hi_all) * 1e-3, 1e-9)
        ax.set_xlim(lo_all - 0.04 * span, hi_all + 0.22 * span)
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([e.name for e in entries])
        ax.set_xlabel("Annual cost (USD) at 10th and 90th input percentile")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FuncFormatter(_money))
        ax.grid(axis="y", visible=False)
        fig.savefig(out)
        plt.close(fig)
    return out


def plot_uncertainty(results: Dict[str, MonteCarloResult], out: Path,
                     title: str = "Cost distribution under input uncertainty",
                     bins: int = 60) -> Path:
    """Overlaid histograms with median markers."""
    out = _prepare(out)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for i, (name, res) in enumerate(results.items()):
            v = res.valid
            if v.size == 0:
                continue
            colour = PALETTE[i % len(PALETTE)]
            ax.hist(v, bins=bins, alpha=0.5, color=colour,
                    label=f"{name} (p50 {_money(res.percentile(50))})",
                    density=True, edgecolor="none")
            ax.axvline(res.percentile(50), color=colour, linestyle="--",
                       linewidth=1.2)
        ax.set_xlabel("Annual total cost (USD)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FuncFormatter(_money))
        ax.set_yticks([])
        ax.legend(loc="upper right")
        fig.savefig(out)
        plt.close(fig)
    return out


def plot_scaling(projection: ScalingProjection, out: Path,
                 title: str = "Unit cost, volume and total spend") -> Path:
    """The Jevons figure: unit cost down, volume up, spend possibly up."""
    out = _prepare(out)
    years = [y.year for y in projection.years]
    unit = [y.unit_cost for y in projection.years]
    volume = [y.volume for y in projection.years]
    spend = [y.total_spend for y in projection.years]
    effective = [y.effective_unit_cost for y in projection.years]
    # The two price lines are the point of the figure since v8.0: the
    # tariff collapses, the price the institution actually pays does not,
    # and demand responds to the second one.
    split = (projection.assumptions.price_inelastic_per_query > 0
             or projection.assumptions.fixed_annual_cost > 0)

    with plt.rc_context(STYLE):
        fig, ax1 = plt.subplots()
        ax1.plot(years, [u / unit[0] for u in unit], color=PALETTE[0],
                 marker="o", markersize=3.5,
                 label="Token tariff (indexed)" if split
                 else "Unit cost (indexed)")
        if split:
            ax1.plot(years, [e / effective[0] for e in effective],
                     color=PALETTE[0], marker="o", markersize=3.5,
                     linestyle="--", alpha=0.75,
                     label="Effective price paid (indexed)")
        ax1.plot(years, [v / volume[0] for v in volume], color=PALETTE[2],
                 marker="s", markersize=3.5, label="Volume (indexed)")
        ax1.plot(years, [s / spend[0] for s in spend], color=PALETTE[1],
                 marker="^", markersize=4, linewidth=2.4,
                 label="Total spend (indexed)")
        ax1.axhline(1.0, color="0.5", linewidth=0.9, linestyle=":")
        ax1.set_yscale("log")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Indexed to year 1 (log scale)")
        ax1.set_xticks(years)
        eps = projection.assumptions.price_elasticity
        ax1.set_title(f"{title}  (elasticity {eps:.2f}, "
                      f"{projection.regime} regime)")
        ax1.legend(loc="upper left")
        fig.savefig(out)
        plt.close(fig)
    return out


def plot_technique_regimes(batches: Sequence[float],
                           series: Dict[str, Sequence[float]],
                           out: Path,
                           reference: Optional[Dict[str, float]] = None,
                           title: str = "Efficiency multipliers are regime-dependent"
                           ) -> Path:
    """Emergent cost multiplier of each technique against batch size.

    ``reference`` draws the fixed multiplier a published table would
    assign, making the divergence between a constant and the measured
    curve visible in one glance.
    """
    out = _prepare(out)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for i, (name, values) in enumerate(series.items()):
            colour = PALETTE[i % len(PALETTE)]
            ax.plot(batches, values, color=colour, marker="o",
                    markersize=3, label=name)
            if reference and name in reference:
                ax.axhline(reference[name], color=colour, linestyle="--",
                           linewidth=1.0, alpha=0.7)
                ax.annotate(f"published {reference[name]:.2f}x",
                            xy=(batches[-1], reference[name]),
                            xytext=(-4, 4), textcoords="offset points",
                            fontsize=7, color=colour, ha="right")
        ax.axhline(1.0, color="0.4", linewidth=0.9, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size (concurrent sequences)")
        ax.set_ylabel("Cost multiplier vs unoptimised baseline")
        ax.set_title(title)
        ax.set_ylim(top=1.14)
        ax.legend(loc="upper left", ncol=3, fontsize=7.5,
                  columnspacing=1.4, handlelength=1.6)
        fig.savefig(out)
        plt.close(fig)
    return out
