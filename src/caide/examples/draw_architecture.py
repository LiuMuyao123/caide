#!/usr/bin/env python3
"""Draw the CAIDE architecture figure (Figure 1 of the software paper).

Kept separate from reproduce_paper.py because it illustrates structure
rather than reporting a result: nothing here is computed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#16202c"
MUTE = "#5b6875"

FILL = {
    "input": "#e9eff7",
    "physics": "#dae5f2",
    "economics": "#fbe6e4",
    "analysis": "#fcf1da",
    "output": "#e4efe7",
}
EDGE = {
    "input": "#7d97b8",
    "physics": "#2f4b7c",
    "economics": "#c0554e",
    "analysis": "#b5851f",
    "output": "#487a55",
}

# Row geometry: (y_bottom, height). Chosen so that every box has room for
# a bold title plus at most three body lines without overflowing.
ROWS = {
    "scenario":  (0.905, 0.075),
    "specs":     (0.735, 0.135),
    "roofline":  (0.560, 0.140),
    "economics": (0.375, 0.150),
    "analysis":  (0.190, 0.150),
    "output":    (0.020, 0.135),
}

LEFT, RIGHT = 0.075, 0.965
FULL_W = RIGHT - LEFT
HALF_W = (FULL_W - 0.035) / 2
THIRD_W = (FULL_W - 0.055) / 3


def box(ax, x, y, w, h, title, body, band, title_size=9.2, body_size=6.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=FILL[band], edgecolor=EDGE[band], linewidth=1.15))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.055, body, ha="center", va="top",
                fontsize=body_size, color=MUTE, linespacing=1.55)


def arrow(ax, x, y_from, y_to, band, rad=0.0, width=1.25):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=10,
        color=EDGE[band], linewidth=width,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def band_tag(ax, band, label):
    y0, h = ROWS[band] if band in ROWS else (0, 0)
    ax.text(LEFT - 0.028, y0 + h / 2, label, ha="center", va="center",
            fontsize=6.8, color=EDGE[band if band in EDGE else "input"],
            fontweight="bold", rotation=90, letterspacing=1.2)


def main(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- input ----------------
    y, h = ROWS["scenario"]
    box(ax, LEFT, y, FULL_W, h, "Scenario  —  declarative YAML, strictly validated",
        "workload classes · candidate architectures · SLO · grid · "
        "cost layers · input distributions",
        "input", title_size=9.6, body_size=7.0)
    ax.text(LEFT - 0.028, y + h / 2, "INPUT", ha="center", va="center",
            fontsize=6.8, color=EDGE["input"], fontweight="bold", rotation=90)

    # ---------------- physics ----------------
    y, h = ROWS["specs"]
    box(ax, LEFT, y, HALF_W, h, "specs  +  catalog",
        "unit-checked model, hardware,\nserving and grid specifications\n"
        "9 archetypes · 6 accelerators · 9 grids",
        "physics")
    box(ax, LEFT + HALF_W + 0.035, y, HALF_W, h, "efficiency",
        "15 techniques as\nDeploymentState → DeploymentState\n"
        "transforms — never as multipliers",
        "physics")

    y, h = ROWS["roofline"]
    box(ax, LEFT, y, FULL_W, h, "roofline",
        "prefill is compute-bound:  2·N·T_in + causal attention\n"
        "decode is memory-bound:  max(weight + KV traffic, arithmetic)\n"
        "GQA-aware KV cache · MoE expert-touch p(B) · M/D/1 TTFT · "
        "SLO-bisected batch",
        "physics", body_size=6.6)
    ax.text(LEFT - 0.028, 0.68, "PHYSICS", ha="center", va="center",
            fontsize=6.8, color=EDGE["physics"], fontweight="bold", rotation=90)

    # ---------------- economics ----------------
    y, h = ROWS["economics"]
    box(ax, LEFT, y, HALF_W, h, "costing  +  perturb",
        "six layers, six scaling laws\nintegral-replica capacity\n"
        "review cost + displaced labour\npublic perturbation API",
        "economics")
    box(ax, LEFT + HALF_W + 0.035, y, HALF_W, h, "routing",
        "exact minimum-cost assignment\nof classes to a tier ladder\n"
        "under quality floors and\nper-tier fixed costs",
        "economics")
    ax.text(LEFT - 0.028, y + h / 2, "ECONOMICS", ha="center", va="center",
            fontsize=6.8, color=EDGE["economics"], fontweight="bold", rotation=90)

    # ---------------- analysis ----------------
    y, h = ROWS["analysis"]
    for i, (title, body) in enumerate([
        ("breakeven", "every crossing on a\nlog-bracketed scan\n"
                      "+ indistinguishable\nband when granular"),
        ("uncertainty", "Monte Carlo over\ninput distributions\n"
                        "+ Spearman rank\nsensitivity"),
        ("scaling", "closed-form price\nelasticity projection\n"
                    "+ elasticity fitted\nfrom own history"),
    ]):
        box(ax, LEFT + i * (THIRD_W + 0.0275), y, THIRD_W, h, title, body,
            "analysis", title_size=8.8)
    ax.text(LEFT - 0.028, y + h / 2, "ANALYSIS", ha="center", va="center",
            fontsize=6.8, color=EDGE["analysis"], fontweight="bold", rotation=90)

    # ---------------- output ----------------
    y, h = ROWS["output"]
    box(ax, LEFT, y, HALF_W, h, "report",
        "Markdown · CSV · single-file HTML\ndashboard with embedded figures\n"
        "stamped with version, digest, seed",
        "output")
    box(ax, LEFT + HALF_W + 0.035, y, HALF_W, h, "cli  /  Python API",
        "run · breakeven · sweep · route\nexamples · catalog · init · validate\n"
        "or import caide and compose",
        "output")
    ax.text(LEFT - 0.028, y + h / 2, "OUTPUT", ha="center", va="center",
            fontsize=6.8, color=EDGE["output"], fontweight="bold", rotation=90)

    # ---------------- flow ----------------
    mid_l = LEFT + HALF_W / 2
    mid_r = LEFT + HALF_W + 0.035 + HALF_W / 2

    arrow(ax, 0.5, ROWS["scenario"][0], ROWS["specs"][0] + ROWS["specs"][1], "physics")
    for x in (mid_l, mid_r):
        arrow(ax, x, ROWS["specs"][0], ROWS["roofline"][0] + ROWS["roofline"][1],
              "physics")
        arrow(ax, x, ROWS["roofline"][0],
              ROWS["economics"][0] + ROWS["economics"][1], "economics")
    for i in range(3):
        x = LEFT + i * (THIRD_W + 0.0275) + THIRD_W / 2
        arrow(ax, x, ROWS["economics"][0],
              ROWS["analysis"][0] + ROWS["analysis"][1], "analysis")
    for x in (mid_l, mid_r):
        arrow(ax, x, ROWS["analysis"][0], ROWS["output"][0] + ROWS["output"][1],
              "output")

    # the feedback loop that is the point of the design
    ax.add_patch(FancyArrowPatch(
        (RIGHT + 0.012, ROWS["roofline"][0] + 0.03),
        (RIGHT + 0.012, ROWS["specs"][0] + 0.03),
        arrowstyle="-|>", mutation_scale=11, color=EDGE["physics"],
        linewidth=1.5, connectionstyle="arc3,rad=-0.55"))
    ax.text(RIGHT + 0.062, (ROWS["roofline"][0] + ROWS["specs"][0]) / 2 + 0.03,
            "multiplier measured here,\nnot supplied",
            ha="center", va="center", fontsize=6.5, color=EDGE["physics"],
            rotation=90, linespacing=1.5)

    fig.text(0.5, 0.002,
             "No cost multiplier is an input. Each is measured by re-running the "
             "roofline on the transformed\ndeployment state, so it changes with "
             "the operating point instead of staying constant.",
             ha="center", va="bottom", fontsize=7.4, color=INK,
             style="italic", linespacing=1.7)

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.18)
    plt.close(fig)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1
                  else "paper_figures/fig1_architecture.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(main(target))
