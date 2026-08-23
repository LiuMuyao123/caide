#!/usr/bin/env python3
"""Reproduce every figure and quantitative claim in the CAIDE paper.

Run from the repository root:

    python examples/reproduce_paper.py --out paper_figures/

Everything is seeded, so the numbers printed here are the numbers in the
paper. If a future change to the model moves them, this script is where
that shows up first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caide import (
    DeploymentState,
    ServingConfig,
    WorkloadClass,
    apply_stack,
    find_break_even,
    get_grid,
    get_hardware,
    get_model,
    load_scenario,
    monte_carlo,
    perturbed_cost,
    self_hosted_query_cost,
    sensitivity,
)
from caide import plotting
from caide.scaling import ScalingAssumptions, project

EXAMPLES = Path(__file__).resolve().parent
SEED = 20260101

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# The serving stack used by Results 3 and 4. It is the "aggressive"
# preset with speculative decoding removed, because at these operating
# points (batch 256, INT4 weights) the derived multiplier for
# speculation exceeds one: the verification arithmetic that v7.0 prices
# makes the technique a net loss exactly where these analyses run.
# Keeping it in would make Result 3 decompose the cost of a mis-tuned
# stack and Result 4 measure the mis-tuning rather than the capacity
# granularity. The preset itself is untouched -- it mirrors what
# practitioners deploy -- and Result 1 is where the mis-tuning is shown.
# The v7 audit report tabulates both readings.
from caide.efficiency import PRESET_STACKS
TUNED_STACK = tuple(k for k in PRESET_STACKS["aggressive"]
                    if k != "speculative_decoding")

# Multipliers a fixed-multiplier table assigns to these techniques.
PUBLISHED = {
    "speculative_decoding": 0.40,
    "int4": 0.65,
    "semantic_caching": 0.50,
}


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# Result 1: efficiency multipliers are regime dependent
# ---------------------------------------------------------------------------

def result_regime_dependence(out: Path) -> dict:
    banner("RESULT 1  Efficiency multipliers are regime dependent")

    grid = get_grid("us-average")
    workload = WorkloadClass("tutoring", 1.0, tokens_in=1500, tokens_out=400)
    series: dict[str, list[float]] = {k: [] for k in PUBLISHED}

    print(f"{'batch':>7}{'baseline $/q':>15}"
          + "".join(f"{k[:14]:>16}" for k in PUBLISHED))
    print("-" * 74)

    for batch in BATCHES:
        state = DeploymentState(
            get_model("dense-70b"), get_hardware("h100-sxm"),
            ServingConfig(n_accelerators=4, max_batch=batch))
        base = self_hosted_query_cost(state, workload, grid,
                                      respect_slo=False).compute_cost
        row = f"{batch:>7}{base:>15.6f}"
        for key in PUBLISHED:
            cost = self_hosted_query_cost(apply_stack(state, [key]), workload,
                                          grid, respect_slo=False).compute_cost
            mult = cost / base
            series[key].append(mult)
            row += f"{mult:>15.3f}x"
        print(row)

    print()
    summary = {}
    for key, values in series.items():
        lo, hi = min(values), max(values)
        published = PUBLISHED[key]
        err_lo = abs(published - lo) / lo
        err_hi = abs(published - hi) / hi
        summary[key] = {"min": lo, "max": hi, "spread": hi / lo,
                        "published": published,
                        "max_relative_error": max(err_lo, err_hi)}
        print(f"  {key:<22} {lo:.3f}x .. {hi:.3f}x  "
              f"(spread {hi/lo:4.1f}x, published constant {published:.2f}x, "
              f"worst-case error {max(err_lo, err_hi):.0%})")

    # The same technique on a mixture-of-experts target, which inverts
    # the shape: the router sees gamma + 1 tokens per sequence, so more
    # experts are touched and the weight stream the technique exists to
    # amortise grows instead. No published configuration exercised this
    # before v9.0, which is how the routing defect survived a round.
    moe_curve = []
    for batch in BATCHES:
        st = DeploymentState(
            get_model("moe-8x7b"), get_hardware("h100-sxm"),
            ServingConfig(n_accelerators=2, max_batch=batch))
        base = self_hosted_query_cost(st, workload, grid,
                                      respect_slo=False).compute_cost
        spec = self_hosted_query_cost(
            apply_stack(st, ["speculative_decoding"]), workload, grid,
            respect_slo=False).compute_cost
        moe_curve.append(spec / base)
    series["speculative_decoding (MoE)"] = moe_curve
    summary["speculative_decoding_moe"] = {
        "min": min(moe_curve), "max": max(moe_curve),
        "at_batch_1": moe_curve[0],
        "best_batch": float(BATCHES[moe_curve.index(min(moe_curve))]),
        "spread": max(moe_curve) / min(moe_curve),
    }
    print(f"\n  speculative_decoding on MoE   {min(moe_curve):.3f}x .. "
          f"{max(moe_curve):.3f}x  (best at batch "
          f"{BATCHES[moe_curve.index(min(moe_curve))]}, "
          f"{moe_curve[0]:.3f}x at batch 1)")
    print("  On a mixture-of-experts target the curve inverts: the router")
    print("  sees gamma + 1 tokens per sequence, so verification touches")
    print("  more experts and enlarges the weight stream the technique")
    print("  exists to amortise. The published 0.40x holds at batch one")
    print("  for a dense model and is worth nothing there for this one.")

    print("\n  A constant is correct at one batch size and wrong elsewhere.")
    print("  Speculative decoding's published 0.40x holds at batch 1 -- an")
    print("  unbatched deployment -- yet it is routinely multiplied against a")
    print("  continuous-batching multiplier that assumes the opposite regime.")

    plotting.plot_technique_regimes(
        BATCHES, series, out / "fig2_technique_regimes.png",
        reference=PUBLISHED)
    return summary


# ---------------------------------------------------------------------------
# Result 2: stacks interact, and the sign of the error is not predictable
# ---------------------------------------------------------------------------

def result_interaction(out: Path) -> dict:
    banner("RESULT 2  Technique stacks interact")

    grid = get_grid("us-average")
    findings = {}

    cases = [
        ("memory-bound (70B, 2xH100, 8k context)",
         DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                         ServingConfig(n_accelerators=2, max_batch=512)),
         WorkloadClass("long", 1.0, 8000, 600)),
        ("headroom-rich (8B, 1xL40S, 1.5k context)",
         DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                         ServingConfig(n_accelerators=1, max_batch=256)),
         WorkloadClass("short", 1.0, 1500, 400)),
    ]

    for label, state, workload in cases:
        def cost(st):
            return self_hosted_query_cost(st, workload, grid,
                                          respect_slo=False).compute_cost

        base = cost(state)
        a = cost(apply_stack(state, ["int4"])) / base
        b = cost(apply_stack(state, ["paged_attention"])) / base
        both = cost(apply_stack(state, ["int4", "paged_attention"])) / base
        product = a * b
        error = (product - both) / both

        findings[label] = {"int4": a, "paged": b, "product": product,
                           "measured": both, "relative_error": error}
        print(f"\n  {label}")
        print(f"    INT4 alone                {a:.4f}x")
        print(f"    PagedAttention alone      {b:.4f}x")
        print(f"    product of the constants  {product:.4f}x")
        print(f"    measured together         {both:.4f}x")
        print(f"    the product errs by       {error:+.0%}")

    print("\n  In the memory-bound case INT4 has already freed the HBM that")
    print("  PagedAttention would have recovered, so the pair delivers far")
    print("  less than the product predicts. Where headroom is ample the")
    print("  interaction all but vanishes. The error has no reliable sign,")
    print("  which is the argument for deriving a multiplier over quoting one.")
    return findings


# ---------------------------------------------------------------------------
# Result 3: duty cycle and overhead separate the model from reference tables
# ---------------------------------------------------------------------------

def result_duty_cycle(out: Path) -> dict:
    banner("RESULT 3  Duty cycle and overhead dominate the unit-cost gap")

    grid = get_grid("us-average")
    workload = WorkloadClass("tutoring", 1.0, 1500, 400)
    model, hw = get_model("dense-70b"), get_hardware("h100-sxm")

    def cost(duty: float, sched: float, overhead: float) -> float:
        cfg = ServingConfig(n_accelerators=4, max_batch=256,
                            demand_duty_cycle=duty, scheduler_efficiency=sched,
                            infra_overhead=overhead)
        state = apply_stack(DeploymentState(model, hw, cfg), TUNED_STACK)
        return self_hosted_query_cost(state, workload, grid,
                                      respect_slo=False).compute_cost

    # A reference table reports a fully loaded replica with no overhead.
    idealised = cost(duty=1.0, sched=1.0, overhead=1.0)
    # A service provisioned for a deadline peak and idle through the term,
    # with the orchestration and redundancy a production deployment needs.
    realistic = cost(duty=0.42, sched=0.45, overhead=1.35)
    # Isolating each factor shows which one carries the gap.
    duty_only = cost(duty=0.42, sched=1.0, overhead=1.0)
    sched_only = cost(duty=1.0, sched=0.45, overhead=1.0)
    overhead_only = cost(duty=1.0, sched=1.0, overhead=1.35)

    print(f"  fully loaded, no overhead                 ${idealised:.6f}/query")
    print(f"  demand duty cycle 0.42 alone              ${duty_only:.6f}  "
          f"({duty_only/idealised:.2f}x)")
    print(f"  scheduler efficiency 0.45 alone           ${sched_only:.6f}  "
          f"({sched_only/idealised:.2f}x)")
    print(f"  infrastructure overhead 1.35 alone        ${overhead_only:.6f}  "
          f"({overhead_only/idealised:.2f}x)")
    print(f"  all three together                        ${realistic:.6f}  "
          f"({realistic/idealised:.2f}x)")
    print()
    print("  Reference unit-cost tables report the first figure. A service")
    print("  provisioned for a deadline peak, served by a scheduler that")
    print("  cannot fill every cycle, and carrying production overhead runs")
    print("  at the last. Neither is wrong; conflating them is.")
    print()
    print("  Note that the optimised stack raises scheduler efficiency but")
    print("  cannot touch demand duty cycle -- no serving optimisation")
    print("  creates traffic that was never sent. Speculative decoding is")
    print("  excluded from the stack here: at batch 256 on INT4 weights its")
    print("  derived multiplier exceeds one (see Result 1), and only a model")
    print("  that derives multipliers can catch that mis-tuning.")

    return {"idealised": idealised, "realistic": realistic,
            "ratio": realistic / idealised,
            "duty_only": duty_only / idealised,
            "scheduler_only": sched_only / idealised,
            "overhead_only": overhead_only / idealised}


# ---------------------------------------------------------------------------
# Result 4: break-even is a band, not a threshold
# ---------------------------------------------------------------------------

def result_break_even(out: Path) -> dict:
    banner("RESULT 4  Granular capacity turns break-even into a band")

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
             "stack": list(TUNED_STACK)}],
    })

    result = find_break_even(
        scenario.cost_curve("api-economy"), scenario.cost_curve("selfhost-70b"),
        label_a="api-economy", label_b="selfhost-70b",
        volume_min=1e7, volume_max=2e10, samples=900)

    bands = result.tie_bands(0.05)
    band = result.tie_band(0.05)
    print(f"  crossings found            {len(result.crossings)}")
    if result.crossings:
        print(f"  first crossing             "
              f"{result.crossings[0].volume:,.0f} q/yr")
        for c in result.crossings:
            print(f"    {c.volume:>16,.0f}  {c.winner_below} -> {c.winner_above}")
    else:
        winner = result.winner_at(1e9)
        print(f"  no crossing in range; {winner} is cheaper throughout")
    if band:
        print(f"  indistinguishable windows  {len(bands)}")
        for lo, hi in bands:
            print(f"      {lo:15,.0f} .. {hi:<15,.0f} ({hi / lo:.2f}x wide)")
        print(f"  widest window              "
              f"{band[0]:,.0f} .. {band[1]:,.0f} q/yr")
        outside = max(result.relative_gap(v) for v in sorted(result.curve_a)
                      if bands[0][0] <= v <= bands[-1][1])
        print(f"  worst gap between windows  {outside:.0%}")
    print()
    print("  A single 'self-host above N queries' threshold would present")
    print("  as a clean decision what is really a set of narrow windows in")
    print("  which cost does not decide, one around each replica riser.")
    print("  Reporting first-to-last of those windows as one band -- which")
    print("  this script did through v8.0 -- claims indistinguishability")
    print("  across stretches where the gap reaches nearly half.")

    plotting.plot_break_even(result, out / "fig3_break_even.png")
    return {"n_crossings": len(result.crossings),
            "first_crossing": (result.crossings[0].volume
                               if result.crossings else None),
            "band": list(band) if band else None,
            "n_tie_windows": len(bands),
            "tie_windows": [list(b) for b in bands]}


# ---------------------------------------------------------------------------
# Result 5: cross-domain application
# ---------------------------------------------------------------------------

def result_cross_domain(out: Path) -> dict:
    banner("RESULT 5  Three domains, three different answers")

    findings = {}
    print(f"{'scenario':<32}{'volume':>12}{'cheapest':>20}"
          f"{'TCO':>12}{'inference':>11}{'FTE':>7}")
    print("-" * 94)

    for name in ("university_tutoring", "hospital_documentation",
                 "public_helpline"):
        scenario = load_scenario(EXAMPLES / f"{name}.yaml")
        results = scenario.evaluate_all()
        feasible = {k: v for k, v in results.items() if v.feasible}
        best_name = min(feasible or results,
                        key=lambda k: (feasible or results)[k].total)
        best = results[best_name]
        share = best.share()
        infra = share.get("compute_serving", 0) + share.get("model_access", 0)

        # Spread over the architectures a planner may actually choose.
        # Ranking one that fails a declared quality floor above one that
        # meets it compares two different services, not two prices.
        pool = feasible or results
        spread = (max(r.total for r in pool.values())
                  / min(r.total for r in pool.values()) - 1)

        findings[name] = {
            "volume": scenario.annual_volume,
            "cheapest": best_name,
            "total": best.total,
            "inference_share": infra,
            "review_fte": best.review_fte,
            "displaced": best.displaced_labour_annual,
            "architecture_spread": spread,
            "n_feasible": float(len(feasible)),
            "n_architectures": float(len(results)),
            "n_fully_evaluated": float(
                sum(1 for r in results.values() if r.fully_evaluated)),
            "winner_fully_evaluated": float(best.fully_evaluated),
        }
        flag = "" if feasible else "  (none feasible)"
        print(f"{name:<30}{scenario.annual_volume:>12,.0f}{best_name:>18}"
              f"${best.total/1e6:>11.2f}M{infra:>9.0%}{best.review_fte:>6.0f}"
              f"{len(feasible):>4}/{len(results)}{flag}")

    print()
    for name, f in findings.items():
        print(f"  {name}: {f['n_feasible']:.0f} of {f['n_architectures']:.0f} "
              f"architectures meet every declared quality floor; those span "
              f"{f['architecture_spread']:.1%}; "
              f"displaced labour ${f['displaced']/1e6:.1f}M/yr")
    unchecked = [n for n, f in findings.items()
                 if not f["winner_fully_evaluated"]]
    if unchecked:
        print(f"\n  Winner's constraints not all evaluated: {', '.join(unchecked)}")
        print("  A commercial endpoint's latency is not modelled here, so its")
        print("  latency objective is recorded as unevaluated rather than met.")
        print("  Through v10.0 it was recorded as met, which let an API")
        print("  architecture satisfy by construction a check every")
        print("  self-hosted candidate had to pass on evidence.")

    print("\n  A quality floor is a constraint, not a preference. Through")
    print("  v9.0 it was enforced when routing and ignored when comparing")
    print("  architectures, so the architecture reported cheapest failed at")
    print("  least one floor in all three scenarios -- and in the helpline")
    print("  case no single architecture meets them all, which is what the")
    print("  routing command had been saying about the same file all along.")

    print("\n  Inference is a minority of every total. The candidate")
    print("  architectures differ from each other by less than the width of")
    print("  the uncertainty on the fixed layers -- so in all three cases the")
    print("  model choice is not the decision that matters most.")
    return findings


# ---------------------------------------------------------------------------
# Result 6: uncertainty changes the reading
# ---------------------------------------------------------------------------

def result_uncertainty(out: Path) -> dict:
    banner("RESULT 6  Point estimates hide the comparison")

    scenario = load_scenario(EXAMPLES / "university_tutoring.yaml")

    mc_results = {}
    for name in ("api-midrange", "selfhost-70b"):
        arch = scenario.architecture(name)
        mc_results[name] = monte_carlo(
            lambda d, a=arch: perturbed_cost(scenario, a, d),
            list(scenario.uncertainty.values()),
            n_samples=4000, seed=SEED, label=name)

    for name, mc in mc_results.items():
        s = mc.summary()
        print(f"  {name:<16} p05 ${s['p05']/1e6:5.2f}M   "
              f"p50 ${s['p50']/1e6:5.2f}M   p95 ${s['p95']/1e6:5.2f}M   "
              f"CV {s['cv']:.2f}")

    a = mc_results["api-midrange"].valid
    b = mc_results["selfhost-70b"].valid
    k = min(a.size, b.size)
    prob = float((a[:k] < b[:k]).mean())
    print(f"\n  api-midrange is cheaper in {prob:.0%} of {k:,} draws")

    # Sensitivity is reported for the SELF-HOSTED architecture. On a
    # commercial-endpoint architecture, accelerator price and utilisation
    # are structurally absent from the cost model, so a near-zero
    # sensitivity there says nothing about how much those inputs matter.
    # The v1.0 paper drew exactly that invalid inference.
    entries = sensitivity(mc_results["selfhost-70b"])
    api_entries = sensitivity(mc_results["api-midrange"])

    print("\n  sensitivity ranking (selfhost-70b)")
    for e in entries[:6]:
        print(f"    {e.name:<22} rho {e.spearman:+.3f}   "
              f"{e.contribution:5.1%} of explained variance")

    print(f"\n  feasible draws: {mc.feasible_fraction:.1%}; "
          f"declared inputs explain {mc.explained_rank_variance():.2f} of "
          f"rank variance before normalisation")
    print("  Percentiles and the ranking above are computed on feasible")
    print("  draws only, so both are conditional on the configuration")
    print("  working -- which matters when that fraction is below one.")

    from caide.perturb import uncovered_draw_keys
    held = uncovered_draw_keys(scenario.uncertainty)
    print(f"\n  inputs held fixed (no declared distribution): "
          f"{', '.join(held) if held else 'none'}")
    print("  A sensitivity ranking answers 'of the inputs we varied, which")
    print("  mattered'. Through v8.0 two of the three factors of per-query")
    print("  review cost -- minutes and wage -- could not be varied at all,")
    print("  so they never appeared here and volume ranked first.")

    accel = next((e for e in entries if e.name == "accelerator_hourly"), None)
    accel_api = next((e for e in api_entries if e.name == "accelerator_hourly"), None)
    if accel and accel_api:
        print(f"\n  accelerator_hourly: {accel.contribution:.1%} on the "
              f"self-hosted architecture, {accel_api.contribution:.1%} on the "
              f"API one.")
        print("  The API figure is structural, not empirical: that cost model")
        print("  contains no accelerator price term at all. Only the")
        print("  self-hosted number is evidence about anything.")

    print(f"\n  Narrowing '{entries[0].name}' alone is worth more than")
    print("  refining every other input in the model.")

    plotting.plot_uncertainty(mc_results, out / "fig4_uncertainty.png")
    plotting.plot_tornado(entries, out / "fig5_tornado.png",
                          title="Sensitivity of annual cost (self-hosted 70B)")
    return {"probability_api_cheaper": prob,
            "top_driver": entries[0].name,
            "top_contribution": entries[0].contribution,
            "accelerator_selfhosted": accel.contribution if accel else None,
            "feasible_fraction": mc.feasible_fraction,
            "explained_rank_variance": mc.explained_rank_variance(),
            "accelerator_api": accel_api.contribution if accel_api else None,
            "p05": mc_results["selfhost-70b"].percentile(5),
            "p50": mc_results["selfhost-70b"].percentile(50),
            "p95": mc_results["selfhost-70b"].percentile(95)}


# ---------------------------------------------------------------------------
# Result 7: the Jevons regime
# ---------------------------------------------------------------------------

def result_jevons(out: Path) -> dict:
    banner("RESULT 7  Falling unit cost, rising total spend")

    scenario = load_scenario(EXAMPLES / "university_tutoring.yaml")
    results = scenario.evaluate_all()
    feasible = {k: v for k, v in results.items() if v.feasible}
    best = (feasible or results)[min(feasible or results,
                                     key=lambda k: (feasible or results)[k].total)]

    # Only part of a query's cost tracks the token tariff. Feeding the
    # blended per-query figure into a tariff decline -- which this script
    # did through v7.0 -- declines reviewer wages at the speed of GPU
    # prices and makes a fixed audit programme scale with volume.
    split = best.scaling_inputs()

    def assume(eps):
        return ScalingAssumptions(
            annual_price_decline=0.38, price_elasticity=eps,
            autonomous_growth=0.12, horizon_years=5,
            price_inelastic_per_query=split["price_inelastic_per_query"],
            fixed_annual_cost=split["fixed_annual"])

    print(f"  cost per query that tracks the token tariff: "
          f"{split['declining_share']:.1%}")
    print("  (reviewer time and fixed programmes carry the rest)\n")

    findings = {"declining_share": split["declining_share"]}
    for eps in (0.6, 1.0, 1.35, 1.8):
        proj = project(split["declining_per_query"], scenario.annual_volume,
                       assume(eps))
        eff = proj.years[-1].effective_unit_cost / proj.years[0].effective_unit_cost
        findings[eps] = {"regime": proj.regime,
                         "volume_ratio": proj.volume_ratio,
                         "spend_ratio": proj.spend_ratio,
                         "effective_price_ratio": eff}
        print(f"  elasticity {eps:.2f} ({proj.regime:<9})  "
              f"volume x{proj.volume_ratio:5.1f}   "
              f"effective price x{eff:4.2f}   "
              f"spend x{proj.spend_ratio:5.2f}")

    print("\n  The crossover at elasticity 1.0 remains exact: spend follows")
    print("  the effective price raised to (1 - eps). What the composition")
    print("  changes is the magnitude -- a 38%/yr tariff decline is a 9%/yr")
    print("  decline in what this institution pays, so demand barely moves")
    print("  and spend rises in every regime once adoption growth is added.")
    print("  An institution forecasting from the unit-cost trend alone gets")
    print("  the direction wrong; one forecasting from a blended per-query")
    print("  figure gets the magnitude wrong by a factor of three or more.")

    proj = project(split["declining_per_query"], scenario.annual_volume,
                   assume(1.35))
    plotting.plot_scaling(proj, out / "fig7_jevons.png")
    return findings


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Result 8: validation against published measurements
#
# Added in v13.0. Through v12.0 the manuscript's validation paragraph -- the
# one place a reader looks to decide whether to trust anything else -- was
# not produced here. Its figures were carried by hand while the script's
# own preamble said every result below was regenerated under a fixed seed.
# They had moved twice: v7.0 changed every predicted-to-measured ratio and
# v10.0 re-derived the framework overhead constants.
# ---------------------------------------------------------------------------

def result_validation(out: Path) -> dict:
    banner("RESULT 8  Validation against published serving measurements")
    from caide.calibration import (
        FRAMEWORK_OVERHEAD_REFERENCE,
        REFERENCE_OBSERVATIONS,
        EXCLUDED_OBSERVATIONS,
        READMITTED_OBSERVATIONS,
        admissible_conventions,
        fit,
        implied_mbu,
        predicted_output_tps,
    )

    observations = REFERENCE_OBSERVATIONS()
    print(f"  {'configuration':<44}{'pred':>8}{'meas':>8}{'ratio':>7}{'MBU':>7}")
    print("  " + "-" * 74)
    rows = []
    for obs in observations:
        predicted = predicted_output_tps(obs)
        ratio = predicted / obs.aggregate_output_tps
        mbu = implied_mbu(obs, "aggregate")
        rows.append({"source": obs.source, "predicted": predicted,
                     "measured": obs.aggregate_output_tps, "ratio": ratio,
                     "implied_mbu": mbu,
                     "conventions": admissible_conventions(obs)})
        print(f"  {obs.source[:44]:<44}{predicted:>8.0f}"
              f"{obs.aggregate_output_tps:>8.0f}{ratio:>7.2f}{mbu:>7.2f}")

    within = [r for r in rows if 0.5 <= r["ratio"] <= 2.0]
    over = [r for r in rows if r["ratio"] > 1.0]
    result = fit(observations)
    summary = result.summary()

    print(f"\n  within a factor of two   {len(within)} of {len(rows)}")
    print(f"  over-predicts            {len(over)} of {len(rows)}")
    print(f"  calibration lifts the within-2x fraction "
          f"{summary['within_2x_before']:.0%} -> {summary['within_2x_after']:.0%}")
    print(f"  mbu_scale                {summary['mbu_scale']:.3f}")
    print(f"  excluded / readmitted    {len(EXCLUDED_OBSERVATIONS)} / "
          f"{len(READMITTED_OBSERVATIONS)}")
    print("\n  The overhead constants are a two-point fit to two parameters:")
    print(f"  {FRAMEWORK_OVERHEAD_REFERENCE['degrees_of_freedom']} degrees of "
          f"freedom, status \"{FRAMEWORK_OVERHEAD_REFERENCE['status']}\".")

    return {
        "observations": rows,
        "n_observations": float(len(rows)),
        "n_within_2x": float(len(within)),
        "n_over_predicting": float(len(over)),
        "within_2x_before": summary["within_2x_before"],
        "within_2x_after": summary["within_2x_after"],
        "mbu_scale": summary["mbu_scale"],
        "log_rmse_before": summary["log_rmse_before"],
        "log_rmse_after": summary["log_rmse_after"],
        "n_excluded": float(len(EXCLUDED_OBSERVATIONS)),
        "n_readmitted": float(len(READMITTED_OBSERVATIONS)),
        "overhead_per_step_seconds":
            FRAMEWORK_OVERHEAD_REFERENCE["per_step_seconds"],
        "overhead_per_sequence_seconds":
            FRAMEWORK_OVERHEAD_REFERENCE["per_sequence_seconds"],
        "overhead_degrees_of_freedom":
            float(FRAMEWORK_OVERHEAD_REFERENCE["degrees_of_freedom"]),
    }


# ---------------------------------------------------------------------------
# Result 9: are the conclusions artefacts of the modelling choices?
#
# The nine-variant sweep the manuscript cites was, through v12.0, re-run by
# hand each round and typed into the text. Same hazard as Result 8.
# ---------------------------------------------------------------------------

VARIANTS = {
    "baseline": {},
    "decode_mfu_half_batch=16": {"decode_mfu_half_batch": 16.0},
    "decode_mfu_half_batch=256": {"decode_mfu_half_batch": 256.0},
    "mfu_prefill=0.30": {"mfu_prefill": 0.30},
    "mfu_prefill=0.60": {"mfu_prefill": 0.60},
    "mbu_decode=0.50": {"mbu_decode": 0.50},
    "mbu_decode=0.90": {"mbu_decode": 0.90},
    "interconnect_ideal": {"tensor_parallel_penalty": 0.0},
    "interconnect_poor": {"tensor_parallel_penalty": 0.08},
}


def result_structural_sensitivity(out: Path) -> dict:
    banner("RESULT 9  The conclusions are not artefacts of the modelling choices")
    grid = get_grid("us-average")
    workload = WorkloadClass("tutoring", 1.0, 1500, 400)
    hardware = get_hardware("h100-sxm")

    def multipliers(model_key, accelerators, technique, overrides):
        out_series = []
        for batch in BATCHES:
            cfg = ServingConfig(n_accelerators=accelerators, max_batch=batch,
                                **overrides)
            state = DeploymentState(get_model(model_key), hardware, cfg)
            base = self_hosted_query_cost(state, workload, grid,
                                          respect_slo=False).compute_cost
            cost = self_hosted_query_cost(
                apply_stack(state, [technique]), workload, grid,
                respect_slo=False).compute_cost
            out_series.append(cost / base)
        return out_series

    print(f"  {'variant':<28}{'dense@1':>9}{'dense@256':>11}{'spread':>8}"
          f"{'moe@1':>8}{'int4 err':>10}")
    print("  " + "-" * 74)
    rows = {}
    for name, overrides in VARIANTS.items():
        dense = multipliers("dense-70b", 4, "speculative_decoding", overrides)
        moe = multipliers("moe-8x7b", 2, "speculative_decoding", overrides)
        int4 = multipliers("dense-70b", 4, "int4", overrides)
        err = max(abs(PUBLISHED["int4"] - min(int4)) / min(int4),
                  abs(PUBLISHED["int4"] - max(int4)) / max(int4))
        rows[name] = {"dense_at_1": dense[0], "dense_at_256": dense[-1],
                      "dense_spread": max(dense) / min(dense),
                      "moe_at_1": moe[0], "moe_best": min(moe),
                      "int4_worst_error": err}
        print(f"  {name:<28}{dense[0]:>9.3f}{dense[-1]:>11.3f}"
              f"{max(dense) / min(dense):>8.2f}{moe[0]:>8.3f}{err:>10.0%}")

    spreads = [r["dense_spread"] for r in rows.values()]
    at_256 = [r["dense_at_256"] for r in rows.values()]
    moe_at_1 = [r["moe_at_1"] for r in rows.values()]
    errors = [r["int4_worst_error"] for r in rows.values()]

    print(f"\n  dense spread            {min(spreads):.2f}x .. {max(spreads):.2f}x")
    print(f"  saturating multiplier   {min(at_256):.3f} .. {max(at_256):.3f}; "
          f"past parity in {sum(1 for x in at_256 if x > 1.0)} of {len(at_256)}")
    print(f"  MoE at batch one        {min(moe_at_1):.3f} .. {max(moe_at_1):.3f}")
    print(f"  int4 worst-case error   {min(errors):.0%} .. {max(errors):.0%}")

    return {
        "variants": rows,
        "n_variants": float(len(rows)),
        "dense_spread_min": min(spreads), "dense_spread_max": max(spreads),
        "dense_at_256_min": min(at_256), "dense_at_256_max": max(at_256),
        "n_past_parity": float(sum(1 for x in at_256 if x > 1.0)),
        "moe_at_1_min": min(moe_at_1), "moe_at_1_max": max(moe_at_1),
        "int4_error_min": min(errors), "int4_error_max": max(errors),
    }


# ---------------------------------------------------------------------------
# Result 10: the provenance record, and the routing answer for the scenario
# that has no single admissible architecture.
#
# Both are capabilities the manuscript describes and no published result
# exercised: report.py and routing.py were at 0% and 25% of statements when
# the reproduction script was measured on its own in v13.0.
# ---------------------------------------------------------------------------

def result_provenance_and_routing(out: Path) -> dict:
    banner("RESULT 10  Provenance, and routing a workload no architecture fits")
    from caide.report import ReportBundle, scenario_digest, write_markdown
    from caide.routing import Tier, optimise_routing

    scenario = load_scenario(EXAMPLES / "public_helpline.yaml")
    results = scenario.evaluate_all()
    bundle = ReportBundle(scenario, seed=20260101)
    bundle.tco = results
    digest = scenario_digest(scenario)
    report_path = write_markdown(bundle, out / "public_helpline_report.md")

    print(f"  scenario digest          {digest}")
    print(f"  admissible architectures {sum(1 for r in results.values() if r.feasible)}"
          f" of {len(results)}")
    print(f"  report written           {report_path.name}")

    # One tier per candidate architecture, priced from the replicas each
    # would actually require, then the minimum-cost assignment.
    tiers = []
    for arch in scenario.architectures:
        if arch.pricing is not None:
            tiers.append(Tier(
                name=arch.name, quality_index=arch.pricing.quality_index,
                cost_fn=lambda w, p=arch.pricing: p.query_cost(w.tokens_in,
                                                               w.tokens_out)))
        else:
            tiers.append(Tier(
                name=arch.name,
                quality_index=arch.state.model.quality_index,
                cost_fn=lambda w, s=arch.state: self_hosted_query_cost(
                    s, w, scenario.grid, scenario.slo).compute_cost))

    plan = optimise_routing(scenario.workloads, tiers, scenario.annual_volume)
    print(f"  tiers opened             {', '.join(plan.tiers_opened)}")
    print(f"  unroutable classes       "
          f"{', '.join(plan.unroutable) if plan.unroutable else 'none'}")
    print(f"  blended quality          {plan.blended_quality:.3f}")
    print("\n  No single architecture meets every quality floor here, so the")
    print("  cost answer is a routing plan rather than a choice. The floors")
    print("  the plan respects are the same ones the comparison enforces.")

    return {
        "digest": digest,
        "n_feasible": float(sum(1 for r in results.values() if r.feasible)),
        "n_architectures": float(len(results)),
        "tiers_opened": list(plan.tiers_opened),
        "n_tiers_opened": float(len(plan.tiers_opened)),
        "unroutable": list(plan.unroutable),
        "blended_quality": plan.blended_quality,
        "routed_annual_cost": plan.annual_cost,
        "plan_is_exact": float(plan.exact),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper_figures")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    findings = {
        "regime_dependence": result_regime_dependence(out),
        "interaction": result_interaction(out),
        "duty_cycle": result_duty_cycle(out),
        "break_even": result_break_even(out),
        "cross_domain": result_cross_domain(out),
        "uncertainty": result_uncertainty(out),
        "jevons": result_jevons(out),
        "validation": result_validation(out),
        "structural_sensitivity": result_structural_sensitivity(out),
        "provenance_and_routing": result_provenance_and_routing(out),
    }

    # headline figure: the layer decomposition of the flagship scenario
    scenario = load_scenario(EXAMPLES / "university_tutoring.yaml")
    results = scenario.evaluate_all()
    plotting.plot_tco_breakdown(
        results, out / "fig6_tco_breakdown.png",
        title="Annual cost of ownership by layer (engineering education)")

    (out / "findings.json").write_text(json.dumps(findings, indent=2,
                                                  default=str))
    banner("DONE")
    for f in sorted(out.glob("*.png")):
        print(f"  {f}")
    print(f"  {out / 'findings.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
