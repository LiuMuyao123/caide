"""Property-based audit: let a search engine look for counterexamples.

Rounds 1 and 2 used fixed-point tests. Those confirm behaviour at the
points someone thought to check. This round states the invariants the
model claims to satisfy and lets Hypothesis hunt for parameter
combinations that break them.
"""

from __future__ import annotations

import math
import sys

from hypothesis import HealthCheck, assume, given, settings, strategies as st

from caide import (
    DeploymentState,
    ModelSpec,
    ServingConfig,
    SLO,
    WorkloadClass,
    apply_stack,
    capacity_batch,
    evaluate_request,
    find_break_even,
    get_grid,
    get_hardware,
    get_model,
    project,
    self_hosted_query_cost,
    solve_batch_for_slo,
)
from caide.scaling import ScalingAssumptions

GRID = get_grid("us-average")
SETTINGS = settings(max_examples=300, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])

MODELS = ["dense-1b", "dense-3b", "dense-8b", "dense-32b", "dense-70b",
          "moe-8x7b", "moe-8x22b", "moe-236b"]
HARDWARE = ["a100-40gb", "a100-80gb", "h100-sxm", "h200-sxm", "l40s",
            "consumer-24gb"]
STACKS = ["none", "baseline_serving", "standard", "aggressive", "maximal"]

FAILURES = []


def record(name, exc):
    FAILURES.append((name, exc))
    print(f"  FAIL  {name}")
    print("        " + str(exc).replace("\n", "\n        ")[:600])


def run(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:                              # noqa: BLE001
        record(name, exc)


# ---------------------------------------------------------------------------
# P1  Utilisation invariant
# ---------------------------------------------------------------------------

@SETTINGS
@given(duty=st.floats(0.01, 1.0), sched=st.floats(0.01, 1.0),
       stack=st.sampled_from(STACKS))
def p1_effective_never_exceeds_demand(duty, sched, stack):
    cfg = ServingConfig(n_accelerators=2, demand_duty_cycle=duty,
                        scheduler_efficiency=sched)
    state = DeploymentState(get_model("dense-8b"), get_hardware("h100-sxm"), cfg)
    after = apply_stack(state, stack).serving
    assert after.demand_duty_cycle == duty, "a technique changed demand"
    assert after.effective_utilisation <= duty + 1e-12


# ---------------------------------------------------------------------------
# P2  Monotonicity of cost in the obvious directions
# ---------------------------------------------------------------------------

@SETTINGS
@given(duty=st.floats(0.05, 0.99), hourly=st.floats(0.1, 20.0),
       tin=st.integers(64, 8000), tout=st.integers(8, 2000))
def p2_cost_falls_as_duty_rises(duty, hourly, tin, tout):
    from dataclasses import replace
    hw = replace(get_hardware("h100-sxm"), hourly_cost=hourly)
    w = WorkloadClass("q", 1.0, tin, tout)

    def cost(d):
        cfg = ServingConfig(n_accelerators=4, max_batch=128,
                            demand_duty_cycle=d, scheduler_efficiency=0.6)
        return self_hosted_query_cost(
            DeploymentState(get_model("dense-8b"), hw, cfg), w, GRID,
            respect_slo=False).compute_cost

    lo, hi = cost(duty * 0.5), cost(duty)
    assert hi <= lo + 1e-15, f"cost rose when duty rose: {lo} -> {hi}"


@SETTINGS
@given(model=st.sampled_from(MODELS), hw=st.sampled_from(HARDWARE),
       n=st.integers(1, 8), tin=st.integers(64, 4000), tout=st.integers(8, 1000))
def p2b_more_accelerators_never_reduce_capacity(model, hw, n, tin, tout):
    m, h = get_model(model), get_hardware(hw)
    w = WorkloadClass("q", 1.0, tin, tout)
    c1 = capacity_batch(m, h, ServingConfig(n_accelerators=n), w.avg_sequence)
    c2 = capacity_batch(m, h, ServingConfig(n_accelerators=n + 1), w.avg_sequence)
    assert c2 >= c1 - 1e-9


# ---------------------------------------------------------------------------
# P3  Roofline outputs stay physical
# ---------------------------------------------------------------------------

@SETTINGS
@given(model=st.sampled_from(MODELS), hw=st.sampled_from(HARDWARE),
       n=st.integers(1, 8), batch=st.integers(1, 512),
       tin=st.integers(16, 16000), tout=st.integers(1, 4000))
def p3_no_nan_or_negative_outputs(model, hw, n, batch, tin, tout):
    state = DeploymentState(get_model(model), get_hardware(hw),
                            ServingConfig(n_accelerators=n, max_batch=batch))
    perf = evaluate_request(state, WorkloadClass("q", 1.0, tin, tout))
    for field in ("ttft", "tpot", "latency", "throughput_qps",
                  "accelerator_seconds"):
        v = getattr(perf, field)
        assert not (isinstance(v, float) and math.isnan(v)), f"{field} is NaN"
        assert v >= 0, f"{field} is negative: {v}"
    if perf.feasible:
        assert perf.throughput_qps > 0
        assert math.isfinite(perf.latency)


@SETTINGS
@given(model=st.sampled_from(MODELS), hw=st.sampled_from(HARDWARE),
       n=st.integers(1, 8), tin=st.integers(16, 8000), tout=st.integers(1, 2000))
def p3b_latency_equals_ttft_plus_decode(model, hw, n, tin, tout):
    state = DeploymentState(get_model(model), get_hardware(hw),
                            ServingConfig(n_accelerators=n, max_batch=64))
    perf = evaluate_request(state, WorkloadClass("q", 1.0, tin, tout))
    if not math.isfinite(perf.latency):
        return
    assert perf.latency == perf.ttft + perf.decode_seconds


# ---------------------------------------------------------------------------
# P4  SLO solver contract
# ---------------------------------------------------------------------------

@SETTINGS
@given(model=st.sampled_from(MODELS), hw=st.sampled_from(HARDWARE),
       n=st.integers(1, 8), tin=st.integers(64, 4000), tout=st.integers(8, 800),
       ttft=st.floats(0.05, 20.0), tpot=st.floats(0.005, 0.5))
def p4_solver_returns_a_feasible_point_or_admits_failure(model, hw, n, tin,
                                                         tout, ttft, tpot):
    state = DeploymentState(get_model(model), get_hardware(hw),
                            ServingConfig(n_accelerators=n, max_batch=256))
    w = WorkloadClass("q", 1.0, tin, tout)
    slo = SLO(ttft_seconds=ttft, tpot_seconds=tpot)
    perf = solve_batch_for_slo(state, w, slo)
    if perf.slo_met:
        assert perf.ttft <= ttft + 1e-9
        assert perf.tpot <= tpot + 1e-9
        assert perf.batch >= 1


# ---------------------------------------------------------------------------
# P5  Efficiency stacks never increase cost
# ---------------------------------------------------------------------------

@SETTINGS
@given(model=st.sampled_from(MODELS), hw=st.sampled_from(HARDWARE),
       n=st.integers(1, 8), tin=st.integers(64, 6000), tout=st.integers(8, 1200),
       batch=st.integers(1, 256))
def p5_richer_stacks_never_cost_more(model, hw, n, tin, tout, batch):
    base = DeploymentState(get_model(model), get_hardware(hw),
                           ServingConfig(n_accelerators=n, max_batch=batch,
                                         demand_duty_cycle=0.7,
                                         scheduler_efficiency=0.5))
    w = WorkloadClass("q", 1.0, tin, tout)

    def cost(stack):
        c = self_hosted_query_cost(apply_stack(base, stack), w, GRID,
                                   respect_slo=False).compute_cost
        return c

    prev = cost("none")
    if not math.isfinite(prev):
        return
    for stack in ("baseline_serving", "standard", "aggressive"):
        cur = cost(stack)
        if not math.isfinite(cur):
            return
        assert cur <= prev * 1.0001, (
            f"{stack} cost more than the previous stack: {prev} -> {cur}")
        prev = cur


# ---------------------------------------------------------------------------
# P6  Break-even scan self-consistency
# ---------------------------------------------------------------------------

@SETTINGS
@given(a0=st.floats(1.0, 1e5), a1=st.floats(1e-6, 1e-2),
       b0=st.floats(1.0, 1e6), b1=st.floats(1e-6, 1e-2))
def p6_reported_crossings_actually_cross(a0, a1, b0, b1):
    fa = lambda v: a0 + a1 * v
    fb = lambda v: b0 + b1 * v
    r = find_break_even(fa, fb, label_a="a", label_b="b",
                        volume_min=1e3, volume_max=1e10, samples=200)
    for c in r.crossings:
        lo, hi = c.volume * 0.98, c.volume * 1.02
        assert (fa(lo) - fb(lo)) * (fa(hi) - fb(hi)) <= 1e-9, (
            "reported crossing has no sign change around it")


# ---------------------------------------------------------------------------
# P7  Scaling projection algebra
# ---------------------------------------------------------------------------

@SETTINGS
@given(c0=st.floats(1e-6, 1.0), v0=st.floats(1e3, 1e10),
       r=st.floats(0.0, 0.95), eps=st.floats(0.0, 3.0),
       g=st.floats(0.0, 0.5), years=st.integers(1, 12))
def p7_spend_matches_closed_form(c0, v0, r, eps, g, years):
    proj = project(c0, v0, ScalingAssumptions(
        annual_price_decline=r, price_elasticity=eps,
        autonomous_growth=g, horizon_years=years))
    for y in proj.years:
        t = y.year - 1
        c = c0 * (1 - r) ** t
        ratio = (c / c0) if c0 > 0 else 1.0
        v = v0 * ratio ** (-eps) * (1 + g) ** t
        assert math.isclose(y.unit_cost, c, rel_tol=1e-9)
        assert math.isclose(y.volume, v, rel_tol=1e-9)
        assert math.isclose(y.variable_spend, c * v, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# P8  Model geometry invariants
# ---------------------------------------------------------------------------

@SETTINGS
@given(params=st.floats(1e8, 1e12), layers=st.integers(1, 200),
       d_model=st.integers(128, 16384), heads=st.integers(1, 128),
       bpp=st.floats(0.25, 4.0))
def p8_weight_and_kv_bytes_are_positive_and_scale(params, layers, d_model,
                                                  heads, bpp):
    assume(d_model % heads == 0)
    m = ModelSpec("t", n_params_total=params, n_layers=layers,
                  d_model=d_model, n_heads=heads, bytes_per_param=bpp)
    assert m.weight_bytes == params * bpp
    assert m.kv_bytes_per_token > 0
    doubled = ModelSpec("t2", n_params_total=params, n_layers=layers * 2,
                        d_model=d_model, n_heads=heads, bytes_per_param=bpp)
    assert math.isclose(doubled.kv_bytes_per_token,
                        2 * m.kv_bytes_per_token, rel_tol=1e-9)


@SETTINGS
@given(model=st.sampled_from(["moe-8x7b", "moe-8x22b", "moe-236b"]),
       batch=st.integers(1, 4096))
def p8b_moe_expert_bytes_bounded(model, batch):
    m = get_model(model)
    touched = m.expert_bytes_touched(batch)
    assert 0 < touched <= m.weight_bytes * (1 + 1e-9)
    assert touched >= m.expert_bytes_touched(1) - 1e-6


def main() -> int:
    checks = [
        ("P1  effective_utilisation <= demand_duty_cycle",
         p1_effective_never_exceeds_demand),
        ("P2  cost is non-increasing in duty cycle", p2_cost_falls_as_duty_rises),
        ("P2b more accelerators never reduce capacity",
         p2b_more_accelerators_never_reduce_capacity),
        ("P3  no NaN or negative roofline outputs", p3_no_nan_or_negative_outputs),
        ("P3b latency == ttft + decode", p3b_latency_equals_ttft_plus_decode),
        ("P4  SLO solver honours its contract",
         p4_solver_returns_a_feasible_point_or_admits_failure),
        ("P5  richer stacks never cost more", p5_richer_stacks_never_cost_more),
        ("P6  reported crossings actually cross", p6_reported_crossings_actually_cross),
        ("P7  scaling matches the closed form", p7_spend_matches_closed_form),
        ("P8  model geometry invariants", p8_weight_and_kv_bytes_are_positive_and_scale),
        ("P8b MoE expert bytes bounded", p8b_moe_expert_bytes_bounded),
    ]
    print("=" * 74)
    print("PROPERTY-BASED AUDIT  (Hypothesis, 300 examples per property)")
    print("=" * 74)
    for name, fn in checks:
        run(name, fn)
    print()
    print(f"failures: {len(FAILURES)} / {len(checks)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
