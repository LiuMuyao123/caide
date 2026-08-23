"""Tests for the roofline performance model.

Several tests check the model against closed-form limits rather than
against remembered numbers, because a regression test that only asserts
"this used to return 0.0034" tells you that something changed but not
whether the change was wrong.
"""

from __future__ import annotations

import math

import pytest

from caide import (
    DeploymentState,
    ModelSpec,
    ServingConfig,
    SLO,
    WorkloadClass,
    capacity_batch,
    evaluate_request,
    get_hardware,
    get_model,
    prefill_flops,
    solve_batch_for_slo,
)
from caide.roofline import decode_step_time


@pytest.fixture
def state():
    return DeploymentState(get_model("dense-8b"), get_hardware("l40s"),
                           ServingConfig(n_accelerators=1, max_batch=128))


@pytest.fixture
def workload():
    return WorkloadClass("q", 1.0, tokens_in=1000, tokens_out=300)


# -- geometry ---------------------------------------------------------------

def test_weight_bytes_matches_hand_calculation():
    m = ModelSpec("t", n_params_total=7e9, n_layers=32, d_model=4096, n_heads=32)
    assert m.weight_bytes == pytest.approx(14e9)          # 7B at bf16
    assert m.with_precision(0.5).weight_bytes == pytest.approx(3.5e9)


def test_kv_bytes_accounts_for_grouped_query_attention():
    mha = ModelSpec("mha", 8e9, 32, 4096, 32, n_kv_heads=32)
    gqa = ModelSpec("gqa", 8e9, 32, 4096, 32, n_kv_heads=8)
    assert mha.kv_bytes_per_token == pytest.approx(4 * gqa.kv_bytes_per_token)


def test_prefill_flops_linear_term_dominates_at_short_context():
    m = get_model("dense-8b")
    linear_only = 2 * m.active_params * 100
    assert prefill_flops(m, 100) == pytest.approx(linear_only, rel=0.02)


def test_prefill_flops_quadratic_term_matters_at_long_context():
    m = get_model("dense-8b")
    linear_only = 2 * m.active_params * 100_000
    assert prefill_flops(m, 100_000) > 1.5 * linear_only


def test_moe_expert_touch_grows_with_batch():
    moe = get_model("moe-8x7b")
    single = moe.expert_bytes_touched(1)
    many = moe.expert_bytes_touched(512)
    assert single < many
    assert many == pytest.approx(moe.weight_bytes, rel=0.01)


def test_dense_model_touches_all_weights_regardless_of_batch():
    dense = get_model("dense-8b")
    assert dense.expert_bytes_touched(1) == dense.expert_bytes_touched(1000)


# -- capacity ---------------------------------------------------------------

def test_capacity_zero_when_weights_do_not_fit():
    cfg = ServingConfig(n_accelerators=1)
    assert capacity_batch(get_model("dense-405b"), get_hardware("l40s"),
                          cfg, 2000) == 0.0


def test_capacity_scales_with_accelerator_count():
    m, hw = get_model("dense-8b"), get_hardware("a100-80gb")
    one = capacity_batch(m, hw, ServingConfig(n_accelerators=1), 2000)
    two = capacity_batch(m, hw, ServingConfig(n_accelerators=2), 2000)
    assert two > 2 * one      # weights are fixed, so headroom more than doubles


def test_infeasible_configuration_reports_infinite_latency():
    st = DeploymentState(get_model("dense-405b"), get_hardware("l40s"),
                         ServingConfig(n_accelerators=1))
    perf = evaluate_request(st, WorkloadClass("q", 1.0, 500, 100))
    assert not math.isfinite(perf.latency)
    assert perf.throughput_qps == 0.0
    assert perf.decode_bound_by == "infeasible"


# -- roofline behaviour -----------------------------------------------------

def test_decode_is_memory_bound_at_small_batch(state):
    _, bound = decode_step_time(state.model, state.hardware, state.serving,
                                batch=1, context_length=1000)
    assert bound == "memory"


def test_decode_becomes_compute_bound_at_large_batch_short_context(state):
    """The roofline must actually turn over; a model that stays memory
    bound at every batch is not a roofline, it is a straight line.

    The turnover requires a short context. Both terms grow linearly in
    batch -- compute as ``2*N*B``, memory as ``kv_per_token*ctx*B`` -- so
    which one wins is set by the context length, not by the batch.
    """
    bounds = [decode_step_time(state.model, state.hardware, state.serving,
                               batch=b, context_length=64)[1]
              for b in (1, 4096)]
    assert bounds[0] == "memory"
    assert bounds[1] == "compute"


def test_long_context_stays_memory_bound_at_any_batch(state):
    """A real and frequently missed property: once the KV cache is large
    enough, streaming it costs more per sequence than the arithmetic does,
    so batching cannot move the workload off the memory roof. Deployments
    that assume batching always reaches the compute roof will over-provision
    FLOPs and under-provision bandwidth."""
    for batch in (1, 64, 4096):
        _, bound = decode_step_time(state.model, state.hardware, state.serving,
                                    batch=batch, context_length=32_000)
        assert bound == "memory"


def test_throughput_increases_with_batch(state, workload):
    previous = 0.0
    for b in (1, 2, 4, 8, 16, 32, 64):
        perf = evaluate_request(state, workload, batch_override=b)
        assert perf.throughput_qps > previous
        previous = perf.throughput_qps


def test_per_query_accelerator_seconds_fall_with_batch(state, workload):
    low = evaluate_request(state, workload, batch_override=1).accelerator_seconds
    high = evaluate_request(state, workload, batch_override=64).accelerator_seconds
    assert high < low / 5      # batching must give a large, not marginal, win


def test_time_per_output_token_degrades_with_batch(state, workload):
    """Throughput and latency trade against each other; if both improved
    together the SLO solver would have nothing to solve."""
    low = evaluate_request(state, workload, batch_override=1).tpot
    high = evaluate_request(state, workload, batch_override=256).tpot
    assert high > low


def test_quantisation_lowers_step_time(state):
    fast, _ = decode_step_time(state.model.with_precision(0.5), state.hardware,
                               state.serving, batch=8, context_length=1000)
    slow, _ = decode_step_time(state.model, state.hardware, state.serving,
                               batch=8, context_length=1000)
    assert fast < slow


# -- SLO solver -------------------------------------------------------------

def test_slo_solver_returns_feasible_batch(state, workload):
    slo = SLO(ttft_seconds=1.0, tpot_seconds=0.05)
    perf = solve_batch_for_slo(state, workload, slo)
    assert perf.slo_met
    assert perf.tpot <= slo.tpot_seconds + 1e-9
    assert perf.ttft <= slo.ttft_seconds + 1e-9


def test_slo_solver_finds_the_boundary(state, workload):
    """Just above the solved batch the SLO must fail, or the solver
    stopped early and is leaving throughput on the table."""
    slo = SLO(ttft_seconds=1.0, tpot_seconds=0.02)
    perf = solve_batch_for_slo(state, workload, slo)
    if perf.batch < state.serving.max_batch - 1:
        beyond = evaluate_request(state, workload, slo,
                                  batch_override=perf.batch * 1.25)
        assert not beyond.slo_met


def test_impossible_slo_reported_not_silently_relaxed(state, workload):
    perf = solve_batch_for_slo(state, workload,
                              SLO(ttft_seconds=1e-9, tpot_seconds=1e-9))
    assert not perf.slo_met


def test_disabled_slo_uses_scheduler_cap(state, workload):
    perf = evaluate_request(state, workload, SLO(ttft_seconds=1e-9,
                                                 tpot_seconds=1e-9,
                                                 enforce=False))
    assert perf.slo_met


# -- queueing ---------------------------------------------------------------

def test_ttft_stays_plausible_at_large_batch(state, workload):
    """A naive model charges the whole batch's prefill to one request and
    reports absurd time-to-first-token. Guard against regressing to it."""
    perf = evaluate_request(state, workload, batch_override=256)
    assert perf.ttft < 10.0


def test_ttft_grows_with_prefill_pressure(state):
    short = evaluate_request(state, WorkloadClass("s", 1.0, 200, 400),
                             batch_override=32).ttft
    long = evaluate_request(state, WorkloadClass("l", 1.0, 8000, 400),
                            batch_override=32).ttft
    assert long > short


# -- speculative decoding ---------------------------------------------------

def test_speculative_decoding_helps_more_at_small_batch(state, workload):
    """The central regime-dependence claim: a fixed multiplier cannot be
    right at both ends of the batch range."""
    spec = ServingConfig(n_accelerators=1, max_batch=128,
                         speculative_gamma=4.0, speculative_acceptance=0.72,
                         draft_param_ratio=0.03)
    spec_state = DeploymentState(state.model, state.hardware, spec)

    def ratio(batch: float) -> float:
        base = evaluate_request(state, workload,
                                batch_override=batch).accelerator_seconds
        fast = evaluate_request(spec_state, workload,
                                batch_override=batch).accelerator_seconds
        return fast / base

    assert ratio(1) < ratio(128)
