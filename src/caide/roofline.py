"""Roofline performance model for autoregressive transformer serving.

The model separates the two phases of a request because they are bound by
different resources, and conflating them is the most common source of
error in back-of-envelope cost estimates:

* **Prefill** processes all input tokens in one pass. It is *compute
  bound*: time scales with ``2 * N_active * T_in`` FLOPs plus a quadratic
  attention term, divided by achievable FLOP/s.

* **Decode** emits one token at a time. It is *memory-bandwidth bound*:
  every step must stream the weights out of HBM regardless of how many
  sequences are in flight. Batching therefore amortises the weight read
  across sequences, which is why decode cost per query falls steeply with
  batch size until the arithmetic roofline takes over.

A pure ``2N`` FLOPs-per-token model -- the approximation used in most
cost analyses -- predicts that a batch of 1 and a batch of 128 cost the
same per token. Measured systems disagree by more than an order of
magnitude. Reproducing that gap is the reason this module exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .specs import (
    SLO,
    DeploymentState,
    HardwareSpec,
    ModelSpec,
    ServingConfig,
    WorkloadClass,
)

__all__ = [
    "PhasePerformance",
    "capacity_batch",
    "prefill_flops",
    "decode_step_time",
    "evaluate_request",
    "solve_batch_for_slo",
    "uniform_routing_imbalance",
]


@dataclass(frozen=True)
class PhasePerformance:
    """Resolved performance of one workload class on one deployment."""

    batch: float
    ttft: float                    # seconds to first token
    tpot: float                    # seconds per output token
    latency: float                 # end-to-end seconds for one request
    throughput_qps: float          # requests/second for the whole replica
    prefill_seconds: float
    decode_seconds: float
    decode_bound_by: str           # "memory" or "compute"
    kv_bytes: float
    memory_headroom: float         # bytes free for KV after weights
    slo_met: bool
    accelerator_seconds: float     # accelerator-seconds consumed per request
    context_overflow: bool = False  # request exceeds the model's context window
    batch_truncated: bool = False   # requested batch exceeded KV-cache capacity

    @property
    def feasible(self) -> bool:
        """False when the configuration cannot serve this request at all."""
        return (not self.context_overflow
                and self.decode_bound_by != "infeasible"
                and math.isfinite(self.latency))

    def as_dict(self) -> dict:
        return {
            "batch": self.batch,
            "ttft_s": self.ttft,
            "tpot_s": self.tpot,
            "latency_s": self.latency,
            "throughput_qps": self.throughput_qps,
            "decode_bound_by": self.decode_bound_by,
            "accelerator_seconds": self.accelerator_seconds,
            "slo_met": self.slo_met,
            "context_overflow": self.context_overflow,
            "batch_truncated": self.batch_truncated,
            "feasible": self.feasible,
        }


def _achievable_flops(hw: HardwareSpec, cfg: ServingConfig, mfu: float) -> float:
    """Aggregate FLOP/s of one replica after utilisation losses.

    No tensor-parallel derate is applied here. Under tensor parallelism
    each accelerator computes its own shard, so the aggregate arithmetic
    rate really is ``N`` times one device's. What tensor parallelism costs
    is communication, and communication is charged where it happens, in
    :func:`_collective_seconds`.
    """
    return hw.effective_flops(cfg.precision) * cfg.n_accelerators * mfu


def _achievable_bandwidth(hw: HardwareSpec, cfg: ServingConfig) -> float:
    """Aggregate HBM bandwidth of one replica.

    Also undermined, in versions up to 5.0, by a tensor-parallel derate.
    Each accelerator streams its own weight shard out of its own HBM in
    parallel with the others; there is no bandwidth lost to sharding. The
    derate charged a communication cost in proportion to weight bytes,
    which is the wrong quantity: for a 405B model at TP=8 it billed 5.75 ms
    per step against a true all-reduce cost of 0.59 ms.
    """
    return hw.memory_bandwidth * cfg.n_accelerators * cfg.mbu_decode


def _collective_seconds(model: ModelSpec, hw: HardwareSpec,
                        cfg: ServingConfig, tokens: float) -> float:
    """Time spent in tensor-parallel all-reduces for ``tokens`` of activation.

    A tensor-parallel transformer layer synchronises twice, once after
    attention and once after the MLP. Each synchronisation moves one
    activation tensor of ``tokens * d_model`` elements. The traffic scales
    with the number of tokens in flight and the model width, and does not
    scale with the parameter count -- which is precisely why expressing it
    as a fractional derate on weight streaming misplaces it.

    Returns zero for a single accelerator, where there is nothing to
    synchronise. ``tensor_parallel_penalty`` remains the knob for
    interconnect quality: it scales this term, so zero recovers an ideal
    interconnect and larger values model a worse one, the same direction
    the parameter had before.
    """
    if cfg.n_accelerators <= 1:
        return 0.0
    activation_bytes = 2.0 * tokens * model.d_model * model.n_layers * 2.0
    # ring all-reduce moves 2(N-1)/N of the payload
    scale = 2.0 * (cfg.n_accelerators - 1) / cfg.n_accelerators
    ideal = activation_bytes * scale / hw.interconnect_bandwidth
    return ideal * (1.0 + cfg.tensor_parallel_penalty * 10.0)


def _draft_weight_bytes(model: ModelSpec, cfg: ServingConfig) -> float:
    """HBM the speculative draft model occupies alongside the target.

    Versions up to 6.0 charged the draft in time -- its weight stream and
    its arithmetic appear in every decode step -- while granting it the
    memory for free: :func:`capacity_batch` subtracted only the target's
    weights, so the KV headroom was computed as if the draft were not
    resident. A mechanism charged in one ledger and not in the other is
    the same consistency defect the decode-attention omission was, at
    about 3% of the weight footprint for the catalogue's default ratio.

    The draft's own KV cache remains out of scope and is documented as
    such: its size depends on the draft's layer count and head geometry,
    which a parameter ratio does not determine. For the catalogue's
    default ratio of 0.03 the omitted traffic is under 2% of the step in
    every regime the paper reports; a user serving a disproportionately
    deep draft should model it as a distinct deployment.
    """
    if cfg.speculative_gamma <= 0 or cfg.draft_param_ratio <= 0:
        return 0.0
    return model.active_params * cfg.draft_param_ratio * model.bytes_per_param


def uniform_routing_imbalance(batch: float, n_experts: int,
                              experts_per_token: int) -> float:
    """Mean-to-peak expert load for a *perfectly uniform* router.

    Even a router with no learned preference produces a straggler: with
    ``batch * k`` token-expert assignments spread over ``E`` experts, the
    busiest expert exceeds the mean by counting fluctuation alone. The
    first-order extreme-value approximation for the maximum of ``E``
    near-Poisson loads with mean ``m = batch * k / E`` is
    ``m + sqrt(2 * m * ln E)``, and the returned ratio is ``m`` over that
    peak, floored so the peak is never below one whole token.

    This is a *ceiling* on achievable balance, not a measurement: a real
    router is at best uniform and generally worse, so a deployment's
    ``expert_imbalance`` should be at or below this value, never above.
    Shipped because v6.0 delivered the parameter with no basis to set it
    -- the same dangling state ``framework_overhead_per_step`` spent one
    release in. At batch 256 over 160 experts with k=6 the formula gives
    a peak near twice the mean, matching Monte Carlo to a few percent.
    """
    if n_experts <= 1:
        return 1.0
    assignments = max(batch, 1.0) * max(experts_per_token, 1)
    mean = assignments / n_experts
    peak = max(mean + math.sqrt(2.0 * mean * math.log(n_experts)), 1.0)
    return min(mean / peak, 1.0) if peak > 0 else 1.0


def capacity_batch(model: ModelSpec, hw: HardwareSpec, cfg: ServingConfig,
                   avg_sequence: float) -> float:
    """Largest batch the KV cache can hold, ignoring the scheduler cap.

    Returns 0.0 when the weights do not fit at all -- the draft model's
    weights included, when speculation is configured -- which the caller
    must treat as an infeasible configuration rather than as zero cost.
    """
    total_memory = hw.memory_bytes * cfg.n_accelerators * cfg.memory_utilisation
    headroom = total_memory - model.weight_bytes - _draft_weight_bytes(model, cfg)
    if headroom <= 0:
        return 0.0
    per_sequence = model.kv_bytes_per_token * max(avg_sequence, 1.0)
    if per_sequence <= 0:
        return float(cfg.max_batch)
    return headroom / per_sequence


def prefill_flops(model: ModelSpec, tokens_in: float) -> float:
    """FLOPs for one prefill pass, including causal attention.

    The linear term is the standard ``2 * N_active * T``. The quadratic
    term ``2 * L * T^2 * d_model`` covers the QK^T and AV matmuls with a
    causal mask; it is negligible for short prompts and dominant for long
    ones, which is exactly the regime where retrieval-augmented pipelines
    operate.
    """
    linear = 2.0 * model.active_params * tokens_in
    attention = 2.0 * model.n_layers * (tokens_in ** 2) * model.d_model
    return linear + attention


def decode_step_time(model: ModelSpec, hw: HardwareSpec, cfg: ServingConfig,
                     batch: float, context_length: float) -> tuple:
    """Seconds for one decode step at the given batch, plus the binding resource.

    Three costs contend:
      1. streaming weights out of HBM (independent of batch),
      2. streaming the KV cache (linear in batch and context),
      3. the arithmetic itself (linear in batch, and in the tokens
         verified per step when speculation is on).
    Memory traffic (1 + 2) and compute (3) overlap on real hardware, so the
    step time is their maximum rather than their sum.
    """
    bandwidth = _achievable_bandwidth(hw, cfg)

    # A verification step under speculative decoding is not a one-token
    # step. The draft proposes ``gamma`` tokens and the target scores all
    # of them plus one bonus token in a single forward pass -- amortising
    # the weight stream over ``gamma + 1`` tokens is the entire mechanism.
    # The memory ledger always reflected that (weights stream once); the
    # arithmetic ledger, up to v6.0, priced one token. The expected-tokens
    # denominator in :func:`_speculative_speedup` assumed gamma + 1 tokens
    # were verified per step while the step price contained the arithmetic
    # of one: the same mechanism on the two sides of a division, modelled
    # on one side only. Invisible wherever memory binds; decisive at large
    # batch, where verification arithmetic is what saturates first.
    speculating = cfg.speculative_gamma > 0 and cfg.draft_param_ratio > 0
    verified_tokens = cfg.speculative_gamma + 1.0 if speculating else 1.0

    # FLOP utilisation follows the tokens in the matmul, not the sequences
    # in the batch: verifying gamma + 1 positions per sequence is a GEMM
    # with ``batch * (gamma + 1)`` rows, and pricing it at the utilisation
    # of a ``batch``-row GEMM would overstate the cost of exactly the
    # regime speculation is for.
    flops = _achievable_flops(hw, cfg, _decode_mfu(cfg, batch * verified_tokens))

    # The router sees every token in the forward pass, not every sequence.
    # A verification step submits ``batch * (gamma + 1)`` tokens, each
    # routed independently, so the expected fraction of experts touched is
    # taken at that count. Version 7.0 corrected the arithmetic ledger and
    # the collective to the verified-token count but left this one call at
    # ``batch``, which handed a mixture-of-experts model the dense model's
    # amortisation for free: for a dense model streaming the weights once
    # over gamma + 1 tokens is the whole benefit, while for MoE more
    # tokens reach more experts and the weight stream *grows*. At batch 1
    # with gamma = 4 the omission understated expert traffic by up to
    # 2.8x -- in precisely the memory-bound regime speculation is for.
    routed_tokens = batch * verified_tokens
    # Streamed, not resident: the input embedding table is gathered a row
    # at a time and does not appear in the weight stream, while the head
    # does and at its own precision. See ModelSpec.decode_weight_bytes.
    weight_bytes = model.decode_weight_bytes(routed_tokens)
    kv_bytes = model.kv_bytes_per_token * context_length * batch
    memory_time = (weight_bytes + kv_bytes) / bandwidth

    # Two arithmetic terms, not one. The GEMM term streams the weights
    # through the tensor cores once per token per sequence in the batch.
    # The attention term scores each new token against every cached key
    # and then weights every cached value: per layer and per sequence that
    # is ``2 * d_model * context`` for QK^T plus the same again for AV.
    #
    # Omitting the second term -- as versions up to 5.0 did -- is harmless
    # wherever the KV cache is being streamed anyway, because the roofline
    # maximum selects the memory term and hides it. It stops being
    # harmless at small batch, where decode FLOP utilisation is a few
    # percent and the machine's effective balance falls below attention's
    # arithmetic intensity of ``n_heads / n_kv_heads`` FLOPs per KV byte.
    # There the term decides which resource binds, and the same physics is
    # already modelled one phase earlier in :func:`prefill_flops`.
    gemm_flops = 2.0 * model.active_params * batch * verified_tokens
    attention_flops = (4.0 * model.n_layers * model.d_model * context_length
                       * batch * verified_tokens)
    compute_time = (gemm_flops + attention_flops) / flops

    if speculating:
        draft_params = model.active_params * cfg.draft_param_ratio
        draft_bytes = draft_params * model.bytes_per_param
        draft_steps = cfg.speculative_gamma
        # The draft's own KV cache. Zero by default -- a parameter ratio
        # does not determine a draft's layer count or head geometry, so
        # the honest default is to declare the term rather than guess it.
        # Set ``draft_kv_ratio`` to price it; the v8 regression test
        # evaluates it at the parameter ratio to bound what the default
        # omits.
        if cfg.draft_kv_ratio > 0:
            memory_time += (draft_steps * cfg.draft_kv_ratio
                            * model.kv_bytes_per_token * context_length
                            * batch) / bandwidth
        # The draft decodes one token at a time at the batch's own row
        # count, so its arithmetic is priced at the batch's utilisation,
        # not the verification GEMM's.
        draft_flops = _achievable_flops(hw, cfg, _decode_mfu(cfg, batch))
        memory_time += draft_steps * draft_bytes / bandwidth
        compute_time += draft_steps * 2.0 * draft_params * batch / draft_flops

    # Under expert parallelism a decode step is not finished when the
    # average expert is finished, it is finished when the busiest one is.
    # ``expert_imbalance`` is the ratio of mean to peak expert load, so
    # dividing by it stretches the *expert* work to the straggler. Only
    # the expert work: the KV stream and the attention arithmetic are not
    # sharded by expert and do not wait per-expert, which is why the v6.0
    # form -- a multiplier on the whole of both terms -- levied the
    # surcharge on the wrong base and grew with context length, a quantity
    # expert skew knows nothing about. ``uniform_routing_imbalance`` gives
    # the counting-statistics ceiling for this parameter; 1.0 remains the
    # default and describes replicated or tensor-sharded experts, which is
    # the configuration the rest of CAIDE prices.
    if model.is_moe and cfg.expert_imbalance < 1.0:
        stretch = 1.0 / cfg.expert_imbalance - 1.0
        shared_bytes = (
            max(model.moe_shared_params - model._high_precision_params, 0.0)
            * model.bytes_per_param
            + model.lm_head_params * model.head_bytes)
        expert_traffic = max(weight_bytes - shared_bytes, 0.0)
        memory_time += (expert_traffic / bandwidth) * stretch
        expert_gemm = 2.0 * max(model.active_params - model.moe_shared_params,
                                0.0) * batch * verified_tokens
        compute_time += (expert_gemm / flops) * stretch

    step = max(memory_time, compute_time)
    bound = "memory" if memory_time >= compute_time else "compute"

    # One decode step advances every sequence in the batch by one token --
    # or verifies ``gamma + 1`` candidate tokens per sequence -- so the
    # activation tensor being synchronised holds that many tokens. The
    # draft is assumed replicated rather than sharded, the standard
    # deployment for a model two orders of magnitude smaller.
    collective = _collective_seconds(model, hw, cfg, batch * verified_tokens)
    if collective > 0:
        step += collective
        if collective > max(memory_time, compute_time):
            bound = "interconnect"

    # A roofline models the accelerator. A serving framework also runs a
    # scheduler, an HTTP layer and a detokeniser on the CPU, and published
    # throughput figures include all of it. vLLM's own profiling of an 8B
    # model on one accelerator attributed 33% of wall time to the API
    # server and 29% to scheduling, leaving 38% for GPU execution. Leaving
    # this at zero gives the pure hardware roofline, which is the right
    # default for comparing configurations and the wrong one for
    # predicting what a benchmark will report.
    # The overhead has two parts, and a single constant cannot express
    # both. Scheduling and the API server run once per step whatever the
    # concurrency; detokenisation and per-request bookkeeping run once per
    # live sequence. Fitting the constant form to the two published
    # 8B-on-A100 points requires 15.0 ms at batch 1 and 30.8 ms at batch 8
    # -- a factor of two apart, which is the signature of a term the
    # constant form is missing rather than of noise.
    overhead = (cfg.framework_overhead_per_step
                + cfg.framework_overhead_per_sequence * batch)
    if overhead > 0:
        step += overhead
        if overhead > max(memory_time, compute_time):
            bound = "framework"
    return step, bound


def _decode_mfu(cfg: ServingConfig, batch: float) -> float:
    """Achievable decode FLOP utilisation, which rises with batch size.

    A decode step at batch 1 is a matrix-vector product: arithmetic
    intensity is one, tensor cores idle, and utilisation is a few percent.
    At batch 256 the same step is a proper matrix-matrix product and
    approaches the utilisation of prefill. Modelling this as a constant --
    as CAIDE 2.0 did -- places the memory-to-compute transition at an
    artificially low batch, which understates how much quantisation still
    helps at production batch sizes.

    The interpolation ``B / (B + B_half)`` is the standard saturating form:
    utilisation is half of prefill's at ``B = B_half`` and approaches it
    asymptotically. ``B_half`` defaults to 64, the region where GEMM
    efficiency turns over on current tensor-core hardware, and is exposed
    as ``ServingConfig.decode_mfu_half_batch`` for calibration.

    At small batch the value barely matters: decode is deeply
    memory-bound there and the roofline maximum selects the memory term
    regardless. It matters at large batch, which is where deployments run.
    """
    b = max(batch, 1.0)
    saturation = b / (b + cfg.decode_mfu_half_batch)
    return max(cfg.mfu_prefill * saturation, 0.005)


def _prefill_queue_factor(prefill_seconds: float, cycle_seconds: float,
                          ceiling: float = 25.0) -> float:
    """Contention multiplier applied to a single request's prefill time.

    A request does not wait for the whole batch to prefill -- modern
    schedulers interleave prefill chunks with decode steps -- but neither
    does it get the accelerator to itself. Treating the prefill stage as
    an M/D/1 queue whose utilisation is the share of the serving cycle
    spent in prefill gives the expected wait ``rho / (2 * (1 - rho))`` in
    units of service time, which is exact in the limits: no contention
    when prefill is a negligible share of the cycle, unbounded as prefill
    saturates the replica.

    The ceiling keeps the return value finite as ``rho -> 1``; callers
    should read a saturated factor as "this configuration is past its
    stable operating point", which the SLO check will then reject.
    """
    if cycle_seconds <= 0:
        return 1.0
    rho = min(max(prefill_seconds / cycle_seconds, 0.0), 0.999)
    if rho >= 0.999:
        return ceiling
    return min(1.0 + rho / (2.0 * (1.0 - rho)), ceiling)


def _speculative_speedup(cfg: ServingConfig) -> float:
    """Expected accepted tokens per verification step.

    With ``gamma`` draft tokens and per-token acceptance ``alpha``, the
    expected number of accepted tokens is the truncated geometric mean
    ``(1 - alpha^(gamma+1)) / (1 - alpha)``. Acceptance of 1.0 degenerates
    to ``gamma + 1``.
    """
    gamma, alpha = cfg.speculative_gamma, cfg.speculative_acceptance
    if gamma <= 0 or alpha <= 0:
        return 1.0
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1.0)) / (1.0 - alpha)


def evaluate_request(state: DeploymentState, workload: WorkloadClass,
                     slo: Optional[SLO] = None,
                     batch_override: Optional[float] = None) -> PhasePerformance:
    """Resolve latency and throughput for one workload class on one deployment."""
    model, hw, cfg = state.model, state.hardware, state.serving

    tokens_in = workload.tokens_in * (1.0 - cfg.prefix_cache_hit)
    tokens_out = workload.tokens_out
    # One derivation: ``WorkloadClass.avg_sequence`` writes the same
    # formula and was never called, which is how two copies of one
    # quantity start to disagree. Prefix caching shortens what is
    # *computed*, not what is resident, so the full prompt is used here.
    avg_context = workload.avg_sequence

    cap = capacity_batch(model, hw, cfg, avg_context)
    if cap <= 0:
        return PhasePerformance(
            batch=0.0, ttft=math.inf, tpot=math.inf, latency=math.inf,
            throughput_qps=0.0, prefill_seconds=math.inf, decode_seconds=math.inf,
            decode_bound_by="infeasible", kv_bytes=0.0, memory_headroom=-1.0,
            slo_met=False, accelerator_seconds=math.inf,
            context_overflow=False,
        )

    # A request longer than the model's window is not a slow request, it is
    # an impossible one. Returning a large latency for it -- which the v1.0
    # model did -- gives a plausible-looking number for a configuration that
    # cannot run, and the scenario-level validator is no help to anyone
    # calling this function directly.
    overflow = (workload.tokens_in + tokens_out) > model.max_context

    if batch_override is not None:
        # A requested batch that the KV cache cannot hold is not a slow
        # configuration, it is an impossible one. v3.0 returned a confident
        # throughput figure for it, which is exactly the failure mode the
        # context-overflow check exists to prevent one level up.
        batch = max(min(batch_override, cap), 1.0)
        batch_truncated = batch_override > cap
    else:
        batch = max(min(cap, cfg.max_batch), 1.0)
        batch_truncated = False

    flops_prefill = _achievable_flops(hw, cfg, cfg.mfu_prefill)
    pf_flops = prefill_flops(model, tokens_in)
    prefill_per_request = (pf_flops / flops_prefill
                           + _collective_seconds(model, hw, cfg, tokens_in))

    step, bound = decode_step_time(model, hw, cfg, batch, avg_context)
    steps = tokens_out / _speculative_speedup(cfg)
    decode_seconds = steps * step

    batch_seconds = prefill_per_request * batch + decode_seconds
    throughput = batch / batch_seconds if batch_seconds > 0 else 0.0

    ttft = prefill_per_request * _prefill_queue_factor(
        prefill_per_request * batch, batch_seconds)

    accel_seconds = (cfg.n_accelerators / throughput) if throughput > 0 else math.inf

    slo_met = not overflow
    if slo_met and slo is not None and slo.enforce:
        slo_met = (ttft <= slo.ttft_seconds) and (step <= slo.tpot_seconds)

    return PhasePerformance(
        batch=batch,
        ttft=ttft,
        tpot=step,
        latency=ttft + decode_seconds,
        throughput_qps=throughput,
        prefill_seconds=prefill_per_request * batch,
        decode_seconds=decode_seconds,
        decode_bound_by=bound,
        kv_bytes=model.kv_bytes_per_token * avg_context * batch,
        memory_headroom=(hw.memory_bytes * cfg.n_accelerators * cfg.memory_utilisation
                         - model.weight_bytes - _draft_weight_bytes(model, cfg)),
        slo_met=slo_met,
        accelerator_seconds=accel_seconds,
        context_overflow=overflow,
        batch_truncated=batch_truncated,
    )


def solve_batch_for_slo(state: DeploymentState, workload: WorkloadClass,
                        slo: SLO, tolerance: float = 1e-3) -> PhasePerformance:
    """Largest batch that still satisfies the SLO, found by bisection.

    Throughput rises monotonically with batch while latency degrades
    monotonically, so the feasible set is an interval anchored at batch 1
    and the boundary can be bracketed exactly. Serving at the SLO boundary
    rather than at the scheduler cap is what separates a cost model that
    respects user experience from one that does not.
    """
    model, hw, cfg = state.model, state.hardware, state.serving
    avg_context = workload.avg_sequence
    cap = min(capacity_batch(model, hw, cfg, avg_context), float(cfg.max_batch))

    if cap < 1.0:
        return evaluate_request(state, workload, slo)

    best = evaluate_request(state, workload, slo, batch_override=1.0)
    if not best.slo_met:
        return best                      # even batch 1 violates the SLO

    top = evaluate_request(state, workload, slo, batch_override=cap)
    if top.slo_met:
        return top

    lo, hi = 1.0, cap
    while hi - lo > max(tolerance, lo * tolerance):
        mid = 0.5 * (lo + hi)
        probe = evaluate_request(state, workload, slo, batch_override=mid)
        if probe.slo_met:
            lo, best = mid, probe
        else:
            hi = mid
    return best
