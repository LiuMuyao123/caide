"""Core specification dataclasses for CAIDE.

Every quantity carries explicit SI-derived units so that the roofline,
costing and carbon layers can be composed without unit ambiguity.

Unit conventions
----------------
* parameters            : count (not billions)
* memory / bytes        : bytes
* bandwidth             : bytes / second
* compute               : FLOP / second
* power                 : watts
* energy                : joules internally, kWh at reporting boundaries
* money                 : USD
* time                  : seconds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

__all__ = [
    "ModelSpec",
    "HardwareSpec",
    "ServingConfig",
    "WorkloadClass",
    "SLO",
    "PricingSpec",
    "GridSpec",
    "DeploymentState",
    "IMPLAUSIBLE_ABOVE",
    "implausible",
]


def _positive(name: str, value: float) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}")
    return float(value)


#: Magnitude bounds beyond which an input is accepted but reported as
#: implausible. The distinction matters: a negative token count is
#: impossible and must be rejected, whereas a thousand-fold infrastructure
#: overhead is merely absurd -- refusing it would prevent legitimate
#: sensitivity sweeps, but accepting it silently violates the same design
#: principle that rejects the negative one.
IMPLAUSIBLE_ABOVE = {
    "self_consistency_k": 20,        # more than 20 samples per query
    "infra_overhead": 5.0,           # more than 5x raw accelerator cost
    "review_minutes": 240.0,         # more than four hours per reviewed item
    "baseline_minutes": 480.0,       # more than a working day per item
    "carbon_intensity": 1.5,         # kg CO2e/kWh; worse than lignite
    "pue": 3.0,                      # power usage effectiveness
    "n_accelerators": 64,            # tensor-parallel degree
    "tokens_in": 2_000_000,
    "tokens_out": 200_000,
}


def implausible(name: str, value: float) -> Optional[str]:
    """Return a warning when a value is possible but not credible."""
    ceiling = IMPLAUSIBLE_ABOVE.get(name)
    if ceiling is None or value <= ceiling:
        return None
    return (f"{name}={value:g} exceeds the plausible ceiling of {ceiling:g}. "
            "The value is accepted so that sensitivity sweeps can reach it, "
            "but check it before quoting the result.")


def _fraction(name: str, value: float, *, upper: float = 1.0) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not (0.0 <= value <= upper):
        raise ValueError(f"{name} must lie in [0, {upper}], got {value!r}")
    return float(value)


#: Smallest difference in ``quality_index`` that should be read as an
#: ordering. The index is a declared scale, not a measurement: the
#: catalogue's values are round numbers to two decimals and no procedure
#: exists that would distinguish 0.856 from 0.860. Any admissibility
#: verdict resting on a margin below this is a verdict the index cannot
#: support, and :attr:`TCOResult.marginal_verdicts` names those.
#:
#: The figure was pinned in v17.0 after the carry-forward ledger's oldest
#: overdue item was settled: three of the fourteen floors the shipped
#: scenarios declare sit within 1.2% of an architecture's index, and all
#: three decide a published answer. One of them -- 0.5% -- is the sole
#: reason a whole scenario has no admissible architecture at all.
QUALITY_INDEX_RESOLUTION = 0.05


@dataclass(frozen=True)
class ModelSpec:
    """Architecture of a decoder-only transformer.

    ``n_params_active`` differs from ``n_params_total`` only for
    mixture-of-experts models, where a router selects ``experts_per_token``
    of ``n_experts`` for every token. Memory must hold all experts while
    arithmetic touches only the active subset -- the asymmetry that makes
    MoE serving economics distinct from dense serving economics.
    """

    name: str
    n_params_total: float
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: Optional[int] = None          # None -> multi-head (= n_heads)
    n_params_active: Optional[float] = None   # None -> dense (= total)
    n_experts: int = 1
    experts_per_token: int = 1
    bytes_per_param: float = 2.0              # bf16 by default
    bytes_per_kv_element: float = 2.0
    vocab_size: int = 128_000
    #: Input embedding and output projection share one matrix. Untied is
    #: the common case for the archetypes bundled here.
    tied_embeddings: bool = False
    #: Bytes per parameter for the embedding table and the language-model
    #: head. ``None`` follows ``bytes_per_param``. Production weight
    #: quantisation leaves both at higher precision because quantising the
    #: head degrades output quality out of proportion to the memory it
    #: saves, so the quantisation techniques set this to bf16 and the
    #: bundled multipliers now reflect that.
    head_bytes_per_param: Optional[float] = None
    max_context: int = 32_768
    #: Relative capability on a **ratio scale**: 1.0 is the reference
    #: frontier model, 0.5 means half. The scale matters because the code
    #: does arithmetic on it -- efficiency techniques multiply it by
    #: ``1 + quality_delta`` and compose those on retention, and a
    #: workload's ``quality_floor`` is compared against the product. The
    #: v10 limits table called it ordinal, which would have made all of
    #: that meaningless; the arithmetic is the older of the two claims and
    #: the documentation was the one that was wrong.
    #:
    #: It is an *idealisation*: capability is not one-dimensional, and no
    #: single number orders two models that differ by task. Treat a floor
    #: as a declared admissibility threshold on a declared index, not as a
    #: measurement.
    quality_index: float = 1.0

    def __post_init__(self) -> None:
        _positive("n_params_total", self.n_params_total)
        _positive("n_layers", self.n_layers)
        _positive("d_model", self.d_model)
        _positive("n_heads", self.n_heads)
        _positive("bytes_per_param", self.bytes_per_param)
        _positive("bytes_per_kv_element", self.bytes_per_kv_element)
        if self.n_kv_heads is not None and self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot exceed n_heads")
        if self.n_params_active is not None:
            if self.n_params_active > self.n_params_total:
                raise ValueError("n_params_active cannot exceed n_params_total")
        if self.experts_per_token > self.n_experts:
            raise ValueError("experts_per_token cannot exceed n_experts")
        # A router that activates k of E experts cannot make the active
        # parameter count fall below k/E of the total, because that bound
        # is attained only when *every* parameter lives in an expert and
        # nothing is shared. A spec below it describes no architecture.
        # Version 5.0 accepted such specs and let a clamp inside
        # :meth:`expert_bytes_touched` absorb them, which returned a
        # confident weight-traffic figure that disagreed with the declared
        # active count by up to 1.65x -- the silent-unit failure mode the
        # v5.0 audit had just removed from the calibration layer.
        if self.n_experts > 1 and self.n_params_active is not None:
            floor = self.n_params_total * self.experts_per_token / self.n_experts
            if self.n_params_active < floor:
                raise ValueError(
                    f"n_params_active={self.n_params_active:g} is below the "
                    f"arithmetic floor {floor:g} for a router activating "
                    f"{self.experts_per_token} of {self.n_experts} experts. "
                    "The floor is reached when every parameter sits in an "
                    "expert and none is shared; below it the geometry "
                    "describes no architecture."
                )

    # -- derived geometry ------------------------------------------------

    @property
    def kv_heads(self) -> int:
        return int(self.n_kv_heads if self.n_kv_heads is not None else self.n_heads)

    @property
    def d_head(self) -> int:
        return int(self.d_model // self.n_heads)

    @property
    def active_params(self) -> float:
        return float(self.n_params_active if self.n_params_active is not None
                     else self.n_params_total)

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1 and self.active_params < self.n_params_total

    # -- embedding geometry -------------------------------------------
    # ``vocab_size`` was declared in v1.0 and read by nothing until v10.0,
    # the fifth dangling parameter this project has found in itself. It is
    # the field that decides how large the two matrices below are, and
    # therefore how much of a model a quantisation technique can actually
    # reach.

    @property
    def embedding_params(self) -> float:
        """Input embedding table: ``V x d_model``, read by gather."""
        return float(self.vocab_size * self.d_model)

    @property
    def lm_head_params(self) -> float:
        """Output projection, zero when it shares the embedding matrix."""
        return 0.0 if self.tied_embeddings else self.embedding_params

    @property
    def head_bytes(self) -> float:
        """Bytes per parameter for the two matrices quantisation skips."""
        return (self.bytes_per_param if self.head_bytes_per_param is None
                else self.head_bytes_per_param)

    @property
    def _high_precision_params(self) -> float:
        """Parameters a weight-quantisation scheme leaves alone.

        Capped at the block they sit in, so that the parameter count is
        preserved exactly whatever the geometry: the point of splitting
        the precision is to move bytes between blocks, never to invent
        or lose parameters. A spec whose vocabulary would exceed its
        shared block is unusual rather than impossible, and the cap makes
        the batch-of-one identity hold for it too.
        """
        return self.embedding_params + self.lm_head_params

    @property
    def _streamed_head_params(self) -> float:
        """The matrix the logit projection streams every decode step.

        Tied or untied, exactly one ``V x d_model`` matrix is multiplied
        against the final hidden state. The *input* table is not: a decode
        step gathers ``batch`` rows from it, which is nothing next to a
        weight stream, and counting the whole table was overstating decode
        traffic by 6.5% on the smallest bundled archetype.
        """
        return self.embedding_params

    @property
    def weight_bytes(self) -> float:
        """Resident weight memory, all experts included.

        Everything must be in memory, but not everything is at the same
        precision: the embedding and the head sit at ``head_bytes``.
        """
        hp = min(self._high_precision_params, self.n_params_total)
        return (self.n_params_total - hp) * self.bytes_per_param + \
            hp * self.head_bytes

    def decode_weight_bytes(self, routed_tokens: float = 1.0) -> float:
        """Weight bytes *streamed* by one decode step.

        Differs from :attr:`weight_bytes` in two ways, both of which the
        model got wrong until v10.0 and which pointed in opposite
        directions -- which is why the total looked reasonable. The input
        embedding is gathered rather than streamed, so it comes out; the
        head stays in, at its own precision rather than the quantised one.
        """
        streamed = self.expert_bytes_touched(routed_tokens)
        # remove the input table, which a decode step does not stream
        gathered = min(self._streamed_head_params,
                       self.moe_shared_params if self.is_moe
                       else self.n_params_total)
        return max(streamed - gathered * self.head_bytes, 0.0)

    @property
    def kv_bytes_per_token(self) -> float:
        """K and V for every layer, GQA-aware."""
        return (2.0 * self.n_layers * self.kv_heads * self.d_head
                * self.bytes_per_kv_element)

    @property
    def moe_shared_params(self) -> float:
        """Parameters outside the experts -- attention, embeddings, router.

        Solved so that a batch of one reads exactly ``active_params``: the
        one token touches the shared parameters plus ``k`` experts' worth
        of the expert pool. The constructor's arithmetic floor guarantees
        the solution is non-negative. For a dense model every parameter is
        shared. Exposed since v7.0 because the expert-parallel straggler
        term must scope its surcharge to the expert share, and deriving
        the split in two places is how two places end up disagreeing.
        """
        if not self.is_moe:
            return float(self.n_params_total)
        return self.n_params_total - (self.n_params_total - self.active_params) \
            * (self.n_experts / max(self.n_experts - self.experts_per_token, 1))

    def expert_bytes_touched(self, batch: float) -> float:
        """Weight bytes actually read for one decode step at a given batch.

        For a dense model this is the full weight footprint. For MoE the
        router spreads tokens over experts, so the expected fraction of
        experts touched by ``batch`` independent tokens is
        ``1 - (1 - k/E)**batch`` -- small batches read few experts, large
        batches converge on reading all of them. This is why MoE loses its
        bandwidth advantage exactly when batching would otherwise help.
        """
        if not self.is_moe:
            return self.weight_bytes
        shared = self.moe_shared_params
        # ``shared`` is solved so that a batch of one reads exactly
        # ``active_params``. The constructor's arithmetic floor guarantees
        # it is non-negative, so the 10%-of-active clamp that stood here
        # until v6.0 can only fire on geometries that are now rejected --
        # and where it did fire it broke the very identity this expression
        # exists to satisfy.
        expert_pool = max(self.n_params_total - shared, 0.0)
        p_touch = 1.0 - (1.0 - self.experts_per_token / self.n_experts) ** max(batch, 1.0)
        # The embedding and the head are shared parameters and sit at
        # their own precision; only the rest of the shared block follows
        # ``bytes_per_param``.
        hp = min(self._high_precision_params, shared)
        return ((shared - hp) * self.bytes_per_param
                + hp * self.head_bytes
                + expert_pool * p_touch * self.bytes_per_param)

    def with_precision(self, bytes_per_param: float,
                       bytes_per_kv_element: Optional[float] = None,
                       head_bytes_per_param: Optional[float] = None
                       ) -> "ModelSpec":
        """Restate the model at a different weight precision.

        ``head_bytes_per_param`` is what the embedding and the language-
        model head are stored at. Passing it is how a caller says "this
        quantisation scheme does not touch those two matrices", which is
        what GPTQ, AWQ and the llama.cpp k-quants all do in practice.
        Leaving it ``None`` quantises everything uniformly, which is what
        versions up to 9.0 did unconditionally.
        """
        return replace(
            self,
            bytes_per_param=bytes_per_param,
            bytes_per_kv_element=(bytes_per_kv_element
                                  if bytes_per_kv_element is not None
                                  else self.bytes_per_kv_element),
            head_bytes_per_param=head_bytes_per_param,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_params_total": self.n_params_total,
            "n_params_active": self.active_params,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.kv_heads,
            "bytes_per_param": self.bytes_per_param,
            "is_moe": self.is_moe,
            "weight_gib": self.weight_bytes / 2**30,
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }


@dataclass(frozen=True)
class HardwareSpec:
    """One accelerator type, plus the node it lives in."""

    name: str
    peak_flops: float                 # dense FLOP/s at the serving precision
    memory_bytes: float               # HBM per accelerator
    memory_bandwidth: float           # bytes/s per accelerator
    power_watts: float                # board power per accelerator, under load
    hourly_cost: float                # USD per accelerator-hour (rental or amortised)
    interconnect_bandwidth: float = 4.5e11   # bytes/s, for tensor-parallel penalty
    idle_power_watts: Optional[float] = None   # board power while provisioned but
                                               # not serving; None -> 15% of load
                                               # power, a documented estimate
    low_precision_speedup: Dict[str, float] = field(
        default_factory=lambda: {"bf16": 1.0, "fp8": 2.0, "int8": 2.0, "int4": 2.0}
    )

    def __post_init__(self) -> None:
        _positive("peak_flops", self.peak_flops)
        _positive("memory_bytes", self.memory_bytes)
        _positive("memory_bandwidth", self.memory_bandwidth)
        _positive("power_watts", self.power_watts)
        if self.hourly_cost < 0:
            raise ValueError("hourly_cost must be non-negative")
        if self.idle_power_watts is not None:
            if not (0.0 <= self.idle_power_watts <= self.power_watts):
                raise ValueError(
                    "idle_power_watts must lie in [0, power_watts]: an idle "
                    "board cannot draw more than a loaded one"
                )

    @property
    def resolved_idle_power(self) -> float:
        """Idle board power, falling back to 15% of load power.

        The fallback is an estimate, not a measurement: published idle
        figures for datacentre accelerators cluster between 10% and 20%
        of board power. Versions up to 6.0 implicitly used 100% -- every
        provisioned second was charged at full load power -- which is the
        one value known to be wrong.
        """
        if self.idle_power_watts is not None:
            return self.idle_power_watts
        return 0.15 * self.power_watts

    def effective_flops(self, precision: str = "bf16") -> float:
        return self.peak_flops * self.low_precision_speedup.get(precision, 1.0)


@dataclass(frozen=True)
class ServingConfig:
    """Everything about *how* the model is served, as opposed to what it is.

    Utilisation is deliberately split into two independent factors,
    because conflating them lets a serving optimisation silently overwrite
    a statement about demand:

    ``demand_duty_cycle``
        The fraction of the year during which requests actually arrive.
        A property of the workload -- deadline-driven academic traffic sits
        near 0.4 whatever software serves it. No efficiency technique may
        change this, because no serving stack creates demand.

    ``scheduler_efficiency``
        The fraction of accelerator time spent on useful work *while*
        requests are arriving. Static batching idles the device waiting for
        the longest sequence in a batch; continuous batching does not. This
        is what serving optimisations improve.

    The product is what costing divides by, and it can never exceed the
    demand duty cycle -- which is the invariant the earlier single-parameter
    design violated.
    """

    n_accelerators: int = 1           # tensor/pipeline parallel degree per replica
    min_replicas: int = 1             # N+1 redundancy floor; capacity is integral
    demand_duty_cycle: float = 0.65   # share of the year with live traffic
    scheduler_efficiency: float = 0.45   # useful-work share while traffic is live
    memory_utilisation: float = 0.90  # fraction of HBM usable after fragmentation
    mfu_prefill: float = 0.45         # model-FLOPs utilisation, compute-bound phase
    mbu_decode: float = 0.70          # memory-bandwidth utilisation, decode phase
    max_batch: int = 256              # scheduler cap on concurrent sequences
    decode_mfu_half_batch: float = 64.0   # batch at which decode reaches half of
                                          # prefill FLOP utilisation
    framework_overhead_per_step: float = 0.0   # seconds of non-GPU time per
                                               # decode step, independent of
                                               # how many sequences are live
    framework_overhead_per_sequence: float = 0.0   # additional seconds per
                                                   # decode step per live
                                                   # sequence (detokenisation,
                                                   # per-request bookkeeping)
    expert_imbalance: float = 1.0     # mean/peak expert load under expert
                                      # parallelism; 1.0 = perfectly balanced
    infra_overhead: float = 1.35      # orchestration, networking, redundancy, storage
    precision: str = "bf16"
    tensor_parallel_penalty: float = 0.03   # per extra accelerator, communication loss
    prefill_chunking: bool = False
    speculative_gamma: float = 0.0    # draft tokens per verification step (0 = off)
    speculative_acceptance: float = 0.0
    draft_param_ratio: float = 0.0    # draft model size / target model size
    # Draft KV traffic and residency, as a fraction of the target's per
    # token. Default 0.0 keeps the v6/v7 scope boundary -- the draft's KV
    # out of scope -- but makes it a *modelled* zero rather than an
    # undocumented omission: a user serving a disproportionately deep
    # draft sets this and gets it priced. The v8 audit turned the old
    # "under 2% of the step" prose claim into a regression test by
    # evaluating this term at the parameter ratio.
    draft_kv_ratio: float = 0.0
    semantic_cache_hit: float = 0.0   # fraction of queries served without inference
    prefix_cache_hit: float = 0.0     # fraction of *prefill tokens* reused
    reserved_discount: float = 1.0    # 1.0 = on-demand, <1 = committed-use discount

    def __post_init__(self) -> None:
        _positive("n_accelerators", self.n_accelerators)
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be >= 1")
        _fraction("memory_utilisation", self.memory_utilisation)
        _fraction("mfu_prefill", self.mfu_prefill)
        _fraction("mbu_decode", self.mbu_decode)
        _fraction("demand_duty_cycle", self.demand_duty_cycle)
        _fraction("scheduler_efficiency", self.scheduler_efficiency)
        _fraction("semantic_cache_hit", self.semantic_cache_hit)
        _fraction("prefix_cache_hit", self.prefix_cache_hit)
        _fraction("speculative_acceptance", self.speculative_acceptance)
        _fraction("draft_kv_ratio", self.draft_kv_ratio)
        if self.demand_duty_cycle <= 0:
            raise ValueError("demand_duty_cycle must be > 0")
        if self.scheduler_efficiency <= 0:
            raise ValueError("scheduler_efficiency must be > 0")
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if self.infra_overhead < 1.0:
            raise ValueError("infra_overhead must be >= 1.0")
        if self.decode_mfu_half_batch <= 0:
            raise ValueError("decode_mfu_half_batch must be > 0")
        if self.framework_overhead_per_step < 0:
            raise ValueError("framework_overhead_per_step must be >= 0")
        if self.framework_overhead_per_sequence < 0:
            raise ValueError("framework_overhead_per_sequence must be >= 0")
        if not (0.0 < self.expert_imbalance <= 1.0):
            raise ValueError(
                "expert_imbalance must lie in (0, 1]; it is the ratio of "
                "mean to peak expert load, so 1.0 is a perfectly balanced "
                "router and smaller values are more skewed"
            )

    @property
    def effective_utilisation(self) -> float:
        """Share of paid accelerator time that does useful work.

        Bounded above by ``demand_duty_cycle``: a perfect scheduler cannot
        make a replica busy when nobody is asking it anything.
        """
        return self.demand_duty_cycle * self.scheduler_efficiency

    @property
    def target_utilisation(self) -> float:
        """Deprecated alias for :attr:`effective_utilisation`.

        Retained so that code written against v1.0 keeps reading the right
        number. It is read-only by design -- assigning a single utilisation
        figure is exactly the ambiguity this split removed.
        """
        return self.effective_utilisation

    # ``parallel_efficiency`` lived here until v16.0: a multiplicative
    # derate, 1 - penalty x (n - 1), claiming a 21% loss at eight
    # accelerators regardless of batch size. The v6.0 audit replaced that
    # model with a per-layer all-reduce whose cost scales with batch and
    # model width, and measured the interconnect share of a decode step at
    # 0.02% to 1.43% over the same range. The old property was never
    # deleted and never called, so for ten releases the package exposed,
    # as public API, a model of its own that it had already rejected --
    # and a batch-independent constant is precisely what this project
    # exists to argue against. Removed rather than corrected: a second
    # derivation of a quantity the roofline already computes has no
    # right answer to be corrected to.


@dataclass(frozen=True)
class SLO:
    """Service-level objectives that bound how far batching may be pushed."""

    ttft_seconds: float = 2.0         # time to first token, p50
    tpot_seconds: float = 0.05        # time per output token (20 tok/s)
    enforce: bool = True

    def __post_init__(self) -> None:
        _positive("ttft_seconds", self.ttft_seconds)
        _positive("tpot_seconds", self.tpot_seconds)


@dataclass(frozen=True)
class WorkloadClass:
    """One recognisable kind of request, with its share of total traffic.

    ``quality_floor`` is the minimum model quality index that may serve this
    class; the router uses it to keep cheap models away from tasks that
    need capability. ``review_rate`` and ``review_minutes`` capture human
    oversight, which for high-assurance deployments dominates compute.

    ``baseline_minutes`` is the human time the same task consumed *before*
    the system existed. It matters because review time is frequently not
    a new cost at all: a clinician editing a generated discharge summary
    is not doing extra work, they are doing less of the work they already
    did. Leaving this at zero -- the default -- makes the analysis report
    gross review cost, which is the right choice for genuinely novel
    services and the wrong one for automation of existing workflows.
    """

    name: str
    share: float                      # fraction of total query volume
    tokens_in: float
    tokens_out: float
    quality_floor: float = 0.0
    review_rate: float = 0.0          # fraction of outputs seen by a human
    review_minutes: float = 0.0       # minutes per reviewed output
    baseline_minutes: float = 0.0     # minutes the task took before the system
    self_consistency_k: int = 1       # repeated sampling for reliability
    cacheable: bool = True
    latency_sensitive: bool = True

    def __post_init__(self) -> None:
        _fraction("share", self.share)
        _positive("tokens_in", self.tokens_in)
        _positive("tokens_out", self.tokens_out)
        _fraction("review_rate", self.review_rate)
        if self.review_minutes < 0 or self.baseline_minutes < 0:
            raise ValueError("review_minutes and baseline_minutes must be >= 0")
        if self.self_consistency_k < 1:
            raise ValueError("self_consistency_k must be >= 1")

    @property
    def avg_sequence(self) -> float:
        return self.tokens_in + self.tokens_out / 2.0


@dataclass(frozen=True)
class PricingSpec:
    """Commercial API price sheet, per million tokens."""

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: Optional[float] = None
    quality_index: float = 1.0
    monthly_platform_fee: float = 0.0

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise ValueError("token prices must be non-negative")

    def query_cost(self, tokens_in: float, tokens_out: float,
                   cached_fraction: float = 0.0) -> float:
        cached_rate = (self.cached_input_per_mtok
                       if self.cached_input_per_mtok is not None
                       else self.input_per_mtok)
        cached_tokens = tokens_in * cached_fraction
        fresh_tokens = tokens_in - cached_tokens
        return (fresh_tokens * self.input_per_mtok
                + cached_tokens * cached_rate
                + tokens_out * self.output_per_mtok) / 1e6


@dataclass(frozen=True)
class GridSpec:
    """Electricity supply for a self-hosted or colocated deployment."""

    name: str = "global-average"
    carbon_intensity: float = 0.436   # kg CO2e per kWh
    pue: float = 1.20                 # power usage effectiveness of the facility
    electricity_cost: float = 0.0     # USD/kWh; 0 if bundled into hourly_cost
    wue: float = 1.8                  # litres of water per kWh, facility level

    def __post_init__(self) -> None:
        if self.carbon_intensity < 0:
            raise ValueError("carbon_intensity must be non-negative")
        if self.pue < 1.0:
            raise ValueError("pue must be >= 1.0")


@dataclass(frozen=True)
class DeploymentState:
    """The mutable bundle that efficiency techniques transform.

    Techniques never mutate: each returns a new ``DeploymentState``, so a
    stack is a fold over an immutable value and is trivially reproducible.
    """

    model: ModelSpec
    hardware: HardwareSpec
    serving: ServingConfig
    notes: tuple = ()

    def evolve(self, *, model: Optional[ModelSpec] = None,
               hardware: Optional[HardwareSpec] = None,
               serving: Optional[ServingConfig] = None,
               note: Optional[str] = None) -> "DeploymentState":
        return DeploymentState(
            model=model if model is not None else self.model,
            hardware=hardware if hardware is not None else self.hardware,
            serving=serving if serving is not None else self.serving,
            notes=self.notes + ((note,) if note else ()),
        )
