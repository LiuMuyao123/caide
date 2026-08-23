"""Efficiency techniques modelled as transformations of physical state.

Published cost analyses tabulate efficiency techniques as fixed
multipliers -- "4-bit quantisation: 0.65x", "speculative decoding: 0.40x"
-- and multiply them together to estimate a stack. That method has two
defects. It cannot express *interaction*, and it cannot express *regime
dependence*.

Interaction: quantisation shrinks the weights, which frees HBM, which
raises the feasible batch, which amortises the weight read further. The
combined effect of quantisation and batching is larger than the product
of their separate multipliers.

Regime dependence: speculative decoding helps enormously at batch 1,
where decode is starved for arithmetic, and helps almost not at all at
batch 256, where the GPU is already compute-saturated and the draft model
competes for the same FLOPs. A single multiplier cannot be right in both
regimes.

CAIDE therefore represents each technique as a function
``DeploymentState -> DeploymentState`` that edits the underlying physics.
The cost multiplier is not an input; it is *measured* by re-running the
roofline model on the transformed state. Stacks compose by function
composition, and their emergent multipliers are outputs of the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Sequence

from .specs import DeploymentState

__all__ = [
    "QUANTISATION_HEAD_BYTES",
    "Technique",
    "TECHNIQUES",
    "get_technique",
    "apply_stack",
    "stack_quality_delta",
    "stack_engineering_hours",
    "available_techniques",
    "resolve_stack",
    "PRESET_STACKS",
]


@dataclass(frozen=True)
class Technique:
    """One efficiency intervention.

    Attributes
    ----------
    quality_delta
        Expected change in the model's relative quality index. Negative
        values mean degradation. These are configurable defaults, not
        universal constants -- override them with local evaluation results.
    engineering_hours
        One-off implementation and validation effort. Techniques with a
        low cost multiplier and a high hour count are frequently the wrong
        first move for a small team, which the report surfaces explicitly.
    maturity
        ``stable`` techniques are in production serving stacks today;
        ``emerging`` ones carry integration risk.
    """

    key: str
    label: str
    apply: Callable[[DeploymentState], DeploymentState]
    quality_delta: float = 0.0
    #: Where ``quality_delta`` comes from, and what it does not cover. A
    #: non-zero delta with no basis is a quoted constant wearing the
    #: clothes of a derived one, which is the practice this package was
    #: written to argue against -- and since v10.0 the quality axis
    #: decides admissibility, so the argument now applies to CAIDE's own
    #: numbers.
    quality_basis: str = ""
    engineering_hours: float = 0.0
    maturity: str = "stable"
    description: str = ""
    conflicts: tuple = ()

    def __call__(self, state: DeploymentState) -> DeploymentState:
        return self.apply(state).evolve(note=self.key)


# ---------------------------------------------------------------------------
# individual transformations
# ---------------------------------------------------------------------------

#: Bytes per parameter the embedding table and the language-model head
#: keep under weight quantisation. GPTQ, AWQ and the llama.cpp k-quants
#: all skip both: the head is the one matrix whose quantisation error
#: lands directly on the output distribution, and it buys the least
#: memory per unit of damage. Versions up to 9.0 quantised every
#: parameter uniformly, which overstated int4's saving by 1.39x on the
#: smallest bundled archetype and by 1.09x on the one the paper sweeps.
QUANTISATION_HEAD_BYTES = 2.0


def _quantise(bytes_per_param: float, precision: str,
              mfu_gain: float = 1.0) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        head = max(QUANTISATION_HEAD_BYTES, bytes_per_param)
        model = state.model.with_precision(bytes_per_param,
                                           head_bytes_per_param=head)
        serving = replace(
            state.serving,
            precision=precision,
            mfu_prefill=min(state.serving.mfu_prefill * mfu_gain, 0.85),
        )
        return state.evolve(model=model, serving=serving)
    return _apply


def _quantise_kv(bytes_per_element: float) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        model = replace(state.model, bytes_per_kv_element=bytes_per_element)
        return state.evolve(model=model)
    return _apply


def _continuous_batching(state: DeploymentState) -> DeploymentState:
    """Raise the achievable batch and the scheduler efficiency.

    Static batching idles the accelerator while the longest sequence in a
    batch finishes. Continuous batching admits new requests at every step,
    which is worth more in utilisation terms than in per-step terms.

    Note what this does *not* touch: ``demand_duty_cycle``. A better
    scheduler cannot manufacture traffic, so the improvement is bounded by
    how much of the live period was previously wasted.
    """
    serving = replace(
        state.serving,
        max_batch=max(state.serving.max_batch, 256),
        scheduler_efficiency=min(state.serving.scheduler_efficiency * 1.9, 0.88),
    )
    return state.evolve(serving=serving)


def _paged_attention(state: DeploymentState) -> DeploymentState:
    """Recover the HBM lost to KV-cache fragmentation.

    Pre-allocating a contiguous KV region per sequence wastes memory
    proportional to the gap between the reserved and realised sequence
    length. Paging that region into blocks recovers most of it, which
    shows up as a larger feasible batch rather than as a faster step.
    """
    serving = replace(
        state.serving,
        memory_utilisation=min(state.serving.memory_utilisation * 1.18, 0.96),
    )
    return state.evolve(serving=serving)


def _flash_attention(state: DeploymentState) -> DeploymentState:
    serving = replace(
        state.serving,
        mfu_prefill=min(state.serving.mfu_prefill * 1.45, 0.80),
    )
    return state.evolve(serving=serving)


def _prefix_caching(hit: float) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        current = state.serving.prefix_cache_hit
        merged = 1.0 - (1.0 - current) * (1.0 - hit)
        return state.evolve(serving=replace(state.serving, prefix_cache_hit=merged))
    return _apply


def _semantic_caching(hit: float) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        current = state.serving.semantic_cache_hit
        merged = 1.0 - (1.0 - current) * (1.0 - hit)
        return state.evolve(serving=replace(state.serving, semantic_cache_hit=merged))
    return _apply


def _speculative(gamma: float, acceptance: float,
                 draft_ratio: float) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        serving = replace(
            state.serving,
            speculative_gamma=gamma,
            speculative_acceptance=acceptance,
            draft_param_ratio=draft_ratio,
        )
        return state.evolve(serving=serving)
    return _apply


def _distil(ratio: float) -> Callable[[DeploymentState], DeploymentState]:
    """Replace the target model with a smaller student of the same shape.

    Depth and width are scaled by ``ratio**(1/3)`` and ``ratio**(1/3)``
    respectively so that the parameter count scales as ``ratio`` while the
    aspect ratio stays realistic; a student that is only shallower or only
    narrower is not representative of published distillation practice.
    """
    def _apply(state: DeploymentState) -> DeploymentState:
        m = state.model
        shrink = ratio ** (1.0 / 3.0)
        n_heads = max(int(round(m.n_heads * shrink)), 1)
        d_model = max(int(round(m.d_model * shrink / n_heads)) * n_heads, n_heads)
        kv_heads = max(int(round(m.kv_heads * shrink)), 1)
        student = replace(
            m,
            name=f"{m.name}-distilled-{ratio:g}",
            n_params_total=m.n_params_total * ratio,
            n_params_active=m.active_params * ratio,
            n_layers=max(int(round(m.n_layers * shrink)), 1),
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=min(kv_heads, n_heads),
            quality_index=m.quality_index * (0.55 + 0.45 * ratio ** 0.25),
        )
        return state.evolve(model=student)
    return _apply


def _grouped_query_attention(groups: int) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        m = state.model
        kv = max(m.n_heads // max(groups, 1), 1)
        return state.evolve(model=replace(m, n_kv_heads=min(kv, m.kv_heads)))
    return _apply


def _chunked_prefill(state: DeploymentState) -> DeploymentState:
    """Interleave prefill chunks with decode steps.

    Removes the head-of-line blocking that leaves the device partly idle
    during long prefills, so it too improves scheduler efficiency rather
    than demand duty cycle.
    """
    serving = replace(
        state.serving,
        prefill_chunking=True,
        scheduler_efficiency=min(state.serving.scheduler_efficiency * 1.25, 0.92),
    )
    return state.evolve(serving=serving)


def _committed_use(discount: float) -> Callable[[DeploymentState], DeploymentState]:
    def _apply(state: DeploymentState) -> DeploymentState:
        hw = replace(state.hardware,
                     hourly_cost=state.hardware.hourly_cost * discount)
        return state.evolve(hardware=hw)
    return _apply


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------

TECHNIQUES: Dict[str, Technique] = {
    t.key: t for t in [
        Technique(
            key="flash_attention",
            label="FlashAttention kernels",
            apply=_flash_attention,
            quality_delta=0.0,
            engineering_hours=8,
            description="IO-aware attention; raises prefill FLOP utilisation.",
        ),
        Technique(
            key="continuous_batching",
            label="Continuous batching",
            apply=_continuous_batching,
            quality_delta=0.0,
            engineering_hours=24,
            description="Admit requests at every decode step; raises duty cycle.",
        ),
        Technique(
            key="paged_attention",
            label="PagedAttention",
            apply=_paged_attention,
            quality_delta=0.0,
            engineering_hours=16,
            description="Block-paged KV cache; recovers fragmented HBM.",
        ),
        Technique(
            key="chunked_prefill",
            label="Chunked prefill",
            apply=_chunked_prefill,
            quality_delta=0.0,
            engineering_hours=20,
            description="Interleave prefill chunks with decode to smooth latency.",
        ),
        Technique(
            key="int8",
            label="INT8 weight quantisation",
            apply=_quantise(1.0, "int8", mfu_gain=1.10),
            quality_delta=-0.003,
            quality_basis=(
                "Quoted constant, not derived. Reference point: a 70B-class dense model at the catalogue's default vocabulary. INT8 weight quantisation; published perplexity deltas are small at this scale and grow as the model shrinks, which this single figure does not express."),
            engineering_hours=24,
            description="8-bit weights; halves weight traffic and footprint.",
            conflicts=("int4", "fp8"),
        ),
        Technique(
            key="int4",
            label="INT4 weight quantisation (GPTQ/AWQ)",
            apply=_quantise(0.5, "int4", mfu_gain=1.10),
            quality_delta=-0.010,
            quality_basis=(
                "Quoted constant, not derived. Reference point: a 70B-class dense model at the catalogue's default vocabulary. INT4 weight quantisation. Damage is strongly size dependent -- a 4-bit 8B model degrades far more than a 4-bit 405B -- and one constant is applied to both. See the limits table."),
            engineering_hours=40,
            description="4-bit weights; quarters weight traffic and footprint.",
            conflicts=("int8", "fp8"),
        ),
        Technique(
            key="fp8",
            label="FP8 weights and activations",
            apply=_quantise(1.0, "fp8", mfu_gain=1.35),
            quality_delta=-0.002,
            quality_basis=(
                "Quoted constant, not derived. Reference point: a 70B-class dense model at the catalogue's default vocabulary. Reduced-precision arithmetic or KV storage; the smallest of the losses in this catalogue and the least size sensitive."),
            engineering_hours=32,
            description="8-bit float on supporting hardware; keeps tensor-core speedup.",
            conflicts=("int8", "int4"),
        ),
        Technique(
            key="kv_fp8",
            label="FP8 KV cache",
            apply=_quantise_kv(1.0),
            quality_delta=-0.002,
            quality_basis=(
                "Quoted constant, not derived. Reference point: a 70B-class dense model at the catalogue's default vocabulary. Reduced-precision arithmetic or KV storage; the smallest of the losses in this catalogue and the least size sensitive."),
            engineering_hours=16,
            description="Halves KV-cache bytes; enlarges feasible batch at long context.",
        ),
        Technique(
            key="gqa",
            label="Grouped-query attention (8 groups)",
            apply=_grouped_query_attention(8),
            quality_delta=-0.004,
            quality_basis=(
                "Quoted constant, not derived. Reference point: a 70B-class dense model at the catalogue's default vocabulary. Grouped-query attention trades attention capacity for KV traffic; the loss depends on the head geometry, which this figure does not read."),
            engineering_hours=0,
            description="Architectural: fewer KV heads, much smaller KV cache.",
        ),
        Technique(
            key="prefix_caching",
            label="Prefix caching",
            apply=_prefix_caching(0.55),
            quality_delta=0.0,
            engineering_hours=24,
            description="Reuse KV for shared system prompts and retrieved blocks.",
        ),
        Technique(
            key="semantic_caching",
            label="Semantic response caching",
            apply=_semantic_caching(0.30),
            quality_delta=-0.012,
            quality_basis=(
                "Quoted constant, not derived. Reference point: a 70B-class dense model at the catalogue's default vocabulary. Semantic caching answers from a near neighbour rather than the query itself; the loss depends on the similarity threshold and the workload's diversity, neither of which this figure reads."),
            engineering_hours=56,
            maturity="emerging",
            description="Serve near-duplicate queries from cache; staleness risk.",
        ),
        Technique(
            key="speculative_decoding",
            label="Speculative decoding",
            apply=_speculative(gamma=4.0, acceptance=0.72, draft_ratio=0.03),
            quality_delta=0.0,
            engineering_hours=72,
            maturity="emerging",
            description="Draft model proposes, target verifies; lossless by construction.",
        ),
        Technique(
            key="distillation_50",
            label="Distillation to 50% parameters",
            apply=_distil(0.5),
            # Zero because ``_distil`` derives the loss from the student's
            # geometry rather than quoting it -- the derivation CAIDE
            # prefers everywhere else. Charging both was double counting.
            quality_delta=0.0,
            engineering_hours=320,
            description="Task-specific student model for high-volume bounded traffic.",
            conflicts=("distillation_25",),
        ),
        Technique(
            key="distillation_25",
            label="Distillation to 25% parameters",
            apply=_distil(0.25),
            quality_delta=0.0,   # derived by _distil; see distillation_50
            engineering_hours=400,
            description="Aggressive student; suitable only for narrow, reviewed tasks.",
            conflicts=("distillation_50",),
        ),
        Technique(
            key="committed_use",
            label="Committed-use discount",
            apply=_committed_use(0.62),
            quality_delta=0.0,
            engineering_hours=8,
            description="Reserved capacity pricing; trades flexibility for unit cost.",
        ),
    ]
}


PRESET_STACKS: Dict[str, tuple] = {
    "none": (),
    "baseline_serving": ("flash_attention", "continuous_batching", "paged_attention"),
    "standard": ("flash_attention", "continuous_batching", "paged_attention",
                 "chunked_prefill", "prefix_caching"),
    "aggressive": ("flash_attention", "continuous_batching", "paged_attention",
                   "chunked_prefill", "int4", "kv_fp8", "prefix_caching",
                   "speculative_decoding"),
    "maximal": ("flash_attention", "continuous_batching", "paged_attention",
                "chunked_prefill", "int4", "kv_fp8", "gqa", "prefix_caching",
                "semantic_caching", "speculative_decoding", "committed_use"),
}


def available_techniques() -> List[Technique]:
    """Every efficiency technique in the catalogue, in declaration order."""
    return list(TECHNIQUES.values())


def get_technique(key: str) -> Technique:
    """Look up one technique by key; raises KeyError listing the alternatives."""
    if key not in TECHNIQUES:
        raise KeyError(
            f"unknown technique {key!r}; available: {sorted(TECHNIQUES)}"
        )
    return TECHNIQUES[key]


def resolve_stack(spec: Sequence[str] | str) -> tuple:
    """Accept either a preset name or an explicit sequence of technique keys."""
    if isinstance(spec, str):
        if spec not in PRESET_STACKS:
            raise KeyError(
                f"unknown preset stack {spec!r}; available: {sorted(PRESET_STACKS)}"
            )
        return PRESET_STACKS[spec]
    return tuple(spec)


def _check_conflicts(keys: Sequence[str]) -> None:
    seen = set(keys)
    for key in keys:
        tech = get_technique(key)
        clash = seen.intersection(tech.conflicts)
        if clash:
            raise ValueError(
                f"technique {key!r} conflicts with {sorted(clash)}; "
                "quantisation formats and distillation ratios are mutually exclusive"
            )


def apply_stack(state: DeploymentState,
                keys: Sequence[str] | str) -> DeploymentState:
    """Fold a stack of techniques over a deployment state.

    Order matters and is preserved: quantisation before paging yields a
    different feasible batch than paging before quantisation only if a
    technique reads a value another one writes, which the catalogue is
    designed to minimise but does not forbid.
    """
    keys = resolve_stack(keys)
    _check_conflicts(keys)
    for key in keys:
        state = get_technique(key)(state)

    # The quality consequence is part of the state transformation, not a
    # figure a caller is expected to fetch separately. Until v11.0 it was
    # the latter: thirteen of fifteen techniques recorded a
    # ``quality_delta`` that ``apply_stack`` ignored, so a library user
    # who quantised a model to four bits and read back
    # ``state.model.quality_index`` got the undegraded number. Distillation
    # was the mirror image -- its transform lowered the index *and* the
    # scenario layer applied its delta again, charging the same loss twice
    # (0.880 to 0.817 to 0.776 for a half-size student).
    delta = stack_quality_delta(keys)
    if delta:
        state = state.evolve(model=replace(
            state.model,
            quality_index=state.model.quality_index * (1.0 + delta)))
    return state


def stack_quality_delta(keys: Sequence[str] | str) -> float:
    """Aggregate quality impact, composed on retention rather than on loss.

    Two techniques that each retain 99% of quality retain 98.01%
    together, not 98%: the second loss applies to an already-reduced
    base, so composing retentions is very slightly gentler than adding
    losses. The gap is immaterial for two techniques and reaches roughly
    a percentage point across a long stack, which is the scale at which
    a quality floor in a routing decision can flip.
    """
    keys = resolve_stack(keys)
    retained = 1.0
    for key in keys:
        retained *= (1.0 + get_technique(key).quality_delta)
    return retained - 1.0


def stack_engineering_hours(keys: Sequence[str] | str) -> float:
    """One-off implementation and validation effort for a stack, in hours.

    A stack with a low cost multiplier and a high hour count is often the
    wrong first move for a small team, which the report surfaces.
    """
    keys = resolve_stack(keys)
    return sum(get_technique(k).engineering_hours for k in keys)
