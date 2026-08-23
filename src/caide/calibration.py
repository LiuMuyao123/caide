"""Calibration of the roofline against measured throughput.

The roofline predicts from first principles, which is what lets it compare
configurations nobody has built. The price is that its two utilisation
parameters -- achievable FLOP rate in prefill and achievable bandwidth in
decode -- are assumptions, and a wrong assumption biases every prediction
in the same direction.

Version 3.0 of this software asserted in its documentation that
predictions land within a factor of two of measured serving stacks. That
claim had never been tested. When it was, against four published
measurements spanning three serving frameworks, only two of the four fell
inside the band and the model over-predicted in three of them.

The right response to that is not to tune the constants until four
heterogeneous data points agree -- that fits noise from different
frameworks, versions and datasets. It is to make the correction an input
the user supplies from their own hardware.

:func:`fit` takes observations of the form "this configuration achieved
this many output tokens per second" and returns the scalar corrections to
``mfu_prefill`` and ``mbu_decode`` that minimise squared error in the log
of the predicted-to-measured ratio. Log space is used because the errors
are multiplicative: being 2x fast and 2x slow are equally wrong, and a
linear fit would treat the first as twice the error of the second.

Two observations are enough to fit one scalar. Fewer, and the function
says so rather than returning a number.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

from .roofline import evaluate_request
from .specs import DeploymentState, HardwareSpec, ModelSpec, ServingConfig, \
    WorkloadClass

__all__ = [
    "admissible_conventions",
    "implied_mbu",
    "FRAMEWORK_OVERHEAD_REFERENCE",
    "FRAMEWORK_OVERHEAD_HISTORY",
    "FRAMEWORK_OVERHEAD_SENSITIVITY",
    "Observation",
    "CONVENTIONS",
    "CalibrationResult",
    "fit",
    "predicted_output_tps",
    "REFERENCE_OBSERVATIONS",
    "EXCLUDED_OBSERVATIONS",
    "READMITTED_OBSERVATIONS",
]


#: How a published benchmark reports throughput. The two conventions
#: differ by exactly the batch size -- a factor of 8 for a small
#: concurrency test and 256 for a large one -- and papers and blogs use
#: both while writing the same word, "throughput".
CONVENTIONS = ("aggregate", "per_request")


@dataclass(frozen=True)
class Observation:
    """One measured throughput figure with the configuration that produced it.

    ``measured_output_tps`` is output tokens per second, excluding prompt
    tokens. ``convention`` says whether that figure is for the whole
    replica (``"aggregate"``) or for one request (``"per_request"``);
    :attr:`aggregate_output_tps` normalises to the former.

    The field is mandatory in spirit and defaulted in practice, because
    getting it wrong is a silent factor-of-batch error and there is no
    value that is safe to guess. Sources that do not state their
    convention should not be used: ``source`` exists so that the
    ambiguity is visible in the record rather than buried in a number.
    """

    model: ModelSpec
    hardware: HardwareSpec
    n_accelerators: int
    batch: int
    tokens_in: float
    tokens_out: float
    measured_output_tps: float
    precision: str = "bf16"
    source: str = ""
    convention: str = "aggregate"

    def __post_init__(self) -> None:
        if self.measured_output_tps <= 0:
            raise ValueError("measured_output_tps must be positive")
        if self.batch < 1 or self.n_accelerators < 1:
            raise ValueError("batch and n_accelerators must be >= 1")
        if self.convention not in CONVENTIONS:
            raise ValueError(
                f"convention must be one of {CONVENTIONS}, got "
                f"{self.convention!r}. A benchmark reporting 'throughput' "
                "without saying which it means differs by a factor of "
                f"{self.batch} here; do not guess."
            )

    @property
    def aggregate_output_tps(self) -> float:
        """Measured throughput normalised to whole-replica output tokens/s."""
        if self.convention == "per_request":
            return self.measured_output_tps * self.batch
        return self.measured_output_tps

    def serving(self, mfu_scale: float = 1.0,
                mbu_scale: float = 1.0) -> ServingConfig:
        base = ServingConfig(n_accelerators=self.n_accelerators,
                             max_batch=self.batch, precision=self.precision)
        return replace(
            base,
            mfu_prefill=min(max(base.mfu_prefill * mfu_scale, 0.005), 0.95),
            mbu_decode=min(max(base.mbu_decode * mbu_scale, 0.005), 0.98),
        )

    def workload(self) -> WorkloadClass:
        return WorkloadClass("observed", 1.0, self.tokens_in, self.tokens_out)


def predicted_output_tps(obs: Observation, mfu_scale: float = 1.0,
                         mbu_scale: float = 1.0) -> Optional[float]:
    """Aggregate output tokens/second the roofline predicts for ``obs``.

    Returns ``None`` when the configuration is infeasible -- the weights do
    not fit, or the request exceeds the context window. An infeasible
    prediction is not a prediction of zero and must not be averaged in.
    """
    state = DeploymentState(obs.model, obs.hardware,
                            obs.serving(mfu_scale, mbu_scale))
    perf = evaluate_request(state, obs.workload(), batch_override=obs.batch)
    if not perf.feasible or perf.decode_seconds <= 0:
        return None
    # Output tokens over the *whole* serving cycle, prefill included. A
    # published tokens-per-second figure divides output tokens by wall
    # time, and wall time contains the prompt pass. Versions up to 6.0
    # divided by decode time alone, which is not the model's own
    # definition of throughput -- :func:`evaluate_request` divides by the
    # full cycle -- so the same quantity had two derivations inside one
    # package, and they disagreed by the prefill share of the cycle: up
    # to 2.2x for the 70B TP=4 observation, whose 512-token prompts at
    # batch 128 put prefill at half the cycle. "Decode-dominated" turned
    # out to describe the requests, not the arithmetic. No convention
    # verdict turned on it, but every published predicted-to-measured
    # ratio did.
    cycle = perf.prefill_seconds + perf.decode_seconds
    if cycle <= 0:
        return None
    return perf.batch * obs.tokens_out / cycle


def _physical_bound_state(obs: Observation) -> DeploymentState:
    """The observation's configuration stripped down to what physics forbids.

    Bandwidth utilisation goes to 1.0, framework overhead to zero and the
    interconnect to ideal. The result is not a prediction of anything --
    no deployment reaches it -- but it is a floor on the decode step that
    no measurement can be under, which is what makes the convention test
    a statement about the hardware rather than about CAIDE's default
    assumptions. Testing against the configured ``mbu_decode`` of 0.70
    instead would reject any benchmark that merely tuned its kernels
    better than the default assumes.
    """
    return DeploymentState(
        obs.model, obs.hardware,
        replace(obs.serving(),
                mbu_decode=1.0,
                mfu_prefill=0.95,
                tensor_parallel_penalty=0.0,
                framework_overhead_per_step=0.0,
                framework_overhead_per_sequence=0.0))


def admissible_conventions(obs: Observation) -> List[str]:
    """Which readings of a reported throughput figure are physically possible.

    A benchmark that reports "throughput" without saying whether it means
    the whole replica or one request leaves two readings that differ by
    the batch size. Version 5.0 treated that as unresolvable and excluded
    such sources. It is not always unresolvable: the roofline is a *lower
    bound* on step time given the declared memory-bandwidth utilisation,
    so a reading that implies a step faster than the bound implies a
    bandwidth utilisation above one, and can be rejected on physics rather
    than on judgement.

    Returns the subset of :data:`CONVENTIONS` that survives. A single
    survivor identifies the convention; two mean the figure really is
    ambiguous; none means the observation contradicts the model under
    either reading, which is a finding about the model or the source
    rather than about the convention.

    The test is one-sided by construction. It can rule a reading out for
    being too fast; it can never rule one out for being too slow, because
    arbitrarily much framework overhead can sit between the hardware bound
    and the wall clock.
    """
    perf = evaluate_request(_physical_bound_state(obs), obs.workload(),
                            batch_override=obs.batch)
    if not perf.feasible or perf.tpot <= 0:
        return list(CONVENTIONS)

    survivors = []
    for convention in CONVENTIONS:
        aggregate = (obs.measured_output_tps * obs.batch
                     if convention == "per_request" else obs.measured_output_tps)
        implied_step = perf.batch / aggregate if aggregate > 0 else math.inf
        if implied_step >= perf.tpot:
            survivors.append(convention)
    return survivors


def implied_mbu(obs: Observation, convention: str) -> float:
    """Memory-bandwidth utilisation a reading of ``obs`` would require.

    Above 1.0 the reading is impossible: it asks the accelerator to move
    bytes faster than its rated bandwidth. This is the quantity that makes
    :func:`admissible_conventions` a physical test rather than a heuristic.
    """
    perf = evaluate_request(_physical_bound_state(obs), obs.workload(),
                            batch_override=obs.batch)
    aggregate = (obs.measured_output_tps * obs.batch
                 if convention == "per_request" else obs.measured_output_tps)
    if aggregate <= 0 or not perf.feasible or perf.tpot <= 0:
        return math.inf
    implied_step = perf.batch / aggregate
    return perf.tpot / implied_step


@dataclass
class CalibrationResult:
    """Scalar corrections to the two utilisation parameters, and their fit."""

    mfu_scale: float
    mbu_scale: float
    n_observations: int
    log_rmse_before: float
    log_rmse_after: float
    ratios_before: List[float] = field(default_factory=list)
    ratios_after: List[float] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.log_rmse_after < self.log_rmse_before

    @property
    def within_factor_after(self) -> float:
        """Fraction of observations predicted within 2x after calibration."""
        if not self.ratios_after:
            return math.nan
        inside = sum(1 for r in self.ratios_after if 0.5 <= r <= 2.0)
        return inside / len(self.ratios_after)

    @property
    def within_factor_before(self) -> float:
        if not self.ratios_before:
            return math.nan
        inside = sum(1 for r in self.ratios_before if 0.5 <= r <= 2.0)
        return inside / len(self.ratios_before)

    def apply(self, cfg: ServingConfig) -> ServingConfig:
        """Return ``cfg`` with the fitted corrections applied."""
        return replace(
            cfg,
            mfu_prefill=min(max(cfg.mfu_prefill * self.mfu_scale, 0.005), 0.95),
            mbu_decode=min(max(cfg.mbu_decode * self.mbu_scale, 0.005), 0.98),
        )

    def summary(self) -> Dict[str, float]:
        return {
            "mfu_scale": self.mfu_scale,
            "mbu_scale": self.mbu_scale,
            "n_observations": float(self.n_observations),
            "log_rmse_before": self.log_rmse_before,
            "log_rmse_after": self.log_rmse_after,
            "within_2x_before": self.within_factor_before,
            "within_2x_after": self.within_factor_after,
        }


def _log_rmse(ratios: Sequence[float]) -> float:
    if not ratios:
        return math.nan
    return math.sqrt(statistics.fmean([math.log(r) ** 2 for r in ratios]))


def fit(observations: Sequence[Observation],
        bounds: tuple = (0.2, 3.0),
        steps: int = 61) -> CalibrationResult:
    """Fit multiplicative corrections to prefill and decode utilisation.

    Decode utilisation is fitted first and alone, because the observations
    that serving benchmarks publish are decode-dominated: a request with
    hundreds of output tokens spends almost all of its time in decode, so
    prefill utilisation is barely identified by them. Fitting both jointly
    on such data would assign the prefill parameter whatever value happens
    to absorb the residual, which looks like a better fit and is not one.

    The search is a grid over the log of the correction, which is adequate
    for a one-dimensional monotone objective and avoids the failure modes
    of a gradient method on a function with feasibility cliffs.
    """
    if len(observations) < 2:
        raise ValueError(
            f"need at least 2 observations to fit a correction, got "
            f"{len(observations)}. A single measurement can be matched "
            "exactly by construction and tells you nothing about the model."
        )

    lo, hi = bounds
    if not (0 < lo < hi):
        raise ValueError("bounds must satisfy 0 < low < high")

    skipped: List[str] = []
    usable: List[Observation] = []
    for obs in observations:
        if predicted_output_tps(obs) is None:
            skipped.append(obs.source or obs.model.name)
        else:
            usable.append(obs)

    if len(usable) < 2:
        raise ValueError(
            f"only {len(usable)} of {len(observations)} observations are "
            "feasible under the model; cannot fit"
        )

    def ratios(mbu_scale: float) -> List[float]:
        out = []
        for obs in usable:
            pred = predicted_output_tps(obs, 1.0, mbu_scale)
            if pred is not None:
                out.append(pred / obs.aggregate_output_tps)
        return out

    before = ratios(1.0)
    best_scale, best_err = 1.0, _log_rmse(before)
    for i in range(steps):
        scale = math.exp(math.log(lo) + (math.log(hi) - math.log(lo))
                         * i / (steps - 1))
        err = _log_rmse(ratios(scale))
        if math.isfinite(err) and err < best_err:
            best_scale, best_err = scale, err

    after = ratios(best_scale)
    return CalibrationResult(
        mfu_scale=1.0,
        mbu_scale=best_scale,
        n_observations=len(usable),
        log_rmse_before=_log_rmse(before),
        log_rmse_after=best_err,
        ratios_before=before,
        ratios_after=after,
        skipped=skipped,
    )


def _obs(model_key: str, hw_key: str, n_acc: int, batch: int, tin: float,
         tout: float, tps: float, precision: str, bytes_per_param: float,
         source: str, convention: str = "aggregate") -> Observation:
    from .catalog import get_hardware, get_model
    model = get_model(model_key)
    if bytes_per_param != model.bytes_per_param:
        model = model.with_precision(bytes_per_param)
    return Observation(model=model, hardware=get_hardware(hw_key),
                       n_accelerators=n_acc, batch=batch, tokens_in=tin,
                       tokens_out=tout, measured_output_tps=tps,
                       precision=precision, source=source,
                       convention=convention)


def REFERENCE_OBSERVATIONS() -> List[Observation]:
    """Published throughput measurements, for validating the model.

    These are not a calibration set. They come from different serving
    frameworks at different versions on different datasets, and the model
    archetypes here are generic shapes rather than the exact architectures
    measured. They are adequate to check the model's order of magnitude and
    inadequate to tune it. Calibrate against your own hardware instead.

    Every entry states its measurement convention explicitly. Sources that
    report "throughput" without saying whether they mean the whole replica
    or one request are excluded, because the two differ by the batch size
    and there is no safe way to guess. Two candidate observations were
    dropped for exactly that reason during the v5.0 audit; the exclusion is
    recorded here rather than silently applied.

    All figures are GPU-plus-framework wall-clock, which is what benchmarks
    report. The bundled model has ``framework_overhead_per_step`` at zero
    by default -- a pure hardware roofline -- so it will over-predict these
    numbers. That gap is a documented property, not a defect: see
    ``docs/model.md`` section 8.
    """
    return [
        _obs("dense-70b", "h100-sxm", 4, 128, 512, 256, 3245, "bf16", 2.0,
             "vLLM, LLaMA-2-70B TP=4, high concurrency (arXiv 2511.17593); "
             "states aggregate token throughput", "aggregate"),
        _obs("dense-405b", "h100-sxm", 8, 64, 1024, 2048, 3089, "fp8", 1.0,
             "TensorRT-LLM, Llama 3.1 405B TP=8 FP8 (arXiv 2509.20241); "
             "table header states Tokens per Second, high concurrency",
             "aggregate"),
        _obs("dense-405b", "h100-sxm", 8, 256, 128, 128, 3732, "fp8", 1.0,
             "TensorRT-LLM, Llama 3.1 405B TP=8 FP8 (arXiv 2509.20241); "
             "same table", "aggregate"),
        _obs("dense-70b", "h100-sxm", 1, 64, 256, 512, 460, "fp8", 1.0,
             "SGLang, Llama 3.1 70B FP8 on 1xH100 (cerebrium.ai); reported "
             "as throughput at batch size 64, read as aggregate",
             "aggregate"),
        _obs("dense-8b", "a100-80gb", 1, 8, 1024, 512, 187, "bf16", 2.0,
             "vLLM, Llama 3 8B on 1xA100, 8 concurrent requests. The source "
             "does not state its convention; the per-request reading is "
             "excluded by admissible_conventions() because it implies a "
             "memory-bandwidth utilisation of 1.60, so the aggregate "
             "reading is the only physically possible one",
             "aggregate"),
    ]


#: Framework overhead measured on one configuration, with its provenance.
#:
#: Version 5.0 added ``framework_overhead_per_step`` with no default and no
#: source, which left users a parameter they had no basis to set. These are
#: the first empirical values. They come from one model on one accelerator
#: with one serving stack and must not be transferred to another
#: deployment; they are published so that the order of magnitude is on the
#: record and so that the fit below can be checked.
#:
#: Derivation: the pure-hardware roofline predicts a decode step of 11.3 ms
#: at batch 1 and 12.0 ms at batch 8 for an 8B model on one A100. The two
#: published wall-clock figures imply per-step wall times of 26.1 ms and
#: 40.9 ms once the modelled prefill (0.12 s per 1,024-token prompt) is
#: taken off the cycle first -- v6.0 skipped that subtraction and folded
#: prefill into "overhead", overstating the batch-8 residual by 6%. The
#: residuals, 14.7 ms and 28.9 ms, differ by a factor of two, so no
#: constant fits both; a constant plus a per-sequence term fits both
#: exactly.
#:
#: Two points and two parameters leave zero degrees of freedom, so this is
#: a consistency statement and not a validation. What supports it
#: independently is vLLM's own profiling of the same model class, which
#: attributes 38% of wall time to GPU execution; the batch-1 figures here
#: put the accelerator at 43% of wall time, a 13% disagreement between two
#: routes that share no data.
#: What the two-point fit has produced at each release. Five successive
#: physics corrections moved these constants, and ``per_sequence_seconds``
#: has returned to where it started -- the v7.0 prefill correction pushed
#: it down and the v10.0 weight-stream correction pushed it back. With two
#: points fitting two parameters the residual is absorbed by construction,
#: so these observations can never contradict a change to the hardware
#: model. That is the property this history exists to make visible.
FRAMEWORK_OVERHEAD_HISTORY = (
    ("6.0.0", 0.0127, 0.00226, "two-point fit introduced"),
    ("7.0.0", 0.0127, 0.00202, "prefill removed from the wall clock"),
    ("10.0.0", 0.01345, 0.00226, "input embedding removed from the stream"),
)

#: The band the constants have occupied across those releases, offered as
#: a declared sensitivity interval. Recommended for four consecutive audit
#: rounds and adopted in v11.0: a parameter that cannot be validated
#: should be presented as a modelling choice with a range, not as a
#: pending calibration that never arrives.
FRAMEWORK_OVERHEAD_SENSITIVITY = {
    "per_step_seconds": (0.0127, 0.01345),
    "per_sequence_seconds": (0.00202, 0.00226),
    "status": "declared modelling choice, not a validated measurement",
}

FRAMEWORK_OVERHEAD_REFERENCE = {
    "config": "vLLM, Llama 3 8B, 1xA100-80GB, bf16",
    # Re-derived in v10.0 after the decode weight stream stopped counting
    # the input embedding table: the hardware step for this 8B model fell,
    # so more of the measured wall clock is residual. The constants are a
    # two-point fit to two parameters and have had zero degrees of freedom
    # since v6.0 -- see ``degrees_of_freedom`` below, and the standing
    # recommendation, now in its fifth round, to obtain a third point.
    "per_step_seconds": 0.01345,
    "per_sequence_seconds": 0.00226,
    "gpu_share_of_wall_at_batch_1": 0.403,
    "vllm_profiled_gpu_share": 0.38,
    "n_points": 2,
    "degrees_of_freedom": 0,
    # With zero degrees of freedom the fit is exact whatever the hardware
    # model says, so these two points measure the residual rather than
    # testing the physics. Stated here so that a reader does not mistake
    # an exact fit for strong evidence -- it is the weakest possible kind.
    "residual_absorber": True,
    "status": FRAMEWORK_OVERHEAD_SENSITIVITY["status"],
}


#: Candidate observations excluded during the v5.0 audit, kept so that the
#: exclusion is auditable. Each states why it was not usable.
EXCLUDED_OBSERVATIONS = (
    ("vLLM, Llama 3 8B on 1xA100, single stream, 38 tok/s",
     "Unambiguous as a per-request figure, but a single-stream measurement "
     "is dominated by framework overhead rather than by the hardware the "
     "roofline models. It is used in FRAMEWORK_OVERHEAD_REFERENCE to "
     "estimate that overhead and is kept out of the validation set, "
     "because a point used to fit a parameter cannot also test it."),
    ("A100 80GB, Llama 2 7B, ~2,500 tok/s batch throughput",
     "Marketing blog with no stated batch size, sequence lengths, or "
     "precision. Not reproducible."),
)

#: Observations excluded by an earlier audit and readmitted later, with the
#: reason the earlier exclusion no longer holds. Kept separate from
#: EXCLUDED_OBSERVATIONS so that a validation set growing over time can be
#: distinguished from one drifting toward the points that happen to agree.
READMITTED_OBSERVATIONS = (
    ("vLLM, Llama 3 8B on 1xA100, 8 concurrent requests, 187 tok/s",
     "Excluded in v5.0 as ambiguous between the aggregate and per-request "
     "conventions. v6.0 shows the ambiguity is decidable: the per-request "
     "reading requires a decode step of 5.3 ms against a hardware bound of "
     "12.0 ms, implying a memory-bandwidth utilisation of 1.60. Readmitted "
     "under the aggregate reading, which the physics does not exclude."),
)
