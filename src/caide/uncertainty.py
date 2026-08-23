"""Uncertainty propagation and sensitivity analysis.

A deployment plan built on point estimates is a plan built on the least
likely outcome. Every input here -- token counts, cache hit rates,
accelerator prices, achievable utilisation -- is uncertain by a factor
that is often larger than the difference between the architectures being
compared. Reporting a single number invites a decision that the evidence
does not support.

Two complementary tools are provided.

:func:`monte_carlo` propagates input distributions through any callable
and returns the output distribution, so a comparison can be stated as
"self-hosting is cheaper in 78% of draws" instead of "self-hosting is
cheaper".

:func:`sensitivity` ranks inputs by how much of the output variance they
explain, using Spearman rank correlation on the same draws. Rank
correlation is used rather than Pearson because the cost model is
strongly non-linear -- step functions, roofline maxima, SLO cliffs --
and a monotone but curved relationship should not be scored as weak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Distribution",
    "uniform",
    "triangular",
    "lognormal",
    "normal",
    "point",
    "MonteCarloResult",
    "monte_carlo",
    "sensitivity",
    "SensitivityEntry",
]


@dataclass(frozen=True)
class Distribution:
    """A named sampler over one scalar input."""

    name: str
    sample: Callable[[np.random.Generator, int], np.ndarray]
    nominal: float
    kind: str = "custom"
    #: The parameters this distribution was built from. A sampler is a
    #: closure, so without this the numbers that define the assumption
    #: exist nowhere a report or a provenance digest can reach them: two
    #: lognormals an order of magnitude apart in spread were
    #: indistinguishable to both until v10.0.
    params: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "kind": self.kind,
                "nominal": self.nominal, **self.params}

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.asarray(self.sample(rng, n), dtype=float)


def uniform(name: str, low: float, high: float) -> Distribution:
    """Flat uncertainty between two bounds; use when only a range is known."""
    if high < low:
        raise ValueError(f"{name}: high must be >= low")
    return Distribution(name, lambda r, n: r.uniform(low, high, n),
                        0.5 * (low + high), "uniform",
                        {"low": low, "high": high})


def triangular(name: str, low: float, mode: float, high: float) -> Distribution:
    """Bounded uncertainty with a most-likely value.

    The natural choice for an expert estimate stated as "usually X, never
    below Y, never above Z" -- which is how utilisation and review rates
    are actually known.
    """
    if not (low <= mode <= high):
        raise ValueError(f"{name}: require low <= mode <= high")
    return Distribution(name, lambda r, n: r.triangular(low, mode, high, n),
                        mode, "triangular",
                        {"low": low, "mode": mode, "high": high})


def normal(name: str, mean: float, sd: float, *, clip_low: Optional[float] = None
           ) -> Distribution:
    """Symmetric additive uncertainty, optionally clipped from below.

    Prefer :func:`lognormal` for prices and other strictly positive
    quantities whose uncertainty is multiplicative.
    """
    if sd < 0:
        raise ValueError(f"{name}: sd must be non-negative")

    def _s(r: np.random.Generator, n: int) -> np.ndarray:
        x = r.normal(mean, sd, n)
        return np.clip(x, clip_low, None) if clip_low is not None else x

    return Distribution(name, _s, mean, "normal",
                        {"mean": mean, "sd": sd})


def lognormal(name: str, median: float, sigma: float) -> Distribution:
    """Multiplicative uncertainty: ``sigma`` is the log-scale spread.

    ``sigma = 0.35`` gives roughly a 2x spread between the 5th and 95th
    percentile, which is a realistic prior for a price quotation obtained
    from one vendor in one region.
    """
    if median <= 0:
        raise ValueError(f"{name}: median must be positive")
    if sigma < 0:
        raise ValueError(f"{name}: sigma must be non-negative")
    mu = math.log(median)
    return Distribution(name, lambda r, n: r.lognormal(mu, sigma, n),
                        median, "lognormal",
                        {"median": median, "sigma": sigma})


def point(name: str, value: float) -> Distribution:
    """A known constant. Excluded from sensitivity analysis, since an input
    that never varies cannot explain any output variance."""
    return Distribution(name, lambda r, n: np.full(n, value), value, "point",
                        {"value": value})


@dataclass
class MonteCarloResult:
    """Output distribution of a model under input uncertainty."""

    samples: np.ndarray
    inputs: Dict[str, np.ndarray]
    n_failed: int = 0
    label: str = "output"
    #: Draws in which an input hit a physical bound and was clamped,
    #: keyed by input name. A clamped draw is neither a failure nor a
    #: faithful sample of the declared distribution, and it is reported
    #: for the same reason failures are: the propagated uncertainty is
    #: then narrower than the uncertainty the scenario declares.
    n_saturated: Dict[str, int] = field(default_factory=dict)

    def saturation_share(self) -> Dict[str, float]:
        n = float(self.samples.size) or 1.0
        return {k: v / n for k, v in self.n_saturated.items() if v}

    @property
    def valid(self) -> np.ndarray:
        return self.samples[np.isfinite(self.samples)]

    @property
    def feasible_fraction(self) -> float:
        """Share of draws that produced a finite result.

        Every percentile below is computed on those draws alone, so the
        spread they describe is conditional on the configuration being
        feasible. That conditioning is harmless at 1.0 and material below
        it, and it was not stated anywhere until v12.0.
        """
        n = float(self.samples.size)
        return (self.valid.size / n) if n else math.nan

    def explained_rank_variance(self) -> float:
        """Sum of squared rank correlations before normalisation.

        ``SensitivityEntry.contribution`` is a share of this quantity, so
        the shares total one however little the declared inputs actually
        account for. Reporting the total lets a reader tell "these five
        inputs explain almost everything, and here is their order" from
        "these five inputs explain a third of it, and here is their
        order".
        """
        return sum(e.spearman ** 2 for e in sensitivity(self)
                   if math.isfinite(e.spearman))

    def percentile(self, q: float) -> float:
        v = self.valid
        return float(np.percentile(v, q)) if v.size else math.nan

    def summary(self) -> Dict[str, float]:
        v = self.valid
        if v.size == 0:
            return {"n": 0.0}
        return {
            "n": float(v.size),
            "n_failed": float(self.n_failed),
            "n_saturated": float(sum(self.n_saturated.values())),
            "feasible_fraction": self.feasible_fraction,
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "p05": self.percentile(5),
            "p25": self.percentile(25),
            "p50": self.percentile(50),
            "p75": self.percentile(75),
            "p95": self.percentile(95),
            "cv": float(np.std(v, ddof=1) / np.mean(v)) if v.size > 1
            and np.mean(v) != 0 else math.nan,
        }

    def probability_below(self, threshold: float) -> float:
        v = self.valid
        return float(np.mean(v < threshold)) if v.size else math.nan


def monte_carlo(model: Callable[[Dict[str, float]], float],
                distributions: Sequence[Distribution],
                n_samples: int = 4000,
                seed: Optional[int] = 20260101,
                label: str = "output",
                saturation: Optional[Callable[[Dict[str, float]],
                                              Sequence[str]]] = None
                ) -> MonteCarloResult:
    """Propagate input distributions through ``model``.

    ``model`` receives one dict of sampled values per draw and returns a
    scalar. Draws that raise or return a non-finite value are counted in
    ``n_failed`` rather than silently dropped, because a configuration
    that is infeasible in 30% of draws is a finding, not a nuisance.

    ``saturation`` is an optional predicate returning the names of inputs
    that hit a physical bound on a given draw. The same principle applies
    to it: an input clamped in most draws means the propagated spread is
    narrower than the declared one, and that belongs in the report rather
    than in the difference between two numbers nobody compared.

    The default seed makes every reported figure reproducible; pass
    ``seed=None`` for an independent run.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(seed)

    inputs = {d.name: d.draw(rng, n_samples) for d in distributions}
    out = np.empty(n_samples, dtype=float)
    failed = 0
    saturated: Dict[str, int] = {}

    for i in range(n_samples):
        row = {k: float(v[i]) for k, v in inputs.items()}
        if saturation is not None:
            for key in saturation(row):
                saturated[key] = saturated.get(key, 0) + 1
        try:
            value = float(model(row))
        except Exception:
            value, failed = math.nan, failed + 1
        else:
            if not math.isfinite(value):
                failed += 1
        out[i] = value

    return MonteCarloResult(out, inputs, failed, label, saturated)


@dataclass(frozen=True)
class SensitivityEntry:
    name: str
    spearman: float
    contribution: float          # normalised share of explained rank variance
    nominal: float
    low_output: float
    high_output: float
    #: Rank association between this input and the *failure* of a draw.
    #: ``nan`` when nothing failed. Reported because the cost correlation
    #: above is computed over surviving draws only: an input whose whole
    #: effect is to make the configuration infeasible has no variation
    #: left to correlate with, and scores zero. Constructed case: an input
    #: that renders 10.7% of draws infeasible received 0.0% of explained
    #: variance while the harmless one received 100%.
    failure_spearman: float = math.nan

    @property
    def swing(self) -> float:
        return self.high_output - self.low_output

    def as_dict(self) -> Dict[str, float]:
        return {
            "input": self.name,
            "failure_spearman": self.failure_spearman,
            "spearman": self.spearman,
            "contribution": self.contribution,
            "low_output": self.low_output,
            "high_output": self.high_output,
            "swing": self.swing,
        }


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # average ties so that repeated values do not bias the correlation
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(counts.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return math.nan
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom > 0 else math.nan


def sensitivity(result: MonteCarloResult,
                quantiles: Tuple[float, float] = (10.0, 90.0)
                ) -> List[SensitivityEntry]:
    """Rank inputs by their monotone influence on the output.

    For each input the draws are split at the given quantiles and the
    median output in each tail is reported, which gives the tornado bars
    a direct reading: "moving this input from its 10th to its 90th
    percentile moves annual cost from A to B".
    """
    mask = np.isfinite(result.samples)
    y = result.samples[mask]
    if y.size < 3:
        return []

    # Failure is an outcome, not an absence of one. ``monte_carlo`` has
    # counted failed draws since v1.0 on the principle that a
    # configuration infeasible in a third of draws is a finding; this
    # function discarded them and then ranked inputs by what was left,
    # which is the same principle with the opposite implementation two
    # definitions apart. The association with failure is reported
    # separately rather than imputed as a cost, because CAIDE does not
    # know what an infeasible year is worth.
    failed = ~mask
    any_failed = bool(failed.any())

    entries: List[SensitivityEntry] = []
    raw: List[Tuple[str, float, float, float, float, float]] = []

    for name, values in result.inputs.items():
        x = values[mask]
        if np.allclose(x, x[0]):
            continue
        rho = _spearman(x, y)
        fail_rho = (_spearman(values, failed.astype(float))
                    if any_failed else math.nan)
        lo_cut = np.percentile(x, quantiles[0])
        hi_cut = np.percentile(x, quantiles[1])
        lo_mask = x <= lo_cut
        hi_mask = x >= hi_cut
        lo_out = float(np.median(y[lo_mask])) if lo_mask.any() else math.nan
        hi_out = float(np.median(y[hi_mask])) if hi_mask.any() else math.nan
        nominal = float(np.median(x))
        raw.append((name, rho, nominal, lo_out, hi_out, fail_rho))

    total = sum(r[1] ** 2 for r in raw if math.isfinite(r[1]))
    for name, rho, nominal, lo_out, hi_out, fail_rho in raw:
        share = (rho ** 2 / total) if total > 0 and math.isfinite(rho) else math.nan
        entries.append(SensitivityEntry(name, rho, share, nominal, lo_out,
                                        hi_out, fail_rho))

    entries.sort(key=lambda e: abs(e.spearman) if math.isfinite(e.spearman) else -1,
                 reverse=True)
    return entries
