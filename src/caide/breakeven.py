"""Break-even analysis between deployment architectures.

The textbook picture is a single crossing: API cheaper below some volume,
self-hosting cheaper above it. That picture is an artefact of assuming
both curves are linear.

They are not. Self-hosted capacity arrives in whole replicas, so its cost
curve is a staircase; the fixed layers make it an *offset* staircase. A
staircase and a line can intersect zero, one, or many times, and the
cheaper architecture can alternate as volume grows -- each new replica
overshoots demand and hands the advantage back to the API until the
replica fills up.

:func:`find_break_even` therefore reports *all* crossings on a bracketed
log-spaced scan rather than solving for one root, and
:func:`dominance_intervals` converts them into the intervals a planner
actually needs: "below X buy, between X and Y build, above Y buy again".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Crossing",
    "BreakEvenResult",
    "find_break_even",
    "dominance_intervals",
    "volume_sweep",
]

CostCurve = Callable[[float], float]


@dataclass(frozen=True)
class Crossing:
    """One volume at which the cheaper architecture changes."""

    volume: float
    cost_at_crossing: float
    winner_below: str
    winner_above: str
    margin_pct: float          # |relative gap| just outside the crossing

    def as_dict(self) -> Dict[str, object]:
        return {
            "volume": self.volume,
            "annual_cost": self.cost_at_crossing,
            "winner_below": self.winner_below,
            "winner_above": self.winner_above,
            "margin_pct": self.margin_pct,
        }


@dataclass
class BreakEvenResult:
    label_a: str
    label_b: str
    crossings: List[Crossing]
    scanned_min: float
    scanned_max: float
    curve_a: Dict[float, float]
    curve_b: Dict[float, float]

    @property
    def primary(self) -> Optional[Crossing]:
        """The crossing, when there is exactly one.

        Returns ``None`` when neither option ever overtakes the other, and
        **raises** when there are several. A staircase against a line
        crosses many times, and "the primary crossing" is the single
        threshold this module exists to argue against: the v9.0 audit
        showed the tie region is a union of narrow windows separated by
        stretches where the gap reaches 47%. The property was never
        called inside the package, so it kept offering that summary to
        anyone importing it without ever being exercised.
        """
        if not self.crossings:
            return None
        if len(self.crossings) > 1:
            raise ValueError(
                f"{len(self.crossings)} crossings: there is no primary one. "
                "Use .crossings for all of them, .tie_bands() for the "
                "volumes where cost does not decide, or .winner_at(volume) "
                "for a specific volume."
            )
        return self.crossings[0]

    def winner_at(self, volume: float) -> str:
        ca = _interp(self.curve_a, volume)
        cb = _interp(self.curve_b, volume)
        return self.label_a if ca <= cb else self.label_b

    def relative_gap(self, volume: float) -> float:
        """|cost_a - cost_b| / min(cost_a, cost_b) at one volume.

        The denominator is the cheaper option, so the figure reads as
        "how much more the loser costs". :attr:`Crossing.margin_pct`
        uses the same denominator since v9.0; until then it divided by
        the mean of the two, which made the same gap look smaller near a
        crossing and gave the module two definitions of one quantity.
        """
        ca = _interp(self.curve_a, volume)
        cb = _interp(self.curve_b, volume)
        floor = min(ca, cb)
        if floor <= 0 or not math.isfinite(floor):
            return math.inf
        return abs(ca - cb) / floor

    def tie_bands(self, tolerance: float = 0.05
                  ) -> List[Tuple[float, float]]:
        """Every *contiguous* volume run over which the options tie.

        The set of volumes where two architectures differ by less than a
        tolerance is a union of intervals, not an interval. Against a
        staircase it is typically several narrow windows, one around each
        riser, separated by stretches where the gap is large: the
        self-hosted curve is flat across a tread while the API line keeps
        rising, so the two agree only briefly near each step.

        Versions up to 8.0 reported ``first .. last`` of the qualifying
        scan points as a single band. On the shipped break-even that span
        contained 175 scan points of which 137 exceeded the tolerance,
        reaching 47% -- a "band in which cost does not decide" inside
        which cost decided by nearly half.
        """
        runs: List[Tuple[float, float]] = []
        current: List[float] = []
        for v in sorted(self.curve_a):
            if self.relative_gap(v) <= tolerance:
                current.append(v)
            else:
                if len(current) >= 2:
                    runs.append((current[0], current[-1]))
                current = []
        if len(current) >= 2:
            runs.append((current[0], current[-1]))
        return runs

    def tie_band(self, tolerance: float = 0.05
                 ) -> Optional[Tuple[float, float]]:
        """The **widest** contiguous tie window, or None if there is none.

        Kept as a single-interval summary for callers that need one, and
        it is now genuinely a range inside which cost does not decide.
        Use :meth:`tie_bands` when the shape matters -- with granular
        capacity the windows are several and narrow, and reporting only
        the widest understates how often the tie recurs.
        """
        runs = self.tie_bands(tolerance)
        if not runs:
            return None
        return max(runs, key=lambda r: r[1] / r[0] if r[0] > 0 else 0.0)

    def is_indistinguishable(self, volume: float,
                             tolerance: float = 0.05) -> bool:
        return self.relative_gap(volume) <= tolerance

    def as_dict(self) -> Dict[str, object]:
        return {
            "a": self.label_a,
            "b": self.label_b,
            "n_crossings": len(self.crossings),
            "crossings": [c.as_dict() for c in self.crossings],
            "scan_range": [self.scanned_min, self.scanned_max],
        }


def _interp(curve: Dict[float, float], x: float) -> float:
    keys = sorted(curve)
    if x <= keys[0]:
        return curve[keys[0]]
    if x >= keys[-1]:
        return curve[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= x <= hi:
            if hi == lo:
                return curve[lo]
            t = (x - lo) / (hi - lo)
            return curve[lo] * (1 - t) + curve[hi] * t
    return curve[keys[-1]]                              # pragma: no cover


def _bisect_crossing(f: Callable[[float], float], lo: float, hi: float,
                     iterations: int = 60) -> float:
    """Locate a sign change of ``f`` on ``[lo, hi]`` in log space.

    Bisecting the logarithm rather than the value keeps resolution
    uniform across the six or seven orders of magnitude that separate a
    departmental pilot from a national platform.
    """
    f_lo = f(lo)
    log_lo, log_hi = math.log(lo), math.log(hi)
    for _ in range(iterations):
        mid = math.exp(0.5 * (log_lo + log_hi))
        f_mid = f(mid)
        if f_mid == 0:
            return mid
        if (f_lo < 0) == (f_mid < 0):
            log_lo, f_lo = math.log(mid), f_mid
        else:
            log_hi = math.log(mid)
    return math.exp(0.5 * (log_lo + log_hi))


def volume_sweep(curves: Dict[str, CostCurve], volumes: Sequence[float]
                 ) -> Dict[str, List[float]]:
    """Evaluate several cost curves on a shared volume grid."""
    return {name: [fn(v) for v in volumes] for name, fn in curves.items()}


def find_break_even(cost_a: CostCurve, cost_b: CostCurve, *,
                    label_a: str = "A", label_b: str = "B",
                    volume_min: float = 1e3, volume_max: float = 1e10,
                    samples: int = 240) -> BreakEvenResult:
    """Find every volume at which the cheaper of two architectures changes.

    Parameters
    ----------
    cost_a, cost_b
        Annual total cost as a function of annual query volume.
    volume_min, volume_max
        Bracket for the scan, log-spaced. Crossings outside the bracket
        are not reported; widen it if the result looks suspiciously empty.
    """
    if volume_min <= 0 or volume_max <= volume_min:
        raise ValueError("require 0 < volume_min < volume_max")

    log_lo, log_hi = math.log(volume_min), math.log(volume_max)
    grid = [math.exp(log_lo + (log_hi - log_lo) * i / (samples - 1))
            for i in range(samples)]

    curve_a = {v: cost_a(v) for v in grid}
    curve_b = {v: cost_b(v) for v in grid}
    delta = {v: curve_a[v] - curve_b[v] for v in grid}

    def f(v: float) -> float:
        return cost_a(v) - cost_b(v)

    crossings: List[Crossing] = []
    for lo, hi in zip(grid, grid[1:]):
        d_lo, d_hi = delta[lo], delta[hi]
        if not (math.isfinite(d_lo) and math.isfinite(d_hi)):
            continue
        if d_lo == 0.0:
            continue
        if (d_lo < 0) != (d_hi < 0):
            x = _bisect_crossing(f, lo, hi)
            cost_here = 0.5 * (cost_a(x) + cost_b(x))
            below = label_a if d_lo < 0 else label_b
            above = label_b if d_lo < 0 else label_a
            # Same denominator as ``relative_gap``: the cheaper option.
            denom = max(min(abs(cost_a(x)), abs(cost_b(x))), 1e-12)
            margin = 100.0 * max(abs(d_lo), abs(d_hi)) / denom
            crossings.append(Crossing(x, cost_here, below, above, margin))

    crossings.sort(key=lambda c: c.volume)
    return BreakEvenResult(label_a, label_b, crossings,
                           volume_min, volume_max, curve_a, curve_b)


def dominance_intervals(result: BreakEvenResult) -> List[Tuple[float, float, str]]:
    """Convert crossings into ``(lower, upper, winner)`` planning intervals."""
    edges = [result.scanned_min] + [c.volume for c in result.crossings] \
        + [result.scanned_max]
    intervals: List[Tuple[float, float, str]] = []
    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo:
            continue
        probe = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        intervals.append((lo, hi, result.winner_at(probe)))

    merged: List[Tuple[float, float, str]] = []
    for lo, hi, who in intervals:
        if merged and merged[-1][2] == who:
            merged[-1] = (merged[-1][0], hi, who)
        else:
            merged.append((lo, hi, who))
    return merged
