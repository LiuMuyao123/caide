"""Multi-year scaling dynamics under falling unit cost.

Institutions plan AI budgets on the assumption that falling token prices
translate into falling bills. They frequently do not, and the reason is
not mysterious: cheaper inference makes previously uneconomic uses
economic, and the new uses consume the saving.

Not all of a query's cost is a token price. Three components behave
differently and must be held apart, which versions up to 7.0 did not:

``declining``     the part that tracks the token tariff, ``c(t) = c0*(1-r)^t``
``price-inelastic`` per-query cost that scales with volume but not with the
                  tariff -- human review at an hourly rate, per-query record
                  retention
``fixed``         layers that scale with neither -- the audit programme, the
                  integration estate

Demand responds to the price the buyer actually faces, which is the sum
of all three per query:

    c_eff(t) = c_decline(t) + c_inelastic + F / V(t)
    V(t)     = V0 * (c_eff(t)/c_eff(0))^(-eps) * (1 + g)^t

``V`` appears on both sides because fixed cost amortises over volume, so
each year is solved by damped iteration. Spend is ``c_eff(t) * V(t)``.

Substituting gives ``S(t) = c_eff(0)^eps * V0 * c_eff(t)^(1-eps) * (1+g)^t``,
so the sign of ``1 - eps`` still decides the direction of spend and the
crossover at ``eps = 1`` is still exact -- that result is structural and
survived the v8 correction unchanged. What the correction moves is the
*magnitude*, by a great deal, because ``c_eff`` falls far more slowly
than the tariff does. In the shipped education scenario only 10.4% of
cost per query tracks the token price; 50.6% is human review and 39.0%
is fixed. A 38%/yr tariff decline is a 9%/yr decline in what the
institution pays. Feeding a blended per-query figure into a token-price
decline -- as the analysis shipped through v7.0 did -- makes reviewer
wages fall at the speed of GPU prices, and makes fixed audit programmes
scale with query volume in both directions at once.

The model remains deliberately transparent: closed form apart from one
scalar fixed point, no fitting. Its purpose is to make the elasticity
assumption explicit and falsifiable rather than to predict a number.
Institutions that log their own volume against their own price changes
can estimate ``eps`` directly with :func:`estimate_elasticity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

_MAX_FIXED_POINT_ITERATIONS = 500

__all__ = [
    "ScalingAssumptions",
    "ScalingYear",
    "ScalingProjection",
    "project",
    "estimate_elasticity",
    "jevons_threshold",
]


@dataclass(frozen=True)
class ScalingAssumptions:
    """Parameters of the demand-response model."""

    annual_price_decline: float = 0.40   # r: fractional fall in unit cost per year
    price_elasticity: float = 1.30       # eps: |dlnV / dlnc|
    autonomous_growth: float = 0.15      # g: adoption growth independent of price
    horizon_years: int = 5
    capacity_ceiling: Optional[float] = None   # saturation on annual volume
    fixed_annual_cost: float = 0.0       # volume-independent layers
    # Per-query cost that scales with volume but not with the token
    # tariff: human review at an hourly rate, per-query record retention.
    # Zero by default, which reproduces the pre-v8 behaviour exactly --
    # and zero is wrong for every scenario CAIDE ships, where this term
    # is the largest of the three.
    price_inelastic_per_query: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.annual_price_decline < 1.0):
            raise ValueError("annual_price_decline must lie in [0, 1)")
        if self.price_elasticity < 0:
            raise ValueError("price_elasticity must be non-negative")
        if self.horizon_years < 1:
            raise ValueError("horizon_years must be >= 1")
        if self.price_inelastic_per_query < 0:
            raise ValueError("price_inelastic_per_query must be non-negative")
        if self.fixed_annual_cost < 0:
            raise ValueError("fixed_annual_cost must be non-negative")


@dataclass(frozen=True)
class ScalingYear:
    year: int
    unit_cost: float                 # the declining (tariff-tracking) component
    volume: float
    variable_spend: float            # declining component x volume
    fixed_spend: float
    inelastic_spend: float = 0.0     # price-inelastic per-query x volume
    converged: bool = True

    @property
    def total_spend(self) -> float:
        return self.variable_spend + self.inelastic_spend + self.fixed_spend

    @property
    def effective_unit_cost(self) -> float:
        """What the buyer actually pays per query, all three parts."""
        return self.total_spend / self.volume if self.volume > 0 else math.nan

    def as_dict(self) -> Dict[str, float]:
        return {
            "year": self.year,
            "unit_cost": self.unit_cost,
            "effective_unit_cost": self.effective_unit_cost,
            "volume": self.volume,
            "variable_spend": self.variable_spend,
            "inelastic_spend": self.inelastic_spend,
            "fixed_spend": self.fixed_spend,
            "total_spend": self.total_spend,
        }


@dataclass
class ScalingProjection:
    years: List[ScalingYear]
    assumptions: ScalingAssumptions
    saturated_from: Optional[int] = None

    @property
    def regime(self) -> str:
        eps = self.assumptions.price_elasticity
        if abs(eps - 1.0) < 1e-9:
            return "neutral"
        return "jevons" if eps > 1.0 else "inelastic"

    @property
    def spend_ratio(self) -> float:
        """Final-year spend divided by first-year spend."""
        if not self.years or self.years[0].total_spend == 0:
            return math.nan
        return self.years[-1].total_spend / self.years[0].total_spend

    @property
    def volume_ratio(self) -> float:
        if not self.years or self.years[0].volume == 0:
            return math.nan
        return self.years[-1].volume / self.years[0].volume

    def table(self) -> List[Dict[str, float]]:
        rows = [y.as_dict() for y in self.years]
        for row in rows:
            row["saturated"] = (self.saturated_from is not None
                                and row["year"] >= self.saturated_from)
        return rows

    def narrative(self) -> str:
        r = self.assumptions.annual_price_decline
        eps = self.assumptions.price_elasticity
        direction = "rises" if self.spend_ratio > 1.0 else "falls"
        text = (
            f"With unit cost falling {r:.0%}/yr and price elasticity "
            f"{eps:.2f} ({self.regime} regime), volume grows "
            f"{self.volume_ratio:.1f}x over {len(self.years)} years while "
            f"total spend {direction} {self.spend_ratio:.2f}x."
        )
        # ``saturated_from`` was computed from v1.0 and read by nothing
        # until v10.0: a projection that hit its declared capacity ceiling
        # reported a flattened volume curve and said nothing about why,
        # so the elasticity appeared to stop working. The same silent-
        # saturation failure the v8 audit found in the draw clamps.
        if self.saturated_from is not None:
            text += (f" Demand reaches the declared capacity ceiling in "
                     f"year {self.saturated_from}; growth after that is "
                     f"the ceiling, not the elasticity.")
        if any(not y.converged for y in self.years):
            text += (" One or more years did not converge; treat the "
                     "volume path as indicative.")
        return text


def _effective_unit_cost(declining: float, assumptions: ScalingAssumptions,
                         volume: float) -> float:
    """Price the buyer faces per query: tariff + labour + amortised fixed."""
    fixed_share = (assumptions.fixed_annual_cost / volume) if volume > 0 else 0.0
    return declining + assumptions.price_inelastic_per_query + fixed_share


def project(initial_unit_cost: float, initial_volume: float,
            assumptions: ScalingAssumptions) -> ScalingProjection:
    """Roll the demand-response model forward year by year.

    ``initial_unit_cost`` is the *declining* component only -- the part
    that tracks the token tariff. Costs that scale with volume but not
    with the tariff go in ``price_inelastic_per_query``; costs that scale
    with neither go in ``fixed_annual_cost``. Passing a blended
    per-query figure here and leaving the other two at zero is what the
    shipped analysis did through v7.0: it declines reviewer wages at the
    speed of accelerator prices, and it makes a fixed audit programme
    grow with query volume.
    """
    if initial_unit_cost < 0 or initial_volume < 0:
        raise ValueError("initial cost and volume must be non-negative")

    years: List[ScalingYear] = []
    saturated: Optional[int] = None
    base_effective = _effective_unit_cost(initial_unit_cost, assumptions,
                                          initial_volume)

    for t in range(assumptions.horizon_years):
        cost = initial_unit_cost * (1.0 - assumptions.annual_price_decline) ** t
        growth = (1.0 + assumptions.autonomous_growth) ** t

        # Demand responds to the effective price, which contains a fixed
        # cost amortised over the very volume being solved for. Damped
        # iteration; with no fixed layer the first pass is exact and the
        # loop is a no-op, which is why pre-v8 scenarios reproduce to the
        # last digit.
        volume, converged = initial_volume, True
        if base_effective <= 0:
            volume = initial_volume * growth
        else:
            converged = False
            for _ in range(_MAX_FIXED_POINT_ITERATIONS):
                effective = _effective_unit_cost(cost, assumptions, volume)
                target = (initial_volume
                          * ((effective / base_effective)
                             ** -assumptions.price_elasticity) * growth)
                if abs(target - volume) <= 1e-10 * max(volume, 1.0):
                    volume, converged = target, True
                    break
                volume = 0.5 * volume + 0.5 * target
            if not converged:
                # Never reached by any shipped scenario or test through
                # v12.0: the damped map is a contraction wherever the
                # fixed cost is finite, so 500 iterations is generous by
                # orders of magnitude. Kept because "generous" is not
                # "proved", and reported through ``converged`` so that a
                # future parameterisation that does diverge says so
                # rather than returning the last iterate as an answer.
                volume = target

        if assumptions.capacity_ceiling is not None:
            if volume > assumptions.capacity_ceiling:
                volume = assumptions.capacity_ceiling
                if saturated is None:
                    saturated = t + 1

        years.append(ScalingYear(
            year=t + 1,
            unit_cost=cost,
            volume=volume,
            variable_spend=cost * volume,
            fixed_spend=assumptions.fixed_annual_cost,
            inelastic_spend=(assumptions.price_inelastic_per_query * volume),
            converged=converged,
        ))

    return ScalingProjection(years, assumptions, saturated)


def jevons_threshold() -> float:
    """Elasticity above which falling unit cost raises total spend."""
    return 1.0


def estimate_elasticity(unit_costs: Sequence[float],
                        volumes: Sequence[float]) -> Dict[str, float]:
    """Least-squares estimate of price elasticity from observed history.

    Regresses ``ln V`` on ``ln c``; the negated slope is the elasticity.
    This is a descriptive fit, not a causal identification: anything that
    moves volume and price together -- a new cohort, a mandate, a
    curriculum change -- is absorbed into the slope. Reported alongside
    the fit are the residual standard error and the sample size so that
    a weak estimate is visibly weak.
    """
    if len(unit_costs) != len(volumes):
        raise ValueError("unit_costs and volumes must have equal length")
    pairs = [(c, v) for c, v in zip(unit_costs, volumes) if c > 0 and v > 0]
    if len(pairs) < 3:
        raise ValueError("need at least 3 positive observations")

    xs = [math.log(c) for c, _ in pairs]
    ys = [math.log(v) for _, v in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("unit costs show no variation; elasticity unidentified")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx

    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    rse = math.sqrt(sum(r ** 2 for r in resid) / dof)
    se_slope = rse / math.sqrt(sxx) if sxx > 0 else math.nan
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (sum(r ** 2 for r in resid) / syy) if syy > 0 else math.nan

    elasticity = -slope
    # The crossover at unit elasticity decides the direction of spend, so
    # a verdict about which side of it an estimate falls on is only worth
    # issuing when the estimate distinguishes the two. Until v12.0 the
    # regime was read off the point estimate while the standard error sat
    # unused in the same dictionary: on three-point histories -- the
    # minimum this function accepts -- the interval straddles one more
    # often than not, and "inelastic" was reported with the same
    # confidence either way.
    half = 1.96 * se_slope if math.isfinite(se_slope) else math.inf
    ci_low, ci_high = elasticity - half, elasticity + half
    if not math.isfinite(half) or (ci_low < 1.0 < ci_high):
        regime = "undetermined"
    elif elasticity > 1.0:
        regime = "jevons"
    else:
        regime = "inelastic"

    return {
        "elasticity": elasticity,
        "std_error": se_slope,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r_squared": r2,
        "n_observations": float(n),
        "regime": regime,
        #: What the point estimate alone would have said. Kept so that the
        #: change from a definite verdict to an honest one is visible
        #: rather than silent.
        "point_regime": "jevons" if elasticity > 1.0 else "inelastic",
    }
