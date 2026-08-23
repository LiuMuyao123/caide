"""Cost layers and total cost of ownership.

The central modelling claim is that the six layers of a deployed LLM
service scale differently with query volume, and that ignoring this is
what makes naive per-token estimates diverge from realised budgets:

======================  ==============================================
layer                   scaling in annual volume V
======================  ==============================================
model access            linear          -- pay per token, forever
compute and serving     step            -- capacity comes in whole nodes
retrieval and data      sublinear       -- index once, query many times
integration and SRE     volume-free     -- driven by system count
assurance and governance mixed          -- fixed programme, linear review
workforce and redesign  front-loaded    -- large year 1, flat thereafter
======================  ==============================================

The assurance row said "volume-free" until v8.0, and it was the row that
mattered most: human review is charged per query, and in all three
shipped scenarios the assurance layer is both the *largest* layer (58%,
91%, 84% of ownership cost) and 83-95% volume-linear. A reader applying
the table to forecast a doubling of volume would have expected the
dominant layer to stand still.

Two layers are substantially linear, therefore, not one. A model that
includes only model access still understates cost at low volume (fixed
layers dominate) *and* overstates the savings from efficiency work at
high volume (the efficient layer shrinks while the others do not) --
but the second error is larger than the table implied, because the layer
that does not shrink with token prices does grow with volume.

Because a table is an assertion and this one was wrong for six releases,
:func:`layer_volume_elasticity` measures each layer's response to volume
instead of quoting it: a finite difference in ``d ln(cost) / d ln(V)``,
returning 1.0 for a linear layer, 0.0 for a fixed one, the exponent for
a sublinear one, and whatever the staircase is actually doing locally
for the stepped one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .roofline import evaluate_request, solve_batch_for_slo
from .specs import (
    SLO,
    DeploymentState,
    GridSpec,
    PricingSpec,
    WorkloadClass,
)

__all__ = [
    "replica_annual_cost",
    "CostLayer",
    "AssuranceProfile",
    "QueryCost",
    "TCOResult",
    "self_hosted_query_cost",
    "api_query_cost",
    "total_cost_of_ownership",
    "layer_volume_elasticity",
    "SIX_LAYERS",
]

SIX_LAYERS = (
    "model_access",
    "compute_serving",
    "retrieval_data",
    "integration_sre",
    "assurance_governance",
    "workforce_redesign",
)

SECONDS_PER_HOUR = 3600.0
JOULES_PER_KWH = 3.6e6


@dataclass(frozen=True)
class CostLayer:
    """One layer of the ownership stack with an explicit scaling law."""

    name: str
    fixed_annual: float = 0.0
    per_query: float = 0.0
    sublinear_coefficient: float = 0.0
    sublinear_exponent: float = 0.35
    step_size: float = 0.0            # queries per capacity unit
    step_cost: float = 0.0            # annual cost of one capacity unit
    front_load_year1: float = 0.0
    decay: float = 0.35               # fraction of year-1 cost recurring later

    def __post_init__(self) -> None:
        """Validate, which this class did not do until v13.0.

        Every other spec dataclass in the package rejects impossible
        values; this one, the class the three configurable ownership
        layers are built from, accepted anything. A ``step_cost`` declared
        without a ``step_size`` contributed exactly zero -- a layer the
        author wrote down, priced, and never saw again -- and negative
        values were absorbed silently. The scenario loader's own comment
        says a silently ignored key is the most expensive kind of typo
        here; the same is true one level down.
        """
        for field_name in ("fixed_annual", "per_query", "sublinear_coefficient",
                           "step_size", "step_cost", "front_load_year1"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(
                    f"{self.name}.{field_name} must be non-negative, got {value}")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(
                f"{self.name}.decay must lie in [0, 1], got {self.decay}")
        if self.sublinear_exponent < 0:
            raise ValueError(
                f"{self.name}.sublinear_exponent must be non-negative")
        if (self.step_cost > 0) != (self.step_size > 0):
            raise ValueError(
                f"{self.name}: step_cost and step_size must be given "
                "together. One without the other describes a capacity "
                "block with no price or a price with no block, and the "
                "layer would contribute nothing."
            )
        if self.sublinear_coefficient > 0 and self.sublinear_exponent >= 1.0:
            raise ValueError(
                f"{self.name}: sublinear_exponent={self.sublinear_exponent} "
                "is not sublinear; use per_query for a linear layer."
            )

    def annual_cost(self, volume: float, year: int = 1) -> float:
        total = self.fixed_annual + self.per_query * volume
        if self.sublinear_coefficient > 0 and volume > 0:
            total += self.sublinear_coefficient * volume ** self.sublinear_exponent
        if self.step_size > 0 and self.step_cost > 0:
            units = math.ceil(volume / self.step_size) if volume > 0 else 0
            total += units * self.step_cost
        if self.front_load_year1 > 0:
            total += (self.front_load_year1 if year <= 1
                      else self.front_load_year1 * self.decay)
        return total


@dataclass(frozen=True)
class AssuranceProfile:
    """Governance and oversight, priced as infrastructure rather than overhead.

    Human review and audit-record retention are charged per query;
    everything else -- the evaluation suite, the audit pipeline, the
    red-team programme -- costs the same whether the service answers ten
    thousand queries a year or ten million.

    The per-query part is not a rounding term. In the three shipped
    scenarios it is 83%, 95% and 93% of this layer, and this layer is the
    largest of the six in all three. It is also the part that does *not*
    fall when token prices fall, because it is priced in reviewer hours:
    see :mod:`caide.scaling`, where treating it as if it tracked the
    token tariff was the v8 audit's largest finding.
    """

    audit_logging_annual: float = 0.0
    evaluation_annual: float = 0.0
    red_team_annual: float = 0.0
    privacy_review_annual: float = 0.0
    incident_response_annual: float = 0.0
    reviewer_hourly_cost: float = 45.0
    storage_per_query: float = 2.0e-6      # audit record retention, USD/query

    @property
    def fixed_annual(self) -> float:
        return (self.audit_logging_annual + self.evaluation_annual
                + self.red_team_annual + self.privacy_review_annual
                + self.incident_response_annual)

    def review_cost_per_query(self, workloads: Sequence[WorkloadClass]) -> float:
        total = 0.0
        for w in workloads:
            minutes = w.review_rate * w.review_minutes
            total += w.share * (minutes / 60.0) * self.reviewer_hourly_cost
        return total

    def displaced_labour_per_query(self,
                                   workloads: Sequence[WorkloadClass]) -> float:
        """Human cost the system removes, priced at the same hourly rate.

        Only classes that declare ``baseline_minutes`` contribute. The
        figure is reported alongside -- never subtracted from -- total
        cost of ownership, because a labour saving and a cash outlay are
        not the same instrument: one appears in a budget line, the other
        in a capacity argument, and netting them silently is how business
        cases stop being auditable.
        """
        total = 0.0
        for w in workloads:
            if w.baseline_minutes <= 0:
                continue
            total += w.share * (w.baseline_minutes / 60.0) * self.reviewer_hourly_cost
        return total

    def review_hours_per_year(self, workloads: Sequence[WorkloadClass],
                              annual_volume: float) -> float:
        """Implied human review hours -- the sanity check that catches
        scenarios demanding more reviewer time than staff exist to supply."""
        total = 0.0
        for w in workloads:
            total += (w.share * annual_volume * w.review_rate
                      * w.review_minutes / 60.0)
        return total


@dataclass(frozen=True)
class QueryCost:
    """Per-query economics for one workload class under one architecture."""

    workload: str
    architecture: str
    compute_cost: float
    energy_joules: float
    carbon_kg: float
    water_litres: float
    latency_seconds: float
    #: ``True`` met, ``False`` missed, ``None`` **not evaluated**. The
    #: third state exists because a commercial endpoint's latency is not
    #: modelled here, and recording that as a pass made every API
    #: architecture satisfy every latency objective by construction --
    #: an asymmetry that decides admissibility now that v10.0 made
    #: feasibility the ranking criterion.
    slo_met: Optional[bool]
    batch: float
    detail: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "workload": self.workload,
            "architecture": self.architecture,
            "compute_cost_usd": self.compute_cost,
            "energy_wh": self.energy_joules / 3600.0,
            "carbon_g": self.carbon_kg * 1000.0,
            "water_l": self.water_litres,
            "latency_s": self.latency_seconds,
            "slo_met": self.slo_met,
            "batch": self.batch,
            **self.detail,
        }


@dataclass
class TCOResult:
    """Annual total cost of ownership, decomposed and attributed."""

    architecture: str
    annual_volume: float
    year: int
    layers: Dict[str, float]
    per_query_blended: float
    compute_per_query: float
    annual_carbon_kg: float
    annual_energy_kwh: float
    annual_water_l: float
    quality_index: float
    slo_violations: List[str] = field(default_factory=list)
    #: Classes whose latency objective could not be evaluated for this
    #: architecture. Reported rather than assumed either way: an
    #: unevaluated constraint is not a satisfied one, and it is not a
    #: violated one either.
    slo_unevaluated: List[str] = field(default_factory=list)
    #: Workload classes whose declared ``quality_floor`` exceeds this
    #: architecture's quality index. Recorded for the same reason SLO
    #: violations are: a scenario that states a class needs a capability
    #: level has stated a constraint, and an architecture that misses it
    #: is not a cheap option, it is not an option. Until v10.0 the floor
    #: was enforced in :mod:`caide.routing` and read nowhere else, so the
    #: architecture comparison that produces every published verdict
    #: ignored it -- and in all three shipped scenarios the architecture
    #: reported as cheapest failed at least one floor.
    quality_violations: List[str] = field(default_factory=list)
    #: How far below each floor this architecture falls, as a fraction of
    #: the floor. Reported because "inadmissible" is a label pressed onto
    #: a continuous quantity, and the distance to the boundary is what
    #: distinguishes a candidate that misses by 1.2% from one that misses
    #: by 22%. The v13 audit found a published claim that had been false
    #: for three releases on a margin of 0.6%; this is the same hazard one
    #: layer up, and v10.0 made it the ranking criterion.
    #: The floors the margins were measured against, so that a margin can
    #: be converted back to an absolute difference in index units.
    _floors: Dict[str, float] = field(default_factory=dict)
    quality_shortfall: Dict[str, float] = field(default_factory=dict)
    #: Signed margin against each declared floor, as a fraction of the
    #: floor: positive where the architecture clears it, negative where it
    #: does not. Reported for every class, not only the failing ones,
    #: because "cleared it by 1.1%" and "cleared it by 49%" are different
    #: statements and the verdict alone makes them the same.
    quality_margin: Dict[str, float] = field(default_factory=dict)
    capacity_units: float = 0.0
    notes: List[str] = field(default_factory=list)
    displaced_labour_annual: float = 0.0
    review_hours_annual: float = 0.0
    #: Per-query cost that scales with volume but not with the token
    #: tariff -- reviewer hours and per-query record retention. Carried
    #: here rather than re-derived by callers, because deriving one
    #: quantity in two places is how two places start to disagree.
    price_inelastic_per_query: float = 0.0

    def scaling_inputs(self) -> Dict[str, float]:
        """The three components :func:`caide.scaling.project` needs.

        ``declining_per_query`` is the share that tracks the token tariff
        (model access and compute); ``price_inelastic_per_query`` is
        reviewer time and record retention, which scale with volume at a
        rate set by wages, not by GPU prices; ``fixed_annual`` is
        everything left. Splitting them is not a refinement: in the
        shipped scenarios the declining share is 10% of cost per query,
        so declining the blended figure at the tariff rate -- which is
        what CAIDE's own published scaling analysis did through v7.0 --
        is a statement about the wrong 90%.
        """
        v = self.annual_volume
        declining = self.layers["model_access"] + self.layers["compute_serving"]
        inelastic = self.price_inelastic_per_query * v
        fixed = max(self.total - declining - inelastic, 0.0)
        return {
            "declining_per_query": declining / v if v > 0 else 0.0,
            "price_inelastic_per_query": self.price_inelastic_per_query,
            "fixed_annual": fixed,
            "declining_share": declining / self.total if self.total > 0 else 0.0,
        }

    @property
    def feasible(self) -> bool:
        """No declared constraint is *violated* by this architecture.

        An unevaluated constraint does not make an architecture
        infeasible -- see :attr:`fully_evaluated`, which says whether the
        verdict rests on evidence for every constraint or only for some.
        """
        return not self.slo_violations and not self.quality_violations

    @property
    def marginal_verdicts(self) -> List[str]:
        """Classes whose admissibility rests on less than the index can say.

        The quality index is a declared scale with a stated resolution;
        a margin below it orders two numbers the scale does not order.
        Every previous round that met a threshold claim added the distance
        to the boundary (v9, v12, v13, v14). This adds the thing those
        distances had to be compared against.
        """
        from .specs import QUALITY_INDEX_RESOLUTION
        return sorted(
            cls for cls, margin in self.quality_margin.items()
            if abs(margin) * self._floor_of(cls) < QUALITY_INDEX_RESOLUTION)

    def _floor_of(self, cls: str) -> float:
        return self._floors.get(cls, 0.0)

    @property
    def fully_evaluated(self) -> bool:
        """Every declared constraint was actually checked."""
        return not self.slo_unevaluated

    @property
    def total(self) -> float:
        return sum(self.layers.values())

    @property
    def net_of_displaced_labour(self) -> float:
        """TCO minus the human cost the system removes.

        Reported separately from :attr:`total` on purpose. A negative
        value means the deployment pays for itself in labour terms; it
        does not mean the budget line disappears.
        """
        return self.total - self.displaced_labour_annual

    @property
    def review_fte(self) -> float:
        """Full-time equivalents implied by the review workload (1700 h/FTE)."""
        return self.review_hours_annual / 1700.0

    @property
    def effective_per_query(self) -> float:
        return self.total / self.annual_volume if self.annual_volume > 0 else math.inf

    def share(self) -> Dict[str, float]:
        total = self.total
        if total <= 0:
            return {k: 0.0 for k in self.layers}
        return {k: v / total for k, v in self.layers.items()}

    def as_dict(self) -> Dict[str, object]:
        out = {
            "architecture": self.architecture,
            "annual_volume": self.annual_volume,
            "year": self.year,
            "total_usd": self.total,
            "usd_per_query": self.effective_per_query,
            "compute_usd_per_query": self.compute_per_query,
            "annual_tco_carbon_kg": self.annual_carbon_kg,
            "annual_energy_kwh": self.annual_energy_kwh,
            "quality_index": self.quality_index,
            "capacity_units": self.capacity_units,
        }
        out.update({f"layer_{k}": v for k, v in self.layers.items()})
        return out


# ---------------------------------------------------------------------------
# per-query costs
# ---------------------------------------------------------------------------

def self_hosted_query_cost(state: DeploymentState, workload: WorkloadClass,
                           grid: GridSpec, slo: Optional[SLO] = None,
                           respect_slo: bool = True) -> QueryCost:
    """Cost, energy and latency of one request on owned or rented accelerators.

    Accelerator-seconds are converted to money through the hourly rate,
    then divided by the target duty cycle: a replica provisioned for peak
    is paid for during the trough as well. Semantic-cache hits bypass
    inference entirely and are removed from the billable fraction.
    """
    if slo is not None and respect_slo and slo.enforce:
        perf = solve_batch_for_slo(state, workload, slo)
    else:
        perf = evaluate_request(state, workload, slo)

    cfg = state.serving
    if not math.isfinite(perf.accelerator_seconds):
        return QueryCost(
            workload=workload.name, architecture="self_hosted",
            compute_cost=math.inf, energy_joules=math.inf, carbon_kg=math.inf,
            water_litres=math.inf, latency_seconds=math.inf, slo_met=False,
            batch=0.0, detail={"infeasible": 1.0},
        )

    samples = float(workload.self_consistency_k)
    miss = 1.0 - (cfg.semantic_cache_hit if workload.cacheable else 0.0)

    accel_seconds = perf.accelerator_seconds * samples * miss
    duty = max(cfg.effective_utilisation, 1e-3)

    hourly = state.hardware.hourly_cost * cfg.reserved_discount
    raw_compute = accel_seconds / SECONDS_PER_HOUR * hourly / duty
    compute_cost = raw_compute * cfg.infra_overhead

    # Money and energy amortise idle time differently, because a rented
    # accelerator bills the same dollars whether it is working or waiting,
    # while its board does not draw the same watts. Versions up to 6.0
    # charged the idle share -- ``accel_seconds * (1/duty - 1)`` -- at
    # full load power, which is the one idle-draw figure known to be
    # wrong. The split below prices working seconds at load power and the
    # provisioned-but-idle share at the board's idle draw. Approximation,
    # stated: scheduler-inefficient gaps inside the live period are also
    # charged at idle power, which understates draw during ramp.
    busy_joules = state.hardware.power_watts * accel_seconds
    idle_joules = (state.hardware.resolved_idle_power
                   * accel_seconds * (1.0 / duty - 1.0))
    device_joules = busy_joules + idle_joules
    facility_joules = device_joules * grid.pue
    energy_kwh = facility_joules / JOULES_PER_KWH

    electricity = energy_kwh * grid.electricity_cost
    compute_cost += electricity

    return QueryCost(
        workload=workload.name,
        architecture="self_hosted",
        compute_cost=compute_cost,
        energy_joules=facility_joules,
        carbon_kg=energy_kwh * grid.carbon_intensity,
        water_litres=energy_kwh * grid.wue,
        latency_seconds=perf.latency,
        slo_met=perf.slo_met,
        batch=perf.batch,
        detail={
            "accelerator_seconds": accel_seconds,
            "throughput_qps": perf.throughput_qps,
            "ttft_s": perf.ttft,
            "tpot_s": perf.tpot,
            "decode_bound_by": perf.decode_bound_by == "memory",
            "bound": perf.decode_bound_by,
            "busy_facility_joules": busy_joules * grid.pue,
            "electricity_usd": electricity,
            "cache_miss_fraction": miss,
        },
    )


def api_query_cost(pricing: PricingSpec, workload: WorkloadClass,
                   grid: GridSpec, prefix_cache_hit: float = 0.0,
                   semantic_cache_hit: float = 0.0,
                   provider_energy_wh_per_ktok: float = 0.30) -> QueryCost:
    """Cost of one request against a commercial endpoint.

    The provider's energy figure is a disclosure-dependent estimate rather
    than a measurement, and is reported so that comparisons between hosted
    and self-hosted architectures are not silently carbon-blind. Treat it
    as an order-of-magnitude anchor and override it when a provider
    publishes audited figures.
    """
    samples = float(workload.self_consistency_k)
    miss = 1.0 - (semantic_cache_hit if workload.cacheable else 0.0)
    per_call = pricing.query_cost(workload.tokens_in, workload.tokens_out,
                                  cached_fraction=prefix_cache_hit)
    cost = per_call * samples * miss

    ktokens = (workload.tokens_in + workload.tokens_out) / 1000.0 * samples * miss
    energy_kwh = ktokens * provider_energy_wh_per_ktok / 1000.0
    joules = energy_kwh * JOULES_PER_KWH

    return QueryCost(
        workload=workload.name,
        architecture="api",
        compute_cost=cost,
        energy_joules=joules,
        carbon_kg=energy_kwh * grid.carbon_intensity,
        water_litres=energy_kwh * grid.wue,
        latency_seconds=float("nan"),
        slo_met=None,          # not modelled for a commercial endpoint
        batch=float("nan"),
        detail={"per_call_usd": per_call, "cache_miss_fraction": miss},
    )


# ---------------------------------------------------------------------------
# total cost of ownership
# ---------------------------------------------------------------------------

def _capacity_units(state: DeploymentState, workloads: Sequence[WorkloadClass],
                    volume: float, slo: Optional[SLO]) -> float:
    """Replicas needed to carry peak traffic, not average traffic."""
    if volume <= 0:
        return 0.0
    seconds_per_year = 365.25 * 24 * 3600
    total = 0.0
    for w in workloads:
        perf = (solve_batch_for_slo(state, w, slo) if slo and slo.enforce
                else evaluate_request(state, w, slo))
        if perf.throughput_qps <= 0:
            return math.inf
        class_qps = volume * w.share / seconds_per_year
        miss = 1.0 - (state.serving.semantic_cache_hit if w.cacheable else 0.0)
        class_qps *= miss * w.self_consistency_k
        total += class_qps / perf.throughput_qps
    return total / max(state.serving.effective_utilisation, 1e-3)


HOURS_PER_YEAR = 365.25 * 24


def replica_annual_cost(state: DeploymentState, grid: GridSpec) -> float:
    """Cost of owning one replica for a year, whether or not it is busy.

    A replica is the indivisible unit of self-hosted capacity: it is the
    set of accelerators that together hold one copy of the weights. You
    cannot rent 0.37 of it, and the fraction of the year it sits idle is
    still billed. Electricity is added when the grid carries a non-zero
    tariff, on the assumption that a bundled hourly rate already includes
    it.
    """
    cfg = state.serving
    hourly = state.hardware.hourly_cost * cfg.reserved_discount
    hardware_cost = (hourly * cfg.n_accelerators * HOURS_PER_YEAR
                     * cfg.infra_overhead)
    if grid.electricity_cost > 0:
        # Board draw follows load: the utilised share of the year at load
        # power, the remainder at idle power. Charging every hour at load
        # power -- as versions up to 6.0 did -- priced electricity the one
        # way an idle board is known not to draw it.
        duty = cfg.effective_utilisation
        mean_watts = (state.hardware.power_watts * duty
                      + state.hardware.resolved_idle_power * (1.0 - duty))
        kwh = (mean_watts * cfg.n_accelerators
               * HOURS_PER_YEAR * grid.pue / 1000.0)
        hardware_cost += kwh * grid.electricity_cost
    return hardware_cost


def _stepped_serving_cost(state: DeploymentState, grid: GridSpec,
                          capacity_units: float, continuous_cost: float) -> float:
    """Charge for whole replicas, never for fractions of one.

    Returns the greater of the integral-replica cost and the continuous
    estimate. The two agree when demand happens to fill an exact number
    of replicas and diverge everywhere else -- and it is that divergence,
    repeated at every capacity boundary, that turns a self-hosted cost
    curve into a staircase and lets a break-even scan find more than one
    crossing.
    """
    if not math.isfinite(capacity_units):
        return math.inf
    replicas = max(math.ceil(capacity_units - 1e-9), state.serving.min_replicas)
    return max(replicas * replica_annual_cost(state, grid), continuous_cost)


def _stepped_annual_energy(state: DeploymentState, grid: GridSpec,
                           capacity_units: float, busy_facility_joules: float,
                           busy_device_seconds: float) -> tuple:
    """Annual facility energy, carbon and water on the replica staircase.

    Working joules come from the per-query accounting; the remainder of
    every provisioned replica's year -- the idle share the dollar ledger
    has always billed -- draws idle power. Kept adjacent to
    :func:`_stepped_serving_cost` on purpose: the two are the same
    staircase read in different units, and an independent reference
    implementation of this layer is what exposed that only one of them
    was being stepped.
    """
    if not math.isfinite(capacity_units):
        return math.inf, math.inf, math.inf
    replicas = max(math.ceil(capacity_units - 1e-9),
                   state.serving.min_replicas)
    provisioned_seconds = (replicas * state.serving.n_accelerators
                           * HOURS_PER_YEAR * SECONDS_PER_HOUR)
    idle_seconds = max(provisioned_seconds - busy_device_seconds, 0.0)
    idle_joules = (idle_seconds * state.hardware.resolved_idle_power
                   * grid.pue)
    total_j = busy_facility_joules + idle_joules
    kwh = total_j / JOULES_PER_KWH
    return total_j, kwh * grid.carbon_intensity, kwh * grid.wue


def layer_volume_elasticity(base: "TCOResult",
                            evaluate: "Callable[[float], TCOResult]",
                            relative_step: float = 0.02) -> Dict[str, float]:
    """Measured ``d ln(cost) / d ln(volume)`` for each of the six layers.

    The six-layer taxonomy is the paper's central modelling claim, and
    until v8.0 it was stated in a docstring rather than checked against
    the code -- which is how the assurance row came to say "volume-free"
    about a layer that is 83-95% linear in every shipped scenario. This
    function derives the classification the way CAIDE derives efficiency
    multipliers: by moving the input and measuring.

    ``evaluate`` re-runs :func:`total_cost_of_ownership` at a given
    volume. A central difference in log space is used, so a linear layer
    returns 1.0, a fixed layer 0.0 and a sublinear layer its exponent.
    The stepped layer returns whatever the staircase is doing locally --
    0.0 inside a tread and a large number across a riser -- which is the
    honest answer for a discontinuous function and the reason the value
    is reported rather than asserted.
    """
    v = base.annual_volume
    if v <= 0:
        return {k: math.nan for k in SIX_LAYERS}
    lo, hi = evaluate(v * (1.0 - relative_step)), evaluate(v * (1.0 + relative_step))
    dlnv = math.log((1.0 + relative_step) / (1.0 - relative_step))
    out: Dict[str, float] = {}
    for k in SIX_LAYERS:
        a, b = lo.layers[k], hi.layers[k]
        if a <= 0 and b <= 0:
            out[k] = 0.0
        elif a <= 0 or b <= 0:
            out[k] = math.inf
        else:
            out[k] = (math.log(b) - math.log(a)) / dlnv
    return out


def total_cost_of_ownership(
    *,
    architecture: str,
    annual_volume: float,
    workloads: Sequence[WorkloadClass],
    grid: GridSpec,
    state: Optional[DeploymentState] = None,
    pricing: Optional[PricingSpec] = None,
    assurance: Optional[AssuranceProfile] = None,
    retrieval: Optional[CostLayer] = None,
    integration: Optional[CostLayer] = None,
    workforce: Optional[CostLayer] = None,
    slo: Optional[SLO] = None,
    year: int = 1,
    platform_engineering_annual: float = 0.0,
    quality_penalty: float = 0.0,
    # Energy a commercial provider spends per thousand tokens. Every API
    # carbon and water figure in the package comes from this one number,
    # which was a function default no caller overrode and no scenario
    # could set until v11.0 -- so the provenance digest could not cover it
    # and no sensitivity analysis could move it.
    provider_energy_wh_per_ktok: float = 0.30,
) -> TCOResult:
    """Assemble the six layers into one annual figure.

    ``architecture`` is ``"self_hosted"`` (requires ``state``) or ``"api"``
    (requires ``pricing``). Layers left as ``None`` contribute zero, which
    makes it possible to isolate compute economics during model
    development and then add governance once the picture is trusted.
    """
    shares = sum(w.share for w in workloads)
    if not math.isclose(shares, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"workload shares must sum to 1.0, got {shares:.6f}"
        )

    assurance = assurance or AssuranceProfile()
    layers = {k: 0.0 for k in SIX_LAYERS}
    violations: List[str] = []
    notes: List[str] = []

    quality_violations: List[str] = []
    unevaluated: List[str] = []
    blended_compute = 0.0
    annual_energy_j = 0.0
    annual_carbon = 0.0
    annual_water = 0.0
    capacity = 0.0
    quality = 1.0

    if architecture == "self_hosted":
        if state is None:
            raise ValueError("self_hosted architecture requires a DeploymentState")
        quality = state.model.quality_index * (1.0 + quality_penalty)
        busy_energy_j = 0.0
        busy_device_seconds = 0.0
        for w in workloads:
            qc = self_hosted_query_cost(state, w, grid, slo)
            # ``slo_met`` is never None on this branch: a modelled
            # deployment always yields a latency. The third state exists
            # for architectures whose latency is not modelled at all.
            if not qc.slo_met and w.latency_sensitive:
                violations.append(w.name)
            elif not qc.slo_met:
                # Declared latency-insensitive, so a miss is recorded and
                # not treated as disqualifying. v10.0 wired ``quality_floor``
                # into this path and left its declared sibling behind; the
                # v9 report had already named the two together.
                notes.append(
                    f"{w.name} misses the latency objective but is declared "
                    "latency-insensitive, so it does not rule the "
                    "architecture out")
            blended_compute += w.share * qc.compute_cost
            busy_energy_j += (w.share * annual_volume
                              * qc.detail.get("busy_facility_joules", 0.0))
            busy_device_seconds += (w.share * annual_volume
                                    * qc.detail.get("accelerator_seconds", 0.0))
        capacity = _capacity_units(state, workloads, annual_volume, slo)
        layers["compute_serving"] = _stepped_serving_cost(
            state, grid, capacity, blended_compute * annual_volume)
        layers["compute_serving"] += platform_engineering_annual
        # Energy walks the same staircase as dollars. Versions up to 6.0
        # summed per-query joules -- a continuous curve -- while charging
        # dollars for whole replicas, so below one full replica the ledger
        # billed the money of 1.0 replicas and the carbon of a fraction of
        # one. The idle remainder of every provisioned replica draws idle
        # power for the rest of the year, and that energy is real.
        annual_energy_j, annual_carbon, annual_water = _stepped_annual_energy(
            state, grid, capacity, busy_energy_j, busy_device_seconds)
    elif architecture == "api":
        if pricing is None:
            raise ValueError("api architecture requires a PricingSpec")
        quality = pricing.quality_index * (1.0 + quality_penalty)
        for w in workloads:
            qc = api_query_cost(
                pricing, w, grid,
                provider_energy_wh_per_ktok=provider_energy_wh_per_ktok)
            if slo is not None and qc.slo_met is None and w.latency_sensitive:
                unevaluated.append(w.name)
            blended_compute += w.share * qc.compute_cost
            annual_energy_j += w.share * annual_volume * qc.energy_joules
            annual_carbon += w.share * annual_volume * qc.carbon_kg
            annual_water += w.share * annual_volume * qc.water_litres
        layers["model_access"] = (blended_compute * annual_volume
                                  + pricing.monthly_platform_fee * 12)
        layers["compute_serving"] = platform_engineering_annual
    else:
        raise ValueError(
            f"unknown architecture {architecture!r}; "
            "expected 'self_hosted' or 'api'"
        )

    if retrieval is not None:
        layers["retrieval_data"] = retrieval.annual_cost(annual_volume, year)
    if integration is not None:
        layers["integration_sre"] = integration.annual_cost(annual_volume, year)
    if workforce is not None:
        layers["workforce_redesign"] = workforce.annual_cost(annual_volume, year)

    # A quality floor is a declared constraint, checked here against the
    # architecture's quality index exactly as routing checks it against a
    # tier's. Reported, never priced: CAIDE does not know what a capability
    # shortfall costs, and inventing a penalty would be worse than naming
    # the classes the architecture cannot serve.
    if unevaluated:
        notes.append(
            "latency objective not evaluated for "
            f"{', '.join(unevaluated)}: this architecture's latency is not "
            "modelled, so its admissibility rests on the quality floor "
            "alone and not on evidence about latency"
        )

    quality_violations = [w.name for w in workloads
                          if w.quality_floor > quality + 1e-12]
    quality_shortfall = {w.name: (w.quality_floor - quality) / w.quality_floor
                         for w in workloads
                         if w.quality_floor > quality + 1e-12}
    quality_margin = {w.name: (quality - w.quality_floor) / w.quality_floor
                      for w in workloads if w.quality_floor > 0}
    declared_floors = {w.name: w.quality_floor for w in workloads
                       if w.quality_floor > 0}
    if quality_violations:
        notes.append(
            "quality floor not met for "
            + ", ".join(f"{n} (short by {quality_shortfall[n]:.1%})"
                        for n in quality_violations)
            + f" (architecture index {quality:.3f}); "
            "this architecture is not a candidate for "
            "those classes, and a mixed workload needs routing rather "
            "than one architecture"
        )

    review = assurance.review_cost_per_query(workloads)
    layers["assurance_governance"] = (
        assurance.fixed_annual
        + (review + assurance.storage_per_query) * annual_volume
    )
    review_hours = assurance.review_hours_per_year(workloads, annual_volume)
    displaced = assurance.displaced_labour_per_query(workloads) * annual_volume

    if review > 0:
        notes.append(
            f"human review contributes ${review:.5f}/query "
            f"(${review * annual_volume:,.0f}/yr, {review_hours:,.0f} h, "
            f"{review_hours / 1700.0:.1f} FTE)"
        )
    if displaced > 0:
        notes.append(
            f"displaced human effort worth ${displaced:,.0f}/yr is reported "
            "separately and is not netted off the total"
        )

    return TCOResult(
        architecture=architecture,
        annual_volume=annual_volume,
        year=year,
        layers=layers,
        per_query_blended=blended_compute,
        compute_per_query=blended_compute,
        annual_carbon_kg=annual_carbon,
        annual_energy_kwh=annual_energy_j / JOULES_PER_KWH,
        annual_water_l=annual_water,
        quality_index=quality,
        slo_violations=violations,
        slo_unevaluated=unevaluated,
        quality_violations=quality_violations,
        quality_shortfall=quality_shortfall,
        quality_margin=quality_margin,
        _floors=declared_floors,
        capacity_units=capacity,
        notes=notes,
        displaced_labour_annual=displaced,
        review_hours_annual=review_hours,
        price_inelastic_per_query=review + assurance.storage_per_query,
    )
