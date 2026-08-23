"""Workload classification and model routing.

A platform that sends every request to one model either overspends on the
easy traffic or underserves the hard traffic. Routing splits a workload
mix across a tier ladder subject to a per-class quality floor, which
converts a single architecture decision into a portfolio decision.

The optimisation is small enough to solve exactly. Each workload class
must be served by some tier whose quality index meets its floor; among
the feasible tiers the cheapest is chosen. Because classes are
independent given the tier ladder, the greedy per-class choice *is* the
global optimum -- no search is required, and the result is auditable
rather than the output of a heuristic.

When a fixed cost is attached to standing up a tier (a self-hosted
replica has to exist before anything can be routed to it), independence
breaks and the problem becomes a set-cover: opening a tier for one class
makes it free for others. :func:`optimise_routing` handles this by
enumerating tier subsets, which is exact for the tier counts that occur
in practice (fewer than about fifteen).

Two things break the per-class independence further, and both were
silently absent until v9.0:

*Capacity limits.* ``Tier.max_share`` was declared from the first
release and never read. A caller capping a tier at 30% of traffic got a
plan routing 100% of it there, with no warning -- the same dangling-
parameter failure the package has now found four times in its own code.
It is honoured now, which costs the greedy optimality argument: with a
cap the assignment is a constrained one and the cheapest tier for a
class may already be full.

*Granular capacity.* A self-hosted tier does not cost
``share x volume x per_query``. Capacity arrives in whole replicas --
that is the package's central costing claim -- so a tier carrying 0.6%
of a replica's throughput still costs a whole replica. Routing priced it
marginally, understating a lightly-loaded self-hosted tier by 13x on the
shipped public-service scenario, and the CLI's ``--tier-fixed-cost``
defaulted to zero, so standing up a replica appeared free. ``Tier``
now accepts ``annual_cost_fn``, which prices the tier from what is
actually routed to it.

Where either applies, :func:`optimise_routing` solves the assignment
exactly by enumeration when the instance is small enough to allow it,
and says so in :attr:`RoutingPlan.notes` when it has to fall back.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .specs import WorkloadClass

__all__ = ["Tier", "RoutingPlan", "route_greedy", "optimise_routing"]


@dataclass(frozen=True)
class Tier:
    """One servable option in the ladder.

    ``cost_fn`` maps a workload class to its per-query cost on this tier,
    which lets the caller plug in either the roofline-derived self-hosted
    cost or a token tariff without this module knowing the difference.
    """

    name: str
    quality_index: float
    cost_fn: Callable[[WorkloadClass], float]
    annual_fixed_cost: float = 0.0
    #: Largest share of total traffic this tier may carry. Honoured
    #: since v9.0; before that it was accepted and ignored.
    max_share: float = 1.0
    latency_ok: bool = True
    #: Optional exact annual cost given the classes routed here and the
    #: annual volume. Supply it for tiers whose cost is not linear in
    #: what they serve -- a self-hosted tier billed in whole replicas is
    #: the case this exists for. When absent the tier is priced as
    #: ``per_query x share x volume + annual_fixed_cost``.
    annual_cost_fn: Optional[Callable[[Sequence[WorkloadClass], float],
                                      float]] = None

    @property
    def is_separable(self) -> bool:
        """True when this tier's cost is linear in what it is given."""
        return self.annual_cost_fn is None and self.max_share >= 1.0

    def cost(self, workload: WorkloadClass) -> float:
        try:
            value = self.cost_fn(workload)
        except Exception:                                # pragma: no cover
            return math.inf
        return value if math.isfinite(value) else math.inf

    def serves(self, workload: WorkloadClass) -> bool:
        if self.quality_index < workload.quality_floor:
            return False
        if workload.latency_sensitive and not self.latency_ok:
            return False
        return math.isfinite(self.cost(workload))


@dataclass
class RoutingPlan:
    """Assignment of workload classes to tiers, with its economics."""

    assignment: Dict[str, str]
    per_query_cost: float
    annual_cost: float
    annual_volume: float
    tiers_opened: Tuple[str, ...]
    blended_quality: float
    unroutable: List[str] = field(default_factory=list)
    detail: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Whether the reported assignment is provably optimal. False when
    #: the instance was too large for exact enumeration under capacity
    #: limits or non-separable tier costs, in which case a greedy plan is
    #: reported and the reason is in :attr:`notes`.
    exact: bool = True
    notes: List[str] = field(default_factory=list)
    #: Share of traffic each opened tier carries, for checking against
    #: the caps that produced it.
    tier_shares: Dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.unroutable

    def summary_rows(self) -> List[Dict[str, object]]:
        rows = []
        for name, tier in self.assignment.items():
            d = self.detail.get(name, {})
            rows.append({
                "workload": name,
                "tier": tier,
                "share": d.get("share", float("nan")),
                "usd_per_query": d.get("cost", float("nan")),
                "annual_usd": d.get("annual", float("nan")),
            })
        return rows


def _blended(workloads: Sequence[WorkloadClass],
             assignment: Dict[str, str],
             tiers: Dict[str, Tier]) -> Tuple[float, float, Dict]:
    per_query = 0.0
    quality = 0.0
    detail: Dict[str, Dict[str, float]] = {}
    for w in workloads:
        tier = tiers[assignment[w.name]]
        c = tier.cost(w)
        per_query += w.share * c
        quality += w.share * tier.quality_index
        detail[w.name] = {"share": w.share, "cost": c,
                          "quality": tier.quality_index}
    return per_query, quality, detail


def _tier_annual_cost(tier: Tier, served: Sequence[WorkloadClass],
                      annual_volume: float) -> float:
    """Annual cost of one tier given what is routed to it.

    ``annual_cost_fn`` wins when supplied, because a tier whose cost is a
    staircase cannot be described by a per-query price. Otherwise the
    separable form applies.
    """
    if not served:
        return 0.0
    if tier.annual_cost_fn is not None:
        return tier.annual_cost_fn(served, annual_volume)
    return (sum(w.share * tier.cost(w) for w in served) * annual_volume
            + tier.annual_fixed_cost)


def _plan_from_assignment(workloads: Sequence[WorkloadClass],
                          assignment: Dict[str, str],
                          table: Dict[str, Tier],
                          annual_volume: float,
                          unroutable: Sequence[str],
                          exact: bool,
                          notes: Sequence[str]) -> RoutingPlan:
    routed = [w for w in workloads if w.name in assignment]
    per_query, quality, detail = _blended(routed, assignment, table)
    opened = tuple(sorted(set(assignment.values())))

    shares: Dict[str, float] = {}
    by_tier: Dict[str, List[WorkloadClass]] = {}
    for w in routed:
        by_tier.setdefault(assignment[w.name], []).append(w)
        shares[assignment[w.name]] = shares.get(assignment[w.name], 0.0) + w.share

    annual = sum(_tier_annual_cost(table[name], served, annual_volume)
                 for name, served in by_tier.items())

    for w in routed:
        detail[w.name]["annual"] = detail[w.name]["cost"] * w.share * annual_volume

    extra = list(notes)
    marginal = per_query * annual_volume + sum(
        table[n].annual_fixed_cost for n in opened)
    if annual > marginal * 1.05:
        extra.append(
            f"tier costs charged from actual capacity are "
            f"{annual / marginal:.1f}x the marginal estimate; a lightly "
            f"loaded self-hosted tier still costs whole replicas"
        )

    return RoutingPlan(
        assignment=assignment,
        per_query_cost=per_query,
        annual_cost=annual,
        annual_volume=annual_volume,
        tiers_opened=opened,
        blended_quality=quality,
        unroutable=list(unroutable),
        detail=detail,
        exact=exact,
        notes=extra,
        tier_shares=shares,
    )


def route_greedy(workloads: Sequence[WorkloadClass],
                 tiers: Sequence[Tier],
                 annual_volume: float) -> RoutingPlan:
    """Cheapest feasible tier per class, ignoring tier fixed costs.

    Optimal when every tier is separable (no capacity cap, no
    ``annual_cost_fn``) *and* fixed costs are ignored, which is what this
    function is for. Capacity caps are respected by filling classes in
    descending share order; that is a heuristic, and
    :func:`optimise_routing` replaces it with exact enumeration whenever
    the instance is small enough.
    """
    table = {t.name: t for t in tiers}
    assignment: Dict[str, str] = {}
    unroutable: List[str] = []
    notes: List[str] = []
    used: Dict[str, float] = {t.name: 0.0 for t in tiers}
    capped = any(t.max_share < 1.0 for t in tiers)

    order = sorted(workloads, key=lambda w: -w.share) if capped else workloads
    for w in order:
        feasible = [t for t in tiers if t.serves(w)
                    and used[t.name] + w.share <= t.max_share + 1e-12]
        if not feasible:
            if any(t.serves(w) for t in tiers):
                notes.append(
                    f"{w.name} could not be placed: every tier meeting its "
                    "quality floor is at its max_share cap"
                )
            unroutable.append(w.name)
            continue
        chosen = min(feasible, key=lambda t: t.cost(w))
        assignment[w.name] = chosen.name
        used[chosen.name] += w.share

    return _plan_from_assignment(workloads, assignment, table, annual_volume,
                                 unroutable, exact=not capped, notes=notes)


#: Largest number of (class, tier) enumerations attempted before falling
#: back to the greedy heuristic.
_MAX_ENUMERATION = 2_000_000


def optimise_routing(workloads: Sequence[WorkloadClass],
                     tiers: Sequence[Tier],
                     annual_volume: float,
                     max_tiers: Optional[int] = None) -> RoutingPlan:
    """Exact minimum-cost routing including the fixed cost of opening tiers.

    When every tier is separable, subset enumeration with a greedy
    assignment inside each subset is exact: with the open set fixed the
    fixed costs are sunk and each class independently takes its cheapest
    feasible tier.

    A capacity cap or a non-separable ``annual_cost_fn`` destroys that
    independence -- what one class takes changes what another may take,
    or what the tier costs. The assignment is then enumerated in full,
    which is exact and affordable at the sizes deployment ladders
    actually reach. Beyond :data:`_MAX_ENUMERATION` combinations the
    greedy plan is returned with ``exact=False`` and a note, rather than
    a heuristic answer presented as an optimum.
    """
    tiers = list(tiers)
    if not tiers:
        raise ValueError("at least one tier is required")
    if len(tiers) > 20:
        raise ValueError(
            f"exact enumeration is capped at 20 tiers, got {len(tiers)}; "
            "use route_greedy for larger ladders"
        )

    table = {t.name: t for t in tiers}
    separable = all(t.is_separable for t in tiers)

    if separable:
        best: Optional[RoutingPlan] = None
        limit = max_tiers or len(tiers)
        for size in range(1, limit + 1):
            for subset in itertools.combinations(tiers, size):
                plan = route_greedy(workloads, subset, annual_volume)
                if not plan.feasible:
                    continue
                if best is None or plan.annual_cost < best.annual_cost:
                    best = plan
        if best is None:
            return route_greedy(workloads, tiers, annual_volume)
        return best

    # --- non-separable: enumerate assignments -----------------------------
    options: List[List[Tier]] = []
    unroutable: List[str] = []
    routable: List[WorkloadClass] = []
    for w in workloads:
        feasible = [t for t in tiers if t.serves(w)]
        if not feasible:
            unroutable.append(w.name)
            continue
        routable.append(w)
        options.append(feasible)

    combinations = 1
    for opts in options:
        combinations *= len(opts)
        if combinations > _MAX_ENUMERATION:
            break
    if combinations > _MAX_ENUMERATION:
        plan = route_greedy(workloads, tiers, annual_volume)
        plan.exact = False
        plan.notes.append(
            f"{combinations:,}+ assignments exceed the enumeration budget; "
            "reporting the greedy plan, which is not proven optimal under "
            "capacity caps or non-separable tier costs"
        )
        return plan

    best_plan: Optional[RoutingPlan] = None
    for combo in itertools.product(*options) if options else [()]:
        used: Dict[str, float] = {}
        ok = True
        for w, tier in zip(routable, combo):
            used[tier.name] = used.get(tier.name, 0.0) + w.share
            if used[tier.name] > tier.max_share + 1e-12:
                ok = False
                break
        if not ok:
            continue
        assignment = {w.name: t.name for w, t in zip(routable, combo)}
        if max_tiers is not None and len(set(assignment.values())) > max_tiers:
            continue
        plan = _plan_from_assignment(workloads, assignment, table,
                                     annual_volume, unroutable, exact=True,
                                     notes=())
        if best_plan is None or plan.annual_cost < best_plan.annual_cost:
            best_plan = plan

    if best_plan is None:
        plan = route_greedy(workloads, tiers, annual_volume)
        plan.notes.append(
            "no assignment satisfies every max_share cap; reporting the "
            "greedy plan with the classes it could not place"
        )
        plan.exact = False
        return plan
    return best_plan
