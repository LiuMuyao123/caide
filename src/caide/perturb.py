"""Public parameter perturbation for uncertainty analysis.

Monte Carlo over a deployment requires re-evaluating a scenario with some
inputs replaced by sampled values. In v1.0 that logic lived as a private
helper inside the command-line module, which meant the script that
reproduces the published results imported a underscore-prefixed function
from a UI layer -- an arrangement in which any refactor of the CLI would
silently break the reproducibility claim.

:func:`perturbed_cost` is that logic, promoted, documented and tested.

Recognised draw keys
--------------------
``volume_scale``           multiplies annual query volume
``tokens_in_scale``        multiplies every class's prompt length
``tokens_out_scale``       multiplies every class's generated length
``review_rate_scale``      multiplies every class's human-review rate
``review_minutes_scale``   multiplies every class's minutes per review
``reviewer_wage_scale``    multiplies the reviewer hourly cost
``accelerator_hourly``     replaces the accelerator hourly price (absolute)
``demand_duty_cycle``      replaces the share of the year with live traffic
``scheduler_efficiency``   replaces the useful-work share while traffic is live
``utilisation``            alias for ``demand_duty_cycle``, kept for v1 scenarios
``mbu``                    replaces achieved memory-bandwidth utilisation
``input_price_scale``      multiplies the API input tariff
``output_price_scale``     multiplies the API output tariff

Unknown keys are ignored rather than rejected, so a scenario may carry
documentation-only entries; :func:`unrecognised_draw_keys` reports them
when a caller wants to check.

Several substitutions are physically bounded -- a review rate cannot
exceed one, a duty cycle cannot exceed one. Clamping them is correct;
clamping them *silently* is not, because the distribution the scenario
declares is then not the distribution that propagates. In the shipped
education scenario the review-rate clamp binds in 71% of draws, on a
workload class carrying 20% of review cost, compressing the upper tail
of that cost by 9%. :func:`saturated_draw_keys` reports which keys hit a
bound on a given draw so that Monte Carlo runs can count it, on the same
principle that already governs failed draws: a constraint biting in a
third of draws is a finding, not a nuisance.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterable, List

from .costing import total_cost_of_ownership
from .specs import DeploymentState

if TYPE_CHECKING:                                        # pragma: no cover
    from .scenario import Architecture, Scenario

__all__ = ["perturbed_cost", "RECOGNISED_DRAW_KEYS", "REVIEW_COST_FACTORS",
           "unrecognised_draw_keys", "saturated_draw_keys",
           "uncovered_draw_keys"]

RECOGNISED_DRAW_KEYS = frozenset({
    "volume_scale", "tokens_in_scale", "tokens_out_scale", "review_rate_scale",
    "review_minutes_scale", "reviewer_wage_scale",
    "accelerator_hourly", "demand_duty_cycle", "scheduler_efficiency",
    "utilisation", "mbu", "input_price_scale", "output_price_scale",
})

#: Inputs whose product forms the per-query human-review cost, which is
#: the largest single term in every shipped scenario. Until v9.0 only the
#: first was perturbable, so the other two were point masses that never
#: appeared in a tornado chart -- a reader could not see they had been
#: held fixed. Adding them changes which input ranks first.
REVIEW_COST_FACTORS = ("review_rate_scale", "review_minutes_scale",
                       "reviewer_wage_scale")


def unrecognised_draw_keys(keys: Iterable[str]) -> List[str]:
    """Draw keys that :func:`perturbed_cost` will silently ignore."""
    return sorted(set(keys) - RECOGNISED_DRAW_KEYS)


def uncovered_draw_keys(declared: Iterable[str]) -> List[str]:
    """Perturbable inputs a scenario left without a distribution.

    A sensitivity ranking answers "of the inputs we varied, which
    mattered", and is silent about the inputs nobody varied. Those are
    implicit point masses: they cannot appear in a tornado chart, so
    their absence is invisible in exactly the output a reader would use
    to check for it. This function names them, so a report can say what
    was held fixed alongside what moved.
    """
    return sorted(RECOGNISED_DRAW_KEYS - set(declared))


def saturated_draw_keys(scenario: "Scenario", architecture: "Architecture",
                        draw: Dict[str, float]) -> List[str]:
    """Draw keys whose substitution hits a physical bound on this draw.

    A clamped input is not an error and not a failure: it is a declared
    distribution meeting a physical ceiling. Reporting it is what keeps
    the propagated uncertainty honest about being narrower than the
    declared uncertainty.
    """
    hit: List[str] = []
    rr = draw.get("review_rate_scale", 1.0)
    if rr != 1.0 and any(w.review_rate * rr > 1.0 for w in scenario.workloads):
        hit.append("review_rate_scale")
    duty = draw.get("demand_duty_cycle", draw.get("utilisation"))
    if duty is not None and not (0.01 <= duty <= 1.0):
        hit.append("demand_duty_cycle" if "demand_duty_cycle" in draw
                   else "utilisation")
    if "scheduler_efficiency" in draw and not (
            0.01 <= draw["scheduler_efficiency"] <= 1.0):
        hit.append("scheduler_efficiency")
    if "mbu" in draw and not (0.01 <= draw["mbu"] <= 1.0):
        hit.append("mbu")
    return hit


def perturbed_cost(scenario: "Scenario", architecture: "Architecture",
                   draw: Dict[str, float], year: int = 1) -> float:
    """Annual total cost of ownership with sampled inputs substituted.

    Parameters
    ----------
    scenario
        The baseline scenario; it is never mutated.
    architecture
        Which candidate to evaluate. Perturbations that do not apply to
        its kind are inert: ``accelerator_hourly`` does nothing to a
        commercial-endpoint architecture, and reading a near-zero
        sensitivity for it there is a statement about the architecture,
        not a finding about accelerator prices.
    draw
        One sample, mapping recognised keys to values.
    """
    volume = scenario.annual_volume * draw.get("volume_scale", 1.0)

    workloads = scenario.workloads
    ti = draw.get("tokens_in_scale", 1.0)
    to = draw.get("tokens_out_scale", 1.0)
    rr = draw.get("review_rate_scale", 1.0)
    rm = draw.get("review_minutes_scale", 1.0)
    if (ti, to, rr, rm) != (1.0, 1.0, 1.0, 1.0):
        workloads = [
            replace(w,
                    tokens_in=w.tokens_in * ti,
                    tokens_out=w.tokens_out * to,
                    review_rate=min(w.review_rate * rr, 1.0),
                    review_minutes=w.review_minutes * rm)
            for w in workloads
        ]

    assurance = scenario.assurance
    wage = draw.get("reviewer_wage_scale", 1.0)
    if wage != 1.0 and assurance is not None:
        assurance = replace(
            assurance,
            reviewer_hourly_cost=assurance.reviewer_hourly_cost * wage)

    state = architecture.state
    if state is not None:
        hardware = state.hardware
        if "accelerator_hourly" in draw:
            hardware = replace(hardware, hourly_cost=draw["accelerator_hourly"])

        updates: Dict[str, float] = {}
        duty = draw.get("demand_duty_cycle", draw.get("utilisation"))
        if duty is not None:
            updates["demand_duty_cycle"] = min(max(duty, 0.01), 1.0)
        if "scheduler_efficiency" in draw:
            updates["scheduler_efficiency"] = min(
                max(draw["scheduler_efficiency"], 0.01), 1.0)
        if "mbu" in draw:
            updates["mbu_decode"] = min(max(draw["mbu"], 0.01), 1.0)

        serving = replace(state.serving, **updates) if updates else state.serving
        state = DeploymentState(state.model, hardware, serving, state.notes)

    pricing = architecture.pricing
    if pricing is not None:
        pricing = replace(
            pricing,
            input_per_mtok=pricing.input_per_mtok
            * draw.get("input_price_scale", 1.0),
            output_per_mtok=pricing.output_per_mtok
            * draw.get("output_price_scale", 1.0),
        )

    return total_cost_of_ownership(
        architecture=architecture.kind,
        annual_volume=volume,
        workloads=workloads,
        grid=scenario.grid,
        state=state,
        pricing=pricing,
        assurance=assurance,
        retrieval=scenario.retrieval,
        integration=scenario.integration,
        workforce=scenario.workforce,
        slo=scenario.slo,
        year=year,
        platform_engineering_annual=architecture.platform_engineering_annual,
        quality_penalty=architecture.quality_penalty,
    ).total
