"""CAIDE -- Cost-Aware Inference Deployment Evaluator.

A deployment planning toolkit for large language model services. CAIDE
turns a declarative description of a workload, a model, an accelerator
and a governance regime into per-query cost, energy and carbon, an
annual total cost of ownership decomposed into six layers, break-even
volumes against alternative architectures, and the sensitivity of all of
it to the inputs that are least well known.

Quick start
-----------
>>> from caide import load_scenario, example_scenario
>>> scenario = load_scenario(example_scenario())
>>> results = scenario.evaluate_all()
>>> cheapest = min(results, key=lambda k: results[k].total)

Command line
------------
``caide run scenario.yaml --out report/``      full analysis
``caide breakeven scenario.yaml -a A -b B``    architecture crossover
``caide sweep scenario.yaml --technique int4`` regime scan
``caide catalog``                              list built-in presets

The design commitment is that no number is asserted without its
derivation being inspectable: every cost traces back to a roofline
evaluation, and every efficiency multiplier is measured on the
transformed physical state rather than supplied as a constant.
"""

from __future__ import annotations

__version__ = "17.1.0"
__author__ = "CAIDE contributors"
__license__ = "MIT"

from .calibration import (
    CONVENTIONS,
    EXCLUDED_OBSERVATIONS,
    CalibrationResult,
    Observation,
    REFERENCE_OBSERVATIONS,
    fit as fit_calibration,
    predicted_output_tps,
)
from .breakeven import (
    BreakEvenResult,
    Crossing,
    dominance_intervals,
    find_break_even,
)
from .catalog import (
    GRIDS,
    HARDWARE,
    MODELS,
    PRICING,
    catalogue_summary,
    get_grid,
    get_hardware,
    get_model,
    get_pricing,
)
from .costing import (
    AssuranceProfile,
    CostLayer,
    QueryCost,
    SIX_LAYERS,
    TCOResult,
    api_query_cost,
    self_hosted_query_cost,
    total_cost_of_ownership,
    layer_volume_elasticity,
)
from .efficiency import (
    PRESET_STACKS,
    TECHNIQUES,
    Technique,
    apply_stack,
    available_techniques,
    get_technique,
    stack_engineering_hours,
    stack_quality_delta,
)
from .roofline import (
    PhasePerformance,
    capacity_batch,
    evaluate_request,
    prefill_flops,
    solve_batch_for_slo,
    uniform_routing_imbalance,
)
from .routing import RoutingPlan, Tier, optimise_routing, route_greedy
from .scaling import (
    ScalingAssumptions,
    ScalingProjection,
    estimate_elasticity,
    project,
)
from .perturb import (RECOGNISED_DRAW_KEYS, REVIEW_COST_FACTORS,
                      perturbed_cost, saturated_draw_keys,
                      uncovered_draw_keys, unrecognised_draw_keys)
from .scenario import (
    Architecture,
    Scenario,
    ScenarioError,
    example_scenario,
    load_scenario,
)
from .specs import (
    SLO,
    DeploymentState,
    GridSpec,
    HardwareSpec,
    ModelSpec,
    PricingSpec,
    ServingConfig,
    WorkloadClass,
)
from .uncertainty import (
    Distribution,
    MonteCarloResult,
    SensitivityEntry,
    lognormal,
    monte_carlo,
    normal,
    point,
    sensitivity,
    triangular,
    uniform,
)

__all__ = [
    "__version__",
    # specs
    "ModelSpec", "HardwareSpec", "ServingConfig", "WorkloadClass", "SLO",
    "PricingSpec", "GridSpec", "DeploymentState",
    # roofline
    "PhasePerformance", "evaluate_request", "solve_batch_for_slo",
    "uniform_routing_imbalance",
    "capacity_batch", "prefill_flops",
    # efficiency
    "Technique", "TECHNIQUES", "PRESET_STACKS", "apply_stack",
    "get_technique", "available_techniques", "stack_quality_delta",
    "stack_engineering_hours",
    # costing
    "CostLayer", "AssuranceProfile", "QueryCost", "TCOResult", "SIX_LAYERS",
    "self_hosted_query_cost", "api_query_cost", "total_cost_of_ownership",
    "layer_volume_elasticity",
    # routing
    "Tier", "RoutingPlan", "route_greedy", "optimise_routing",
    # break-even
    "Crossing", "BreakEvenResult", "find_break_even", "dominance_intervals",
    # scaling
    "ScalingAssumptions", "ScalingProjection", "project", "estimate_elasticity",
    # uncertainty
    "Distribution", "MonteCarloResult", "SensitivityEntry", "monte_carlo",
    "sensitivity", "uniform", "triangular", "normal", "lognormal", "point",
    # scenario
    "Scenario", "Architecture", "ScenarioError", "load_scenario",
    "example_scenario",
    # perturbation
    "perturbed_cost", "RECOGNISED_DRAW_KEYS", "unrecognised_draw_keys",
    "saturated_draw_keys", "uncovered_draw_keys",
    "REVIEW_COST_FACTORS",
    # calibration
    "Observation", "CalibrationResult", "fit_calibration",
    "predicted_output_tps", "REFERENCE_OBSERVATIONS", "CONVENTIONS",
    "EXCLUDED_OBSERVATIONS",
    # catalogue
    "MODELS", "HARDWARE", "PRICING", "GRIDS", "get_model", "get_hardware",
    "get_pricing", "get_grid", "catalogue_summary",
]
