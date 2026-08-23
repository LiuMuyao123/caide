"""Declarative scenarios: the user-facing surface of CAIDE.

A scenario is a YAML document describing what is being served, to whom,
under what constraints, and against which alternatives. Everything the
analysis needs is in the file, which means a result can be reproduced by
sending one artefact to a colleague rather than a screenshot of a
spreadsheet.

Validation is strict and errors name the offending path. A cost model
that silently accepts ``tokens_in: -500`` or workload shares summing to
1.4 is worse than no model, because its output looks equally plausible.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from . import catalog
from .costing import AssuranceProfile, CostLayer, TCOResult, total_cost_of_ownership
from .efficiency import apply_stack, resolve_stack, stack_engineering_hours
from .scaling import ScalingAssumptions
from .specs import (
    SLO,
    implausible,
    DeploymentState,
    GridSpec,
    HardwareSpec,
    ModelSpec,
    PricingSpec,
    ServingConfig,
    WorkloadClass,
)
from .uncertainty import Distribution, lognormal, normal, point, triangular, uniform

__all__ = [
    "Architecture",
    "Scenario",
    "load_scenario",
    "ScenarioError",
    "example_scenario",
]


class ScenarioError(ValueError):
    """Raised when a scenario document is malformed."""


#: Suffixes that mark a string as a path rather than an inline document.
_SCENARIO_SUFFIXES = {".yaml", ".yml", ".json"}

#: Keys recognised at each level. Anything else is reported, because a
#: silently ignored key is the most expensive kind of typo here: mistyping
#: ``review_minutes`` as ``review_minuts`` zeroes a cost component that
#: dominates the total in every shipped example.
_KNOWN_KEYS = {
    "root": {"name", "description", "annual_volume", "grid", "slo", "workloads",
             "architectures", "layers", "assurance", "scaling", "uncertainty",
             "provider_energy_wh_per_ktok"},
    "workload": {"name", "share", "tokens_in", "tokens_out", "quality_floor",
                 "review_rate", "review_minutes", "baseline_minutes",
                 "self_consistency_k", "cacheable", "latency_sensitive"},
    "architecture": {"name", "type", "pricing", "model", "hardware", "serving",
                     "stack", "platform_engineering_annual",
                     "_applied_stack", "quality_penalty"},
    "layers": {"retrieval", "integration", "workforce"},
}

#: Fields renamed since v1.0. Reported with the replacement rather than
#: silently accepted, so a stale scenario fails loudly instead of quietly
#: reverting to a default.
_RENAMED_FIELDS = {
    "target_utilisation": (
        "serving.target_utilisation was split in v2.0 into "
        "'demand_duty_cycle' (share of the year with live traffic, a "
        "property of the workload) and 'scheduler_efficiency' (useful-work "
        "share while traffic is live, improved by continuous batching). "
        "Set demand_duty_cycle to your old value and leave "
        "scheduler_efficiency at its default"
    ),
}


def _check_keys(data: Dict[str, Any], kind: str, path: str,
                warnings: List[str]) -> None:
    known = _KNOWN_KEYS.get(kind, set())
    for key in data:
        if key in known:
            continue
        if key in _RENAMED_FIELDS:
            continue                      # handled with a hard error elsewhere
        suggestion = _closest(key, known)
        warnings.append(
            f"{path}: unrecognised field {key!r} was ignored"
            + (f" -- did you mean {suggestion!r}?" if suggestion else "")
        )


def _closest(key: str, options: Any) -> Optional[str]:
    import difflib
    matches = difflib.get_close_matches(key, sorted(options), n=1, cutoff=0.72)
    return matches[0] if matches else None


def _require(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ScenarioError(f"{path}: missing required field {key!r}")
    return data[key]


def _as_float(value: Any, path: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ScenarioError(f"{path}: expected a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ScenarioError(f"{path}: value must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class Architecture:
    """One candidate way of delivering the workload."""

    name: str
    kind: str                                   # "api" | "self_hosted"
    pricing: Optional[PricingSpec] = None
    state: Optional[DeploymentState] = None
    stack: tuple = ()
    platform_engineering_annual: float = 0.0
    quality_penalty_override: Optional[float] = None

    @property
    def quality_penalty(self) -> float:
        """Quality adjustment *beyond* what the applied stack already did.

        Since v11.0 the stack's quality cost lives where the stack's other
        effects live: in the deployment state, applied by
        :func:`caide.efficiency.apply_stack`. This property therefore
        returns zero unless a scenario declares an explicit override --
        an organisation's own evaluation of a fine-tune, say, which no
        technique catalogue can derive.

        Until v11.0 it re-derived the stack delta here and the caller
        multiplied it into a quality index that, for distillation, had
        already been reduced by the transform. One quantity, two ledgers,
        and every technique landed in exactly one of them or in both.
        """
        if self.quality_penalty_override is not None:
            return self.quality_penalty_override
        return 0.0

    @property
    def engineering_hours(self) -> float:
        return stack_engineering_hours(self.stack) if self.stack else 0.0

    def describe(self) -> Dict[str, Any]:
        base = {"name": self.name, "kind": self.kind,
                "stack": list(self.stack),
                "engineering_hours": self.engineering_hours}
        if self.kind == "api" and self.pricing is not None:
            base.update({"tier": self.pricing.name,
                         "input_per_mtok": self.pricing.input_per_mtok,
                         "output_per_mtok": self.pricing.output_per_mtok})
        if self.kind == "self_hosted" and self.state is not None:
            base.update({"model": self.state.model.name,
                         "hardware": self.state.hardware.name,
                         "accelerators": self.state.serving.n_accelerators})
        return base


@dataclass
class Scenario:
    """A complete, self-contained deployment analysis."""

    name: str
    annual_volume: float
    workloads: List[WorkloadClass]
    architectures: List[Architecture]
    grid: GridSpec
    slo: Optional[SLO] = None
    assurance: AssuranceProfile = field(default_factory=AssuranceProfile)
    retrieval: Optional[CostLayer] = None
    integration: Optional[CostLayer] = None
    workforce: Optional[CostLayer] = None
    scaling: Optional[ScalingAssumptions] = None
    uncertainty: Dict[str, Distribution] = field(default_factory=dict)
    #: Energy a commercial provider spends per thousand tokens, in watt
    #: hours. A disclosure-dependent estimate, not a measurement, and the
    #: sole determinant of every API carbon and water figure the package
    #: produces -- so it belongs in the scenario, where the digest and the
    #: uncertainty machinery can reach it, rather than in a function
    #: default that nothing overrode.
    provider_energy_wh_per_ktok: float = 0.30
    description: str = ""
    source_path: Optional[Path] = None
    parse_warnings: List[str] = field(default_factory=list)

    # -- validation -----------------------------------------------------

    def validate(self) -> List[str]:
        """Return human-readable warnings; hard errors raise at load time."""
        warnings: List[str] = []
        total_share = sum(w.share for w in self.workloads)
        if not math.isclose(total_share, 1.0, abs_tol=1e-6):
            raise ScenarioError(
                f"workload shares sum to {total_share:.4f}, expected 1.0"
            )
        if not self.architectures:
            raise ScenarioError("scenario defines no architectures to compare")

        warnings.extend(self.parse_warnings)

        stale = catalog.stale_price_warning()
        if stale:
            warnings.append(stale)

        for w in self.workloads:
            for key in ("self_consistency_k", "review_minutes",
                        "baseline_minutes", "tokens_in", "tokens_out"):
                msg = implausible(key, float(getattr(w, key)))
                if msg:
                    warnings.append(f"workload {w.name!r}: {msg}")
        for arch in self.architectures:
            if arch.state is None:
                continue
            for key, value in (("infra_overhead", arch.state.serving.infra_overhead),
                               ("n_accelerators",
                                arch.state.serving.n_accelerators)):
                msg = implausible(key, float(value))
                if msg:
                    warnings.append(f"{arch.name}: {msg}")
        for key, value in (("carbon_intensity", self.grid.carbon_intensity),
                           ("pue", self.grid.pue)):
            msg = implausible(key, float(value))
            if msg:
                warnings.append(f"grid {self.grid.name!r}: {msg}")

        hours = self.assurance.review_hours_per_year(self.workloads,
                                                     self.annual_volume)
        fte = hours / 1700.0
        if fte > 250.0:
            warnings.append(
                f"review workload implies {hours:,.0f} hours/yr "
                f"(~{fte:,.0f} FTE reviewers). Confirm that many reviewers "
                "exist before trusting the total -- an implausible review "
                "rate is the most common way a scenario silently becomes "
                "fiction, and it dominates every other cost when it does."
            )
        for w in self.workloads:
            if w.review_rate > 0 and w.review_minutes > 0 and w.baseline_minutes == 0:
                warnings.append(
                    f"workload {w.name!r} charges review time but declares no "
                    "baseline_minutes; if this task replaces existing manual "
                    "work, the analysis overstates its net labour cost"
                )
                break

        for arch in self.architectures:
            if arch.kind == "self_hosted" and arch.state is not None:
                model = arch.state.model
                hw = arch.state.hardware
                cfg = arch.state.serving
                capacity = hw.memory_bytes * cfg.n_accelerators * cfg.memory_utilisation
                if model.weight_bytes >= capacity:
                    warnings.append(
                        f"{arch.name}: {model.name} weights "
                        f"({model.weight_bytes / 2**30:.0f} GiB) do not fit in "
                        f"{cfg.n_accelerators}x {hw.name} "
                        f"({capacity / 2**30:.0f} GiB usable) -- "
                        "add accelerators or quantise"
                    )
                for w in self.workloads:
                    if w.tokens_in + w.tokens_out > model.max_context:
                        warnings.append(
                            f"{arch.name}: workload {w.name!r} needs "
                            f"{w.tokens_in + w.tokens_out:.0f} tokens but "
                            f"{model.name} supports {model.max_context}"
                        )
        return warnings

    # -- evaluation -----------------------------------------------------

    def evaluate(self, architecture: str | Architecture,
                 volume: Optional[float] = None, year: int = 1) -> TCOResult:
        arch = (architecture if isinstance(architecture, Architecture)
                else self.architecture(architecture))
        volume = self.annual_volume if volume is None else volume

        return total_cost_of_ownership(
            architecture=arch.kind,
            annual_volume=volume,
            workloads=self.workloads,
            grid=self.grid,
            state=arch.state,
            pricing=arch.pricing,
            assurance=self.assurance,
            retrieval=self.retrieval,
            integration=self.integration,
            workforce=self.workforce,
            slo=self.slo,
            year=year,
            platform_engineering_annual=arch.platform_engineering_annual,
            quality_penalty=arch.quality_penalty,
            provider_energy_wh_per_ktok=self.provider_energy_wh_per_ktok,
        )

    def evaluate_all(self, volume: Optional[float] = None,
                     year: int = 1) -> Dict[str, TCOResult]:
        return {a.name: self.evaluate(a, volume, year) for a in self.architectures}

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Round-trippable representation of this scenario.

        ``load_scenario(s.to_dict())`` reproduces ``s``. This exists so that
        a report can carry its own inputs: a digest proves the inputs did
        not change, but only the inputs themselves let the recipient re-run
        the analysis. A cost figure a colleague cannot regenerate is a
        screenshot, whatever provenance metadata surrounds it.
        """
        def _serving(cfg: ServingConfig) -> Dict[str, Any]:
            default = ServingConfig()
            out = {}
            # Every field, not a whitelist: the stack writes cache-hit and
            # speculative-decoding fields onto the state, and omitting them
            # made the round trip lossy in exactly the configurations where
            # the stack does the most work.
            for key in default.__dataclass_fields__:
                value = getattr(cfg, key)
                if value != getattr(default, key):
                    out[key] = value
            return out

        def _arch(a: Architecture) -> Dict[str, Any]:
            base: Dict[str, Any] = {"name": a.name, "type": a.kind}
            if a.platform_engineering_annual:
                base["platform_engineering_annual"] = a.platform_engineering_annual
            if a.kind == "api" and a.pricing is not None:
                base["pricing"] = {
                    "name": a.pricing.name,
                    "input_per_mtok": a.pricing.input_per_mtok,
                    "output_per_mtok": a.pricing.output_per_mtok,
                    "cached_input_per_mtok": a.pricing.cached_input_per_mtok,
                    "quality_index": a.pricing.quality_index,
                    "monthly_platform_fee": a.pricing.monthly_platform_fee,
                }
                return base
            if a.state is not None:
                m = a.state.model
                base["model"] = {
                    "name": m.name, "n_params_total": m.n_params_total,
                    "n_params_active": m.active_params, "n_layers": m.n_layers,
                    "d_model": m.d_model, "n_heads": m.n_heads,
                    "n_kv_heads": m.kv_heads, "n_experts": m.n_experts,
                    "experts_per_token": m.experts_per_token,
                    "bytes_per_param": m.bytes_per_param,
                    "bytes_per_kv_element": m.bytes_per_kv_element,
                    "max_context": m.max_context,
                    "quality_index": m.quality_index,
                }
                h = a.state.hardware
                base["hardware"] = {
                    "name": h.name, "peak_flops": h.peak_flops,
                    "memory_bytes": h.memory_bytes,
                    "memory_bandwidth": h.memory_bandwidth,
                    "power_watts": h.power_watts, "hourly_cost": h.hourly_cost,
                    "interconnect_bandwidth": h.interconnect_bandwidth,
                    # Serialised explicitly: dropping it here would let a
                    # round-tripped scenario fall back to the 15% estimate
                    # and disagree with itself by the difference between
                    # that estimate and the catalogue's measured value.
                    "idle_power_watts": h.idle_power_watts,
                }
                serving = _serving(a.state.serving)
                if serving:
                    base["serving"] = serving
            # The stack is already baked into the state above, so it is
            # recorded for the reader and deliberately not re-applied --
            # re-applying it would transform an already-transformed state.
            # Its quality cost is not recoverable from the state, so it is
            # carried explicitly.
            if a.stack:
                base["_applied_stack"] = list(a.stack)
                base["quality_penalty"] = a.quality_penalty
            return base

        doc: Dict[str, Any] = {
            "name": self.name,
            "annual_volume": self.annual_volume,
            "grid": {"name": self.grid.name,
                     "carbon_intensity": self.grid.carbon_intensity,
                     "pue": self.grid.pue,
                     "electricity_cost": self.grid.electricity_cost,
                     "wue": self.grid.wue},
            "workloads": [
                {"name": w.name, "share": w.share, "tokens_in": w.tokens_in,
                 "tokens_out": w.tokens_out, "quality_floor": w.quality_floor,
                 "review_rate": w.review_rate, "review_minutes": w.review_minutes,
                 "baseline_minutes": w.baseline_minutes,
                 "self_consistency_k": w.self_consistency_k,
                 "cacheable": w.cacheable,
                 "latency_sensitive": w.latency_sensitive}
                for w in self.workloads
            ],
            "architectures": [_arch(a) for a in self.architectures],
        }
        if self.description:
            doc["description"] = self.description
        if self.slo is not None:
            doc["slo"] = {"ttft_seconds": self.slo.ttft_seconds,
                          "tpot_seconds": self.slo.tpot_seconds,
                          "enforce": self.slo.enforce}
        layers = {}
        for key, layer in (("retrieval", self.retrieval),
                           ("integration", self.integration),
                           ("workforce", self.workforce)):
            if layer is not None:
                layers[key] = {
                    "fixed_annual": layer.fixed_annual,
                    "per_query": layer.per_query,
                    "sublinear_coefficient": layer.sublinear_coefficient,
                    "sublinear_exponent": layer.sublinear_exponent,
                    "step_size": layer.step_size, "step_cost": layer.step_cost,
                    "front_load_year1": layer.front_load_year1,
                    "decay": layer.decay,
                }
        if layers:
            doc["layers"] = layers
        a = self.assurance
        doc["assurance"] = {
            "audit_logging_annual": a.audit_logging_annual,
            "evaluation_annual": a.evaluation_annual,
            "red_team_annual": a.red_team_annual,
            "privacy_review_annual": a.privacy_review_annual,
            "incident_response_annual": a.incident_response_annual,
            "reviewer_hourly_cost": a.reviewer_hourly_cost,
            "storage_per_query": a.storage_per_query,
        }
        if self.scaling is not None:
            sc = self.scaling
            doc["scaling"] = {
                "annual_price_decline": sc.annual_price_decline,
                "price_elasticity": sc.price_elasticity,
                "autonomous_growth": sc.autonomous_growth,
                "horizon_years": sc.horizon_years,
                "capacity_ceiling": sc.capacity_ceiling,
                "fixed_annual_cost": sc.fixed_annual_cost,
            }
        return doc

    def to_yaml(self) -> str:
        """Serialise to YAML that :func:`load_scenario` accepts."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False,
                              default_flow_style=False, allow_unicode=True)

    def architecture(self, name: str) -> Architecture:
        for a in self.architectures:
            if a.name == name:
                return a
        raise KeyError(
            f"unknown architecture {name!r}; "
            f"defined: {[a.name for a in self.architectures]}"
        )

    def cost_curve(self, name: str, year: int = 1):
        arch = self.architecture(name)

        def _curve(volume: float) -> float:
            return self.evaluate(arch, volume, year).total

        return _curve

    def blended_tokens(self) -> Dict[str, float]:
        return {
            "tokens_in": sum(w.share * w.tokens_in for w in self.workloads),
            "tokens_out": sum(w.share * w.tokens_out for w in self.workloads),
        }


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _parse_workloads(items: Any, path: str,
                     warnings: Optional[List[str]] = None) -> List[WorkloadClass]:
    if not isinstance(items, list) or not items:
        raise ScenarioError(f"{path}: expected a non-empty list of workloads")
    warnings = warnings if warnings is not None else []
    out: List[WorkloadClass] = []
    for i, raw in enumerate(items):
        p = f"{path}[{i}]"
        if not isinstance(raw, dict):
            raise ScenarioError(f"{p}: expected a mapping")
        _check_keys(raw, "workload", p, warnings)
        try:
            out.append(WorkloadClass(
                name=str(_require(raw, "name", p)),
                share=_as_float(_require(raw, "share", p), f"{p}.share"),
                tokens_in=_as_float(_require(raw, "tokens_in", p), f"{p}.tokens_in"),
                tokens_out=_as_float(_require(raw, "tokens_out", p), f"{p}.tokens_out"),
                quality_floor=_as_float(raw.get("quality_floor", 0.0),
                                        f"{p}.quality_floor"),
                review_rate=_as_float(raw.get("review_rate", 0.0), f"{p}.review_rate"),
                review_minutes=_as_float(raw.get("review_minutes", 0.0),
                                         f"{p}.review_minutes"),
                baseline_minutes=_as_float(raw.get("baseline_minutes", 0.0),
                                           f"{p}.baseline_minutes"),
                self_consistency_k=int(raw.get("self_consistency_k", 1)),
                cacheable=bool(raw.get("cacheable", True)),
                latency_sensitive=bool(raw.get("latency_sensitive", True)),
            ))
        except ValueError as exc:
            raise ScenarioError(f"{p}: {exc}") from None
    return out


def _parse_model(raw: Any, path: str) -> ModelSpec:
    if isinstance(raw, str):
        try:
            return catalog.get_model(raw)
        except KeyError as exc:
            raise ScenarioError(f"{path}: {exc}") from None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a preset name or a mapping")
    base = raw.get("preset")
    spec = catalog.get_model(base) if base else None
    fields = {k: v for k, v in raw.items() if k != "preset"}
    if spec is None:
        try:
            return ModelSpec(**fields)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"{path}: {exc}") from None
    try:
        return replace(spec, **fields)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{path}: {exc}") from None


def _parse_hardware(raw: Any, path: str) -> HardwareSpec:
    if isinstance(raw, str):
        try:
            return catalog.get_hardware(raw)
        except KeyError as exc:
            raise ScenarioError(f"{path}: {exc}") from None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a preset name or a mapping")
    base = raw.get("preset")
    fields = {k: v for k, v in raw.items() if k != "preset"}
    try:
        if base:
            return replace(catalog.get_hardware(base), **fields)
        return HardwareSpec(**fields)
    except (TypeError, ValueError, KeyError) as exc:
        raise ScenarioError(f"{path}: {exc}") from None


def _parse_pricing(raw: Any, path: str) -> PricingSpec:
    if isinstance(raw, str):
        try:
            return catalog.get_pricing(raw)
        except KeyError as exc:
            raise ScenarioError(f"{path}: {exc}") from None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a tier name or a mapping")
    base = raw.get("preset")
    fields = {k: v for k, v in raw.items() if k != "preset"}
    try:
        if base:
            return replace(catalog.get_pricing(base), **fields)
        return PricingSpec(**fields)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{path}: {exc}") from None


def _parse_architectures(items: Any, path: str,
                         warnings: Optional[List[str]] = None) -> List[Architecture]:
    if not isinstance(items, list) or not items:
        raise ScenarioError(f"{path}: expected a non-empty list of architectures")
    warnings = warnings if warnings is not None else []
    out: List[Architecture] = []
    for i, raw in enumerate(items):
        p = f"{path}[{i}]"
        if not isinstance(raw, dict):
            raise ScenarioError(f"{p}: expected a mapping")
        _check_keys(raw, "architecture", p, warnings)
        name = str(_require(raw, "name", p))
        kind = str(raw.get("type", "self_hosted"))
        if kind not in {"api", "self_hosted"}:
            raise ScenarioError(
                f"{p}.type: expected 'api' or 'self_hosted', got {kind!r}"
            )

        stack_raw = raw.get("stack", ())
        try:
            stack = resolve_stack(stack_raw) if stack_raw else ()
        except KeyError as exc:
            raise ScenarioError(f"{p}.stack: {exc}") from None

        if kind == "api":
            pricing = _parse_pricing(_require(raw, "pricing", p), f"{p}.pricing")
            out.append(Architecture(
                name=name, kind=kind, pricing=pricing, stack=(),
                platform_engineering_annual=_as_float(
                    raw.get("platform_engineering_annual", 0.0),
                    f"{p}.platform_engineering_annual"),
            ))
            continue

        model = _parse_model(_require(raw, "model", p), f"{p}.model")
        hardware = _parse_hardware(_require(raw, "hardware", p), f"{p}.hardware")
        serving_raw = raw.get("serving", {}) or {}
        if not isinstance(serving_raw, dict):
            raise ScenarioError(f"{p}.serving: expected a mapping")
        for legacy, guidance in _RENAMED_FIELDS.items():
            if legacy in serving_raw:
                raise ScenarioError(f"{p}.serving: {guidance}")
        try:
            serving = ServingConfig(**serving_raw)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"{p}.serving: {exc}") from None

        state = DeploymentState(model, hardware, serving)
        if stack:
            try:
                state = apply_stack(state, stack)
            except (KeyError, ValueError) as exc:
                raise ScenarioError(f"{p}.stack: {exc}") from None

        qp = raw.get("quality_penalty")
        out.append(Architecture(
            name=name, kind=kind, state=state, stack=tuple(stack),
            platform_engineering_annual=_as_float(
                raw.get("platform_engineering_annual", 0.0),
                f"{p}.platform_engineering_annual"),
            quality_penalty_override=(_as_float(qp, f"{p}.quality_penalty")
                                      if qp is not None else None),
        ))
    return out


def _parse_layer(raw: Any, path: str, name: str) -> Optional[CostLayer]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a mapping")
    try:
        return CostLayer(name=name, **raw)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{path}: {exc}") from None


_DIST_BUILDERS = {
    "uniform": lambda n, k: uniform(n, _f(k, "low"), _f(k, "high")),
    "triangular": lambda n, k: triangular(n, _f(k, "low"), _f(k, "mode"),
                                          _f(k, "high")),
    "normal": lambda n, k: normal(n, _f(k, "mean"), _f(k, "sd"),
                                  clip_low=k.get("clip_low")),
    "lognormal": lambda n, k: lognormal(n, _f(k, "median"), _f(k, "sigma")),
    "point": lambda n, k: point(n, _f(k, "value")),
}


def _f(d: Dict[str, Any], key: str) -> float:
    if key not in d:
        raise ScenarioError(f"distribution missing field {key!r}")
    return float(d[key])


def _parse_uncertainty(raw: Any, path: str) -> Dict[str, Distribution]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a mapping of name -> distribution")
    out: Dict[str, Distribution] = {}
    for name, spec in raw.items():
        p = f"{path}.{name}"
        if not isinstance(spec, dict):
            raise ScenarioError(f"{p}: expected a mapping")
        kind = spec.get("kind", "lognormal")
        if kind not in _DIST_BUILDERS:
            raise ScenarioError(
                f"{p}.kind: unknown distribution {kind!r}; "
                f"available: {sorted(_DIST_BUILDERS)}"
            )
        params = {k: v for k, v in spec.items() if k != "kind"}
        try:
            out[name] = _DIST_BUILDERS[kind](name, params)
        except (ScenarioError, ValueError) as exc:
            raise ScenarioError(f"{p}: {exc}") from None
    return out


def load_scenario(source: Union[str, Path, Dict[str, Any]]) -> Scenario:
    """Load and validate a scenario from a YAML file, string, or mapping."""
    path: Optional[Path] = None
    if isinstance(source, dict):
        data = copy.deepcopy(source)
    else:
        text = str(source)
        # A YAML document and a filesystem path are both strings, so the
        # intent has to be inferred. A document contains newlines or is
        # long; a path does not and carries a scenario suffix. Getting this
        # wrong in the safe direction -- treating a mistyped path as a YAML
        # scalar -- produces "root must be a mapping", which sends the user
        # looking for a formatting problem that does not exist.
        looks_like_document = "\n" in text or len(text) > 4096
        if not looks_like_document:
            try:
                candidate = Path(text)
                exists = candidate.exists()
            except (OSError, ValueError):
                candidate, exists = None, False
            if exists and candidate is not None:
                path = candidate
                text = candidate.read_text(encoding="utf-8")
            elif candidate is not None and candidate.suffix.lower() in _SCENARIO_SUFFIXES:
                raise ScenarioError(
                    f"scenario file not found: {text!r}"
                    + (f" (resolved to {candidate.resolve()})"
                       if not candidate.is_absolute() else "")
                    + ". Check the path, or run 'caide examples --extract .' "
                      "to write the bundled example scenarios into the "
                      "current directory."
                )
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ScenarioError(f"invalid YAML: {exc}") from None

    if not isinstance(data, dict):
        raise ScenarioError(
            "scenario root must be a mapping (a YAML document with top-level "
            "keys such as 'name', 'annual_volume', 'workloads')"
        )

    parse_warnings: List[str] = []
    _check_keys(data, "root", "root", parse_warnings)

    grid_raw = data.get("grid", "global-average")
    if isinstance(grid_raw, str):
        try:
            grid = catalog.get_grid(grid_raw)
        except KeyError as exc:
            raise ScenarioError(f"grid: {exc}") from None
    elif isinstance(grid_raw, dict):
        try:
            grid = GridSpec(**grid_raw)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"grid: {exc}") from None
    else:
        raise ScenarioError("grid: expected a name or a mapping")

    slo_raw = data.get("slo")
    slo = None
    if slo_raw is not None:
        if not isinstance(slo_raw, dict):
            raise ScenarioError("slo: expected a mapping")
        try:
            slo = SLO(**slo_raw)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"slo: {exc}") from None

    assurance_raw = data.get("assurance") or {}
    if not isinstance(assurance_raw, dict):
        raise ScenarioError("assurance: expected a mapping")
    try:
        assurance = AssuranceProfile(**assurance_raw)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"assurance: {exc}") from None

    scaling_raw = data.get("scaling")
    scaling = None
    if scaling_raw is not None:
        if not isinstance(scaling_raw, dict):
            raise ScenarioError("scaling: expected a mapping")
        try:
            scaling = ScalingAssumptions(**scaling_raw)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"scaling: {exc}") from None

    layers = data.get("layers") or {}
    if not isinstance(layers, dict):
        raise ScenarioError("layers: expected a mapping")
    _check_keys(layers, "layers", "layers", parse_warnings)

    scenario = Scenario(
        name=str(data.get("name", "unnamed-scenario")),
        description=str(data.get("description", "")),
        annual_volume=_as_float(_require(data, "annual_volume", "root"),
                                "annual_volume"),
        workloads=_parse_workloads(_require(data, "workloads", "root"),
                                   "workloads", parse_warnings),
        architectures=_parse_architectures(
            _require(data, "architectures", "root"), "architectures",
            parse_warnings),
        grid=grid,
        slo=slo,
        assurance=assurance,
        retrieval=_parse_layer(layers.get("retrieval"), "layers.retrieval",
                               "retrieval_data"),
        integration=_parse_layer(layers.get("integration"), "layers.integration",
                                 "integration_sre"),
        workforce=_parse_layer(layers.get("workforce"), "layers.workforce",
                               "workforce_redesign"),
        scaling=scaling,
        uncertainty=_parse_uncertainty(data.get("uncertainty"), "uncertainty"),
        provider_energy_wh_per_ktok=_as_float(
            data.get("provider_energy_wh_per_ktok", 0.30),
            "provider_energy_wh_per_ktok"),
        source_path=path,
        parse_warnings=parse_warnings,
    )
    scenario.validate()
    return scenario


def example_scenario() -> Dict[str, Any]:
    """A minimal but complete scenario, used by ``caide init`` and the tests."""
    return {
        "name": "minimal-example",
        "description": "Two-architecture comparison with one workload class.",
        "grid": "us-average",
        "annual_volume": 5_000_000,
        "workloads": [
            {"name": "assistant_turn", "share": 1.0,
             "tokens_in": 1200, "tokens_out": 350},
        ],
        "slo": {"ttft_seconds": 2.0, "tpot_seconds": 0.05},
        "architectures": [
            {"name": "api-midrange", "type": "api", "pricing": "api-midrange"},
            {"name": "selfhost-8b", "type": "self_hosted",
             "model": "dense-8b", "hardware": "l40s",
             "serving": {"n_accelerators": 1, "max_batch": 128},
             "stack": "standard"},
        ],
    }
