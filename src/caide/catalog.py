"""Built-in reference catalogue.

Two kinds of number appear here and they carry very different epistemic
weight.

**Architectural and physical specifications** -- parameter counts, layer
counts, HBM capacity, memory bandwidth, board power -- are published by
vendors and model authors and change only when a new part or a new model
ships. They are reproduced here as documented constants.

**Prices** -- accelerator rental rates, token tariffs, electricity, staff
time -- move continuously and differ by region, contract and commitment.
They are supplied as *illustrative order-of-magnitude anchors so that
examples run out of the box*, and they are the first thing any serious
analysis should override. Every preset therefore carries a ``source``
and a ``price_epoch`` field, and :func:`stale_price_warning` reports how
old a figure is.

Nothing in CAIDE reads these presets implicitly. A scenario file that
does not name a preset does not get one.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Dict, List, Optional

from .specs import GridSpec, HardwareSpec, ModelSpec, PricingSpec

__all__ = [
    "MODELS",
    "HARDWARE",
    "PRICING",
    "GRIDS",
    "get_model",
    "get_hardware",
    "get_pricing",
    "get_grid",
    "PRICE_EPOCH",
    "stale_price_warning",
    "catalogue_summary",
    "hardware_source",
    "HardwareEntry",
]

#: Prices in this module were compiled at this date and are illustrative only.
PRICE_EPOCH = _dt.date(2026, 8, 1)


def stale_price_warning(today: Optional[_dt.date] = None) -> Optional[str]:
    """Return a warning string when the bundled price anchors are old."""
    today = today or _dt.date.today()
    months = (today.year - PRICE_EPOCH.year) * 12 + (today.month - PRICE_EPOCH.month)
    if months < 6:
        return None
    return (
        f"Bundled price anchors are ~{months} months old (epoch "
        f"{PRICE_EPOCH.isoformat()}). Override accelerator and token prices "
        "with current quotations before relying on absolute figures; "
        "relative comparisons and break-even structure remain valid."
    )


# ---------------------------------------------------------------------------
# model archetypes
# ---------------------------------------------------------------------------
# Shapes follow the conventions of published open-weight decoder-only
# transformers: GQA with 8 key/value groups above 7B, SwiGLU feed-forward
# with a ~3.5x expansion, and head dimension 128.

MODELS: Dict[str, ModelSpec] = {
    "dense-1b": ModelSpec(
        name="dense-1b", n_params_total=1.24e9, n_layers=16, d_model=2048,
        n_heads=16, n_kv_heads=8, max_context=131_072, quality_index=0.42,
    ),
    "dense-3b": ModelSpec(
        name="dense-3b", n_params_total=3.21e9, n_layers=28, d_model=3072,
        n_heads=24, n_kv_heads=8, max_context=131_072, quality_index=0.55,
    ),
    "dense-8b": ModelSpec(
        name="dense-8b", n_params_total=8.03e9, n_layers=32, d_model=4096,
        n_heads=32, n_kv_heads=8, max_context=131_072, quality_index=0.68,
    ),
    "dense-32b": ModelSpec(
        name="dense-32b", n_params_total=32.8e9, n_layers=64, d_model=5120,
        n_heads=40, n_kv_heads=8, max_context=131_072, quality_index=0.82,
    ),
    "dense-70b": ModelSpec(
        name="dense-70b", n_params_total=70.6e9, n_layers=80, d_model=8192,
        n_heads=64, n_kv_heads=8, max_context=131_072, quality_index=0.88,
    ),
    "dense-405b": ModelSpec(
        name="dense-405b", n_params_total=405.9e9, n_layers=126, d_model=16384,
        n_heads=128, n_kv_heads=8, max_context=131_072, quality_index=0.96,
    ),
    "moe-8x7b": ModelSpec(
        name="moe-8x7b", n_params_total=46.7e9, n_params_active=12.9e9,
        n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
        n_experts=8, experts_per_token=2, max_context=32_768, quality_index=0.75,
    ),
    "moe-8x22b": ModelSpec(
        name="moe-8x22b", n_params_total=141.0e9, n_params_active=39.0e9,
        n_layers=56, d_model=6144, n_heads=48, n_kv_heads=8,
        n_experts=8, experts_per_token=2, max_context=65_536, quality_index=0.89,
    ),
    "moe-236b": ModelSpec(
        name="moe-236b", n_params_total=236.0e9, n_params_active=21.0e9,
        n_layers=60, d_model=5120, n_heads=128, n_kv_heads=8,
        n_experts=160, experts_per_token=6, max_context=131_072, quality_index=0.91,
    ),
}


# ---------------------------------------------------------------------------
# accelerators
# ---------------------------------------------------------------------------
# peak_flops is dense bf16 without sparsity. memory_bandwidth and
# memory_bytes are vendor-published. hourly_cost is an illustrative
# on-demand cloud anchor and varies by more than 3x across providers,
# regions and commitment terms. idle_power_watts is the board's draw
# while provisioned but not serving; the figures here are estimates in
# the 10-20% band that published idle measurements cluster in, carried
# as explicit constants so that energy accounting never silently falls
# back to charging idle hours at load power.

@dataclass(frozen=True)
class HardwareEntry:
    spec: HardwareSpec
    source: str
    price_epoch: _dt.date = PRICE_EPOCH


_HW_ENTRIES: Dict[str, HardwareEntry] = {
    "a100-40gb": HardwareEntry(
        HardwareSpec(
            name="a100-40gb", peak_flops=312e12, memory_bytes=40 * 2**30,
            memory_bandwidth=1.555e12, power_watts=400, hourly_cost=1.10,
            interconnect_bandwidth=6.0e11, idle_power_watts=55,
            low_precision_speedup={"bf16": 1.0, "fp8": 1.0, "int8": 2.0, "int4": 2.0},
        ),
        source="NVIDIA A100 datasheet; no FP8 tensor cores",
    ),
    "a100-80gb": HardwareEntry(
        HardwareSpec(
            name="a100-80gb", peak_flops=312e12, memory_bytes=80 * 2**30,
            memory_bandwidth=2.039e12, power_watts=400, hourly_cost=1.60,
            interconnect_bandwidth=6.0e11, idle_power_watts=60,
            low_precision_speedup={"bf16": 1.0, "fp8": 1.0, "int8": 2.0, "int4": 2.0},
        ),
        source="NVIDIA A100 80GB datasheet",
    ),
    "h100-sxm": HardwareEntry(
        HardwareSpec(
            name="h100-sxm", peak_flops=989e12, memory_bytes=80 * 2**30,
            memory_bandwidth=3.35e12, power_watts=700, hourly_cost=3.20,
            interconnect_bandwidth=9.0e11, idle_power_watts=90,
            low_precision_speedup={"bf16": 1.0, "fp8": 2.0, "int8": 2.0, "int4": 2.0},
        ),
        source="NVIDIA H100 SXM datasheet, dense bf16",
    ),
    "h200-sxm": HardwareEntry(
        HardwareSpec(
            name="h200-sxm", peak_flops=989e12, memory_bytes=141 * 2**30,
            memory_bandwidth=4.8e12, power_watts=700, hourly_cost=4.00,
            interconnect_bandwidth=9.0e11, idle_power_watts=95,
            low_precision_speedup={"bf16": 1.0, "fp8": 2.0, "int8": 2.0, "int4": 2.0},
        ),
        source="NVIDIA H200 SXM datasheet",
    ),
    "l40s": HardwareEntry(
        HardwareSpec(
            name="l40s", peak_flops=362e12, memory_bytes=48 * 2**30,
            memory_bandwidth=8.64e11, power_watts=350, hourly_cost=1.00,
            interconnect_bandwidth=6.4e10, idle_power_watts=40,
            low_precision_speedup={"bf16": 1.0, "fp8": 2.0, "int8": 2.0, "int4": 2.0},
        ),
        source="NVIDIA L40S datasheet; PCIe-class interconnect",
    ),
    "consumer-24gb": HardwareEntry(
        HardwareSpec(
            name="consumer-24gb", peak_flops=165e12, memory_bytes=24 * 2**30,
            memory_bandwidth=1.008e12, power_watts=450, hourly_cost=0.35,
            interconnect_bandwidth=3.2e10, idle_power_watts=30,
            low_precision_speedup={"bf16": 1.0, "fp8": 1.6, "int8": 2.0, "int4": 2.0},
        ),
        source="Consumer 24GB class; on-premises amortised estimate",
    ),
}

HARDWARE: Dict[str, HardwareSpec] = {k: v.spec for k, v in _HW_ENTRIES.items()}


# ---------------------------------------------------------------------------
# commercial API tiers
# ---------------------------------------------------------------------------
# Deliberately generic. Vendor tariffs change often enough that naming
# products here would encode staleness into the software; capability
# tiers are stable and let a scenario say what it means.

PRICING: Dict[str, PricingSpec] = {
    "api-economy": PricingSpec(
        name="api-economy", input_per_mtok=0.15, output_per_mtok=0.60,
        cached_input_per_mtok=0.0375, quality_index=0.66,
    ),
    "api-midrange": PricingSpec(
        name="api-midrange", input_per_mtok=1.00, output_per_mtok=4.00,
        cached_input_per_mtok=0.25, quality_index=0.84,
    ),
    "api-frontier": PricingSpec(
        name="api-frontier", input_per_mtok=3.00, output_per_mtok=15.00,
        cached_input_per_mtok=0.75, quality_index=1.0,
    ),
    "api-frontier-premium": PricingSpec(
        name="api-frontier-premium", input_per_mtok=10.00, output_per_mtok=40.00,
        cached_input_per_mtok=2.50, quality_index=1.05,
    ),
}


# ---------------------------------------------------------------------------
# electricity grids
# ---------------------------------------------------------------------------
# carbon_intensity in kg CO2e/kWh, annual average operating margin.
# Regional averages hide large hourly variation; a deployment that can
# shift batch work in time sees a materially different figure.

GRIDS: Dict[str, GridSpec] = {
    "global-average": GridSpec("global-average", 0.436, pue=1.20,
                               electricity_cost=0.00, wue=1.8),
    "nordic-hydro": GridSpec("nordic-hydro", 0.030, pue=1.10,
                             electricity_cost=0.09, wue=0.3),
    "france": GridSpec("france", 0.056, pue=1.15, electricity_cost=0.19, wue=1.4),
    "eu-average": GridSpec("eu-average", 0.251, pue=1.16,
                           electricity_cost=0.22, wue=1.6),
    "us-average": GridSpec("us-average", 0.369, pue=1.15,
                           electricity_cost=0.13, wue=1.9),
    "us-west": GridSpec("us-west", 0.213, pue=1.14, electricity_cost=0.15, wue=2.2),
    "india": GridSpec("india", 0.713, pue=1.45, electricity_cost=0.09, wue=3.1),
    "china-average": GridSpec("china-average", 0.581, pue=1.35,
                              electricity_cost=0.08, wue=2.4),
    "coal-heavy": GridSpec("coal-heavy", 0.820, pue=1.50,
                           electricity_cost=0.07, wue=3.4),
}


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------

def _lookup(table: Dict[str, object], key: str, kind: str):
    if key not in table:
        raise KeyError(
            f"unknown {kind} {key!r}; available: {sorted(table)}"
        )
    return table[key]


def get_model(key: str) -> ModelSpec:
    """Look up a bundled model archetype by key; raises KeyError listing all keys."""
    return _lookup(MODELS, key, "model")            # type: ignore[return-value]


def get_hardware(key: str) -> HardwareSpec:
    """Look up a bundled accelerator by key. Specifications are vendor-published;
    the hourly cost is an illustrative anchor and should be overridden."""
    return _lookup(HARDWARE, key, "hardware")       # type: ignore[return-value]


def get_pricing(key: str) -> PricingSpec:
    """Look up a commercial pricing tier by key. Tariffs are illustrative anchors."""
    return _lookup(PRICING, key, "pricing tier")    # type: ignore[return-value]


def get_grid(key: str) -> GridSpec:
    """Look up an electricity grid by key. Intensities are annual regional averages
    and hide substantial hourly variation."""
    return _lookup(GRIDS, key, "grid")              # type: ignore[return-value]


def hardware_source(key: str) -> str:
    return _HW_ENTRIES[key].source


def catalogue_summary() -> Dict[str, List[str]]:
    """Every bundled preset key, grouped by kind. Backs ``caide catalog --json``."""
    return {
        "models": sorted(MODELS),
        "hardware": sorted(HARDWARE),
        "pricing": sorted(PRICING),
        "grids": sorted(GRIDS),
    }
