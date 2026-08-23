# CAIDE User Manual

**Version 17.1.0** · Cost-, energy- and carbon-aware deployment planning for large language model services

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Functional Modules](#4-functional-modules)
5. [API Reference](#5-api-reference)
6. [Operation Guide](#6-operation-guide)
7. [Command-Line Interface](#7-command-line-interface)
8. [Validation and Accuracy](#8-validation-and-accuracy)
9. [Troubleshooting](#9-troubleshooting)
10. [Support and Version Information](#10-support-and-version-information)
11. [Appendix](#11-appendix)

---

## 1. Introduction

### 1.1 Overview

CAIDE answers four questions you should be able to answer before you deploy a large language model service:

- **What will it cost?** — per-query cost, energy and carbon, and an annual total.
- **Where does the money actually go?** — the total decomposed into six layers that scale differently with volume.
- **At what volume does the cheaper architecture change?** — every break-even point between candidate architectures.
- **How much of the answer is guesswork?** — a sensitivity ranking that says which unknown input is worth measuring next.

The distinctive design commitment is that **no cost multiplier is ever supplied as an input**. Analyses of LLM serving cost are usually assembled from published constants — "4-bit quantisation, 0.65×", "speculative decoding, 0.40×" — multiplied together and applied to a token price. Those constants are correct at one operating point and wrong everywhere else, and they cannot express how two techniques interact. CAIDE instead models each technique as a *function from deployment state to deployment state* that edits the underlying physics (bytes per parameter, usable memory, achievable batch, decode steps), then re-runs a roofline model on the transformed state. The efficiency multiplier is *measured*, not asserted, and comes out different at every operating point.

### 1.2 Key Features

- **Roofline-derived efficiency** — prefill (compute-bound) and decode (memory-bandwidth-bound) modelled separately, with GQA-aware KV cache, mixture-of-experts expert-touch probability, tensor-parallel communication loss, and an M/D/1 queue for time-to-first-token.
- **Six-layer total cost of ownership** — model access, compute and serving, retrieval and data, integration and SRE, assurance and governance, and workforce and redesign — each with its own scaling law, so fixed layers are not amortised away and step-shaped capacity is not linearised.
- **Break-even that admits indecision** — self-hosted capacity is a staircase and API cost is a line, so they can cross many times; CAIDE reports every crossing and marks the windows where cost does not decide within a tolerance.
- **Uncertainty, not point estimates** — Monte Carlo propagation with Spearman rank sensitivity, reporting which input drives variance.
- **Traffic routing** — assign each workload class to the cheapest model tier that still meets its quality floor.
- **Scaling dynamics** — a closed-form demand-response projection that makes the price-elasticity assumption explicit and can surface the Jevons regime, where falling unit cost raises total spend.
- **Calibration** — fit the two utilisation constants to measurements from your own hardware and stack.
- **Reproducible reports** — a Markdown report, a CSV of results, and a single-file self-contained HTML dashboard, each carrying the scenario, a digest and the random seed.

### 1.3 Technical Architecture

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.9 |
| Numerical core | NumPy |
| Scenario format | YAML (PyYAML) |
| Figures | Matplotlib (Agg backend; no display required) |
| Interfaces | `caide` command-line tool and a Python API |
| Packaging | setuptools / PEP 621 (`pyproject.toml`) |
| Continuous integration | GitHub Actions — Ubuntu, macOS, Windows × Python 3.9, 3.11, 3.13 |
| Tests | 487 tests, ~90% line coverage |
| License | MIT |

No GPU is required: CAIDE *models* accelerators, it does not use them.

### 1.4 Two Ways to Run an Analysis

|  | Intended for | How to run |
|---|---|---|
| **`caide` command line** | Anyone comfortable at a terminal; reproducible reports | `caide run scenario.yaml --out report/` |
| **`caide` Python API** | Scripted or programmatic analysis | `from caide import load_scenario` |

The two share one engine and give identical numbers. Use the CLI for a self-contained report you can archive or hand to a reviewer; use the Python API to script analyses across many scenarios, embed CAIDE in a larger tool, or reach the lower-level roofline and costing functions directly.

---

## 2. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.13 |
| Operating system | Linux, macOS, Windows | any of the three (all CI-tested) |
| Memory | 1 GB | 4 GB |
| Disk | 100 MB | 250 MB including figures |

**Required dependencies** (installed automatically):

| Package | Minimum version |
|---|---|
| numpy | 1.22 |
| PyYAML | 6.0 |
| matplotlib | 3.5 |

**Optional dependency group:**

| Group | Packages | Needed for |
|---|---|---|
| `dev` | pytest ≥ 7.0, pytest-cov ≥ 4.0 | running the test suite with coverage |

No GPU, no network access and no external services are required. All computation is local and deterministic given a seed.

---

## 3. Installation

### 3.1 From PyPI

```bash
pip install caide
```

### 3.2 From Source

```bash
git clone https://github.com/LiuMuyao123/caide
cd caide
pip install -e ".[dev]"
```

### 3.3 Verify the Installation

```bash
python -c "import caide; print(caide.__version__)"
# 17.1.0

caide --version
# caide 17.1.0

pytest -q --cov=caide
```

Then run a bundled scenario end to end:

```bash
caide examples --extract .          # write the bundled scenarios to ./examples/
caide run examples/university_tutoring.yaml --layers --out report/
```

If `report/` contains a Markdown report, a CSV and a single-file `report.html`, the installation is complete.

### 3.4 Windows Note

The test suite reads UTF-8 files. On Windows, Python defaults to the legacy code page for file I/O; the CI job sets `PYTHONUTF8=1` for consistency, and the shipped tests specify UTF-8 explicitly so a bare `pytest` also passes on Windows.

---

## 4. Functional Modules

| Module | Purpose |
|---|---|
| `caide.specs` | Immutable descriptions of a model, accelerator, serving config, workload, SLO, pricing and grid, plus the combined `DeploymentState` |
| `caide.roofline` | Prefill/decode performance model, batch-for-SLO solver, capacity |
| `caide.efficiency` | Serving techniques as state transformations, and preset stacks |
| `caide.costing` | Per-query cost and the six-layer total cost of ownership |
| `caide.routing` | Assign workload classes to the cheapest quality-adequate tier |
| `caide.breakeven` | Locate every crossing between two cost curves; dominance windows |
| `caide.scaling` | Closed-form demand-response projection over a horizon |
| `caide.uncertainty` | Distributions, Monte Carlo propagation, rank-based sensitivity |
| `caide.perturb` | One-at-a-time perturbation of a scenario's cost inputs |
| `caide.calibration` | Fit utilisation constants to your own throughput measurements |
| `caide.catalog` | Built-in models, accelerators, price tiers and grids |
| `caide.scenario` | Load, validate and evaluate a declarative YAML scenario |
| `caide.report` | Markdown, CSV and single-file HTML reports |

### 4.1 `caide.specs` — Specifications

The vocabulary of an analysis. All are frozen dataclasses.

- **`ModelSpec`** — parameter count, layers, `d_model`, attention heads (with optional grouped-query `n_kv_heads`), mixture-of-experts fields (`n_experts`, `experts_per_token`, `n_params_active`), precision (`bytes_per_param`), and an ordinal `quality_index`.
- **`HardwareSpec`** — peak FLOP/s, HBM capacity and bandwidth, board power, hourly cost, interconnect bandwidth and idle power.
- **`ServingConfig`** — how the model is served: accelerator count, batch cap, the two utilisation quantities (`demand_duty_cycle` and `scheduler_efficiency`), precision, speculative-decoding parameters, caching hit-rates and infrastructure overhead.
- **`WorkloadClass`** — one class of traffic: its `share`, input/output token counts, `quality_floor`, human-review parameters, self-consistency `k`, and whether it is `latency_sensitive`.
- **`SLO`** — time-to-first-token and time-per-output-token targets, and whether they are enforced.
- **`PricingSpec`**, **`GridSpec`** — a commercial endpoint's tariff, and an electricity grid's carbon intensity, PUE, price and water-use effectiveness.
- **`DeploymentState`** — a `ModelSpec` + `HardwareSpec` + `ServingConfig` bundled together; the object every efficiency technique transforms.

### 4.2 `caide.roofline` — Inference Performance

Prefill and decode are bound by different resources and modelled separately.

- **`prefill_flops`** — `2·N_active·T_in` plus the causal-attention term `2·L·T_in²·d`, negligible for short prompts and dominant for retrieval-augmented ones.
- **`evaluate_request`** — full per-request performance at a given or solved batch: time-to-first-token, time-per-output-token, throughput, and which resource binds.
- **`solve_batch_for_slo`** — bisects for the *largest* batch that still meets the latency SLO, rather than assuming the scheduler cap is reachable. Throughput rises and latency degrades monotonically with batch, so the feasible set is an interval and the boundary is exact.
- **`capacity_batch`** — the batch a given memory budget admits once weights and KV cache are accounted for.
- **`uniform_routing_imbalance`** — the expected fraction of experts touched by a batch, which is why an MoE model loses its bandwidth advantage exactly when batching would help.

### 4.3 `caide.efficiency` — Serving Techniques

Each technique is a `Technique` that rewrites a `DeploymentState`; the cost effect falls out of re-running the roofline on the result.

- **`available_techniques`**, **`get_technique`** — the catalogue of techniques (quantisation, paged/flash attention, continuous and chunked batching, KV-cache compression, GQA, prefix and semantic caching, speculative decoding, distillation, committed-use discounts).
- **`apply_stack`** — apply a named preset stack or an explicit list of technique keys, composing their transformations in order.
- **`stack_quality_delta`**, **`stack_engineering_hours`** — the quality cost and the one-off engineering effort a stack implies.

### 4.4 `caide.costing` — Cost and TCO

- **`self_hosted_query_cost`**, **`api_query_cost`** — per-query compute cost, energy, carbon, water and latency for a self-hosted deployment or a commercial endpoint.
- **`total_cost_of_ownership`** — the annual total decomposed into the six layers, with human review charged explicitly and displaced labour reported alongside but never netted out.
- **`layer_volume_elasticity`** — how each layer responds to a small change in volume, exposing which layers are fixed, linear or stepped.

### 4.5 `caide.routing` — Traffic Routing

- **`optimise_routing`**, **`route_greedy`** — assign each workload class to the cheapest tier whose `quality_index` meets the class's floor and whose capacity share is respected, returning a `RoutingPlan`.

### 4.6 `caide.breakeven` — Architecture Crossover

- **`find_break_even`** — sample two cost curves across a volume range and return every crossing as a `BreakEvenResult`.
- **`dominance_intervals`** — collapse the crossings into the windows a planner acts on, including the windows where the two options are within tolerance and cost does not decide.

### 4.7 `caide.scaling` — Scaling Dynamics

- **`project`** — a closed-form projection: with unit cost falling at rate `r` and price elasticity `ε`, spend moves as `(c/c₀)^(1−ε)`. Above `ε = 1`, falling prices *raise* total spend.
- **`estimate_elasticity`** — a descriptive fit of elasticity from your own volume-and-cost history.

### 4.8 `caide.uncertainty` — Uncertainty and Sensitivity

- **`uniform`**, **`triangular`**, **`normal`**, **`lognormal`**, **`point`** — input `Distribution`s.
- **`monte_carlo`** — propagate the distributions through any cost function to a `MonteCarloResult`.
- **`sensitivity`** — Spearman rank sensitivity, chosen because the model is full of step functions and roofline maxima that a linear correlation would misread.

### 4.9 `caide.calibration` — Calibration

- **`fit_calibration`** — from two or more throughput `Observation`s on your own stack, fit the multiplicative correction to decode-bandwidth utilisation that minimises log-ratio error, returning a `CalibrationResult` you apply to a `ServingConfig`.
- **`predicted_output_tps`** — the throughput the roofline predicts for one observation, for direct comparison.

### 4.10 `caide.scenario` — Scenarios

- **`load_scenario`**, **`example_scenario`** — load a scenario from a YAML file, a path, or a dictionary; or get the bundled example as a dictionary.
- **`Scenario.validate`**, **`Scenario.evaluate`**, **`Scenario.evaluate_all`**, **`Scenario.to_yaml`** — validate the inputs, evaluate one or all architectures to `TCOResult`s, and round-trip the scenario back to YAML.

---

## 5. API Reference

### 5.1 Scenarios (the usual entry point)

```python
load_scenario(source: Union[str, Path, Dict]) -> Scenario
example_scenario() -> Dict          # the bundled example as a dict

Scenario.validate() -> List[str]                                  # [] means valid
Scenario.evaluate(architecture, volume=None, year=1) -> TCOResult
Scenario.evaluate_all(volume=None, year=1) -> Dict[str, TCOResult]
Scenario.to_yaml() -> str
```

### 5.2 Specifications

```python
ModelSpec(name, n_params_total, n_layers, d_model, n_heads,
          n_kv_heads=None, n_params_active=None, n_experts=1,
          experts_per_token=1, bytes_per_param=2.0, quality_index=1.0, ...)

HardwareSpec(name, peak_flops, memory_bytes, memory_bandwidth,
             power_watts, hourly_cost, interconnect_bandwidth=4.5e11,
             idle_power_watts=None, ...)

ServingConfig(n_accelerators=1, max_batch=256,
              demand_duty_cycle=0.65, scheduler_efficiency=0.45,
              mfu_prefill=0.45, mbu_decode=0.7, precision="bf16",
              speculative_gamma=0.0, prefix_cache_hit=0.0, ...)

WorkloadClass(name, share, tokens_in, tokens_out, quality_floor=0.0,
              review_rate=0.0, review_minutes=0.0, baseline_minutes=0.0,
              self_consistency_k=1, latency_sensitive=True)

SLO(ttft_seconds=2.0, tpot_seconds=0.05, enforce=True)
DeploymentState(model, hardware, serving)
```

### 5.3 Roofline

```python
prefill_flops(model, tokens_in) -> float
evaluate_request(state, workload, slo=None, batch_override=None) -> PhasePerformance
solve_batch_for_slo(state, workload, slo, tolerance=1e-3) -> PhasePerformance
capacity_batch(model, hw, cfg, avg_sequence) -> float
```

### 5.4 Efficiency

```python
available_techniques() -> List[Technique]
get_technique(key) -> Technique
apply_stack(state, keys) -> DeploymentState      # keys: preset name or list
```

### 5.5 Costing

```python
self_hosted_query_cost(state, workload, grid, slo=None, respect_slo=True) -> QueryCost
api_query_cost(pricing, workload, grid, prefix_cache_hit=0.0,
               semantic_cache_hit=0.0, provider_energy_wh_per_ktok=0.3) -> QueryCost

total_cost_of_ownership(*, architecture, annual_volume, workloads, grid,
                        state=None, pricing=None, assurance=None,
                        retrieval=None, integration=None, workforce=None,
                        slo=None, year=1, quality_penalty=0.0) -> TCOResult

layer_volume_elasticity(base, evaluate, relative_step=0.02) -> Dict[str, float]
```

`total_cost_of_ownership` is keyword-only. Supply `state=` for `architecture="self_hosted"` or `pricing=` for `architecture="api"`.

### 5.6 Break-even, Routing, Scaling

```python
find_break_even(cost_a, cost_b, *, label_a="A", label_b="B",
                volume_min=1000.0, volume_max=1e10, samples=240) -> BreakEvenResult
dominance_intervals(result) -> List[Tuple[float, float, str]]

optimise_routing(workloads, tiers, annual_volume, max_tiers=None) -> RoutingPlan
route_greedy(workloads, tiers, annual_volume) -> RoutingPlan

project(initial_unit_cost, initial_volume, assumptions) -> ScalingProjection
estimate_elasticity(unit_costs, volumes) -> Dict[str, float]
```

### 5.7 Uncertainty

```python
uniform(name, low, high)                 triangular(name, low, mode, high)
normal(name, mean, sd, *, clip_low=None) lognormal(name, median, sigma)
point(name, value)

monte_carlo(model, distributions, n_samples=4000, seed=20260101,
            label="output", saturation=None) -> MonteCarloResult
sensitivity(result, quantiles=(10.0, 90.0)) -> List[SensitivityEntry]
```

### 5.8 Calibration and Catalogue

```python
fit_calibration(observations, bounds=(0.2, 3.0), steps=61) -> CalibrationResult
predicted_output_tps(obs, mfu_scale=1.0, mbu_scale=1.0) -> Optional[float]

get_model(key) -> ModelSpec        get_hardware(key) -> HardwareSpec
get_pricing(key) -> PricingSpec    get_grid(key) -> GridSpec
catalogue_summary() -> Dict[str, List[str]]
```

---

## 6. Operation Guide

### 6.1 Evaluate a Scenario (the short path)

```python
from caide import load_scenario, example_scenario

scenario = load_scenario(example_scenario())
results = scenario.evaluate_all()                 # {architecture: TCOResult}
cheapest = min(results, key=lambda k: results[k].layers and
               sum(results[k].layers.values()))
for name, r in results.items():
    print(name, sum(r.layers.values()), r.per_query_blended)
```

### 6.2 Build a Deployment From the Catalogue

```python
from caide import (DeploymentState, ServingConfig, WorkloadClass, SLO,
                   get_model, get_hardware, solve_batch_for_slo)

state = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                        ServingConfig(n_accelerators=4, max_batch=256))
work = WorkloadClass("tutor", share=1.0, tokens_in=1500, tokens_out=400)
perf = solve_batch_for_slo(state, work, SLO(ttft_seconds=1.5, tpot_seconds=0.045))
print(perf.batch, perf.ttft_seconds, perf.throughput_qps)
```

### 6.3 Apply a Serving Stack

```python
from caide import apply_stack, self_hosted_query_cost, get_grid

tuned = apply_stack(state, "aggressive")          # or ["int4", "paged_attention", "gqa"]
qc = self_hosted_query_cost(tuned, work, get_grid("us-average"))
print(qc.compute_cost, qc.energy_joules, qc.slo_met)
```

Because the stack rewrites the physical state and the roofline is re-run on the result, the effective multiplier differs at every batch and context length rather than being a fixed number.

### 6.4 Six-Layer Total Cost of Ownership

```python
from caide import total_cost_of_ownership, get_grid

tco = total_cost_of_ownership(
    architecture="self_hosted", annual_volume=9_000_000,
    workloads=[work], grid=get_grid("us-average"), state=tuned)
for layer, dollars in tco.layers.items():
    print(f"{layer:22s} {dollars:>14,.0f}")
print("displaced labour (reported, not netted):", tco.displaced_labour_annual)
```

### 6.5 Break-even Between Two Architectures

```python
from caide import load_scenario, find_break_even, dominance_intervals

s = load_scenario("scenario.yaml")
be = find_break_even(s.cost_curve("api-frontier"), s.cost_curve("selfhost-70b"),
                     label_a="api-frontier", label_b="selfhost-70b")
for lo, hi, winner in dominance_intervals(be):
    print(f"{lo:,.0f} .. {hi:,.0f}: {winner}")
```

### 6.6 Uncertainty and Sensitivity

```python
from caide import monte_carlo, sensitivity, triangular, lognormal

dists = [triangular("utilisation", 0.22, 0.42, 0.65),
         lognormal("accelerator_hourly", 3.20, 0.38)]
mc = monte_carlo(lambda d: my_cost_model(d), dists, n_samples=4000)
for e in sensitivity(mc):
    print(e.name, e.rank_correlation)
```

### 6.7 Calibrate to Your Own Hardware

```python
from caide import Observation, fit_calibration

result = fit_calibration([
    Observation(model=..., hardware=..., n_accelerators=4, batch=128,
                tokens_in=512, tokens_out=256, measured_output_tps=3245),
    # two or more, from your stack
])
print(result.summary())
tuned_config = result.apply(serving_config)
```

### 6.8 Bundled Example Scenarios

| Scenario | Volume | What dominates |
|---|---|---|
| `university_tutoring.yaml` | 9M/yr | Assurance and workforce; inference is ~16% |
| `hospital_documentation.yaml` | 6.6M/yr | Mandatory clinician review; displaced effort exceeds total cost ~4× |
| `public_helpline.yaml` | 420M/yr | Fixed layers dominate; no architecture meets every quality floor |

Extract them with `caide examples --extract .`. Two runnable scripts also ship in the examples: `reproduce_paper.py` regenerates the paper's figures and findings, and `draw_architecture.py` draws the architecture diagram.

---

## 7. Command-Line Interface

The `caide` command exposes the whole engine without writing Python. Every report it writes carries the CAIDE version, a scenario digest and the random seed, so a figure can always be traced back to the inputs that produced it.

### 7.1 Commands

| Command | Question it answers |
|---|---|
| `caide run` | What will this cost, where does the money go, how uncertain is it |
| `caide breakeven` | At what volume does the cheaper architecture change |
| `caide sweep` | How does a technique's benefit vary across the operating range |
| `caide route` | Which model tier should serve which class of traffic |
| `caide catalog` | What presets are available |
| `caide examples` | Give me the bundled scenarios to run or edit |
| `caide init` | Give me a scenario file to start from |
| `caide validate` | Is this scenario well formed |

### 7.2 Usage

```bash
caide run scenario.yaml [--out DIR] [--samples N] [--seed S] [--layers] [--strict]
caide breakeven scenario.yaml [-a A] [-b B] [--min-volume V] [--max-volume V]
                              [--tolerance T] [--json]
caide sweep scenario.yaml [--technique KEY] [--architecture NAME] [--out DIR]
caide route scenario.yaml [--tier-fixed-cost COST]
caide catalog [--json]
caide examples [--extract DIR] [--force]
caide init [path] [--force]
caide validate scenario.yaml
```

`caide run --out report/` writes three artefacts: a Markdown report, a CSV of results, and a **single-file HTML dashboard** with figures embedded as base64 data URIs — no external assets, so it still renders years later inside a procurement record. `--layers` adds the six-layer breakdown; `--samples` sets the Monte Carlo draw count; `--strict` turns input warnings into errors.

### 7.3 A Typical Session

```bash
caide examples --extract .
caide validate examples/university_tutoring.yaml
caide run examples/university_tutoring.yaml --layers --out report/
caide breakeven examples/university_tutoring.yaml -a api-frontier -b selfhost-70b
caide sweep examples/university_tutoring.yaml --technique speculative_decoding
caide route examples/public_helpline.yaml
```

### 7.4 Scenario Format

```yaml
name: university-engineering-genai
grid: us-average
annual_volume: 9_000_000

slo: { ttft_seconds: 1.5, tpot_seconds: 0.045 }

workloads:
  - name: tutoring_turn
    share: 0.58
    tokens_in: 1500
    tokens_out: 400
    quality_floor: 0.62
  - name: code_feedback
    share: 0.42
    tokens_in: 3000
    tokens_out: 1000
    quality_floor: 0.80
    self_consistency_k: 3
    review_rate: 0.12
    review_minutes: 3.0
    baseline_minutes: 11.0        # what it cost before the system existed

architectures:
  - name: api-frontier
    type: api
    pricing: api-frontier
  - name: selfhost-70b
    type: self_hosted
    model: dense-70b
    hardware: h100-sxm
    serving:
      n_accelerators: 4
      max_batch: 256
      demand_duty_cycle: 0.42     # bursty: idle between deadlines
      scheduler_efficiency: 0.45  # raised by the stack, capped by demand
    stack: aggressive

assurance:
  evaluation_annual: 95_000
  reviewer_hourly_cost: 52.0

uncertainty:
  utilisation: { kind: triangular, low: 0.22, mode: 0.42, high: 0.65 }
  accelerator_hourly: { kind: lognormal, median: 3.20, sigma: 0.38 }
```

Validation is strict and errors name the offending path (`workloads[2].tokens_in: value must be finite`). A cost model that silently accepts `tokens_in: -500` is worse than no model, because its output looks equally plausible.

---

## 8. Validation and Accuracy

### 8.1 Against Published Serving Measurements

Uncalibrated, CAIDE's roofline was checked against published throughput measurements spanning three serving frameworks (`docs/model.md` §8; the table is regenerated by the reproduction script, not transcribed):

| Configuration | Predicted | Measured | Ratio | Implied MBU |
|---|---|---|---|---|
| 70B, 4×H100, bf16, high concurrency | 3,178 | 3,245 | 0.98 | 0.33 |
| 405B, 8×H100, FP8, 1024→2048 | 2,078 | 3,089 | 0.67 | 0.90 |
| 405B, 8×H100, FP8, 128→128 | 3,349 | 3,732 | 0.90 | 0.31 |
| 70B, 1×H100, FP8, batch 64 | 1,120 | 460 | 2.44 | 0.26 |
| 8B, 1×A100, bf16, 8 concurrent | 601 | 187 | 3.21 | 0.19 |

Figures are aggregate output tokens per second over the whole serving cycle (prefill included), which is what a wall-clock benchmark divides by.

> Uncalibrated, CAIDE predicts serving throughput within roughly a factor of two, worst case three, with no one-sided bias. That is adequate for **ranking architectures against each other** and inadequate for **absolute capacity commitments**.

The bundled model archetypes are generic shapes rather than the exact architectures measured, so these four points are not a calibration set — tuning constants until they agree would fit the heterogeneity, not the physics.

### 8.2 Calibration

Supplying two or more measurements from *your own* stack to `caide.calibration.fit()` returns the multiplicative correction to decode-bandwidth utilisation that minimises log-ratio error. On the reference points it lifts the fraction inside the factor-of-two band from 50% to 75% and reduces log-RMSE from 0.665 to 0.560 — better, not good, which is what a correction fitted to heterogeneous data should look like.

### 8.3 Reproducibility

`examples/reproduce_paper.py` regenerates every figure and the machine-readable `paper_figures/findings.json` from the shipped scenarios, so the paper's numbers are recomputed rather than quoted. `Scenario.to_yaml()` round-trips exactly, and every report embeds the scenario that produced it alongside a digest and the seed.

### 8.4 Scope and Limitations

- **A model, not a measurement.** Ranking is robust; absolute capacity commitments require calibration against your own stack.
- **The roofline models the accelerator, not the serving stack.** `framework_overhead_per_step` defaults to zero, correct for comparing configurations and wrong for predicting a specific benchmark's wall-clock.
- **Prices are illustrative anchors.** They vary by more than 3× across providers, regions and commitment terms, carry an epoch date, and trigger a warning when more than six months old. Override them before quoting absolute figures; relative structure is robust to price level, the absolute total is not.
- **Implausible magnitudes are warned about, not rejected**, so legitimate sensitivity sweeps are not blocked; read the warnings.
- **Unknown scenario keys are warned about, not rejected**, with a suggested correction.
- **Quality indices are ordinal placeholders.** Replace them with your own evaluation results before a routing decision rests on them.
- **Elasticity is assumed, not identified.** `estimate_elasticity` reports a descriptive fit.
- **Provider-side energy is a disclosure-dependent estimate**, reported so comparisons are not silently carbon-blind.

---

## 9. Troubleshooting

### 9.1 `caide validate`/`run` Cannot Find the Example File

**Cause:** the example scenarios ship *inside* the package, not in the working directory.

**Fix:**

```bash
caide examples --extract .
caide run examples/university_tutoring.yaml --layers
```

### 9.2 One Architecture Reports Every Class as an SLO Violation

**Cause:** the latency SLO is unreachable at any batch for that deployment — usually too few accelerators, or a batch cap forcing a compute-bound regime.

**Fix:** raise `n_accelerators`, relax the SLO, or inspect `solve_batch_for_slo`; a latency-*insensitive* class records a note instead of a disqualifying violation.

### 9.3 The Efficiency Stack Barely Changes the Cost

**Expected** in some regimes. A technique that frees HBM does little where the deployment is already compute-bound; INT4 after PagedAttention finds less fragmented memory left to recover. Use `caide sweep --technique KEY` to see where in the operating range the technique actually helps.

### 9.4 Total Cost Looks Too High at Low Volume

**Cause:** the fixed layers (integration and SRE, assurance and governance) dominate below the volume where per-token cost catches up. This is a property of the six-layer model, not an error.

**Fix:** run `caide breakeven` — the fixed layers are exactly why the cheaper architecture changes with volume.

### 9.5 Implied Reviewer Headcount Is Implausible

**Cause:** `review_rate × review_minutes × annual_volume` implies more full-time reviewers than an organisation could staff. CAIDE surfaces this rather than hiding it in a dollar total.

**Fix:** revisit `review_rate` and `review_minutes`, or the self-consistency gating that triggers review, in the workload class.

### 9.6 A Price Warning Appears on Every Run

**Cause:** the bundled prices carry an epoch date and warn when more than six months old.

**Fix:** override the price fields with current quotations before quoting absolute figures. Relative structure is unaffected.

### 9.7 Windows: `UnicodeDecodeError` When Running `pytest`

**Cause:** legacy code-page file I/O on Windows.

**Fix:** the shipped tests already specify UTF-8; if you still hit this in your own scripts, set `PYTHONUTF8=1` or pass `encoding="utf-8"` to your file reads.

### 9.8 Monte Carlo Results Change Between Runs

**Cause:** no seed was fixed.

**Fix:** `monte_carlo(..., seed=20260101)` (the default) or any fixed integer makes the draw reproducible.

---

## 10. Support and Version Information

### 10.1 Contact

- **Repository:** https://github.com/LiuMuyao123/caide
- **Issue tracker:** https://github.com/LiuMuyao123/caide/issues

### 10.2 Version

- **Current version:** 17.1.0
- **Released:** 2026-08-22
- **Development status:** Production/Stable
- **License:** MIT

### 10.3 Author

| Name | Affiliation |
|---|---|
| Muyao Liu | Academy of Future Education (AOFE), Xi'an Jiaotong-Liverpool University |

Contact: Muyao.Liu25@student.xjtlu.edu.cn

### 10.4 Citation

If you use CAIDE in published work, please cite the archived release:

> Liu, M. (2026). *CAIDE: Cost-Aware Inference Deployment Evaluator* (v17.1.0). Zenodo. https://doi.org/10.5281/zenodo.22066252

DOI: 10.5281/zenodo.22066252. Machine-readable metadata is in `CITATION.cff`.

### 10.5 Testing

```bash
pytest -q --cov=caide
```

Continuous integration runs the full suite (487 tests, ~90% coverage) on Ubuntu, macOS and Windows across Python 3.9, 3.11 and 3.13, and separately smoke-tests every CLI command against the bundled examples.

---

## 11. Appendix

### A. Complete Public API

**Specifications:** `ModelSpec`, `HardwareSpec`, `ServingConfig`, `WorkloadClass`, `SLO`, `PricingSpec`, `GridSpec`, `DeploymentState`

**Roofline:** `prefill_flops`, `evaluate_request`, `solve_batch_for_slo`, `capacity_batch`, `uniform_routing_imbalance`, `PhasePerformance`

**Efficiency:** `available_techniques`, `get_technique`, `apply_stack`, `stack_quality_delta`, `stack_engineering_hours`, `Technique`, `TECHNIQUES`, `PRESET_STACKS`

**Costing:** `self_hosted_query_cost`, `api_query_cost`, `total_cost_of_ownership`, `layer_volume_elasticity`, `QueryCost`, `TCOResult`, `CostLayer`, `AssuranceProfile`, `SIX_LAYERS`

**Routing:** `optimise_routing`, `route_greedy`, `Tier`, `RoutingPlan`

**Break-even:** `find_break_even`, `dominance_intervals`, `Crossing`, `BreakEvenResult`

**Scaling:** `project`, `estimate_elasticity`, `ScalingAssumptions`, `ScalingProjection`

**Uncertainty:** `monte_carlo`, `sensitivity`, `uniform`, `triangular`, `normal`, `lognormal`, `point`, `Distribution`, `MonteCarloResult`, `SensitivityEntry`

**Calibration:** `fit_calibration`, `predicted_output_tps`, `Observation`, `CalibrationResult`, `REFERENCE_OBSERVATIONS`, `CONVENTIONS`

**Catalogue:** `get_model`, `get_hardware`, `get_pricing`, `get_grid`, `catalogue_summary`, `MODELS`, `HARDWARE`, `PRICING`, `GRIDS`

**Scenarios:** `load_scenario`, `example_scenario`, `Scenario`, `Architecture`, `ScenarioError`

### B. Output Field Glossary

| Field | Meaning |
|---|---|
| `layers` | The six-layer breakdown, a `{layer_name: annual_dollars}` mapping |
| `per_query_blended` | Blended cost per query across all workload classes |
| `compute_per_query` | The compute-and-serving component alone |
| `price_inelastic_per_query` | Per-query cost from layers that do not track token price |
| `annual_carbon_kg`, `annual_energy_kwh`, `annual_water_l` | Annual environmental totals |
| `capacity_units` | Whole replicas provisioned (the staircase step) |
| `slo_violations` / `slo_unevaluated` | Classes missing / not able to be evaluated against the SLO |
| `quality_violations`, `quality_shortfall`, `quality_margin` | Classes below their quality floor, and by how much |
| `displaced_labour_annual` | Human time the system displaces, reported but never netted from cost |
| `review_hours_annual` | Implied annual human-review hours |
| `slo_met` (QueryCost) | Whether the latency SLO was met; `None` when latency is not modelled |
| `detail` (QueryCost) | Per-query intermediate quantities (device seconds, facility joules, …) |

### C. Built-in Catalogue

| Category | Keys |
|---|---|
| **Models** | `dense-1b`, `dense-3b`, `dense-8b`, `dense-32b`, `dense-70b`, `dense-405b`, `moe-8x7b`, `moe-8x22b`, `moe-236b` |
| **Hardware** | `a100-40gb`, `a100-80gb`, `h100-sxm`, `h200-sxm`, `l40s`, `consumer-24gb` |
| **Pricing** | `api-economy`, `api-midrange`, `api-frontier`, `api-frontier-premium` |
| **Grids** | `china-average`, `coal-heavy`, `eu-average`, `france`, `global-average`, `india`, `nordic-hydro`, `us-average`, `us-west` |
| **Techniques** | `flash_attention`, `continuous_batching`, `paged_attention`, `chunked_prefill`, `int8`, `int4`, `fp8`, `kv_fp8`, `gqa`, `prefix_caching`, `semantic_caching`, `speculative_decoding`, `distillation_50`, `distillation_25`, `committed_use` |
| **Preset stacks** | `none`, `baseline_serving`, `standard`, `aggressive`, `maximal` |
| **Six layers** | `model_access`, `compute_serving`, `retrieval_data`, `integration_sre`, `assurance_governance`, `workforce_redesign` |

List them at any time with `caide catalog` or `catalogue_summary()`.

### D. Repository Layout

```
caide/
├── src/caide/                  # package source
│   ├── __init__.py             # public API
│   ├── specs.py                # model / hardware / workload / SLO specs
│   ├── roofline.py             # prefill + decode performance model
│   ├── efficiency.py           # techniques as state transformations
│   ├── costing.py              # per-query cost + six-layer TCO
│   ├── routing.py              # tier routing
│   ├── breakeven.py            # architecture crossovers
│   ├── scaling.py              # demand-response projection
│   ├── uncertainty.py          # Monte Carlo + sensitivity
│   ├── perturb.py              # one-at-a-time perturbation
│   ├── calibration.py          # fit utilisation constants
│   ├── catalog.py              # built-in presets
│   ├── scenario.py             # YAML scenario load / validate / evaluate
│   ├── report.py               # Markdown / CSV / HTML reports
│   ├── plotting.py             # figures (Agg backend)
│   └── examples/               # bundled scenarios + reproduction scripts
├── docs/model.md               # the derivation and accuracy evidence
├── audit/                      # self-audit tooling (ledger, mutation, scans)
├── paper_figures/              # generated figures + findings.json
├── tests/                      # pytest suite (487 tests)
├── .github/workflows/ci.yml    # CI matrix
├── pyproject.toml
├── CITATION.cff
├── CAIDE_USER_MANUAL.md        # this manual
├── LICENSE                     # MIT
└── README.md
```

---

*CAIDE 17.1.0 · MIT License · Copyright © 2026 Muyao Liu*
