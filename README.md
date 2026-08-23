# CAIDE — Cost-Aware Inference Deployment Evaluator

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-487%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](tests/)

> **🔗 Live demo — [open the interactive explorer](https://liumuyao123.github.io/caide/)**
> (runs entirely in the browser, no install; every figure is real CAIDE output).
> No internet? Open `caide_explorer.html` from the repository directly in a browser.

**Answer four questions before you deploy a large language model service:
what will it cost, where does the money actually go, at what volume does
the cheaper architecture change, and how much of the answer is guesswork.**

CAIDE takes a declarative description of a workload, a model, an
accelerator fleet and a governance regime, and returns per-query cost,
energy and carbon; an annual total cost of ownership decomposed into six
layers; break-even volumes against alternative architectures; and a
sensitivity ranking that says which unknown is worth measuring next.

```bash
pip install caide
caide examples --extract .                  # bundled scenarios, three domains
caide run examples/university_tutoring.yaml --layers --out report/
```

---

## Why this exists

Deployment cost analyses for LLM services are usually built from
**published constant multipliers**: "4-bit quantisation, 0.65×",
"speculative decoding, 0.40×", "combined, 0.03–0.08×". Multiply them
together, apply to a token price, done.

That method has two defects that CAIDE was written to remove.

**Constants cannot express regime dependence.** Speculative decoding
delivers 0.40× at batch 1, where decode is starved of arithmetic, and
1.09× at batch 256 — past parity, a net loss — where the accelerator is
already compute-saturated
and the draft model competes for the same FLOPs. A single number is right
at one operating point and wrong everywhere else — and it is usually
quoted next to a batching multiplier that assumes the opposite regime.

**Constants cannot express interaction.** INT4 quantisation frees HBM;
PagedAttention recovers fragmented HBM. Stack them and the second one
finds much less left to recover, so the pair delivers 0.21× where the
product of the constants predicts 0.15×. Elsewhere the interaction runs
the other way. The error has no reliable sign.

CAIDE therefore models each technique as a **function from deployment
state to deployment state** that edits the underlying physics — bytes per
parameter, usable memory, achievable batch, decode steps. The cost
multiplier is never an input. It is measured by re-running a roofline
model on the transformed state, so it comes out different at every
operating point, as it should.

---

## What it models

### Inference performance (`caide.roofline`)

Prefill and decode are separated because they are bound by different
resources:

- **Prefill** is compute bound: `2·N_active·T_in` FLOPs plus a causal
  attention term `2·L·T_in²·d_model` that is negligible for short prompts
  and dominant for retrieval-augmented ones.
- **Decode** is memory-bandwidth bound: every step streams the weights
  out of HBM regardless of how many sequences are in flight, so batching
  amortises that read — until the arithmetic roofline takes over.

Also modelled: GQA-aware KV cache sizing, MoE expert-touch probability as
a function of batch (`1 − (1 − k/E)^B`, which is why MoE loses its
bandwidth advantage exactly when batching would help), tensor-parallel
communication loss, and an M/D/1 queue for time-to-first-token.

```python
from caide import *

state = DeploymentState(get_model("dense-70b"), get_hardware("h100-sxm"),
                        ServingConfig(n_accelerators=4, max_batch=256))
perf = solve_batch_for_slo(state, WorkloadClass("tutor", 1.0, 1500, 400),
                           SLO(ttft_seconds=1.5, tpot_seconds=0.045))
print(perf.batch, perf.ttft, perf.throughput_qps)
```

`solve_batch_for_slo` bisects for the **largest batch that still meets the
latency SLO** rather than assuming the scheduler cap is reachable.
Throughput rises monotonically with batch and latency degrades
monotonically, so the feasible set is an interval and the boundary is
exact.

### Demand duty cycle is not scheduler efficiency

Utilisation is held as two independent quantities, because collapsing them
into one lets a serving optimisation silently overwrite a statement about
demand:

| | Meaning | Who sets it |
|---|---|---|
| `demand_duty_cycle` | Share of the year with live traffic | The workload. Deadline-driven academic traffic sits near 0.4 whatever software serves it. |
| `scheduler_efficiency` | Useful-work share *while* traffic is live | The serving stack. This is what continuous batching improves. |

Costing divides by the product, and the product can never exceed the demand
duty cycle — no serving optimisation creates traffic that was never sent.
Version 1.0 used a single `target_utilisation`, which continuous batching
multiplied by 1.9: a scenario declaring 0.42 was silently served at 0.88,
and three different demand assumptions produced one identical cost. It is
now read-only and reports the product.

### Calibration against your own hardware

The roofline predicts from first principles, which is what lets it compare
configurations nobody has built. The price is that two utilisation
parameters are assumptions, and a wrong assumption biases every prediction
the same way.

```python
from caide import Observation, fit_calibration
result = fit_calibration([
    Observation(model=..., hardware=..., n_accelerators=4, batch=128,
                tokens_in=512, tokens_out=256, measured_output_tps=3245),
    ...                                    # two or more, from your stack
])
print(result.summary())                    # scales, log-RMSE, fraction within 2x
tuned_config = result.apply(serving_config)
```

### Reports you can re-run, not just read

Every report embeds the scenario that produced it. A digest proves the
inputs did not change; the inputs themselves are what let the recipient
regenerate the numbers. `Scenario.to_yaml()` round-trips exactly —
verified across two successive trips on all three shipped examples.

### Six-layer total cost of ownership (`caide.costing`)

Each layer scales differently, and only the first is linear:

| Layer | Scaling in annual volume |
|---|---|
| Model access | linear — pay per token, forever |
| Compute and serving | **step** — capacity arrives in whole replicas |
| Retrieval and data | sublinear — index once, query many times |
| Integration and SRE | volume-free — driven by system count |
| Assurance and governance | volume-free — an audit programme is fixed |
| Workforce and redesign | front-loaded — large in year 1, decaying |

Modelling only the linear layer understates cost at low volume, where
fixed layers dominate, **and** overstates the savings from efficiency work
at high volume, where the efficient layer shrinks and the others do not.

### Human review, netted honestly

`review_rate × review_minutes` is charged as cost. `baseline_minutes` —
the human time the task consumed *before* the system existed — is
reported alongside as displaced effort and is **never silently subtracted**
from the total, because a labour saving and a cash outlay are different
instruments.

CAIDE also reports the implied reviewer headcount and warns when a
scenario demands more FTEs than an organisation could plausibly staff.
That check exists because it caught two of the three shipped examples
during development: an early draft quietly assumed 359 full-time
reviewers, and the dollar total looked entirely reasonable.

### Break-even that admits when it cannot decide

Because self-hosted capacity is a staircase and API cost is a line, the
two can cross many times: each new replica overshoots demand and hands
the advantage back until it fills. CAIDE reports **all** crossings, then
collapses them into the summary a planner can act on:

```
5 crossings, the first at 640M queries/yr

4 windows in which cost does not decide (within 5%),
the widest 1.83B .. 2.01B queries/yr. Between them the two
options differ by up to 47%. Choose on latency, data
residency, control, staffing or exit risk inside a window.
```

### Uncertainty, not point estimates

Monte Carlo propagation with Spearman rank sensitivity — rank
correlation because the model is full of step functions, roofline maxima
and SLO cliffs, and a monotone-but-curved relationship should not be
scored as weak.

```
the commercial endpoint is cheaper in 100% of 4,000 draws
dominant driver: review_minutes (33% of explained variance)
all draws feasible; declared inputs explain 0.97 of rank variance
```

### Scaling dynamics

Closed-form demand response: with unit cost falling at rate `r` and price
elasticity `ε`, spend moves as `(c/c₀)^(1−ε)`. Above `ε = 1`, falling
prices **raise** total spend. Three parameters, no fitting — the point is
to make the elasticity assumption explicit and falsifiable, and
`estimate_elasticity` fits it from your own volume history when you have
one.

---

## Command line

| Command | Question it answers |
|---|---|
| `caide run` | What will this cost, where does the money go, how uncertain is it |
| `caide breakeven` | At what volume does the cheaper architecture change |
| `caide sweep` | How does a technique's benefit vary across the operating range |
| `caide route` | Which model tier should serve which class of traffic |
| `caide examples` | Give me the bundled scenarios to run or edit |
| `caide catalog` | What presets are available |
| `caide init` | Give me a scenario file to edit |
| `caide validate` | Is this scenario well formed |

`caide run --out report/` writes a Markdown report, a CSV of results, and
a **single-file HTML dashboard** with figures embedded as base64 data
URIs — no external assets, so it still renders years later when it
surfaces in a procurement record.

Every report carries the CAIDE version, a scenario digest and the random
seed. A cost figure without provenance cannot be defended in a budget
review.

---

## Scenario format

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
    self_consistency_k: 3      # repeated sampling gates review
    review_rate: 0.12
    review_minutes: 3.0
    baseline_minutes: 11.0     # what it cost before the system existed

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
      demand_duty_cycle: 0.42      # bursty: idle between deadlines
      scheduler_efficiency: 0.45   # raised by the stack, capped by demand
    stack: aggressive

layers:
  integration: { fixed_annual: 265_000 }
  workforce:   { front_load_year1: 340_000, decay: 0.32 }

assurance:
  evaluation_annual: 95_000
  reviewer_hourly_cost: 52.0

uncertainty:
  utilisation: { kind: triangular, low: 0.22, mode: 0.42, high: 0.65 }
  accelerator_hourly: { kind: lognormal, median: 3.20, sigma: 0.38 }
```

Validation is strict and errors name the offending path
(`workloads[2].tokens_in: value must be finite`). A cost model that
silently accepts `tokens_in: -500` is worse than no model, because its
output looks equally plausible.

---

## Shipped examples

Three domains, chosen because the shape of the answer inverts between
them:

| Example | Volume | What dominates |
|---|---|---|
| `university_tutoring.yaml` | 9M/yr | Assurance and workforce; inference is 16% |
| `hospital_documentation.yaml` | 6.6M/yr | Mandatory clinician review — but displaced effort exceeds total cost 4× |
| `public_helpline.yaml` | 420M/yr | Fixed layers dominate; no architecture meets every quality floor |

The examples ship **inside the package**, so they are available after
`pip install` and not only from a repository checkout:

```bash
caide examples                              # list what is bundled
caide examples --extract .                  # write them to ./examples/
caide run examples/hospital_documentation.yaml --layers --out out/
caide sweep examples/university_tutoring.yaml --technique speculative_decoding
caide route examples/public_helpline.yaml
python examples/reproduce_paper.py --out figures/
```

---

## On the bundled numbers

Two kinds of constant ship with CAIDE and they carry very different
weight.

**Architectural and physical specifications** — parameter counts, HBM
capacity, memory bandwidth, board power — are published and stable. They
are reproduced as documented constants.

**Prices** — accelerator rental, token tariffs, electricity, staff time —
move continuously and vary by more than 3× across providers, regions and
commitment terms. They are illustrative anchors so that examples run out
of the box, they carry an epoch date, and CAIDE warns you when they are
more than six months old.

> Override them before quoting absolute figures. Relative structure —
> which layer dominates, where curves cross, which input drives variance —
> is robust to price level; the absolute total is not.

---

## Installation

```bash
pip install caide                     # from PyPI
pip install -e ".[dev]" && pytest     # from source, with tests
```

Requires Python 3.9+, NumPy, PyYAML and Matplotlib. No GPU required —
CAIDE models accelerators, it does not use them.

---

## Limitations

- **A model, not a measurement.** Tested against four published
  measurements from three serving frameworks, the uncalibrated roofline
  lands within a factor of two in two of four cases and over-predicts in
  three — call it a factor of two to three, biased optimistic. That is
  adequate for *ranking* architectures, since the bias is largely common
  to all of them, and inadequate for absolute capacity commitments.
  `caide.calibration.fit()` corrects it against measurements from your own
  stack; see `docs/model.md` §8 for the evidence.
- **The roofline models the accelerator, not the serving stack.** Published
  throughput is wall-clock and includes the scheduler and HTTP layer —
  vLLM measured only 38% of its own wall time as GPU execution on an 8B
  model. `framework_overhead_per_step` defaults to zero, which is correct
  for comparing configurations and wrong for predicting a benchmark.
- **Implausible magnitudes are warned about, not rejected.** A
  thousand-fold infrastructure overhead is absurd but not impossible, and
  refusing it would block legitimate sensitivity sweeps. Read the warnings.
- **Unknown scenario keys are warned about, not rejected.** A misspelled
  field is reported with a suggested correction, but the run continues.
  Read the warnings.
- **Quality indices are ordinal placeholders.** They order models
  sensibly and should be replaced with your own evaluation results before
  a routing decision rests on them.
- **Elasticity is assumed, not identified.** `estimate_elasticity`
  reports a descriptive fit; anything moving price and volume together is
  absorbed into the slope.
- **Provider-side energy is a disclosure-dependent estimate.**
  Self-hosted energy is derived from board power; hosted energy is not
  measurable from outside and is reported so that comparisons are not
  silently carbon-blind.

---

## Citation

If CAIDE contributes to published work, please cite the software paper
(see `CITATION.cff`).

## License

MIT — see [LICENSE](LICENSE).
