# The CAIDE cost model

This document states every equation CAIDE uses, the assumption behind
it, and the regime in which it stops being reliable. It is written so
that a reviewer can disagree with a specific line rather than with the
software as a whole.

Notation: `N` parameters, `L` layers, `d` model dimension, `h` attention
heads, `h_kv` key/value heads, `T_in` prompt tokens, `T_out` generated
tokens, `B` batch (concurrent sequences), `w` bytes per parameter, `F`
achievable FLOP/s, `BW` achievable bytes/s, `V` annual query volume.

---

## 1. Inference arithmetic

### 1.1 Prefill

Prefill processes the whole prompt in one pass:

```
FLOPs_prefill = 2·N_active·T_in  +  2·L·T_in²·d
                └── linear ──┘     └── causal attention ──┘
```

The linear term is the standard forward-pass approximation: two FLOPs
(one multiply, one add) per parameter per token. The quadratic term
covers the `QK^T` and `AV` matmuls; the factor 2 rather than 4 reflects
causal masking, which halves the work.

The quadratic term is under 5% of the total below about 2,000 tokens for
an 8B model and exceeds the linear term above roughly 30,000. Analyses
that omit it are therefore safe for chat and wrong for
retrieval-augmented or long-document workloads — which is the case most
deployment planning actually faces.

```
t_prefill = FLOPs_prefill / (F_peak · n_acc · MFU · η_TP)
```

### 1.2 Decode

Each decode step must stream the weights out of memory, read the KV
cache, and do the arithmetic. Memory traffic and compute overlap on real
hardware, so the step time is their **maximum**, not their sum:

```
t_memory  = (bytes_weights(B) + kv_per_token · ctx · B) / BW_eff
t_compute = 2·N_active·B / F_eff
t_step    = max(t_memory, t_compute)
```

This is the model's most consequential structural choice. A `2N`
FLOPs-per-token model predicts that batch 1 and batch 128 cost the same
per token; measured systems differ by more than an order of magnitude.
Reproducing that gap is the reason the memory term exists.

**Decode utilisation depends on batch.** The arithmetic term uses an
achievable FLOP rate that rises with batch:

```
MFU_decode(B) = MFU_prefill · B / (B + B_half)
```

A decode step at `B = 1` is a matrix-vector product: arithmetic intensity
is one, tensor cores idle, utilisation is a few percent. At `B = 256` it
is a proper matrix-matrix product approaching prefill's utilisation.
`B_half` defaults to 64. Modelling this as a constant -- as CAIDE 2.0 did
-- places the transition below at an artificially low batch, which
understates how much quantisation still helps where deployments actually
run. At small batch the value barely matters, because the roofline maximum
selects the memory term regardless.

**Where the roofline turns over.** Both terms are linear in `B`, so which
one binds is set by the *context length*, not by the batch:

```
KV-bound  ⟺  kv_per_token · ctx / BW_eff  >  2·N_active / F_eff
```

Below the crossover, batching eventually reaches the compute roof. Above
it, the workload stays memory bound at every batch — a real and
frequently missed property. Deployments that assume batching always
reaches the compute roof over-provision FLOPs and under-provision
bandwidth.

### 1.3 KV cache

```
kv_per_token = 2 · L · h_kv · d_head · bytes_kv
```

The factor 2 is K and V. Using `h_kv` rather than `h` makes the model
grouped-query aware: an 8-group GQA model has a KV cache four times
smaller than the multi-head equivalent, which changes the feasible batch
by the same factor.

### 1.4 Mixture of experts

All experts must be resident in memory; only the routed subset does
arithmetic. The expected fraction of experts touched by `B` independent
tokens is

```
p_touch(B) = 1 − (1 − k/E)^B
bytes_weights(B) = (N_shared + N_experts · p_touch(B)) · w
```

At `B = 1` an MoE model reads a small fraction of its weights, which is
its entire bandwidth advantage. At `B = 128` with `k/E = 2/8`, `p_touch`
is already above 0.99. **MoE loses its bandwidth advantage exactly when
batching would otherwise deliver one.** A model that uses active
parameters at every batch will systematically overstate MoE throughput at
production batch sizes.

### 1.5 Capacity

```
memory_available = M · n_acc · μ − N_total · w
B_max            = memory_available / (kv_per_token · ctx_avg)
```

`μ` is usable memory after fragmentation. PagedAttention is modelled as
an increase in `μ`, not as a reduction in step time — which is what it
actually does, and why its benefit appears as a larger batch rather than
a faster one.

`memory_available ≤ 0` means the weights do not fit; CAIDE returns
infinite latency and zero throughput rather than a small number.

### 1.6 Time to first token

A request does not wait for the whole batch to prefill — schedulers
interleave prefill chunks with decode — but neither does it get the
accelerator alone. Treating prefill as an M/D/1 queue whose utilisation
is its share of the serving cycle:

```
ρ    = t_prefill_total / t_cycle
TTFT = t_prefill_one · (1 + ρ / (2(1 − ρ)))
```

Exact in both limits: no contention when prefill is a negligible share,
unbounded as prefill saturates the replica. Capped at 25× so that a
saturated configuration returns a finite number the SLO check can then
reject.

### 1.7 Speculative decoding

With `γ` draft tokens and per-token acceptance `α`, the expected accepted
tokens per verification step is the truncated geometric mean:

```
E[accepted] = (1 − α^(γ+1)) / (1 − α)
```

Decode steps are divided by this. The draft model's own weight read and
arithmetic are added to `t_memory` and `t_compute` — which is why the
benefit shrinks at large batch, where the target model is already
compute-bound and the draft competes for the same FLOPs.

---

## 2. Cost

### 2.1 Self-hosted per query

```
qps            = B / (B·t_prefill_one + T_out·t_step)
accel_seconds  = n_acc / qps  ·  k_self_consistency  ·  (1 − cache_hit)
utilisation    = demand_duty_cycle · scheduler_efficiency
cost_per_query = accel_seconds/3600 · price_hourly / utilisation · overhead
                 + energy_kWh · price_electricity
```

**Utilisation is two quantities, not one.** `demand_duty_cycle` is the
share of the year during which requests arrive; it is a property of the
workload and no serving optimisation may change it. `scheduler_efficiency`
is the share of accelerator time doing useful work *while* requests are
arriving; continuous batching and chunked prefill improve this. Their
product is what cost is divided by, and it can never exceed the demand
duty cycle.

Version 1.0 collapsed both into a single `target_utilisation`, which
continuous batching multiplied by 1.9 and capped at 0.85. A scenario
declaring 0.42 was therefore served at 0.88, and three different demand
assumptions produced one identical cost. The split is why v2.0 reports
higher self-hosted unit costs than v1.0 did.

Dividing by utilisation is what makes the model disagree with published
reference tables, which assume a fully loaded replica. A service
provisioned for a deadline peak, served by a scheduler that cannot fill
every cycle, and carrying production overhead runs at roughly 3.2× the
fully-loaded figure.

`overhead` covers orchestration, networking, redundancy and storage.

### 2.2 Integral capacity

You cannot rent a fraction of a replica:

```
replicas    = max(⌈capacity_units⌉, min_replicas)
cost_annual = max(replicas · replica_annual_cost, continuous_estimate)
```

This is what makes the self-hosted cost curve a staircase, and therefore
what allows a break-even scan to find more than one crossing.

### 2.3 API per query

```
cost = (T_in·(1−f_cached)·p_in + T_in·f_cached·p_cached + T_out·p_out) / 10⁶
```

### 2.4 Energy and carbon

```
E_device   = P_board · n_acc · accel_seconds / duty
E_facility = E_device · PUE
carbon     = E_facility/3.6e6 · intensity
water      = E_facility/3.6e6 · WUE
```

Idle capacity draws power, so energy is divided by duty cycle for the
same reason cost is.

Provider-side energy for hosted APIs is **not measurable from outside**.
CAIDE reports a disclosure-dependent estimate (default 0.30 Wh per
thousand tokens) so that hosted-versus-self-hosted comparisons are not
silently carbon-blind, and flags it as an anchor rather than a
measurement. Override it when a provider publishes audited figures.

---

## 3. Six-layer total cost of ownership

```
model_access(V)          = c_token · V + fee
compute_serving(V)       = ⌈capacity(V)⌉ · replica_cost + platform_eng
retrieval_data(V)        = F_r + a·V^b,           b ≈ 0.3–0.4
integration_sre(V)       = F_i
assurance(V)             = F_a + (c_review + c_storage)·V
workforce(V, y)          = W₁          if y = 1
                         = W₁·δ        otherwise
```

Only `model_access` is linear. A model containing only that layer
understates cost at low volume, where fixed layers dominate, **and**
overstates the benefit of efficiency work at high volume, where the
efficient layer shrinks and the others do not.

### 3.1 Human review

```
c_review    = Σ_i share_i · rate_i · minutes_i / 60 · wage
hours_year  = Σ_i share_i · V · rate_i · minutes_i / 60
FTE         = hours_year / 1700
```

`FTE` is reported and checked. A scenario demanding more reviewer hours
than an organisation could staff is arithmetic, not planning — and the
dollar total will look entirely reasonable while it happens.

### 3.2 Displaced labour

```
displaced = Σ_i share_i · V · baseline_minutes_i / 60 · wage
```

Reported **alongside**, never subtracted from, total cost. A labour
saving and a cash outlay are different instruments: one appears in a
budget line, the other in a capacity argument. Netting them silently is
how a business case stops being auditable.

---

## 4. Break-even

For candidate cost curves `C_A(V)` and `C_B(V)`, CAIDE evaluates
`Δ(V) = C_A − C_B` on a log-spaced grid, brackets every sign change, and
bisects in log space. Bisecting the logarithm keeps resolution uniform
across the six or seven orders of magnitude separating a pilot from a
national platform.

A staircase and a line can intersect many times. When they do,
enumerating the crossings implies precision the model does not have, so
CAIDE reports the **indistinguishable band** — the volume range over
which the options differ by less than a tolerance (default 5%) — and says
plainly that inside it cost does not decide.

---

## 5. Scaling dynamics

```
c(t) = c₀·(1 − r)^t
V(t) = V₀·(c(t)/c₀)^(−ε)·(1 + g)^t
S(t) = V(t)·c(t) = V₀·c₀·(c(t)/c₀)^(1−ε)·(1 + g)^t
```

The sign of `1 − ε` decides the direction of spend. Below `ε = 1`,
falling prices cut the bill. Above it — the regime reported for
compute-like inputs whenever a backlog of unmet uses exists — falling
prices **raise** it. The crossover is sharp and the model is closed form,
which is the point: three parameters, no fitting, an assumption made
explicit rather than buried.

`estimate_elasticity` regresses `ln V` on `ln c` and returns the negated
slope with its standard error and R². This is a **descriptive fit, not a
causal identification**: a new cohort, a mandate or a curriculum change
that moves price and volume together is absorbed into the slope.

---

## 6. Uncertainty

Monte Carlo propagation with uniform, triangular, normal, lognormal and
point distributions. Draws that raise or return non-finite values are
counted as failures rather than dropped, because a configuration
infeasible in 30% of draws is a finding.

Sensitivity uses **Spearman rank correlation**, not Pearson. The cost
model is full of step functions, roofline maxima and SLO cliffs; a
relationship that is monotone but strongly curved should not be scored as
weak. Contributions are normalised squared rank correlations, and tornado
bars report the median output at the 10th and 90th input percentiles so
they read directly: "moving this input across its plausible range moves
annual cost from A to B."

---

## 7. Where this model stops being reliable

| Assumption | Breaks when |
|---|---|
| Uniform token distribution within a class | Traffic is strongly bimodal; split it into two classes instead |
| Steady-state batching | Traffic is so bursty that queueing, not throughput, sets cost |
| One model per replica | Multi-LoRA serving shares a base model across adapters |
| Fixed accelerator price | Spot or preemptible capacity, where price is a distribution |
| Independent workload classes | Classes contend for the same replica and interfere |
| Linear tensor-parallel penalty | Beyond about 8-way TP, where topology dominates |
| Constant elasticity | Demand saturates; use `capacity_ceiling` |
| Quality indices as cardinal | They are ordinal placeholders — substitute your own evaluations |
| `B_half = 64` for decode utilisation | Hardware with a different GEMM efficiency curve; calibrate `decode_mfu_half_batch` |
| Plausibility ceilings | They warn, they do not refuse; a sweep may legitimately pass them |
| Uncalibrated utilisation constants | Any absolute capacity claim; calibrate against your own stack first |
| `framework_overhead_per_step = 0` | Predicting what a benchmark will report, as opposed to comparing configurations |
| Published figures without a stated convention | Anything; aggregate and per-request differ by the batch size |
| Generic model archetypes | A specific model whose shape differs materially from the archetype |

## 8. How accurate is it, actually

Version 3.0 of this document asserted that predictions land within a
factor of two of measured serving stacks. That claim had never been
tested. Version 4.0 tested it against four published measurements
spanning three serving frameworks:

| Configuration | Predicted | Measured | Ratio |
|---|---|---|---|
| 70B, 4×H100, bf16, high concurrency | 6,502 | 3,245 | 2.00 |
| 405B, 8×H100, FP8, 1024→2048 | 2,003 | 3,089 | 0.65 |
| 405B, 8×H100, FP8, 128→128 | 5,544 | 3,732 | 1.49 |
| 70B, 1×H100, FP8, batch 64 | 1,213 | 460 | 2.64 |

Aggregate output tokens per second. Sources are recorded in
`caide.calibration.REFERENCE_OBSERVATIONS`.

**Two of four fall inside the factor-of-two band, and the model
over-predicts in three of them.** The honest statement is therefore:

> Uncalibrated, CAIDE predicts serving throughput within roughly a factor
> of two to three, with a bias toward optimism. That is adequate for
> ranking architectures against each other, because the bias is largely
> common to all of them, and inadequate for absolute capacity commitments.

The four measurements are not a calibration set — different frameworks,
versions and datasets, and the bundled model archetypes are generic shapes
rather than the exact architectures measured. Tuning the constants until
they agree would fit that heterogeneity, not the physics.

The supported remedy is `caide.calibration`: supply two or more
measurements from *your own* hardware and stack, and `fit()` returns the
multiplicative correction to decode bandwidth utilisation that minimises
log-ratio error. On the four reference points it lifts the fraction inside
the band from 50% to 75% and reduces log-RMSE from 0.665 to 0.560 — which
is what a correction fitted to heterogeneous data should look like:
better, not good.

```python
from caide import fit_calibration, REFERENCE_OBSERVATIONS, Observation
result = fit_calibration(my_observations)   # your measurements
tuned = result.apply(serving_config)
```

**The roofline models the accelerator, not the serving stack.** A
published throughput figure is wall-clock and includes the scheduler, the
HTTP layer and detokenisation. vLLM's own profiling of an 8B model on one
accelerator attributed 33% of wall time to the API server and 29% to
scheduling, leaving 38% for GPU execution. `framework_overhead_per_step`
defaults to zero — a pure hardware roofline, which is the right default
for *comparing* configurations and the wrong one for *predicting a
benchmark*. Set it from your own measurements before doing the latter.
Because the cost is per step rather than per token, its relative weight
falls as batch rises: single-stream benchmarks test the framework, and
high-concurrency ones test the hardware.

**Measurement convention is not optional.** A benchmark reporting
"throughput" may mean the whole replica or one request, and the two differ
by exactly the batch size. `Observation.convention` must state which;
there is no safe default to guess, and three candidate observations were
excluded from the reference set during the v5.0 audit for failing to state
it. The exclusions are recorded in `EXCLUDED_OBSERVATIONS` rather than
silently applied.

Only decode utilisation is fitted. Published benchmarks are
decode-dominated, so prefill utilisation is barely identified by them;
fitting both jointly would assign the prefill parameter whatever absorbs
the residual, which looks like a better fit and is not one.

Use CAIDE to decide **what to benchmark**, then benchmark it, then
calibrate.
