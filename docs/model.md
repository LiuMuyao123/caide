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
t_memory  = (bytes_streamed(B) + kv_per_token · ctx · B) / BW_eff
t_compute = (2·N_active·B + 4·L·d_model·ctx·B) / F_eff
t_step    = max(t_memory, t_compute) + t_collective(B) + t_framework(B)
```

**Streamed, not resident.** `bytes_streamed` is not the weight footprint.
Two `V · d_model` matrices sit at the ends of the stack and behave
differently from the body and from each other:

```
bytes_streamed(B) = (N_body + N_expert·p_touch(B)) · w  +  N_head · w_head
weight_bytes      = (N_total − N_embed − N_head) · w   +  (N_embed + N_head) · w_head
```

The **input embedding** is a gather: a decode step reads `B` rows of it,
not the table. Counting the whole table in the stream overstated decode
traffic by 6.5% on the 8B archetype. The **head** is a real matmul and
streams every step — at `w_head`, which weight quantisation does not
touch. GPTQ, AWQ and the llama.cpp k-quants all skip the embedding and
the head, because the head is the one matrix whose quantisation error
lands directly on the output distribution and it buys the least memory
per unit of damage. Quantising it uniformly, as versions up to 9.0 did,
overstated int4's saving by 1.39× on the 8B archetype and 1.09× on the
70B one the paper sweeps.

The two errors pointed in opposite directions, which is why the total
looked reasonable for nine releases. `vocab_size` decides how large both
matrices are, and was declared in v1.0 and read by nothing until v10.0 —
the fifth dangling parameter this project has found in itself, and the
reason the omission had nowhere to become visible.

**Two arithmetic terms, not one.** The first streams the weights through
the tensor cores once per sequence. The second scores the new token
against every cached key and then weights every cached value: per layer
and per sequence that is `2·d_model·ctx` for QK^T and the same again for
AV. Versions up to 5.0 omitted the second term. The omission is invisible
wherever the KV cache is being streamed anyway, because the maximum
selects the memory term; it stops being invisible below batch 16 at
contexts past 32k, where decode FLOP utilisation is under one percent and
the machine's effective balance falls below attention's arithmetic
intensity of `n_heads / n_kv_heads` FLOPs per KV byte. In a sweep of 960
configurations the omission moved the step time by more than 10% in 77 of
them and flipped the binding resource in 69.

The same physics appears in the prefill term above. Modelling it in one
phase and not the other was an inconsistency independent of its numerical
size.

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

### 1.2b Tensor-parallel communication

Versions up to 5.0 charged tensor parallelism as a fractional derate on
both achievable bandwidth and achievable FLOP/s:

```
BW_eff = BW · N · MBU · (1 - 0.03·(N-1))        # up to v5.0
```

This is the wrong quantity. Under tensor parallelism each accelerator
streams its own weight shard out of its own HBM in parallel with the
others, so the aggregate bandwidth really is `N · BW`; nothing is lost to
sharding. What tensor parallelism costs is synchronisation, and a
synchronisation moves activations, not weights:

```
t_collective = 2 · L · (B · d_model · 2 bytes) · 2(N-1)/N / BW_interconnect
               · (1 + tensor_parallel_penalty · 10)
```

Two collectives per layer, one after attention and one after the MLP,
each moving one activation tensor, with the `2(N-1)/N` factor of a ring
all-reduce.

The difference is not cosmetic, because the two forms scale with
different things. The derate scaled with the parameter count; the real
cost scales with batch and model width. For a 405B model at TP=8 and
batch 64 the derate billed 5.75 ms per step against a true all-reduce
cost of 0.59 ms — an overcharge of 9.8×. That overcharge was large enough
to push one published measurement below the model's own hardware floor:
before the fix it implied a memory-bandwidth utilisation of 1.08, which
is to say the model asserted the measurement was impossible.

`tensor_parallel_penalty` survives as the interconnect-quality knob and
keeps its direction — zero is an ideal fabric, larger is worse — but it
now scales a term that is dimensionally the right one.

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

The count `B` is **tokens submitted to the router in one forward pass**,
not sequences in the batch. Under speculative decoding a verification
step submits `B·(γ+1)` tokens, each routed independently, so the expert
traffic is taken at that count. This matters because it inverts the
technique's benefit: for a dense model, verifying `γ+1` tokens streams
the weights once and traffic per token falls, whereas for MoE more
tokens reach more experts and the weight stream *grows*. Versions up to
v7.0 evaluated `p_touch` at `B`, handing MoE the dense amortisation for
free — up to 2.8× understated expert traffic at batch 1.

Under expert parallelism a step ends when the busiest expert finishes,
not the average one. `expert_imbalance` (mean/peak load) stretches the
*expert* share of weight traffic and arithmetic by `1/imbalance − 1`,
additively — the KV stream and the attention arithmetic are not sharded
by expert and take no part in the wait. Even a perfectly uniform router
has a straggler from counting fluctuation alone:
`uniform_routing_imbalance(B, E, k)` gives that ceiling in closed form
(`peak ≈ mean + √(2·mean·ln E)`), about 2× at batch 256 over 160
experts. Real routers sit at or below the ceiling. The default of 1.0
describes replicated or tensor-sharded experts, which is the
configuration the rest of CAIDE prices.

### 1.5 Capacity

```
memory_available = M · n_acc · μ − weight_bytes − N_draft · w
B_max            = memory_available / (kv_per_token · ctx_avg)
```

Residency uses `weight_bytes`, which holds everything including the input
embedding: the table has to be in memory even though a step only gathers
from it. Streaming and residency are different quantities and §1.2 gives
both.

`N_draft` is the speculative draft's parameter count (zero when
speculation is off). Until v7.0 the draft was charged in time — its
weight stream appears in every step — while its residency was free,
which is the same one-ledger inconsistency the decode-attention omission
was.

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

Decode steps are divided by this — and the step being divided into must
therefore price the arithmetic of `γ + 1` verified tokens, or the two
sides of the division describe different steps (they did, until v7.0).
One verification pass streams the weights once and computes
`(γ + 1) · (GEMM + attention)` per sequence, at the FLOP utilisation of
a `B·(γ+1)`-row matmul; the draft adds `γ` weight streams and `γ`
one-token arithmetic passes at its own row count; the all-reduce
synchronises `B·(γ+1)` tokens. The draft's KV cache is a **modelled zero**:
`draft_kv_ratio` defaults to 0 because a parameter ratio does not
determine a draft's layer count or head geometry, and set to a value it
is priced. The v7.0 text asserted the omission was "under 2% of the
step"; evaluating it at the parameter ratio gives 3.3% at the paper's
own operating points and 10.8% across the catalogue at the same context
lengths. A disproportionately deep draft should be modelled as its own
deployment.

At batch 1 the weight stream dominates and speculation divides it by the
accepted-token count: transformative, and unchanged by the v7.0
correction. At saturating batch the verification arithmetic is what
binds, the draft competes for the same units, and **the multiplier
crosses one** — the technique costs more than it saves, which is what
measured serving systems report at high concurrency.

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
E_busy     = P_board · accel_seconds
E_idle     = P_idle  · accel_seconds · (1/duty − 1)
E_facility = (E_busy + E_idle) · PUE
carbon     = E_facility/3.6e6 · intensity
water      = E_facility/3.6e6 · WUE
```

Idle capacity draws power — but not board power. `P_idle` is the
board's idle draw (`HardwareSpec.idle_power_watts`; catalogue values
sit in the 10–20% band published idle measurements cluster in, and are
marked as estimates). Until v7.0 the idle share was charged at `P_board`,
the one idle figure known to be wrong. Approximation, stated: scheduler
gaps inside the live period are also charged at idle draw, which
understates ramp power.

At the ownership level, energy walks the same staircase as dollars:
annual joules are working joules plus the idle remainder of every
*provisioned* replica's year at `P_idle`. Until v7.0 dollars were
stepped and joules were continuous, so below one full replica the model
billed the money of 1.0 replicas and the carbon of a fraction of one.

Provider-side energy for hosted APIs is **not measurable from outside**.
CAIDE reports a disclosure-dependent estimate (default 0.30 Wh per
thousand tokens) so that hosted-versus-self-hosted comparisons are not
silently carbon-blind, and flags it as an anchor rather than a
measurement. Override it when a provider publishes audited figures.

---

## 3. Six-layer total cost of ownership

A workload class may declare a `quality_floor` and a `latency_sensitive`
flag. A missed floor is reported with its **distance**: `quality_shortfall`
gives the gap as a fraction of the floor, because "inadmissible" is a
label pressed onto a continuous quantity and a candidate short by 1.2%
is a different proposition from one short by 22%. Both occur in the
shipped scenarios. Both are constraints, both are checked on every path since v11.0,
and a class declared latency-insensitive that misses the objective is
recorded rather than disqualifying.

The quality index an architecture is checked against comes from the
deployment state, and the efficiency techniques edit it the way they edit
every other attribute: `apply_stack` composes the stack's `quality_delta`
on retention. Lossless techniques declare zero — speculative decoding is
exact, so it costs nothing here — and distillation derives its loss from
the student's geometry rather than quoting it.

 An architecture whose
quality index falls below it is **not admissible for that class** — the
classes are listed in `TCOResult.quality_violations` and the result is
`feasible == False`. The shortfall is named, never priced: CAIDE does not
know what a capability gap costs. Through v9.0 the floor was enforced
only when routing, so the architecture comparison ranked options that
could not serve the workload; in all three shipped scenarios the one
reported cheapest failed a floor, and in one no single architecture meets
them all. That case is what `caide route` exists for.


```
model_access(V)          = c_token · V + fee
compute_serving(V)       = ⌈capacity(V)⌉ · replica_cost + platform_eng
retrieval_data(V)        = F_r + a·V^b,           b ≈ 0.3–0.4
integration_sre(V)       = F_i
assurance(V)             = F_a + (c_review + c_storage)·V
workforce(V, y)          = W₁          if y = 1
                         = W₁·δ        otherwise
```

Two layers are substantially linear in `V`, not one: `model_access`, and
the review-and-retention part of `assurance`. Until v8.0 this section
and the module docstring both called assurance volume-free, which was
the wrong claim about the wrong layer — in all three shipped scenarios
assurance is the **largest** of the six (58%, 91%, 84% of ownership
cost) and 83–95% of it moves with volume.

A model containing only `model_access` therefore understates cost at low
volume, where fixed layers dominate, **and** overstates the benefit of
efficiency work at high volume — the second error being larger than the
old table implied, because the layer that does not shrink when token
prices fall does grow when volume rises.

Because a table is an assertion, `layer_volume_elasticity` measures
`d ln(cost) / d ln(V)` per layer by central difference instead: 1.0 for
a linear layer, 0.0 for a fixed one, `b` for the sublinear one, and
whatever the staircase is locally doing for the stepped one. Measured on
the shipped scenarios it returns 0.82, 0.95 and 0.93 for assurance.

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
CAIDE reports the volumes over which the options differ by less than a
tolerance (default 5%) and says plainly that there cost does not decide.

That set is a **union of intervals, not an interval**. Against a
staircase it is several narrow windows, one around each riser: the
self-hosted curve is flat across a tread while the API line keeps
rising, so the two agree only briefly near each step. Versions up to
8.0 reported first-to-last of the qualifying points as a single band,
which on the shipped comparison spanned 175 scan points of which 137
exceeded the tolerance and the worst reached 47%. `tie_bands` returns
every window; `tie_band` returns the widest one and is documented as
such. Since v15.0 every consumer uses the former: the terminal, the
Markdown report and the HTML dashboard each list the windows, their
widths and the worst gap between them, having until then described the
whole span as a single band.

Both `relative_gap` and `Crossing.margin_pct` divide by the cheaper of
the two options, so a figure reads as "how much more the loser costs".
Until v9.0 the second divided by the mean instead.

---

## 5. Scaling dynamics

Not all of a query's cost is a token price, and the parts behave
differently:

```
c_decl(t)  = c₀·(1 − r)^t                     tracks the token tariff
c_inel                                        per query, at wage rates
F                                             per year, volume-free

c_eff(t)   = c_decl(t) + c_inel + F/V(t)      what the buyer pays
V(t)       = V₀·(c_eff(t)/c_eff(0))^(−ε)·(1 + g)^t
S(t)       = c_eff(t)·V(t)
           = c_eff(0)^ε·V₀·c_eff(t)^(1−ε)·(1 + g)^t
```

`V` appears on both sides because fixed cost amortises over the volume
being solved for; each year is closed by damped iteration, and with
`c_inel = F = 0` the first pass is exact, which is why pre-v8 scenarios
reproduce to the last digit.

The sign of `1 − ε` still decides the direction of spend, and the
crossover at `ε = 1` is still exact — that is structural and survived
the v8.0 correction untouched. What the composition changes is
**magnitude**. In the shipped education scenario only 10.4% of cost per
query tracks the tariff, so a 38%/yr tariff decline is a 9%/yr decline
in what the institution pays: five-year spend at `ε = 1.35` moves from
3.07× to 1.82×, and at `ε = 0.6` from 0.73× to **1.38×** — a change of
sign. Feeding a blended per-query figure into a tariff decline, as
versions up to 7.0 did, declines reviewer wages at the speed of GPU
prices.

`estimate_elasticity` regresses `ln V` on `ln c` and returns the negated
slope with its standard error, its 95% interval and R². This is a
**descriptive fit, not a causal identification**: a new cohort, a mandate
or a curriculum change that moves price and volume together is absorbed
into the slope.

The reported `regime` is `undetermined` when that interval contains one.
The crossover decides the direction of spend, so a verdict about which
side of it an estimate falls on is worth issuing only when the estimate
can tell the two apart; on three-point histories — the minimum the
function accepts — it usually cannot. Through v11.0 the regime was read
off the point estimate while the standard error sat unused in the same
returned dictionary, so a slope of 0.91 ± 0.14 was labelled "inelastic"
as firmly as one of 0.70 ± 0.05. `point_regime` records what that rule
would have said.

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

A ranking answers "of the inputs we varied, which mattered", and it is
silent about three further things until they are reported alongside it.

It is silent about **failed draws**: they are excluded from the
correlation, so an input whose whole effect is to make the configuration
infeasible has no variation left among survivors and scores zero. In a
constructed case an input that broke 10.7% of draws received 0.0% of
explained variance. `SensitivityEntry.failure_spearman` gives its
association with failure instead; the failure is never imputed a cost.

It is silent about **how much is explained**: contributions are shares of
the sum of squared rank correlations and total one regardless.
`explained_rank_variance` reports that sum — 0.97 on the education
scenario, meaning the order is worth reading.

And it is silent about inputs nobody varied. Those are implicit point masses, and
their absence is invisible in exactly the chart a reader would use to
notice it. Per-query review cost is `rate × minutes × wage`; through
v8.0 only the rate had a draw key, so the other two factors of the
largest cost term in every shipped scenario could not move at all.
Declaring all three changes which input ranks first — review minutes
displaces volume. `uncovered_draw_keys` lists the perturbable inputs a
scenario left without a distribution, so a report can state what was
held fixed alongside what moved.

---

## 7. Where this model stops being reliable

| Assumption | Breaks when |
|---|---|
| Uniform token distribution within a class | Traffic is strongly bimodal; split it into two classes instead |
| Steady-state batching | Traffic is so bursty that queueing, not throughput, sets cost |
| One model per replica | Multi-LoRA serving shares a base model across adapters |
| Fixed accelerator price | Spot or preemptible capacity, where price is a distribution |
| Independent workload classes | Classes contend for the same replica and interfere |
| Ring all-reduce for tensor-parallel sync | Beyond about 8-way TP, where topology dominates and a ring is no longer the right collective |
| Constant elasticity | Demand saturates; use `capacity_ceiling` |
| Quality indices as cardinal | They are ordinal placeholders — substitute your own evaluations |
| `B_half = 64` for decode utilisation | Hardware with a different GEMM efficiency curve; calibrate `decode_mfu_half_batch` |
| Plausibility ceilings | They warn, they do not refuse; a sweep may legitimately pass them |
| Uncalibrated utilisation constants | Any absolute capacity claim; calibrate against your own stack first |
| A single tensor-parallel derate | Removed in v16.0: interconnect cost scales with batch and model width, so it is computed as a per-layer all-reduce, not applied as a constant |
| `framework_overhead_per_step = 0` | Predicting what a benchmark will report, as opposed to comparing configurations |
| `expert_imbalance = 1.0` | Expert-parallel serving, where the step ends with the busiest expert, not the average one |
| Uniform expert routing in `expert_bytes_touched` | Routers with learned skew, which touch fewer distinct experts than the binomial model assumes |
| The embedding and head held at bf16 under quantisation | Schemes that do quantise them, or models with tied embeddings and unusual vocabularies; set `head_bytes_per_param` |
| Quoted `quality_delta` constants, size-independent | A four-bit 8B model degrades more than a four-bit 405B; the catalogue applies one delta to both |
| A commercial endpoint's latency | Not modelled at all; recorded as unevaluated, never as met, and an architecture whose constraints were not all checked is flagged |
| `provider_energy_wh_per_ktok` | A disclosure-dependent estimate and the sole determinant of every API carbon figure; set it per scenario |
| A quality index as a scalar admissibility test | Anything finer than "this tier is capable enough". The index is a declared ratio scale, not a measurement, and capability is not one-dimensional |
| Size-independent `quality_delta` | A four-bit 8B model degrades far more than a four-bit 405B; one constant is applied to both, and every non-zero delta now states that in `Technique.quality_basis` |
| Speculative decoding as a batch-one win | Mixture-of-experts targets, where verification routes γ+1 tokens per sequence and the multiplier at batch one is 1.005×, not 0.40× |
| A sensitivity ranking as a ranking of the model's inputs | Any input without a declared distribution; check `uncovered_draw_keys` before reading a tornado chart |
| Per-class independence in routing | A tier with `max_share < 1` or a non-separable `annual_cost_fn`; the plan is then enumerated exactly, or flagged `exact=False` |
| `draft_kv_ratio = 0` | Speculative drafts deep enough for their KV cache to matter; at the parameter ratio the omitted term is 3.3% of the step at the paper's operating points and 10.8% across the catalogue |
| A single price decline applied to blended cost | Any forecast: reviewer wages and audit programmes do not track token tariffs, and separating them changes the five-year answer by a factor of three |
| Constant `c_inel` over the horizon | Wage inflation, or a review policy that tightens as volume grows |
| Published figures without a stated convention | Anything; aggregate and per-request differ by the batch size |
| Generic model archetypes | A specific model whose shape differs materially from the archetype |

## 8. How accurate is it, actually

Version 3.0 of this document asserted that predictions land within a
factor of two of measured serving stacks. That claim had never been
tested. Version 4.0 tested it against four published measurements
spanning three serving frameworks:

| Configuration | Predicted | Measured | Ratio | Implied MBU |
|---|---|---|---|---|
| 70B, 4×H100, bf16, high concurrency | 3,178 | 3,245 | 0.98 | 0.33 |
| 405B, 8×H100, FP8, 1024→2048 | 2,078 | 3,089 | 0.67 | 0.90 |
| 405B, 8×H100, FP8, 128→128 | 3,349 | 3,732 | 0.90 | 0.31 |
| 70B, 1×H100, FP8, batch 64 | 1,120 | 460 | 2.44 | 0.26 |
| 8B, 1×A100, bf16, 8 concurrent | 601 | 187 | 3.21 | 0.19 |

Aggregate output tokens per second, over the **whole serving cycle** —
prefill included, because that is what a wall-clock benchmark divides
by, and it is the model's own definition of throughput. Up to v6.0 this
table divided by decode time alone, a second derivation of the same
quantity that disagreed with `evaluate_request` by the prefill share of
the cycle (54% for the first row). Sources are recorded in
`caide.calibration.REFERENCE_OBSERVATIONS`; the Implied MBU column is
unchanged, because the physical-bound test is about step time, not
cycle time.

The fifth row was excluded by the v5.0 audit as ambiguous between the
aggregate and per-request conventions. Version 6.0 readmitted it after
showing the ambiguity is decidable: the roofline is a *lower* bound on
step time, so a reading implying a memory-bandwidth utilisation above 1.0
can be rejected on physics rather than on judgement. The per-request
reading implies 1.60. See `admissible_conventions()`.

The **implied MBU** column is the bandwidth utilisation each measurement
would require of the hardware. It is the same information as the ratio,
recast so that the physical ceiling is visible: a ratio of 0.65 reads as
"modestly conservative", while the implied MBU of 1.08 it corresponded to
in v5.0 reads as "the model says this measurement cannot exist". That
reading is what exposed the tensor-parallel penalty as mis-specified —
see below — and after the fix the same row implies 0.90, which is high
but possible.

**Three of five fall inside the factor-of-two band; the model
over-predicts in two and under-predicts in three.** Since v13.0 this
table is regenerated by Result 8 of the reproduction script rather than
transcribed: through v12.0 it was carried by hand, and two of its figures
had drifted since the v10.0 weight-stream correction without anyone
noticing, because nothing recomputed them. The honest statement is
therefore:

> Uncalibrated, CAIDE predicts serving throughput within roughly a factor
> of two, worst case three, with no longer a one-sided bias. That is
> adequate for ranking architectures against each other and still
> inadequate for absolute capacity commitments. Calibration against a
> user's own measurements halves the log error (0.68 to 0.54) without
> moving any observation across the two-times boundary: the fraction
> inside it stays at 60%, the worst observation calibrating to 2.01.

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

`FRAMEWORK_OVERHEAD_REFERENCE` supplies a starting pair, and since v11.0
it is labelled for what it is: **a declared modelling choice, not a
validated measurement**. Two observations fit two parameters, so the
residual is exact by construction and these points cannot contradict any
change to the hardware model. `FRAMEWORK_OVERHEAD_HISTORY` records what
that produced across five releases:

| version | per_step | per_sequence | what moved |
|---|---|---|---|
| 6.0.0 | 0.0127 | 0.00226 | two-point fit introduced |
| 7.0.0 | 0.0127 | 0.00202 | prefill removed from the wall clock |
| 10.0.0 | 0.01345 | 0.00226 | input embedding removed from the stream |

`per_sequence_seconds` has returned to where it began: one correction
pushed it down and another pushed it back. A residual that can absorb a
correction and then absorb its reversal is measuring the model rather
than the framework. `FRAMEWORK_OVERHEAD_SENSITIVITY` gives the band the
constants have occupied; treat that as the interval, not the point.

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
