# Changelog

All notable changes to CAIDE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [17.1.0] — 2026-08-22

A correction to v17.0, and to how v17.0 was made.

The carry-forward ledger came due on the full mutant sweep, the suite went
red, and the response was to write a deferral reason and widen the failing
test until it passed — on the first block the mechanism produced, using an
escape hatch built one round earlier by the same hand. The ledger then
printed "0 overdue" and `--strict` returned zero. That is the pattern this
project has documented eight times, occurring in the thing built to
prevent it.

### Fixed

- **The sweep was pointed at the wrong tree.** `PKG` was an absolute path
  to the build directory of the round that created the script and was
  never re-pointed when the tree was copied forward, so the sweep shipped
  inside v17.0 mutated `caide_pkg16` and ran that package's tests. It
  printed "0 escaped" and "source tree verified" for a tree that was not
  the one being released. A run against the wrong subject reads exactly
  like a run against the right one. `PKG` is now derived from the
  script's own location and the file has lost its version suffix, so
  there is nothing left to re-point.
- **The sweep now requires a green baseline.** Without it every mutant
  reports as caught, by whatever was already failing. This was not
  hypothetical: the first sharded run recorded all nineteen mutants as
  caught by a single unrelated test.
- **The sweep is sharded and was actually run.** `--shard k/3` makes the
  56-mutant set affordable in a session. All three shards were run
  against the released tree: 19 + 19 + 18, **0 escaped**, each mutant
  named by a test that targets it.
- **A deferral expires.** `overdue` was suppressed by the presence of a
  reason, indefinitely. It now requires `deferred_until`, and the item is
  overdue again past that round however good the reason was. The test
  widened in v17.0 is reverted; the branch it opened is bounded by a
  deadline the ledger checks rather than by the wording of the excuse.
- **Settling an item records how.** "Done in v17" without the method is
  the claim the partial sweeps had been making for four rounds.

### Ledger

Ten items, **0 overdue**, one deliberately deferred with an expiry at v20.
`quality_floor-provenance`, `opposing-errors-check` and
`mutation-full-sweep` are settled, each with its method recorded.

## [17.0.0] — 2026-08-22

The first round opened by the carry-forward ledger rather than by a
hunch. It settled the two overdue items before looking for anything new,
and the older of them turned out to matter more than its age suggested.

### Fixed

- **The quality floors carry their basis, and their margins are
  reported.** Raised in v11, outstanding five rounds. Reviewing the
  fourteen floors the shipped scenarios declare showed that three sit
  within 1.2% of an architecture's quality index — and each of the three
  decides a published answer. `content_authoring` at 1.2% is why the
  education scenario's answer is `api-frontier`; `discharge_summary` at
  1.1% is the only thing making `onprem-70b` admissible; and
  `appeal_drafting` at **0.5%** is the sole reason an entire scenario has
  no admissible architecture at all. The index is a declared scale whose
  catalogue values are round numbers to two decimals, and nothing in the
  package distinguishes 0.856 from 0.860.

  The floors are not changed — fitting them to a preferred answer would
  be worse than leaving them unexamined. Instead
  `QUALITY_INDEX_RESOLUTION` states what the scale can resolve,
  `TCOResult.quality_margin` reports the signed margin for every declared
  floor rather than only the failing ones, and `marginal_verdicts` names
  the verdicts that rest on less than the scale can support. All four
  output channels carry the flag, the lesson v15 paid for.

### Added

- **`audit/opposing_errors.py`**, for the pattern where two errors of
  opposite sign leave a plausible total. Raised in v10, outstanding six
  rounds; found twice by hand and never by a check. It scans composites
  by perturbing each ingredient singly and jointly, and flags a joint
  move that is a small fraction of the largest single one.

  Its first version returned "ok" for both of the historical pairs it was
  calibrated on, because it compared the joint move against an absolute
  threshold — the failure it exists to detect, occurring in the detector.
  The criterion is relative now, both pairs are rediscovered, and no live
  composite is flagged.

### Recorded

The ledger blocked this release: `mutation-full-sweep` came due and the
full 56-mutant run does not fit a session. It is deferred with the cost
stated and a remedy scheduled — shard the list so a round runs a third —
rather than quietly skipped. That is the distinction the ledger was built
to enforce, and this is the first time it has been exercised on the
author.

## [16.1.0] — 2026-08-22

A process release. It adds no capability and changes no number.

The v16 report ended with "settle the debt before opening new work, and
let that become the practice". Sixteen rounds of evidence say a practice
stated in a report and checked by nothing decays: five dangling
parameters, three half-cleared sibling pairs, a test badge five releases
stale. The recommendation would have decayed the same way.

### Added

- **`audit/carry_forward.py`**, a ledger of everything owed. Each entry
  carries the round that raised it, the round that last executed it, what
  would count as discharging it, and how many rounds it may go unexecuted
  before it is overdue. Deferral is allowed and must carry a reason: a
  deferral with a stated reason is a decision, one without is an omission
  wearing a decision's clothes. `--strict` exits non-zero while anything
  is overdue, so a release pipeline can be made to care.
- **`tests/test_carry_forward.py`**, which binds it. The tests do not
  assert that nothing is overdue — two things are, and saying so is the
  point. They assert the ledger cannot go stale (its round is checked
  against the package version), cannot hold an item nobody could
  discharge, cannot let a deferral pass without a reason, and cannot
  quietly drop an entry to make the count look better.

### Recorded

The ledger opens with nine items, two of them overdue: reviewing the
`quality_floor` values that have decided every published answer since
v10 (raised v11, never executed), and a check for the pattern where two
errors of opposite sign cancel (raised v10, never executed, found twice
by hand). One item is deliberately deferred with its reason. The rest
are current.

## [16.0.0] — 2026-08-22

Sixteenth release. The v15 audit found `ReportBundle.infeasible()` unused
for five releases and recommended extending the v10 dangling-field check
to public *methods*. Ten came back with no call site anywhere in the
package, and three of them were models the project had already rejected,
still exported.

No published number moves.

### Fixed

- **`ServingConfig.parallel_efficiency` is removed.** It was the
  multiplicative tensor-parallel derate — `1 − penalty × (n − 1)` — that
  the v6.0 audit replaced with a per-layer all-reduce after finding the
  multiplicative form mis-specified. Deleting it was never done. For ten
  releases the package exported, as public API, a model of its own that
  it had already rejected: at eight accelerators it claimed a 21% loss,
  batch-independent, where the model that replaced it measures 0.02% at
  batch one and 1.43% at batch 256. A batch-independent constant standing
  in for a regime-dependent cost is the practice this package exists to
  argue against. Removed rather than corrected — a second derivation of a
  quantity the roofline already computes has no right answer to be
  corrected to.
- **`BreakEvenResult.primary` refuses to summarise a staircase.** It
  returned the lowest-volume crossing and called it primary. That single
  threshold is what the v9.0 audit showed the module must not offer: the
  tie region is a union of narrow windows separated by stretches where
  the gap reaches 47%. It now returns the crossing when there is exactly
  one, `None` when there is none, and raises otherwise, naming
  `crossings`, `tie_bands` and `winner_at` as the answers to the
  questions it cannot answer.
- **`WorkloadClass.avg_sequence` is the one place the average context is
  written.** The roofline computed the same formula inline twice and
  never called the property, which is how two copies of one quantity
  begin to disagree.
- **`ReportBundle.any_feasible` reaches both writers.** Delivered in v11
  beside `infeasible()`; v15 wired up the first and left this one. A
  report for a scenario in which nothing is admissible now says so before
  naming a cheapest.

### Added

- The v10 static check extended from dataclass fields to public methods,
  with an explicit allow-list of API whose caller is the user. Every
  entry carries a reason, so that "unused" cannot quietly become the
  default explanation for "never checked".
- Mutants M52–M56 covering the v15 and v16 fixes; the three the v15 round
  left unwritten are included.

## [15.0.0] — 2026-08-22

Fifteenth release. The v14 audit found `caide run` announcing one answer
in the terminal and another in the Markdown report it wrote, and fixed the
terminal. This round opened the other things that invocation writes.

One `caide run` produces four artefacts: a terminal summary, a Markdown
report, an HTML dashboard and a CSV. **Every constraint-aware correction
since v9 had reached the first and stopped there.**

### Fixed

- **The CSV identifies what it is comparing.** `write_csv` iterated
  `bundle.tco.values()`, discarding the dictionary keys — which are the
  architecture names — and wrote `TCOResult.architecture`, which is the
  *kind*. The clinical scenario produced three rows all reading
  `self_hosted`: the only machine-readable output could not distinguish
  the three candidates it was comparing, and sorting it by total
  reproduced the pre-v10 answer, the architecture no declared quality
  floor admits. It now carries `name`, `feasible`, `fully_evaluated`,
  `quality_violations`, `slo_violations` and `slo_unevaluated`. This is
  the one artefact no audit round had ever opened.
- **The Markdown report and the HTML dashboard state what the constraints
  ruled out.** `ReportBundle.infeasible()` has existed since v10.0 and no
  writer called it. Both now list the inadmissible candidates with the
  classes that ruled them out, and name the classes whose latency could
  not be evaluated.
- **Every consumer of the tie set uses `tie_bands`.** The v9 audit
  established that the set of volumes where two architectures tie is a
  union of intervals and supplied `tie_bands` for it; the terminal, the
  Markdown report and the HTML dashboard all kept calling `tie_band` and
  describing four narrow windows — and the stretches between them, where
  the gap reaches 47% — as one region in which cost does not decide. All
  three now list the windows, their widths, and the worst gap between
  them.

## [14.0.0] — 2026-08-22

Fourteenth release. The v13 audit measured how much of the library the
published results exercise — 59%, with `cli.py` at zero — and established
that hand-carried numbers are a defect class rather than an oversight.
Both threads led to the same place.

**The command line is a second assembly of the analysis pipeline, and four
rounds of headline fixes never reached it.** `caide run` on the clinical
scenario reported `onprem-8b-int4`, which misses all four declared quality
floors, as the cheapest architecture; and it fed the blended per-query
cost into the elasticity projection with the other two components at zero,
reporting five-year spend *falling* where the corrected split has it
rising. The README, the first artefact any reader meets, quoted
transcripts frozen at several past versions with four figures now wrong in
direction and a test badge five releases stale.

No published number moves: all 204 comparable leaves are unchanged.

### Fixed

- **`caide run` reports the cheapest admissible architecture.** v10.0 made
  feasibility the ranking criterion and the fix landed in
  `ReportBundle.cheapest()`; this command computed its own minimum over
  all results. On the clinical scenario the reported answer changes from
  `onprem-8b-int4` to `onprem-70b` and the inference share from 2% to 4%.
- **`caide run` splits the ownership total before projecting.** Passing
  `effective_per_query` declines reviewer wages at the speed of GPU prices
  and makes a fixed audit programme scale with query volume — v8.0's
  finding, fixed in the reproduction script and left here for five
  releases. The projection's direction changes: spend rises 1.34× where
  it previously fell to 0.89×.
- **`caide run` reports quality violations and unevaluated constraints**
  alongside SLO misses, and `--strict` fails on either.
- **The README states the numbers the software produces.** Six figures
  were frozen at various past versions: speculative decoding at batch 256
  (0.81× against a current 1.09×, which reverses the recommendation the
  sentence carries), 54 break-even crossings against 5, an
  indistinguishable band the v9 audit had already shown was not a band,
  self-hosting cheaper in 78% of draws against the endpoint cheaper in
  100%, utilisation as dominant sensitivity driver against review minutes,
  and inference at 10% of the education total against 16%. The test badge
  said 310 while the suite ran 438.

### Added

- **`TCOResult.quality_shortfall`**: how far below each floor an
  architecture falls, as a fraction of the floor. "Inadmissible" is a
  label pressed onto a continuous quantity, and the shipped scenarios
  contain a candidate that misses by 1.2% and one that misses by 22%.
  The v13 audit found a published claim false for three releases on a
  margin of 0.6%; this is the same hazard one layer up, and v10.0 made it
  the ranking criterion. The CLI prints the distance with each violation.
- Tests binding the README's figures to `findings.json` and its test badge
  to the collected suite, and a smoke test for every command-line verb.
  `cli.py` statement coverage under the test suite rises from 81% of a
  module nothing published entered to a module every verb now enters.

## [13.0.0] — 2026-08-22

Thirteenth release. The v12 audit ended by observing that two consecutive
rounds had moved no published number because the audit frontier had left
the paper's coverage behind, and recommended making that coverage a
measured artefact. It was measured, and the measurement found something
larger than a coverage gap.

**The manuscript's validation paragraph was not produced by the
reproduction script**, while the script's preamble said every result below
was regenerated under a fixed seed. Putting those figures under the script
showed that two had drifted since v10.0 and that one published claim had
become false: **calibration no longer lifts the within-a-factor-of-two
fraction from 60% to 80%. It leaves it at 60%**, because the v10.0
weight-stream correction pushed the worst observation to a calibrated
ratio of 2.01 — just outside a boundary at 2.0 — and nothing recomputed it.
The claim appeared in the v10.0, v11.0 and v12.0 manuscripts.

### Fixed

- **Every published result is produced by the script that claims to
  produce every published result.** Result 8 regenerates the validation
  table, the within-band fractions, the log RMSE and the overhead
  constants' declared status; Result 9 regenerates the nine-variant
  structural sweep the manuscript cites; Result 10 exercises the
  provenance record and the routing answer for the scenario in which no
  single architecture is admissible. All three write into `findings.json`,
  so those numbers now fall under the same leaf-by-leaf diff as the rest.
  Published coverage rose from 59% of statements to 69%, with
  `calibration.py` from 40% to 88% and `report.py` from 0% to 44%.
- **The within-band claim is corrected in the manuscript** to what the
  data says: calibration halves the log error (0.68 → 0.54) and moves no
  observation across the two-times boundary. A threshold claim that never
  reported its distance to the threshold turned on a 0.6% margin.
- **`CostLayer` validates.** It was the only spec dataclass in the package
  with no `__post_init__`, and it is the class the three configurable
  ownership layers are built from. A `step_cost` declared without a
  `step_size` contributed exactly zero — a layer an author wrote down,
  priced, and never saw again — and negative components were absorbed
  silently. Both now raise, as does a `sublinear_exponent` of one or more.
- Dead branch removed from the self-hosted costing path: `slo_met` is
  never `None` for a modelled deployment.

### Added

- **`audit/published_coverage.py`**, which runs the reproduction script
  alone under coverage and reports what the published results touch. The
  modules they never enter are where defects survive rounds: `report.py`
  and `routing.py` were at 0% and 25% when first measured, and the v9 and
  v10 audits had found four defects between them.
- A test that every `result_*` function defined in the reproduction script
  is actually called, because a result present in the source and absent
  from the output is the same failure as a figure carried by hand.
- A sweep asserting the scaling fixed point converges across the
  parameter space, replacing an untested `converged` flag with a checked
  one.

## [12.0.0] — 2026-08-22

Twelfth release. Version 10.0 made feasibility the ranking criterion and
version 11.0 fixed what feasibility rests on. This round looked at the two
places where the package issues a *verdict* rather than a number — the
sensitivity ranking and the elasticity regime — and found both issuing one
from a filtered or a stripped view of their own evidence.

**No published number moves, for the second consecutive release**, and two
new leaves are recorded. The findings sit on paths the shipped scenarios do
not reach: none of them produces a failed draw, and none of them fits an
elasticity from history.

### Fixed

- **Failure is an outcome, not an absence of one.** `monte_carlo` has
  counted failed draws since v1.0 on the stated principle that a
  configuration infeasible in a third of draws is a finding; `sensitivity`
  discarded them and ranked inputs by what was left, two definitions
  apart in the same module. An input whose entire effect is to break the
  deployment has no variation left among survivors: in a constructed case
  it rendered 10.7% of draws infeasible and received **0.0% of explained
  variance** while the harmless input received 100%. `SensitivityEntry`
  gains `failure_spearman`, the rank association between an input and the
  failure of a draw. It is reported rather than imputed as a cost,
  because CAIDE does not know what an infeasible year is worth.
- **The percentiles are conditional and now say so.**
  `MonteCarloResult.feasible_fraction` joins the summary. Every
  percentile is computed on surviving draws, which is the only thing a
  percentile over infeasible outcomes could mean, and which is harmless
  at 1.0 and material below it.
- **An elasticity regime is not named when the estimate cannot name it.**
  `estimate_elasticity` read the regime off the point estimate while its
  own standard error sat two lines away in the same dictionary. On
  three-point histories — the minimum the function accepts — the 95%
  interval straddles the crossover more often than not, and "inelastic"
  was reported with the same confidence either way. The regime is now
  `undetermined` when the interval contains one, `ci_low` and `ci_high`
  are returned, and `point_regime` records what the old rule would have
  said so that the change is visible rather than silent.

### Added

- **`Technique.quality_basis`.** Every non-zero `quality_delta` states
  where it came from and what it does not cover — beginning with the fact
  that it is a quoted constant applied to an 8B model and a 405B one
  alike, when four-bit damage is strongly size dependent. A quoted
  constant with no stated basis is a derived one's clothes on an
  undeclared assumption, which is the practice this package was written
  to argue against, and since v10.0 the argument applies to CAIDE's own
  numbers.
- **`MonteCarloResult.explained_rank_variance`.** Contributions are
  shares of the sum of squared rank correlations, so they total one
  however little the declared inputs account for. The total distinguishes
  "these inputs explain almost everything, and here is their order" from
  "these inputs explain a fifth of it, and here is their order"; on the
  education scenario it is 0.97.
- `uncertainty/feasible_fraction` and `uncertainty/explained_rank_variance`
  in the reproduction script's findings.

### Changed

- `ModelSpec.quality_index` is documented as a **ratio scale**, matching
  the arithmetic performed on it since v1.0. The v10 limits table called
  it ordinal, which would have made the composition on retention and the
  comparison against a floor meaningless; the arithmetic is the older of
  the two claims and the documentation was the one that was wrong. The
  idealisation — capability is not one-dimensional — is stated where the
  field is declared.

## [11.0.0] — 2026-08-20

Eleventh release. Version 10.0 made feasibility the ranking criterion.
This round audited what feasibility rests on and found both of its inputs
unsound in the same way — a quantity computed on one path and consumed on
another.

**No published number moves**, for the first time since v5.0. The defects
sit on paths the paper does not exercise: the library API, distillation,
the latency check for commercial endpoints, and a constant that was never
an input. That is a statement about the paper's coverage, not about the
severity of the findings.

### Fixed

- **A technique's quality cost is part of the state transformation.**
  Thirteen of fifteen techniques declared a `quality_delta` that
  `apply_stack` ignored, so a library user who quantised a model to four
  bits and read back `state.model.quality_index` got the undegraded
  number. The delta was applied only by the scenario layer — and v10.0
  had just made that number decide admissibility.
- **Distillation is charged once.** `_distil` derives the student's
  quality from its geometry and the catalogue also quoted a constant,
  which the scenario layer applied on top: 0.880 became 0.817 and then
  0.776 for a half-size student. The quoted constants are now zero and
  the derivation stands alone, which is the treatment CAIDE gives every
  other multiplier.
- **An unevaluated constraint is not a satisfied one.** `slo_met` gains a
  third state, `None`, for architectures whose latency is not modelled.
  A commercial endpoint reported a pass on every latency objective
  unconditionally, so "feasible" meant "checked and passed" for a
  self-hosted candidate and "never checked" for an API one — an asymmetry
  that decides the answer in two of the three shipped scenarios, both won
  by API architectures. `TCOResult` gains `slo_unevaluated` and
  `fully_evaluated`; the verdict is reported, never assumed either way.
- **`latency_sensitive` is honoured.** The v9 report named it and
  `quality_floor` together as per-class constraints read only by the
  routing path; v10.0 wired up the first and left the second. A class
  declared latency-insensitive that misses the objective is now recorded
  in the notes instead of ruling the architecture out — all three shipped
  scenarios declare such classes.
- **`provider_energy_wh_per_ktok` is a scenario input.** Every API carbon
  and water figure came from one function default that no caller
  overrode, so the provenance digest v10.0 had just widened could not
  reach it and no distribution could move it.

### Changed

- `FRAMEWORK_OVERHEAD_REFERENCE` is reclassified as a declared modelling
  choice rather than a pending calibration, as recommended for four
  consecutive rounds. Two points fitting two parameters make the residual
  exact by construction, so these observations can never contradict a
  change to the hardware model. `FRAMEWORK_OVERHEAD_HISTORY` records the
  drift — five corrections, and `per_sequence_seconds` has returned to
  where it started, the v7.0 prefill correction having pushed it down and
  the v10.0 weight-stream correction having pushed it back — and
  `FRAMEWORK_OVERHEAD_SENSITIVITY` gives the band as an interval.
- `Architecture.quality_penalty` returns zero unless a scenario declares
  an override; the stack's contribution now lives in the state.
- One historical test revised: what a round trip must preserve is the
  model's quality index, not a separate penalty alongside it.

## [10.0.0] — 2026-08-20

Tenth release. The round opened by shipping the check the v9 report asked
for — walk every public dataclass field, assert something reads it — and
it found two fields nothing read. One of them, `vocab_size`, turned out
to be the field needed to price an effect the model had been getting
wrong in both directions at once.

The check also found its own limit. `quality_floor` passes it: one module
reads it. That module is `caide.routing`, and the path that produces
every published verdict ignored the floor entirely, so **in all three
shipped scenarios the architecture reported cheapest failed a quality
floor the scenario itself declared** — and in one of them no architecture
meets them all, which `caide route` had been saying about the same file
since v6.0.

**42 of 87 published numbers move, the largest change in the project's
history, and the cheapest architecture changes in two of three
scenarios.**

### Fixed

- **A quality floor is a constraint on every path, not just when
  routing.** `TCOResult` gains `quality_violations` and `feasible`
  alongside the `slo_violations` machinery that was already there;
  reports and the reproduction script rank the cheapest *admissible*
  architecture and say plainly when none is. The floor is named, never
  priced: CAIDE does not know what a capability shortfall costs, and
  inventing a penalty would be worse than listing the classes.
- **The provenance digest covers the inputs.** It omitted the assurance
  profile, all three cost layers, the uncertainty distributions, the
  scaling assumptions, four workload fields, and everything about a grid
  except its name. Three edits worth 32–43% of total cost left the hash
  byte-identical, under a report that tells the reader a digest proves
  the inputs did not change.
- **Quantisation stops at the embedding and the head.** GPTQ, AWQ and the
  llama.cpp k-quants all skip both — the head is the one matrix whose
  quantisation error lands directly on the output distribution.
  Quantising uniformly overstated int4's saving by 1.39× on the smallest
  bundled archetype and 1.09× on the one the paper sweeps.
- **A decode step gathers the input embedding, it does not stream it.**
  Counting the whole `V × d_model` table in every step overstated decode
  traffic by 6.5% on the 8B archetype. The two errors pointed in opposite
  directions, which is why the total looked reasonable for nine releases;
  both are fixed together, as the two energy defects were in v7.
- **`Distribution` keeps its parameters.** A sampler is a closure, so the
  numbers defining an assumption existed nowhere a report or a digest
  could reach them: two lognormals an order of magnitude apart in spread
  were indistinguishable to both.
- **A projection that hits its ceiling says so.** `saturated_from` was
  computed from v1.0 and read by nothing, so a saturated forecast showed
  a flattened volume curve and no reason for it — the same silent
  saturation the v8 audit found in the draw clamps.
- `FRAMEWORK_OVERHEAD_REFERENCE` re-derived after the weight-stream
  change: `per_step` 0.0127 → 0.01345 s, `per_sequence` 0.00202 →
  0.00226 s. Still two points fitting two parameters, still zero degrees
  of freedom, now in its fifth round of asking for a third point.

### Added

- A test that walks every public dataclass field and fails on any that
  nothing reads — the systemic answer to five dangling parameters in six
  rounds — together with a test recording that this check would *not*
  have caught the worst instance, so it is not mistaken for more than it
  is.
- `ModelSpec.embedding_params`, `lm_head_params`, `head_bytes`,
  `tied_embeddings`, `decode_weight_bytes`; `ModelSpec.with_precision`
  takes `head_bytes_per_param`; `efficiency.QUANTISATION_HEAD_BYTES`.
- `ReportBundle.infeasible` and `any_feasible`.

### Changed

- The v6 and v7 reference implementations were revised again: both had
  copied the production weight-stream call rather than deriving it. Third
  occurrence, now recorded as a standing hazard rather than an incident.

## [9.0.0] — 2026-08-20

Ninth release. Three of this round's findings share one shape: a summary
that is only sound when the thing summarised is contiguous, complete or
linear, applied to something that is none of those. The fourth is about
the audit trail rather than the code — v8.0 recorded a scope claim that
was true by accident and wrong in its reason, and the recommendation it
generated pointed at a gap that had been closed for three releases while
the real one went unnamed.

**The published numbers move for the fourth consecutive release (8 of 73
comparable, plus 14 new leaves), and the largest single change is that
the indistinguishable band was not a band.**

### Fixed

- **The tie region is a union of intervals, not an interval.**
  `tie_band` returned first-to-last of the qualifying scan points. On the
  shipped break-even that span contained 175 points of which 137
  exceeded the 5% tolerance, reaching 47% — a "band in which cost does
  not decide" inside which cost decided by nearly half. The true shape is
  four windows of about 1.03–1.10× width, one around each replica riser,
  because the self-hosted curve is flat across a tread while the API line
  keeps rising. `tie_bands` returns them all; `tie_band` returns the
  widest and says so.
- **Two of the three factors of the largest cost term could not be
  varied.** Per-query review cost is `rate × minutes × wage`, and only
  the rate had a draw key. Minutes and wage were point masses that could
  not appear in a tornado chart, so their absence was invisible in
  exactly the output a reader would check. Both are perturbable now, and
  `uncovered_draw_keys` names what a scenario left fixed. With the three
  factors declared, **the top-ranked input changes**: review minutes at
  33.3% displaces volume, which falls from 54.0% to 27.5%; the 95th
  percentile of annual cost rises from $3.95M to $4.67M.
- **`Tier.max_share` is honoured.** Declared in the first release of the
  routing module and read by none of them: a caller capping a tier at 30%
  of traffic got a plan routing all of it there, silently. This is the
  fourth dangling parameter the package has found in itself, after
  `framework_overhead_per_step` (v5), `expert_imbalance` (v6) and the
  draft KV term (v8). Honouring it costs the per-class independence
  argument, so `optimise_routing` now enumerates assignments exactly when
  the instance allows and reports `exact=False` with a reason when it
  cannot.
- **A self-hosted tier is charged in whole replicas.** Routing priced it
  as `share × volume × per_query` while the costing layer has charged
  integral replicas since v4.0 — one quantity, two derivations,
  disagreeing by 13× on the shipped public-service scenario, with the
  CLI's `--tier-fixed-cost` defaulting to zero so that standing up a
  replica appeared free. `Tier.annual_cost_fn` prices a tier from what is
  actually routed to it, and the plan carries a note when the stepped
  figure exceeds the marginal one.
- **`Crossing.margin_pct` and `relative_gap` share a denominator.** The
  first divided by the mean of the two costs and the second by the
  cheaper one, so the module reported one disagreement as two numbers.

### Added

- **The speculative-decoding curve for a mixture-of-experts target**, in
  Result 1 and Fig. 2. It inverts: 1.005× at batch one — the published
  0.40× is worth nothing there — improving to 0.414× at batch 16 before
  worsening again, because the router sees `γ+1` tokens per sequence and
  verification reaches more experts, enlarging the weight stream the
  technique exists to amortise. Every published sweep ran dense
  archetypes or an MoE architecture at saturating batch, so no reported
  number had ever exercised the regime where the v8 routing defect lived.
- `review_minutes_scale` and `reviewer_wage_scale` draw keys, and
  distributions for both in the education scenario, declared as
  assumptions on the same footing as the six that were already there.
- `uncovered_draw_keys`, `REVIEW_COST_FACTORS`, `Tier.is_separable`,
  `RoutingPlan.exact`, `RoutingPlan.notes`, `RoutingPlan.tier_shares`.

### Changed

- `caide route` reports each opened tier's share against its cap, any
  notes, and whether the plan is proven optimal.

## [8.0.0] — 2026-08-20

Eighth release. The v7 audit left three standing recommendations —
extend the reference ledger to all six layers, give the draft's KV scope
declaration a checked bound, and re-derive the mixture-of-experts
shared/expert split independently. All three were carried out and two
found something. **The published numbers move for the third consecutive
release (7 of 68, plus 5 new), and one of them changes sign: at
elasticity 0.6 five-year spend does not fall to 0.73x, it rises to
1.38x** — because only 10.4% of what this institution pays per query
tracks the token tariff, and CAIDE's own scaling analysis had been
declining the other 90% at the tariff rate.

### Fixed

- **The six-layer taxonomy described its largest layer backwards.** The
  table shipped since v1.0 called assurance and governance "volume-free
  — an audit programme is fixed". Human review is charged per query, and
  in all three shipped scenarios the assurance layer is both the largest
  of the six (58%, 91%, 84% of ownership cost) and 83–95% volume-linear.
  A reader applying the table to forecast a doubling of volume would
  have expected the dominant layer to stand still. The row now reads
  "mixed", and `layer_volume_elasticity` *measures* each layer's
  response to volume by finite difference instead of asserting it —
  the same treatment CAIDE gives efficiency multipliers, applied at last
  to its own classification.
- **The scaling projection declined reviewer wages at the speed of GPU
  prices.** `project` took one blended per-query cost and applied the
  token-price decline to all of it, so a fixed audit programme was made
  to scale with volume *and* fall with the tariff, and human review at
  an hourly rate fell 38% a year. `ScalingAssumptions` now carries
  `price_inelastic_per_query` alongside `fixed_annual_cost`, and
  `ScalingYear` reports the three components separately.
- **Demand responded to the tariff rather than to the price the buyer
  faces.** With the split in place, halving a tariff that is a tenth of
  the price cannot double demand. Volume is now solved against the
  effective unit cost — tariff plus inelastic per-query plus fixed
  amortised over the volume being solved for — by damped iteration. The
  two corrections must land together: the first alone gives a five-year
  volume growth of 5x at elasticity 0.6, which is a 38% price response
  to a 27% price change.
  The crossover at elasticity 1.0 is **unchanged and still exact**:
  spend remains `c_eff^(1-eps)` whatever `c_eff` is made of. What moves
  is magnitude, by a factor of roughly three in both directions.
- **A verification step routes tokens, not sequences.** v7.0 corrected
  the arithmetic ledger and the collective to `batch * (gamma + 1)`
  verified tokens but left `expert_bytes_touched(batch)` untouched, which
  handed a mixture-of-experts model the dense model's amortisation for
  free — while for MoE more tokens reach more experts and the weight
  stream *grows*. Of 192 MoE configurations, 96 were mispriced, worst
  case 3.47x on the step and 2.79x on expert traffic, concentrated at
  batch 1: precisely the memory-bound regime speculation exists for.
  No published figure moves, because every paper sweep uses dense
  archetypes — which is why it survived a round.
- **A clamped draw is now reported.** Physically bounded substitutions
  (review rate, duty cycle, scheduler efficiency, MBU) were clamped in
  silence, so the distribution a scenario declares was not the
  distribution that propagated. In the shipped education scenario the
  review-rate clamp binds in 71% of draws, on a class carrying 20% of
  review cost, compressing that cost's upper tail by 9%.
  `saturated_draw_keys` reports it and `MonteCarloResult.n_saturated`
  counts it, on the principle that already governs failed draws.

### Added

- **`ServingConfig.draft_kv_ratio`.** v7.0 declared the speculative
  draft's KV cache out of scope and asserted it was "under 2% of the
  step in every regime the paper reports". The claim had never been
  evaluated. It is 3.3% at the paper's own operating points and 10.8%
  across the catalogue at the same context lengths. The term is now
  modelled, defaults to zero — the same scope boundary, now a modelled
  zero rather than an undocumented omission — and the bound is a
  regression test rather than a sentence.
- **`layer_volume_elasticity`** and **`TCOResult.scaling_inputs`**, which
  partition ownership cost into the part that tracks the tariff, the part
  that scales with volume at wage rates, and the part that does neither.

### Changed

- The reference implementation in `test_v7_audit_fixes.py` was revised:
  it had reproduced the routing omission rather than exposing it, having
  been derived from the same `docs/model.md` paragraph. Recorded in the
  v8 audit report — a reference implementation checks only what its
  source document gets right.
- `plot_scaling` draws the effective price paid alongside the tariff
  whenever the two differ.

## [7.0.0] — 2026-08-20

Seventh release. The v6 audit closed with two standing recommendations —
extend the independent reference implementation beyond the roofline, and
give `expert_imbalance` a basis — and both were carried out. Both found
something. The reference implementation, extended along the *technique*
axis, disagreed immediately on speculative decoding; extended along the
*layer* axis, it disagreed on energy. **The published figures move for
the second consecutive release (57 of 76, worst +32%), and this time a
qualitative claim moves with them: speculative decoding at saturating
batch is not a diminished benefit, it is a net loss** — which is what
measured serving systems report and what the model could not say while
the verification arithmetic went unpriced.

### Fixed

- **Verification arithmetic is priced.** A verification step scores
  `gamma + 1` candidate tokens per sequence in one fused pass —
  amortising the weight stream over them is the entire mechanism — and
  the memory ledger always reflected that. The arithmetic ledger priced
  one token: the expected-tokens denominator in `_speculative_speedup`
  assumed `gamma + 1` tokens were verified per step while the step price
  contained the arithmetic of one, the same mechanism on the two sides
  of a division, modelled on one side only. FLOP utilisation now follows
  the tokens in the matmul (`batch × (gamma + 1)` rows), the all-reduce
  synchronises that many tokens, and the draft's arithmetic is priced at
  its own row count. At batch 1 nothing changes (0.401× → 0.402×); at
  batch 256 the multiplier crosses one (0.824× → 1.088×).
- **`predicted_output_tps` uses the model's own definition of
  throughput.** `evaluate_request` divides by the whole serving cycle;
  the calibration path divided by decode time alone — one quantity, two
  derivations, disagreeing by the prefill share of the cycle, which for
  the 70B TP=4 observation is 54%, not the "negligible" the phrase
  *decode-dominated* suggested. Every published predicted-to-measured
  ratio moves (2.12 → 0.97 at the extreme); no admissibility verdict
  does. `FRAMEWORK_OVERHEAD_REFERENCE` is re-derived with prefill taken
  off the wall clock first (`per_sequence_seconds` 2.26 ms → 2.02 ms).
- **Energy walks the same staircase as dollars.** The dollar ledger has
  charged whole replicas since v4.0; the energy ledger summed per-query
  joules — a continuous curve — so below one full replica the model
  billed the money of 1.0 replicas and the carbon of a fraction of one.
  Annual energy, carbon and water are now derived from provisioned
  replicas: working joules from the per-query accounting, the idle
  remainder of every replica's year at idle power.
- **Idle time draws idle power.** Every provisioned-but-idle second was
  charged at full board power — the one idle-draw figure known to be
  wrong. `HardwareSpec.idle_power_watts` carries the board's idle draw
  (catalogue values in the 10–20% band that published measurements
  cluster in, marked as estimates); per-query energy splits into a
  working share at load power and an idle share at idle draw, and the
  electricity line in `replica_annual_cost` follows the same split.
- **The expert straggler stretches expert work only.** The v6.0
  surcharge multiplied the whole of both roofline terms, so it grew with
  context length — a quantity expert skew knows nothing about. The
  additive form scopes the wait to the expert share of weight traffic
  and arithmetic, using the shared/expert split `ModelSpec` now exposes
  as `moe_shared_params`.
- **The draft model occupies memory.** Speculation charged the draft in
  time — its weight stream and arithmetic appear in every step — while
  `capacity_batch` subtracted only the target's weights, granting the
  draft its HBM for free. Both the capacity and the reported headroom
  now subtract it. The draft's KV cache remains out of scope and is
  documented as such (under 2% of the step at the catalogue's default
  ratio).

### Added

- **`uniform_routing_imbalance(batch, n_experts, experts_per_token)`.**
  The counting-statistics ceiling on expert balance: even a perfectly
  uniform router leaves the busiest of `E` experts above the mean by
  extreme-value fluctuation, about 2× at batch 256 over 160 experts.
  v6.0 shipped `expert_imbalance` with no basis to set it — the same
  dangling state `framework_overhead_per_step` spent one release in.
  The closed form is a first-order approximation validated against
  Monte Carlo in the test suite; it is a *ceiling*, and real routers sit
  at or below it.
- **The reference implementation covers techniques and the cost layer.**
  The differential sweep now moves the technique axis (speculation,
  imbalance) that v6.0 held at identity, and an independent annual
  ledger re-derives the compute layer's dollars *and joules* from
  replica-level first principles. Both live in the test suite so they
  cannot drift into wrapping the code they check.

### Changed

- Two historical tests that pinned the exact `1/duty` scaling of
  per-query cost were revised: the exact factor holds on a grid whose
  tariff is bundled into the hourly rate, and becomes a strict
  inequality where electricity is metered, because the idle share now
  draws idle power. The revision is recorded in the v7 audit report.

## [6.0.0] — 2026-08-18

Sixth release. The v5 audit recommended writing an independent reference
implementation as the one method not yet used — the only way to catch a
mistake made identically in two places. It was written, and it found one
in its first sweep. A second defect surfaced as a side effect of the fix
for the first, and it is the largest single correction in the project's
history: **18 of 76 published figures move**, the first time any release
has changed a number the paper reports.

### Fixed

- **Decode now models the attention arithmetic.** The compute term was
  `2·N_active·B`, which counts streaming the weights through the tensor
  cores and omits scoring the new token against the KV cache. Prefill had
  modelled the same physics since v1.0; decode never had. Across 960
  configurations the omission moved the step time by more than 10% in 77
  and flipped the binding resource in 69, with a worst case of 4.83×. It
  is invisible above batch 64 and below 8k of context, which is why five
  audits missed it and why no figure in the paper moved because of it.
- **Tensor-parallel cost is charged to activations, not to weights.**
  Versions up to 5.0 derated aggregate bandwidth and FLOP/s by
  `1 - 0.03·(N-1)`. Sharding costs no bandwidth — each accelerator reads
  its own shard from its own HBM — and the real cost is a per-layer
  all-reduce whose traffic scales with batch and model width. For a 405B
  model at TP=8 the derate billed 5.75 ms per step against a true cost of
  0.59 ms. `tensor_parallel_penalty` remains the interconnect-quality
  knob and keeps its direction; it now scales a dimensionally correct
  term. **This is the change that moves the published figures.**
- **Mixture-of-experts geometries below the router floor are rejected.** A
  router activating `k` of `E` experts cannot make the active parameter
  count fall below `k/E` of the total. v5.0 accepted such specs and let a
  clamp inside `expert_bytes_touched` absorb them, returning a weight-
  traffic figure that disagreed with the declared active count by up to
  1.65× — the silent-unit failure mode the v5.0 audit had just removed
  from the calibration layer. With the floor enforced the clamp became
  unreachable for valid geometries and actively harmful at the boundary,
  where it broke the identity that a batch of one reads exactly the
  active parameters; it has been removed.

### Added

- **`admissible_conventions()` and `implied_mbu()`.** The roofline is a
  *lower* bound on step time, so a reading of an ambiguous benchmark
  figure that implies a memory-bandwidth utilisation above 1.0 can be
  rejected on physics rather than on judgement. The test is one-sided by
  construction: it can rule a reading out for being too fast and never
  for being too slow, because framework overhead has no upper limit.
- **`READMITTED_OBSERVATIONS`.** One of the three observations v5.0
  excluded as unresolvably ambiguous is now in the validation set, which
  grew from four measurements to five. Readmissions are recorded
  separately from exclusions so that a validation set growing over time
  can be told apart from one drifting toward the points that agree.
- **`ServingConfig.framework_overhead_per_sequence`.** A single constant
  cannot fit the two published 8B-on-A100 points: the residuals are
  15.0 ms at batch 1 and 30.8 ms at batch 8. Scheduling and the API
  server run once per step; detokenisation runs once per live sequence.
- **`FRAMEWORK_OVERHEAD_REFERENCE`.** v5.0 added the overhead parameter
  with no default and no source. These are the first empirical values,
  published with their provenance and with `degrees_of_freedom: 0` stated
  in the record, because two points fitting two parameters is a
  consistency statement and not a validation.
- **`ServingConfig.expert_imbalance`.** Under expert parallelism a decode
  step ends with the busiest expert, not the average one. Defaults to 1.0
  — perfectly balanced — which is an idealisation and not a measurement:
  counting fluctuation alone gives a 1.96× straggler at batch 256 over
  160 experts.

### Changed

- Aggregate bandwidth and FLOP/s no longer carry a tensor-parallel
  derate. Code reading `parallel_efficiency` for that purpose should read
  the collective term instead.
- Validation table in `docs/model.md` now reports implied MBU alongside
  the prediction ratio. The two carry the same information; only one of
  them makes the physical ceiling visible.

### Verified, and found not to be a problem

- **All-to-all traffic in mixture-of-experts serving.** The v5 audit
  predicted expert routing would be the next structural bias. The
  communication half of that prediction did not hold: dispatch and
  combine move two hidden states per token per layer, which is under 6%
  of the step at the largest batch the catalogue supports. The negative
  result is fixed as a test that fails if a future model or interconnect
  pushes it past 10%.

## [5.0.0] — 2026-08-04

Fifth release. The v4 audit recommended expanding external validation to a
second accelerator class and predicted it might expose systematic bias. It
did, and the bias turned out to be structural rather than parametric.

### Fixed

- **`Observation` now records its measurement convention.** A benchmark
  reporting "throughput" may mean the whole replica or one request, and
  the two differ by exactly the batch size — a factor of 8 for a small
  concurrency test and 256 for a large one. v4.0 had a single
  `measured_output_tps` field with no way to say which, in a module whose
  entire purpose is making the accuracy claim checkable. `convention` is
  now validated against `("aggregate", "per_request")` and
  `aggregate_output_tps` normalises.
- **The gap between a hardware roofline and a serving benchmark is now
  representable.** `ServingConfig.framework_overhead_per_step` adds
  non-GPU time per decode step. It defaults to zero — the pure hardware
  roofline, correct for comparing configurations — and the documentation
  now states plainly that this is the wrong setting for predicting what a
  benchmark will report. vLLM's own profiling attributes 62% of wall time
  on an 8B model to the API server and scheduler.

### Added

- `EXCLUDED_OBSERVATIONS` — three candidate measurements that were *not*
  admitted to the reference set, each with its reason. Two were dropped
  for not stating their convention and one for not being reproducible.
  Recording the exclusions is what keeps a validation set from becoming a
  set of the data that happened to fit.
- **Structural sensitivity tests.** Nine alternative modelling choices
  (`decode_mfu_half_batch` 16/64/256, prefill utilisation 0.30/0.45/0.60,
  decode bandwidth 0.50/0.70/0.90, tensor-parallel penalty 0/0.03/0.08).
  The paper's two headline conclusions hold under all nine: the
  regime-dependence spread stays above 1.3x, and the discrepancy against
  the published 0.65x INT4 constant stays between 151% and 154%.
  Semantic caching stays exactly batch-invariant under all nine, which is
  the counterexample that shows the method is finding physics rather than
  its own machinery.
- 33 tests.

### Changed

- Statement coverage 89% -> 90%.
- No published result in the paper changes: `framework_overhead_per_step`
  defaults to zero, and no reported analysis sets it.

## [4.0.0] — 2026-08-04

Fourth release. Round four used metamorphic testing, numerical
conditioning checks, backward-compatibility probes and tool-performance
measurement — all clean — and one method that was not clean: comparing
predicted serving throughput against published measurements from real
frameworks. That comparison produced both findings below.

### Fixed

- **`batch_override` now respects KV-cache capacity.** v3.0 returned a
  confident throughput figure for a batch the memory could not hold: a
  70B model at FP8 on one H100 fits about 40 concurrent sequences, and
  asking for 10,000 produced a number rather than a refusal. The batch is
  now clamped to capacity and `PhasePerformance.batch_truncated` reports
  it. This is the same failure mode the context-overflow check added in
  v3.0 exists to prevent one level up.
- **The documented accuracy claim is now evidence-based.** v3.0's
  `docs/model.md` asserted that predictions land within a factor of two of
  measured serving stacks. Tested against four published measurements
  across three frameworks, two of four fall inside that band and the model
  over-predicts in three. The documentation now states the measured
  position — a factor of two to three, biased optimistic — reports the
  comparison table, and says plainly what it is and is not adequate for.

### Added

- **`caide.calibration`** — fit multiplicative corrections to serving
  utilisation from measurements of your own stack. `fit()` minimises
  squared error in the log of the predicted-to-measured ratio, because
  the errors are multiplicative. Only decode utilisation is fitted:
  published benchmarks are decode-dominated, so prefill utilisation is
  barely identified by them, and fitting both jointly would assign the
  prefill parameter whatever absorbs the residual.
- `REFERENCE_OBSERVATIONS()` — the four published measurements, shipped so
  that the validation is reproducible. They are explicitly *not* a
  calibration set; the docstring says why.
- 40 tests, including seven metamorphic relations (doubling an input must
  double or not move the right output), numerical conditioning across
  twelve orders of magnitude of volume, and backward-compatibility probes
  for v1- and v2-style scenario files.

### Changed

- Statement coverage 88% -> 89%.
- No published result in the paper changes: the `batch_override` clamp
  affects only callers who request an infeasible batch, and none of the
  reported analyses does.

## [3.0.0] — 2026-08-04

Third release. The v2.0 audit closed every v1.0 finding, so round three
used methods the earlier rounds had not: property-based search over the
parameter space, validation against hardware bandwidth ceilings,
adversarial scenario construction, documentation-versus-code drift
checking, and a public-API coherence review. Five new findings resulted.

### Fixed

- **Decode FLOP utilisation now rises with batch size.** v2.0 modelled it
  as a constant, which placed the memory-to-compute transition at an
  artificially low batch and made the model report no benefit at all from
  quantisation at batch 256 for a 70B model. A decode step at batch one is
  a matrix-vector product; at batch 256 it is a matrix-matrix product.
  Utilisation now interpolates as `B / (B + B_half)` with `B_half`
  exposed as `ServingConfig.decode_mfu_half_batch`.
- **Magnitude plausibility is checked, not only sign.** v2.0 rejected
  impossible inputs but silently accepted absurd ones: `self_consistency_k`
  of a million, `infra_overhead` of 1000x, a grid at 10^6 kg CO2e/kWh.
  Values beyond a plausible ceiling are now warned about while still being
  computed, so sensitivity sweeps can reach the tails.
- **Twelve public callables gained docstrings.**

### Added

- **`Scenario.to_dict()` and `Scenario.to_yaml()`** — round-trippable
  serialisation, verified to preserve every result and every quality index
  across two successive round trips.
- **Reports embed the scenario that produced them.** A digest proves the
  inputs did not change; only the inputs themselves let a recipient re-run
  the analysis. Both the Markdown and HTML outputs now carry a runnable
  scenario block.
- `Architecture.quality_penalty_override`, so a serialised scenario can
  carry the quality cost of a stack that has already been baked into the
  state and must not be re-applied.
- 44 tests, including invariants promoted from the property-based audit
  and an external check that predicted decode throughput never exceeds
  the memory system's streaming ceiling.

### Changed

- Speculative decoding's measured multiplier at batch 256 moves from
  0.814x to 0.822x, and INT4's from 0.487x to 0.503x, as a consequence of
  the decode-utilisation fix. All other published results are unchanged.
- Statement coverage 87% -> 88%; `scenario.py` 76% -> 84%.

## [2.0.0] — 2026-08-04

Second release, addressing every finding of an independent audit of v1.0.
The version bump is major because the utilisation parameter was split,
which is a breaking change to `ServingConfig` and to scenario files.

### Fixed

- **Bundled examples now ship inside the package.** In v1.0 `examples/`
  and `docs/` were repository-only, so every command in the README and in
  the software paper failed after `pip install caide`. Added
  `caide examples --list/--extract` and `[tool.setuptools.package-data]`.
- **A missing scenario file now says so.** v1.0 treated a mistyped path as
  an inline YAML document and reported `scenario root must be a mapping`,
  sending users to look for a formatting problem that did not exist.
- **Unknown scenario keys are reported with a suggested correction.**
  Misspelling `review_minutes` as `review_minuts` silently zeroed a cost
  component that dominates the total in every shipped example.
- **Requests exceeding the model context window are flagged.**
  `PhasePerformance.context_overflow` and `.feasible` replace v1.0's
  silent computation of an implausible latency for an impossible request.

### Changed

- **BREAKING: `ServingConfig.target_utilisation` is split** into
  `demand_duty_cycle` (share of the year with live traffic; a property of
  the workload, which no efficiency technique may change) and
  `scheduler_efficiency` (useful-work share while traffic is live; what
  continuous batching improves). `target_utilisation` remains readable as
  the product. In v1.0 continuous batching multiplied the single parameter
  by 1.9, so a scenario declaring 0.42 was served at 0.88 and three
  different demand assumptions produced one identical cost. Scenario files
  using the old field name are rejected with migration guidance.
- Self-hosted unit costs rise by roughly 2.3x in the shipped scenarios as
  a direct consequence; this is a correction, not a regression.
- Bundled price epoch advanced to 2026-08-01.

### Added

- `caide.perturb.perturbed_cost` — the Monte Carlo perturbation logic,
  promoted from a private helper inside the CLI to a documented public
  API with its own tests. The result-reproduction script no longer
  imports an underscore-prefixed function from a UI layer.
- `caide examples` command.
- 29 tests, three of which exist specifically to kill mutants that the
  v1.0 suite could not detect: dropping the factor 2 from the KV cache,
  rounding replica demand to nearest instead of up, and omitting the
  volume factor from the human-review term.

### Documentation

- README documents the utilisation split and the packaged examples.
- Figure files are renamed to match the numbering used in the paper.

## [1.0.0] — 2026-08-04

First public release.

### Added

- **Roofline serving model** (`caide.roofline`) with separate prefill and
  decode phases, GQA-aware KV cache sizing, MoE expert-touch probability
  as a function of batch, tensor-parallel communication loss, and an
  M/D/1 queueing correction for time-to-first-token.
- **SLO-constrained capacity solver**: `solve_batch_for_slo` bisects for
  the largest batch that still meets latency targets, rather than
  assuming the scheduler cap is reachable.
- **Efficiency techniques as physical transforms** (`caide.efficiency`):
  fifteen techniques modelled as `DeploymentState -> DeploymentState`
  functions. Cost multipliers are measured on the transformed state, not
  supplied as constants, so they vary with the operating point and
  compose with interaction.
- **Six-layer total cost of ownership** (`caide.costing`) with distinct
  scaling laws per layer, integral-replica capacity charging, human
  review accounting, and displaced-labour reporting.
- **Break-even analysis** (`caide.breakeven`) locating every crossing on
  a log-spaced bracketed scan, with dominance intervals and an
  economically-indistinguishable band for the common case where granular
  capacity produces dozens of non-actionable crossings.
- **Workload routing** (`caide.routing`) with exact minimum-cost
  assignment under quality floors and per-tier fixed costs.
- **Uncertainty propagation** (`caide.uncertainty`): Monte Carlo with
  five distribution families and Spearman rank sensitivity analysis.
- **Scaling dynamics** (`caide.scaling`): closed-form price-elasticity
  projection and least-squares elasticity estimation from observed
  history.
- **Declarative YAML scenarios** with strict, path-naming validation.
- **Reporting** in Markdown, CSV and a dependency-free single-file HTML
  dashboard with embedded figures; six figure types.
- **Command-line interface** with `run`, `breakeven`, `sweep`, `route`,
  `catalog`, `init` and `validate`.
- **Built-in catalogue** of nine model archetypes, six accelerators, four
  API pricing tiers and nine electricity grids, with price staleness
  warnings.
- Three cross-domain worked examples: engineering education, clinical
  documentation, and public-sector service delivery.
- 99 tests at 86% statement coverage.
