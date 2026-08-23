#!/usr/bin/env python3
"""The carry-forward ledger: what each round left undone, and since when.

Sixteen rounds produced a recurring failure that no amount of care fixed:
a principle stated in a report and checked by nothing decays. Dangling
parameters were found five times. A pair of named siblings was half-fixed
three times. The README's test badge sat five releases stale. Each time
the correction was real and each time the *practice* it implied was left
as prose.

"Settle the debt before starting new work" is such a practice. This file
is the mechanism that keeps it from becoming another one.

Every unfinished item and every standing recommendation lives here with
three facts: the round that raised it, the round that last executed it,
and what would count as executing it. ``tests/test_carry_forward.py``
asserts the ledger is well-formed and that nothing has gone unexecuted
past its declared tolerance -- so an item cannot quietly age from "next
round" into "continuously deferred for seven rounds", which is where six
of the entries below already are.

Usage:  python audit/carry_forward.py            # report
        python audit/carry_forward.py --strict   # non-zero exit if overdue
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

#: The round whose deliverables are current. Bump it in the same commit
#: that bumps the package version; the test below checks the two agree.
CURRENT_ROUND = 17


@dataclass(frozen=True)
class Item:
    """One thing owed, and the evidence that would discharge it."""

    key: str
    #: What the item is, in the words the report used.
    description: str
    #: Round in which it was first raised.
    raised: int
    #: Round in which it was last actually executed, or ``None`` if never.
    last_done: Optional[int] = None
    #: How would a reader know it had been done? A command, a test name,
    #: a file. An item with no discharge condition cannot be settled, only
    #: forgotten, which is how six of these reached seven rounds.
    discharge: str = ""
    #: Rounds it may go unexecuted before it is overdue. One means "every
    #: round"; a larger number is a declared, deliberate cadence rather
    #: than a slow drift.
    tolerance: int = 1
    #: How the item was settled, when it was. Kept because "done in v17"
    #: without the how is the same claim the partial sweeps were making.
    _settled_note: str = ""
    #: A deferral that a later round discharged instead. Retained so the
    #: record shows the deferral was made and then honoured, not deleted.
    _superseded_deferral: str = ""
    #: True for per-round hygiene -- work that is done again every round
    #: rather than done once and closed. Declared rather than inferred
    #: from ``tolerance``: inferring it would make "never started" and
    #: "not done this round" the same state, and they are not.
    recurring: bool = False
    #: Set when a round decided *not* to do it and said why. A reason is
    #: necessary and not sufficient: it must come with ``deferred_until``.
    #:
    #: v17.0 is why. The ledger came due on the full mutant sweep, the
    #: suite went red, and the author's response was to write a reason and
    #: widen the test until it passed -- using, on the first block, an
    #: escape hatch he had built one round earlier. A deferral that
    #: suppresses the alarm indefinitely is not a decision; it is the
    #: omission with better paperwork. It now expires.
    deferred_because: str = ""
    #: The round by which a deferral must be settled. Past it the item is
    #: overdue again however good the reason was.
    deferred_until: Optional[int] = None

    @property
    def rounds_outstanding(self) -> int:
        return CURRENT_ROUND - (self.last_done
                                if self.last_done is not None else self.raised)

    @property
    def deferral_live(self) -> bool:
        """A deferral is live only while it is within its own deadline."""
        return bool(self.deferred_because) and (
            self.deferred_until is not None
            and CURRENT_ROUND <= self.deferred_until)

    @property
    def overdue(self) -> bool:
        return self.rounds_outstanding > self.tolerance and not self.deferral_live

    def line(self) -> str:
        age = self.rounds_outstanding
        state = ("done in v%d" % self.last_done if self.last_done is not None
                 else "never executed")
        if self.overdue:
            flag = "OVERDUE"
        elif self.deferral_live:
            flag = "till v%d" % self.deferred_until
        else:
            flag = "ok"
        return (f"  [{flag:<8}] {self.key:<28} raised v{self.raised:<3}"
                f"{state:<16} {age} round(s) outstanding")


#: Standing debt. Ordered by how long it has been outstanding, because
#: that ordering is the finding.
LEDGER: List[Item] = [
    Item(
        key="quality_floor-provenance",
        description=(
            "Review the quality_floor and latency_sensitive values in the "
            "three shipped scenarios and record why each was chosen. Since "
            "v10 they decide every published answer, and v14 showed one "
            "scenario turns on a 1.2% margin."),
        raised=11, last_done=17, tolerance=99,
        _settled_note=(
            "Settled in v17: every floor carries a basis note, the three "
            "that decide an answer are marked, and margins below the "
            "index's declared resolution are flagged in all four output "
            "channels. Re-review only when a scenario changes."),
        discharge="a provenance note per floor in each scenario YAML"),
    Item(
        key="opposing-errors-check",
        description=(
            "A check for the pattern where two errors of opposite sign "
            "cancel. Found twice by hand (v7 energy, v10 embedding/head); "
            "both times the combined figure looked reasonable."),
        raised=10, last_done=17, tolerance=3, recurring=True,
        discharge="python audit/opposing_errors.py, which must rediscover "
                  "both historical pairs and report no live one"),
    Item(
        key="prefill-utilisation-evidence",
        description=(
            "An external constraint on mfu_prefill, the largest gap in the "
            "validation coverage."),
        raised=8, last_done=None, tolerance=3,
        deferred_until=20,
        deferred_because=(
            "No usable public measurement has appeared in eight rounds. "
            "Deferred deliberately: the recommended close-out is to "
            "reclassify it as a declared modelling choice with an "
            "interval, as v11 did for the framework overhead constants."),
        discharge="either a third measurement, or reclassification"),
    Item(
        key="audit-tooling-audited",
        description=(
            "The audit tooling has never been audited. v14 found one "
            "instance: an interrupted mutation run left a mutated source "
            "tree behind."),
        raised=14, last_done=None, tolerance=3,
        discharge="a round that treats mutation_run, published_coverage "
                  "and the build scripts as subjects"),
    Item(
        key="documented-numbers-bound",
        description=(
            "Bind the figures quoted in CHANGELOG and in the audit reports "
            "to findings.json, as v14 did for the README."),
        raised=14, last_done=None, tolerance=3,
        discharge="a test comparing quoted numerals against findings.json"),
    Item(
        key="mutation-full-sweep",
        recurring=True,
        description=(
            "Run the complete mutant set rather than only the new ones. "
            "Partial runs are cheap and give a coverage signal that reads "
            "identically to a full one."),
        raised=13, last_done=17, tolerance=3,
        _settled_note=(
            "Settled in v17.1 by sharding: --shard k/3 runs a third of the "
            "set, and all three were run against a green baseline with the "
            "target derived from the script's own location rather than "
            "hard-coded to a build directory. 56 mutants, 0 escaped."),
        _superseded_deferral=(
            "Deferred at v17 with a stated cost, not skipped. The sweep "
            "runs 56 mutants against the full suite at roughly 25 seconds "
            "each -- about 23 minutes -- which does not fit the session "
            "budget a round is worked in, so partial runs have quietly "
            "substituted for it since v13. The remedy is to make it "
            "affordable rather than to keep deferring it: shard the mutant "
            "list so a round runs a third of it and three rounds cover the "
            "set, and record which shard was run. First item of v18."),
        discharge="python audit/mutation_run.py --shard k/3 for k in 1..3, "
                  "each against a green baseline, 0 escaped"),
    Item(
        key="zero-call-scope",
        description=(
            "Extend the zero-call check from class methods to module-level "
            "functions and __all__ exports."),
        raised=16, last_done=None, tolerance=2,
        discharge="the check in tests/test_v16_audit_fixes.py covering both"),
    Item(
        key="opposing-errors-scan",
        description=(
            "Run the cancelling-pair scan each round. Both historical "
            "pairs stayed for six and nine releases producing plausible "
            "totals, so the value is in running it, not in having it."),
        raised=17, last_done=17, tolerance=1, recurring=True,
        discharge="python audit/opposing_errors.py"),
    Item(
        key="published-coverage",
        recurring=True,
        description=(
            "Measure what the published results exercise, at the start of "
            "each round, and use it to choose where to look."),
        raised=13, last_done=17, tolerance=1,
        discharge="python audit/published_coverage.py"),
    Item(
        key="findings-diff",
        recurring=True,
        description="Regenerate findings.json and diff it leaf by leaf.",
        raised=6, last_done=17, tolerance=1,
        discharge="python src/caide/examples/reproduce_paper.py"),
]


def overdue() -> List[Item]:
    return [i for i in LEDGER if i.overdue]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when anything is overdue")
    args = parser.parse_args()

    print(f"carry-forward ledger, current round v{CURRENT_ROUND}")
    print("-" * 78)
    for item in sorted(LEDGER, key=lambda i: -i.rounds_outstanding):
        print(item.line())
    late = overdue()
    print("-" * 78)
    print(f"{len(late)} overdue, {sum(1 for i in LEDGER if i.deferred_because)} "
          f"deliberately deferred, {len(LEDGER)} tracked")
    if late:
        print("\nSettle these before opening new lines of enquiry:")
        for item in late:
            print(f"  * {item.key}: {item.description}")
            print(f"    discharged by: {item.discharge}")
    return 1 if (late and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
