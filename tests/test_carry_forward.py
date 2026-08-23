"""The ledger is checked, or it is prose.

Sixteen rounds established that a principle nothing checks decays: five
dangling parameters, three half-cleared sibling pairs, a test badge five
releases stale. "Settle the debt before opening new work" is a principle
of exactly that kind, so the ledger that records the debt is bound here.

These tests do not assert that nothing is overdue -- two things are, and
saying so is the ledger's purpose. They assert that the ledger cannot go
stale, cannot hold an item nobody could discharge, and cannot drift out
of step with the release it describes.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import caide

ROOT = Path(caide.__file__).resolve().parent.parent.parent
AUDIT = ROOT / "audit"

pytestmark = pytest.mark.skipif(
    not (AUDIT / "carry_forward.py").exists(),
    reason="audit tooling is not part of an installed wheel")


@pytest.fixture(scope="module")
def ledger():
    sys.path.insert(0, str(AUDIT))
    try:
        import carry_forward
    finally:
        sys.path.pop(0)
    return carry_forward


def test_the_ledger_tracks_the_current_release(ledger):
    """The commit that bumps the version bumps the round. Without this the
    ledger ages backwards: every item looks fresher each release."""
    version = re.search(r'version = "(\d+)\.',
                        (ROOT / "pyproject.toml").read_text()).group(1)
    assert ledger.CURRENT_ROUND == int(version)


def test_every_item_can_be_discharged(ledger):
    """An item with no discharge condition cannot be settled, only
    forgotten -- which is how the two overdue entries reached five and six
    rounds while appearing in every report."""
    for item in ledger.LEDGER:
        assert item.discharge, item.key
        assert item.description.strip(), item.key
        assert item.raised <= ledger.CURRENT_ROUND, item.key
        if item.last_done is not None:
            assert item.raised <= item.last_done <= ledger.CURRENT_ROUND, item.key


def test_a_deferral_expires(ledger):
    """A reason is necessary and not sufficient.

    Until v17.1 ``overdue`` was suppressed by the presence of a reason,
    so a deferral silenced the alarm for as long as nobody rewrote it.
    That is the omission with better paperwork, and it was used once, by
    the author, on the first release the ledger blocked.
    """
    for item in ledger.LEDGER:
        if item.deferred_because:
            assert item.deferred_until is not None, item.key
            assert item.deferred_until > item.raised, item.key
        assert item.deferral_live == bool(
            item.deferred_because and item.deferred_until is not None
            and ledger.CURRENT_ROUND <= item.deferred_until)


def test_an_expired_deferral_is_overdue_again(ledger):
    from dataclasses import replace
    live = next((i for i in ledger.LEDGER if i.deferred_because), None)
    assert live is not None, "no deferral to check the expiry rule against"
    expired = replace(live, deferred_until=ledger.CURRENT_ROUND - 1,
                      last_done=None, tolerance=0)
    assert not expired.deferral_live
    assert expired.overdue


def test_settling_an_item_records_how(ledger):
    """"Done in v17" without the how is the claim the partial mutant
    sweeps were making for four rounds."""
    for item in ledger.LEDGER:
        if item.last_done is not None and not item.recurring:
            assert item._settled_note, item.key


def test_deferral_carries_a_reason(ledger):
    """Deferral with a stated reason is a decision. Deferral without one
    is an omission wearing a decision's clothes, and the ledger must not
    let the second pass as the first."""
    for item in ledger.LEDGER:
        if item.deferred_because:
            assert len(item.deferred_because) > 40, item.key
        assert item.tolerance >= 1, item.key


def test_keys_are_unique_and_stable(ledger):
    keys = [i.key for i in ledger.LEDGER]
    assert len(keys) == len(set(keys))
    assert all(re.fullmatch(r"[a-z0-9_-]+", k) for k in keys)


def test_the_two_known_debts_are_still_recorded(ledger):
    """Pinned by presence. The failure mode this guards against is not
    forgetting to do them -- it is quietly dropping them from the list so
    that the report stops showing an uncomfortable number."""
    keys = {i.key for i in ledger.LEDGER}
    assert "quality_floor-provenance" in keys
    assert "opposing-errors-check" in keys
    for key in ("quality_floor-provenance", "opposing-errors-check"):
        item = next(i for i in ledger.LEDGER if i.key == key)
        assert item.overdue or item.last_done == ledger.CURRENT_ROUND


def test_every_round_settles_its_per_round_items(ledger):
    """Items declared as every-round work must actually have been done in
    the current round. This is the mechanism the phrase 'settle before you
    start' turns into."""
    recurring = [i for i in ledger.LEDGER if i.recurring]
    assert recurring, "no per-round hygiene is tracked"
    for item in recurring:
        assert item.last_done is not None, item.key
        if item.deferral_live:
            # A recurring item may lapse only behind a deferral that has
            # not expired. v17.0 widened this branch to accept a bare
            # reason, on the first block, in the same round that produced
            # the block -- so the branch is now bounded by a deadline the
            # ledger checks, not by the wording of the excuse.
            assert item.deferred_until is not None, item.key
            continue
        assert item.last_done >= ledger.CURRENT_ROUND - item.tolerance, item.key


def test_the_script_runs_and_reports(ledger):
    proc = subprocess.run([sys.executable, str(AUDIT / "carry_forward.py")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0
    assert "carry-forward ledger" in proc.stdout
    assert "overdue" in proc.stdout


def test_strict_mode_fails_while_anything_is_overdue(ledger):
    """So that a release pipeline can be made to care, once the two
    outstanding items are settled."""
    proc = subprocess.run(
        [sys.executable, str(AUDIT / "carry_forward.py"), "--strict"],
        capture_output=True, text=True, cwd=ROOT)
    assert (proc.returncode != 0) == bool(ledger.overdue())
