#!/usr/bin/env python3
"""Find modelling errors that cancel each other.

Twice this project has shipped a pair of mistakes of opposite sign whose
combined figure looked reasonable, and both times the pair was found by
hand after a physical review, not by any check:

* v7.0 -- annual energy summed a continuous per-query curve while dollars
  walked a replica staircase (understating), and charged idle hours at
  full board power (overstating). Fixing only the first gave 20.8x the
  original figure. Together they had produced a plausible number for six
  releases.
* v10.0 -- the decode weight stream counted the whole input embedding
  table, which a step only gathers from (overstating), and quantised the
  language-model head, which production schemes leave alone
  (understating). Nine releases of a plausible total.

The pattern has a signature: perturb one ingredient of a composite
quantity and the result moves a lot; perturb all of them and it barely
moves. This scans for that signature.

A flagged pair is not a defect. It says the quantity is held up by two
assumptions that lean against each other, so neither has ever been tested
on its own -- which is the condition both historical cases were in.

Usage:  python audit/opposing_errors.py
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from caide import (
    DeploymentState,
    ServingConfig,
    WorkloadClass,
    get_grid,
    get_hardware,
    get_model,
)
from caide.roofline import decode_step_time

#: A single-ingredient perturbation counts as large above this relative
#: move.
LARGE = 0.05
#: The signature is *relative*: the joint move is a small fraction of the
#: largest single one. The first version of this file compared the joint
#: move against an absolute threshold and returned "ok" for both of the
#: historical pairs it was written to rediscover -- a check whose own
#: calibration cases pass silently, which is the failure it exists to
#: detect, occurring in the detector. The two pairs sit at 0.33 and 0.11
#: on this ratio.
CANCELLATION_RATIO = 0.5


@dataclass(frozen=True)
class Toggle:
    """One modelling decision, and the alternative it was chosen over."""

    name: str
    apply: Callable[[Dict[str, float]], None]
    rationale: str


@dataclass
class Composite:
    """A quantity assembled from several independent decisions."""

    name: str
    evaluate: Callable[[Dict[str, float]], float]
    toggles: Sequence[Toggle]

    def _value(self, active: Sequence[str]) -> float:
        knobs: Dict[str, float] = {}
        for toggle in self.toggles:
            if toggle.name in active:
                toggle.apply(knobs)
        return self.evaluate(knobs)

    def scan(self) -> List[Dict[str, object]]:
        base = self._value([])
        singles = {t.name: self._value([t.name]) for t in self.toggles}
        findings = []
        for a, b in itertools.combinations([t.name for t in self.toggles], 2):
            da = (singles[a] - base) / base
            db = (singles[b] - base) / base
            joint = (self._value([a, b]) - base) / base
            largest = max(abs(da), abs(db))
            if largest > LARGE and abs(joint) < CANCELLATION_RATIO * largest:
                findings.append({
                    "quantity": self.name, "a": a, "b": b,
                    "single_a": da, "single_b": db, "joint": joint,
                    "opposite_signs": da * db < 0,
                    "ratio": abs(joint) / largest,
                })
        return findings


# ---------------------------------------------------------------------------
# The two historical pairs, as calibration: a scan that cannot rediscover
# them is not measuring anything.
# ---------------------------------------------------------------------------

def _decode_stream() -> Composite:
    model = get_model("dense-8b")

    def evaluate(knobs: Dict[str, float]) -> float:
        streamed = model.decode_weight_bytes(1.0)
        if knobs.get("stream_embedding"):
            streamed += model.embedding_params * model.head_bytes
        if knobs.get("quantise_head"):
            head = model.embedding_params + model.lm_head_params
            streamed -= head * (model.head_bytes - 0.5)
        return streamed

    return Composite(
        name="decode weight stream (dense-8b)",
        evaluate=evaluate,
        toggles=[
            Toggle("stream_embedding",
                   lambda k: k.__setitem__("stream_embedding", 1.0),
                   "counting the input table a step only gathers from"),
            Toggle("quantise_head",
                   lambda k: k.__setitem__("quantise_head", 1.0),
                   "quantising the head production schemes leave alone"),
        ])


def _annual_energy() -> Composite:
    """The v7 pair, reconstructed arithmetically rather than by patching."""
    load_w, idle_w, duty, hours = 350.0, 40.0, 0.05, 8766.0
    # The scenario the v7 audit was looking at: a deployment filling 5% of
    # one replica, so the staircase and the idle draw both bite hard.
    busy_seconds = duty * hours * 3600.0

    def evaluate(knobs: Dict[str, float]) -> float:
        # Toggles revert to the pre-v7 assumption, so the base is today's
        # model and a flagged pair says "this figure is held up by two
        # assumptions neither of which has been tested alone".
        provisioned = (duty if knobs.get("continuous") else 1.0) * hours * 3600.0
        idle = load_w if knobs.get("idle_at_load") else idle_w
        return load_w * busy_seconds + idle * max(provisioned - busy_seconds,
                                                  0.0)

    return Composite(
        name="annual facility energy",
        evaluate=evaluate,
        toggles=[
            Toggle("continuous", lambda k: k.__setitem__("continuous", 1.0),
                   "billing a fraction of a replica instead of a whole one"),
            Toggle("idle_at_load", lambda k: k.__setitem__("idle_at_load", 1.0),
                   "charging idle hours at full board power"),
        ])


# ---------------------------------------------------------------------------
# Live surfaces: composites currently assembled from more than one
# independent decision.
# ---------------------------------------------------------------------------

def _speculative_step() -> Composite:
    model = get_model("moe-8x7b")
    hardware = get_hardware("h100-sxm")

    def evaluate(knobs: Dict[str, float]) -> float:
        gamma = 4.0
        cfg = ServingConfig(n_accelerators=2, max_batch=1,
                            speculative_gamma=gamma,
                            speculative_acceptance=0.72,
                            draft_param_ratio=0.03)
        step, _ = decode_step_time(model, hardware, cfg, 1, 2048.0)
        bandwidth = hardware.memory_bandwidth * 2 * cfg.mbu_decode
        if knobs.get("route_by_sequence"):
            step -= (model.expert_bytes_touched(1.0 * (gamma + 1))
                     - model.expert_bytes_touched(1.0)) / bandwidth
        if knobs.get("verify_one_token"):
            flops = hardware.effective_flops("bf16") * 2 * 0.3
            step -= 2.0 * model.active_params * gamma / flops
        return step

    return Composite(
        name="speculative decode step (moe-8x7b, batch 1)",
        evaluate=evaluate,
        toggles=[
            Toggle("route_by_sequence",
                   lambda k: k.__setitem__("route_by_sequence", 1.0),
                   "routing batch tokens instead of batch x (gamma+1)"),
            Toggle("verify_one_token",
                   lambda k: k.__setitem__("verify_one_token", 1.0),
                   "pricing one verified token instead of gamma+1"),
        ])


def _per_query_cost() -> Composite:
    grid = get_grid("us-average")
    workload = WorkloadClass("q", 1.0, 1500, 400)

    def evaluate(knobs: Dict[str, float]) -> float:
        from caide.costing import self_hosted_query_cost
        duty = 0.42 if not knobs.get("ignore_duty") else 1.0
        overhead = 1.35 if not knobs.get("ignore_overhead") else 1.0
        cfg = ServingConfig(n_accelerators=4, max_batch=256,
                            demand_duty_cycle=duty, scheduler_efficiency=1.0,
                            infra_overhead=overhead)
        state = DeploymentState(get_model("dense-70b"),
                                get_hardware("h100-sxm"), cfg)
        return self_hosted_query_cost(state, workload, grid,
                                      respect_slo=False).compute_cost

    return Composite(
        name="self-hosted cost per query",
        evaluate=evaluate,
        toggles=[
            Toggle("ignore_duty", lambda k: k.__setitem__("ignore_duty", 1.0),
                   "pricing a fully loaded replica"),
            Toggle("ignore_overhead",
                   lambda k: k.__setitem__("ignore_overhead", 1.0),
                   "dropping the infrastructure markup"),
        ])


COMPOSITES = [_decode_stream, _annual_energy, _speculative_step,
              _per_query_cost]


def scan() -> List[Dict[str, object]]:
    found: List[Dict[str, object]] = []
    for factory in COMPOSITES:
        found.extend(factory().scan())
    return found


def main() -> int:
    print(f"opposing-error scan: single move > {LARGE:.0%}, joint move "
          f"< {CANCELLATION_RATIO:.0%} of it")
    print("-" * 78)
    for factory in COMPOSITES:
        composite = factory()
        hits = composite.scan()
        mark = "PAIR" if hits else "ok  "
        print(f"  [{mark}] {composite.name}")
        for hit in hits:
            sign = "opposite signs" if hit["opposite_signs"] else "same sign"
            print(f"          {hit['a']} {hit['single_a']:+.1%} and "
                  f"{hit['b']} {hit['single_b']:+.1%} "
                  f"-> together {hit['joint']:+.1%} "
                  f"(ratio {hit['ratio']:.2f}, {sign})")
    print("-" * 78)
    hits = scan()
    print(f"{len(hits)} cancelling pair(s). The two historical ones are "
          "included deliberately: a scan that cannot rediscover them is "
          "not measuring anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
