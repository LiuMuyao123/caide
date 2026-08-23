#!/usr/bin/env python3
"""Which source statements do the *published* results exercise?

The v12 audit ended with the observation that two consecutive rounds had
moved no published number because the audit frontier had left the paper's
coverage behind. That is a measurable statement, and this script measures
it: run the reproduction script under coverage, alone, and report what it
touches.

The first measurement, in v13.0, found 59% of statements, ``report.py`` at
0% and ``routing.py`` at 25% -- two modules the manuscript describes
capabilities of, and the two in which the v9 and v10 audits had found four
defects between them. It also found that the manuscript's validation
paragraph was not produced here at all.

Usage:  python audit/published_coverage.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "src" / "caide" / "examples" / "reproduce_paper.py"


def main() -> int:
    subprocess.run([sys.executable, "-m", "coverage", "run",
                    "--source=src/caide", str(SCRIPT), "--out", "/tmp/pubcov"],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "coverage", "json",
                    "-o", "/tmp/pubcov.json"], cwd=ROOT, check=True,
                   capture_output=True)
    data = json.loads(Path("/tmp/pubcov.json").read_text())

    print(f"{'module':<40}{'covered':>9}{'stmts':>8}{'pct':>7}")
    print("-" * 64)
    for name, info in sorted(data["files"].items()):
        s = info["summary"]
        print(f"{name:<40}{s['covered_lines']:>9}{s['num_statements']:>8}"
              f"{s['percent_covered']:>6.0f}%")
    total = data["totals"]["percent_covered"]
    print("-" * 64)
    print(f"{'TOTAL':<40}{data['totals']['covered_lines']:>9}"
          f"{data['totals']['num_statements']:>8}{total:>6.0f}%")
    print("\nModules the published results never enter are where defects "
          "survive rounds: see the v9, v10 and v13 audit reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
