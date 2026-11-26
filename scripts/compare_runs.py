"""Compares runs saved as .npy directories (e.g. by the Swift CLI's --out) against the golden runs.

    python scripts/compare_runs.py runs/swift_fixed
"""

import argparse
import json
import sys
from pathlib import Path

from nemodiar import GOLDEN_DIR
from nemodiar.compare import compare_runs
from nemodiar.golden import Run, load_run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--golden", default="hf_low_latency_float32")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    print(f"{'file':34s} {'max|dp|':>9s} {'mean|dp|':>9s} {'p99.9':>9s} {'dec':>4s} {'segs':>5s} {'state':>6s}")
    results, ok = {}, True
    for d in sorted(p for p in args.run_dir.iterdir() if (p / "logits.npy").exists()):
        ref = Run.load(GOLDEN_DIR / args.golden / f"{d.name}.npz")
        c = compare_runs(ref, load_run_dir(d))
        state = "same" if c.state_identical else f"@{c.first_state_divergence}"
        print(f"{d.name[:34]:34s} {c.max_abs_prob:9.2e} {c.mean_abs_prob:9.2e} {c.p999_abs_prob:9.2e} "
              f"{c.decision_mismatch_frames:4d} {'same' if c.segments_identical else 'DIFF':>5s} {state:>6s}")
        results[d.name] = c.as_dict()
        ok &= c.state_identical and c.segments_identical
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
