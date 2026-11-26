"""Diarization accuracy of candidate runs next to the original.

  vs truth:    DER against ground-truth labels (synthetic meetings, data/labels)
  vs original: DER of the candidate's segments against the original's segments (agreement), every recording

    python scripts/der_report.py runs/swift_live runs/ane_py
"""

import argparse
from pathlib import Path

import numpy as np

from nemodiar import DATA_DIR, GOLDEN_DIR
from nemodiar.der import activity, der
from nemodiar.golden import Run, load_run_dir

LABELS = DATA_DIR.parent / "labels"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--golden", default="hf_low_latency_float32")
    args = parser.parse_args()
    golden = GOLDEN_DIR / args.golden
    names = sorted(p.stem for p in golden.glob("*.npz"))
    systems = {"original": {n: Run.load(golden / f"{n}.npz") for n in names}}
    for r in args.runs:
        systems[r.name] = {n: load_run_dir(r / n) for n in names if (r / n / "logits.npy").exists()}

    print("== DER vs ground truth (synthetic meetings)")
    print(f"{'file':28s}" + "".join(f"{s:>16s}" for s in systems))
    totals = {s: [0.0, 0] for s in systems}
    for n in names:
        if not (LABELS / f"{n}.npz").exists():
            continue
        ref = np.load(LABELS / f"{n}.npz")["activity"]
        row = f"{n[:28]:28s}"
        for s, runs in systems.items():
            d = der(ref, activity(runs[n].probs))
            totals[s][0] += d.der * d.speech_frames
            totals[s][1] += d.speech_frames
            row += f"{d.der * 100:15.2f}%"
        print(row)
    print(f"{'overall':28s}" + "".join(f"{t[0] / t[1] * 100:15.2f}%" for t in totals.values()))

    print("\n== disagreement with the original (DER with the original's segments as reference)")
    print(f"{'file':28s}" + "".join(f"{s:>16s}" for s in list(systems)[1:]) + "   flipped frames")
    agg = {s: [0.0, 0, 0, 0] for s in list(systems)[1:]}
    for n in names:
        ref = activity(systems["original"][n].probs)
        row, flips = f"{n[:28]:28s}", []
        for s in list(systems)[1:]:
            hyp = activity(systems[s][n].probs)
            d = der(ref, hyp)
            f = int((ref != hyp).any(1).sum())
            agg[s][0] += d.der * d.speech_frames
            agg[s][1] += d.speech_frames
            agg[s][2] += f
            agg[s][3] += len(ref)
            row += f"{d.der * 100:15.3f}%"
            flips.append(str(f))
        print(row + "   " + "/".join(flips))
    print(f"{'overall':28s}" + "".join(f"{a[0] / a[1] * 100:15.3f}%" for a in agg.values())
          + "   " + "/".join(f"{a[2]} of {a[3]} ({a[2] / a[3] * 100:.2f}%)" for a in agg.values()))


if __name__ == "__main__":
    main()
