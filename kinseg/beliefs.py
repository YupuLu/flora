"""Score kinematic cuts with Flora-TempSeg's boundary beliefs and merge phases across weak cuts.

``segment.py --probe`` records, at every start or end TempSeg writes, the
probability of each value on its 0.5 s output grid. Here:

* **Belief of a cut.** For each kinematic cut ``c`` (an internal phase
  boundary), the largest probability that any recorded start or end puts on
  the grid points within 0.25 s of ``c`` (the nearest grid point). Grid points
  at a window's visible edges are left out: a value there means "continues
  beyond this window", not a boundary.
* **Separation.** A cut is positive when it is the nearest cut to a reference
  boundary within 0.5 s, negative otherwise. The ROC AUC of the belief is
  compared with that of the distance to TempSeg's nearest stitched boundary.
* **Merging.** Keeping only cuts whose belief reaches a threshold and joining
  the phases across the others gives intervals scored with the WGO protocol,
  for a range of thresholds.

    python kinseg/beliefs.py --autolabel <checkout> --kinematic <exp_wgo_bench run> \\
        --probe <segment.py --probe output dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from merge import scorer

THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
HALF_CELL_S = 0.25
MATCH_S = 0.5


def boundary_beliefs(record: Dict[str, object]) -> List[Tuple[float, float]]:
    """``(time_s, probability)`` for every grid value of every recorded start/end, edges excluded."""
    out = []
    for window in record["probes"]:
        origin = window["view_start_us"] / 1e6
        valid_start, valid_end = window["envelope"][0], window["envelope"][1]
        for belief in window["beliefs"]:
            logs = belief["log_probs"]
            top = max(logs)
            weights = [math.exp(x - top) for x in logs]
            total = sum(weights)
            for value, w in zip(belief["values"], weights):
                if value in (valid_start, valid_end):
                    continue
                out.append((origin + value / 20, w / total))
    return out


def cut_belief(cut: float, beliefs: Sequence[Tuple[float, float]]) -> float:
    """Largest probability any single prediction puts within half a grid cell of ``cut``."""
    return max((p for t, p in beliefs if abs(t - cut) <= HALF_CELL_S), default=0.0)


def internal_cuts(intervals: Sequence[Tuple[float, float]]) -> List[float]:
    return sorted({e for _, e in intervals[:-1]})


def labels(cuts: Sequence[float], references: Sequence[Tuple[float, float]]) -> List[int]:
    """1 for the cut nearest each reference boundary within MATCH_S, else 0."""
    positive = set()
    for r in internal_cuts(references):
        near = [c for c in cuts if abs(c - r) <= MATCH_S]
        if near:
            positive.add(min(near, key=lambda c: abs(c - r)))
    return [int(c in positive) for c in cuts]


def auc(scores: Sequence[float], y: Sequence[int]) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos = [s for s, v in zip(scores, y) if v]
    neg = [s for s, v in zip(scores, y) if not v]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def merge_phases(phases: Sequence[Tuple[float, float]], keep: Sequence[float]) -> List[Tuple[float, float]]:
    """Join consecutive phases across every internal cut not in ``keep``."""
    kept = set(keep)
    out = [phases[0]]
    for s, e in phases[1:]:
        if s in kept:
            out.append((s, e))
        else:
            out[-1] = (out[-1][0], e)
    return out


def analyse(kinematic_run: Path, probe_dir: Path, score_family) -> Dict[str, object]:
    per_family: Dict[str, Dict[str, list]] = {}
    for path in sorted((kinematic_run / "predictions").glob("*/*.json")):
        case = json.loads(path.read_text())
        probe_path = probe_dir / f"{case['id']}.json"
        if not probe_path.exists():
            continue
        record = json.loads(probe_path.read_text())
        phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
        refs = [(r["start_sec"], r["end_sec"]) for r in case["references"]]
        flora_cuts = internal_cuts(sorted((s["start_s"], s["end_s"]) for s in record["segments"]))
        cuts = internal_cuts(phases)
        beliefs = boundary_beliefs(record)
        fam = per_family.setdefault(case["family"], {"belief": [], "near_flora": [], "y": [], "cases": []})
        fam["belief"] += [cut_belief(c, beliefs) for c in cuts]
        fam["near_flora"] += [-min((abs(c - f) for f in flora_cuts), default=1e9) for c in cuts]
        fam["y"] += labels(cuts, refs)
        fam["cases"].append(dict(phases=phases, cuts=cuts, scores=[cut_belief(c, beliefs) for c in cuts],
                                 references=case["references"]))
    result: Dict[str, object] = {}
    for family, fam in per_family.items():
        rows = {}
        for th in THRESHOLDS:
            items = []
            for c in fam["cases"]:
                keep = [cut for cut, s in zip(c["cuts"], c["scores"]) if s >= th]
                merged = merge_phases(c["phases"], keep)
                items.append({"predictions": [{"start_sec": s, "end_sec": e} for s, e in merged],
                              "references": c["references"]})
            s = score_family(items)
            rows[str(th)] = {"n_pred": s["segment@0.75"]["n_pred"], "n_ref": s["segment@0.75"]["n_ref"],
                             "f1@0.75": s["segment@0.75"]["f1"], "f1@0.5": s["segment@0.5"]["f1"]}
        result[family] = {"cuts": len(fam["y"]), "positives": sum(fam["y"]),
                          "auc_belief": auc(fam["belief"], fam["y"]),
                          "auc_near_flora": auc(fam["near_flora"], fam["y"]), "thresholds": rows}
    return result


def report(result: Dict[str, object]) -> str:
    lines = ["| Family | Cuts | Positives | AUC belief | AUC distance to TempSeg cut |", "|---|---:|---:|---:|---:|"]
    for family, r in result.items():
        lines.append(f"| {family} | {r['cuts']} | {r['positives']} | {r['auc_belief']:.3f} | {r['auc_near_flora']:.3f} |")
    lines += ["", "| Family | Threshold | Pred / ref | F1@0.75 | F1@0.5 |", "|---|---:|---:|---:|---:|"]
    for family, r in result.items():
        for th, row in r["thresholds"].items():
            lines.append(f"| {family} | {th} | {row['n_pred']} / {row['n_ref']} | {row['f1@0.75']:.3f} | {row['f1@0.5']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    ap.add_argument("--kinematic", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    result = analyse(Path(args.kinematic), Path(args.probe), scorer(Path(args.autolabel)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "beliefs.json").write_text(json.dumps(result, indent=1) + "\n")
    text = report(result)
    (out / "report.md").write_text("# TempSeg beliefs at kinematic cuts\n\n" + text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
