"""WGO scoring that forgives where a free movement is attached: how much of the error is convention.

A movement between two actions (moving away after a release, reaching, lifting after a grasp) can be
attached to the previous or the next subtask without being wrong. Around every internal reference cut,
the *transition region* is the arm movement next to the cut that contains no gripper event: the phases
touching the cut (and ``--neighbours`` more on each side), cut back at the nearest gripper event so that
a region never contains one. A predicted segment endpoint inside a region is moved onto that region's
reference cut before scoring; everything else follows the WGO protocol (outer edges snapped, tIoU at
least 0.75, one-to-one matching, counts pooled per family). A method whose cuts fall anywhere inside the
regions then scores as if it had followed the reference convention.

    python kinseg/transition_eval.py --autolabel <checkout> --kinematic <run> --tempseg <dir> \\
        --segments NAME=DIR ... [--out <file>]

``--segments`` folders are ``combine.py score --segments`` output (C1, C4) or caption-free
``<episode>.json`` files with a ``segments`` list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

Interval = Tuple[float, float]


def regions(case: Dict[str, object], neighbours: int = 0) -> List[Tuple[float, Interval]]:
    """(reference cut, transition region) for the internal reference cuts of a kinematic prediction."""
    from merge import gripper_events

    phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
    events = gripper_events(case)
    out = []
    for r in case["references"][:-1]:
        c = r["end_sec"]
        touching = [k for k, (a, b) in enumerate(phases) if a <= c <= b]
        if not touching:
            continue
        lo_k = max(0, min(touching) - neighbours)
        hi_k = min(len(phases) - 1, max(touching) + neighbours)
        lo, hi = phases[lo_k][0], phases[hi_k][1]
        for _, _, e0, e1 in events:  # never cross a gripper event
            if e1 <= c:
                lo = max(lo, e1)
            elif e0 >= c:
                hi = min(hi, e0)
            else:  # the cut lies inside an event: no free movement there
                lo, hi = c, c
        if hi > lo:
            out.append((c, (lo, hi)))
    return out


def merge_intervals_of(cut_regions: Sequence[Tuple[float, Interval]]) -> List[Tuple[None, Interval]]:
    return [(None, r) for r in merge_intervals([r for _, r in cut_regions])]


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def forgive(pred: Sequence[Interval], cut_regions: Sequence[Tuple[float, Interval]]) -> List[Interval]:
    """``pred`` with every endpoint that lies inside a transition region moved onto its reference cut."""
    def move(t: float) -> float:
        for cut, (lo, hi) in cut_regions:
            if lo <= t <= hi:
                return cut
        return t
    out = [(move(a), move(b)) for a, b in pred]
    return [(a, b) if b > a else (a, a) for a, b in out]


def tiou(a: Interval, b: Interval) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if inter > 0 and union > 0 else 0.0


def matches(pred: Sequence[Interval], ref: Sequence[Interval], cut_regions, snap, threshold=0.75) -> int:
    """One-to-one matches (Kuhn) over pairs with tIoU at least ``threshold`` after forgiving endpoints
    inside transition regions, outer edges snapped."""
    scored = snap(forgive(pred, cut_regions), ref)
    edges = [[j for j, r in enumerate(ref) if tiou(p, r) >= threshold] for p in scored]
    owner: Dict[int, int] = {}

    def augment(i: int, seen: set) -> bool:
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(scored)))


def main(argv=None) -> int:
    from merge import boundaries, gripper_events, split_at_grasps

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    ap.add_argument("--kinematic", required=True)
    ap.add_argument("--tempseg", required=True)
    ap.add_argument("--segments", nargs="+", default=[], help="NAME=DIR segment folders")
    ap.add_argument("--keys", nargs="+", default=["C1", "C4"], help="segment lists to score in each file")
    ap.add_argument("--neighbours", type=int, default=0, help="phases beyond those touching a cut")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.autolabel)))
    from modules.exp_wgo_bench.score import snap_outer_edges

    kin = {}
    for path in Path(args.kinematic).glob("predictions/*/*.json"):
        case = json.loads(path.read_text())
        kin[case["id"]] = case
    sources = dict(s.split("=", 1) for s in args.segments)
    episodes = sorted(set.intersection(*[{p.stem for p in Path(d).glob("*.json")} for d in sources.values()])
                      if sources else kin)
    counts: Dict[Tuple[str, str], List[int]] = {}
    masked_time: Dict[str, List[float]] = {}

    def add(fam: str, name: str, pred: List[Interval], refs: List[Interval], mask: List[Interval]) -> None:
        c = counts.setdefault((fam, name), [0, 0, 0, 0, 0])
        c[0] += matches(pred, refs, [], snap_outer_edges)
        c[1] += matches(pred, refs, mask, snap_outer_edges)
        c[4] += sum(1 for a, b in pred for t in (a, b) if any(lo <= t <= hi for _, (lo, hi) in mask))
        c[2] += len(pred)
        c[3] += len(refs)

    for episode in episodes:
        case = kin[episode]
        fam = case["family"]
        refs = [(r["start_sec"], r["end_sec"]) for r in case["references"]]
        mask = regions(case, args.neighbours)
        t = masked_time.setdefault(fam, [0.0, 0.0])
        t[0] += sum(hi - lo for _, (lo, hi) in merge_intervals_of(mask))
        t[1] += refs[-1][1] - refs[0][0]
        segs = sorted((s["start_s"], s["end_s"]) for s in
                      json.loads((Path(args.tempseg) / f"{episode}.json").read_text())["segments"])
        cuts = boundaries([(p["start_sec"], p["end_sec"]) for p in case["predictions"]])
        add(fam, "TempSeg", segs, refs, mask)
        add(fam, "TempSeg + gripper split", split_at_grasps(segs, gripper_events(case), cuts), refs, mask)
        for name, folder in sources.items():
            rec = json.loads((Path(folder) / f"{episode}.json").read_text())
            for key in args.keys if "segments" not in rec else ["segments"]:
                if key in rec:
                    add(fam, f"{name} {key}", [(x["start_s"], x["end_s"]) for x in rec[key]], refs, mask)
    lines = ["| Family | Method | F1@0.75 | F1@0.75, transitions forgiven | pred/ref | endpoints moved |",
             "|---|---|---:|---:|---:|---:|"]
    for (fam, name), (tp, tp_forgiven, n_pred, n_ref, moved) in sorted(counts.items()):
        f1 = 2 * tp / (n_pred + n_ref)
        f1m = 2 * tp_forgiven / (n_pred + n_ref)
        lines.append(f"| {fam} | {name} | {f1:.3f} | {f1m:.3f} | {n_pred}/{n_ref} | {moved} |")
    lines += [""] + [f"{fam}: transition regions cover {t[0] / t[1]:.1%} of the annotated time ({len(episodes)} episodes in all)"
                     for fam, t in sorted(masked_time.items())]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
