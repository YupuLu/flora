"""Combine autolabel's kinematic activity phases with Flora-TempSeg segments.

The kinematic phases cut where the robot's motion changes and contain most
reference boundaries (high merge recall), but are finer than subtasks;
TempSeg groups the video into subtasks but places boundaries on a 4 Hz frame
clock. The rules below take the grouping from one and the boundaries from the
other:

* ``kinematic``: the phases alone.
* ``flora``: the TempSeg segments alone.
* ``snap@d``: TempSeg segments whose endpoints move to the nearest phase
  boundary within ``d`` seconds.
* ``group``: phases joined by the TempSeg segment that overlaps each of them
  most; a phase overlapping none joins its preceding group. The result keeps
  the kinematic boundaries and TempSeg's granularity.
* ``split-first`` / ``split-mid``: TempSeg segments that contain a gripper
  close and a later open of the same arm (an acquisition and a delivery,
  which TempSeg's own policy keeps distinct) are split at a kinematic cut
  between the two: the first cut after the close, or the cut nearest the
  middle between close and open. Repeated while a part still holds such a pair.

Scores use autolabel's WGO-Bench scorer (``modules/exp_wgo_bench/score.py``)::

    python kinseg/merge.py --kinematic <exp_wgo_bench run> --flora <segment.py output dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

AUTOLABEL = Path(os.environ.get("AUTOLABEL_ROOT", "/media/data6/yplu/Github/manip/prep/autolabel"))
sys.path.insert(0, str(AUTOLABEL))

from modules.exp_wgo_bench.score import score_family  # noqa: E402

Interval = Tuple[float, float]
SNAP_TOLERANCES_S = (0.5, 1.0)


def boundaries(phases: Sequence[Interval]) -> List[float]:
    """Sorted cut points of the phases, outer edges included."""
    return sorted({t for p in phases for t in p})


def snap(segments: Sequence[Interval], cuts: Sequence[float], tolerance: float) -> List[Interval]:
    """Move each endpoint to the nearest cut point within ``tolerance``."""
    def nearest(t: float) -> float:
        k = bisect_left(cuts, t)
        near = min((c for c in cuts[max(0, k - 1):k + 1]), key=lambda c: abs(c - t), default=t)
        return near if abs(near - t) <= tolerance else t

    out = []
    for s, e in segments:
        s2, e2 = nearest(s), nearest(e)
        out.append((s2, e2) if e2 > s2 else (s, e))
    return out


def group(phases: Sequence[Interval], segments: Sequence[Interval]) -> List[Interval]:
    """Join consecutive phases assigned to the same segment (largest overlap)."""
    labels: List[int] = []
    for s, e in phases:
        overlaps = [max(0.0, min(e, b) - max(s, a)) for a, b in segments]
        best = max(range(len(overlaps)), key=overlaps.__getitem__, default=-1)
        labels.append(best if best >= 0 and overlaps[best] > 0 else (labels[-1] if labels else -1))
    out: List[Interval] = []
    for (s, e), label, prev in zip(phases, labels, [None] + labels[:-1]):
        if out and label == prev:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


Event = Tuple[str, str, float, float]  # arm, kind, start, end


def split_at_grasps(segments: Sequence[Interval], events: Sequence[Event], cuts: Sequence[float],
                    where: str = "first") -> List[Interval]:
    """Split segments holding a close and a later open of the same arm at a cut between them."""
    def split(a: float, b: float) -> List[Interval]:
        for arm, kind, c0, c1 in sorted(events, key=lambda ev: ev[2]):
            if kind != "close" or not a <= c0 < b:
                continue
            opens = [o0 for arm2, kind2, o0, _ in events if arm2 == arm and kind2 == "open" and c1 < o0 < b]
            if not opens:
                continue
            o0 = min(opens)
            between = [t for t in cuts if max(a, c1) < t < o0]
            if not between:
                continue
            k = between[0] if where == "first" else min(between, key=lambda t: abs(t - (c1 + o0) / 2))
            return [(a, k)] + split(k, b)
        return [(a, b)]

    return [part for s, e in segments for part in split(s, e)]


def rules(phases: Sequence[Interval], segments: Sequence[Interval],
          events: Sequence[Event] = ()) -> Dict[str, List[Interval]]:
    cuts = boundaries(phases)
    out = {"kinematic": list(phases), "flora": list(segments), "group": group(phases, segments)}
    for d in SNAP_TOLERANCES_S:
        out[f"snap@{d:g}"] = snap(segments, cuts, d)
    for where in ("first", "mid"):
        out[f"split-{where}"] = split_at_grasps(segments, events, cuts, where)
    return out


def gripper_events(case: Dict[str, object]) -> List[Event]:
    """Distinct gripper events of a prediction file (each phase lists the events it holds)."""
    seen = {(g["arm"], g["kind"], g["start_sec"], g["end_sec"])
            for p in case["predictions"] for g in p.get("gripper_events", [])}
    return sorted(seen, key=lambda ev: ev[2])


def _as_items(intervals: Sequence[Interval]) -> List[Dict[str, float]]:
    return [{"start_sec": s, "end_sec": e} for s, e in intervals]


def evaluate(kinematic_run: Path, flora_dir: Path) -> Dict[str, Dict[str, object]]:
    """Scores per family and rule, plus the combined intervals per episode."""
    cases: Dict[str, Dict[str, list]] = {}
    combined: Dict[str, Dict[str, object]] = {}
    for path in sorted((kinematic_run / "predictions").glob("*/*.json")):
        case = json.loads(path.read_text())
        flora_path = flora_dir / f"{case['id']}.json"
        if not flora_path.exists():
            continue
        flora = json.loads(flora_path.read_text())
        phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
        segments = sorted((s["start_s"], s["end_s"]) for s in flora["segments"])
        per_rule = rules(phases, segments, gripper_events(case))
        combined[case["id"]] = {"family": case["family"], **{k: v for k, v in per_rule.items()}}
        for name, intervals in per_rule.items():
            cases.setdefault(case["family"], {}).setdefault(name, []).append(
                {"predictions": _as_items(intervals), "references": case["references"]})
    scores = {family: {name: score_family(items) for name, items in by_rule.items()}
              for family, by_rule in cases.items()}
    return {"scores": scores, "combined": combined}


def report(scores: Dict[str, Dict[str, object]]) -> str:
    head = ["| Family | Rule | Episodes | Pred / ref | F1@0.75 | P / R @0.75 | F1@0.5 | Merge recall@0.75 "
            "| Boundary recall ±0.5 s | Offset median abs |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for family, by_rule in scores.items():
        for name, s in by_rule.items():
            seg, seg5 = s["segment@0.75"], s["segment@0.5"]
            rows.append(f"| {family} | {name} | {s['episodes']} | {seg['n_pred']} / {seg['n_ref']} | {seg['f1']:.3f} "
                        f"| {seg['precision']:.3f} / {seg['recall']:.3f} | {seg5['f1']:.3f} "
                        f"| {s['merge_recall@0.75']['recall']:.3f} | {s['boundary_recall@0.5s']['recall']:.3f} "
                        f"| {s['boundary_offset_s']['median_abs']:.2f} s |")
    return "\n".join(head + rows) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kinematic", required=True, help="autolabel exp_wgo_bench run directory")
    ap.add_argument("--flora", required=True, help="kinseg/segment.py output directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = evaluate(Path(args.kinematic), Path(args.flora))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "merge.json").write_text(json.dumps(result, indent=1) + "\n")
    text = report(result["scores"])
    (out / "report.md").write_text(f"# Kinematic phases and Flora-TempSeg on WGO-Bench\n\n"
                                   f"- kinematic: `{args.kinematic}`\n- flora: `{args.flora}`\n\n{text}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
