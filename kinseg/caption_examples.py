"""Markdown pages of Captioner examples: each run's caption of every phase, and each method's segments.

For every episode (of any run, or only those all runs share, ``--shared``): the task; a table of the
phases (time, moving arm, gripper events, reference subtask, one caption per run); and a table of
segmentations (the references, TempSeg, TempSeg + gripper split, and every run under every join rule
in ``JOINS``) with the episode's counts under the WGO protocol. A segment the protocol's one-to-one
matching pairs with a reference is marked ✓. An index at the top lists the episodes with their scores.

    python kinseg/caption_examples.py --autolabel <checkout> --kinematic <run> --tempseg out/tempseg \\
        --runs A=<dir> B_both=<dir> ... --out examples.md [--episodes ...] [--videos <dir>] [--full]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from caption_merge import join_operations, motion_only, normalise  # noqa: E402
from merge import boundaries, gripper_events, split_at_grasps  # noqa: E402

Interval = Tuple[float, float]
# Join rules shown for every run: phases (with captions) -> segments with a label.
JOINS: Dict[str, Callable[[List[Dict[str, object]]], List[Dict[str, object]]]] = {
    "operation join": lambda phases: join_operations(phases),
    "motion absorbed": lambda phases: join_operations(phases, motion_only),
}


def protocol(autolabel: Path):
    """``snap_outer_edges`` and ``match_one_to_one`` of the WGO scorer in an autolabel checkout."""
    sys.path.insert(0, str(autolabel))
    from modules.exp_wgo_bench.score import match_one_to_one, snap_outer_edges
    return snap_outer_edges, match_one_to_one


def matched(pred: Sequence[Interval], ref: Sequence[Interval], snap, match) -> set:
    """Indices of the predictions the protocol matches (outer edges snapped, tIoU >= 0.75, one-to-one)."""
    return {i for i, _ in match(snap(pred, ref), ref, 0.75)}


def f1_text(tp: int, n_pred: int, n_ref: int) -> str:
    f1 = 2 * tp / (n_pred + n_ref) if n_pred + n_ref else 0.0
    return f"{f1:.2f} ({tp}/{n_pred}/{n_ref})"


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def segment_list(intervals: Sequence[Interval], labels: Sequence[str], hits: set) -> str:
    return "<br>".join(" ".join(part for part in (f"`{a:.1f}–{b:.1f}`", cell(label), "✓" if k in hits else "") if part)
                       for k, ((a, b), label) in enumerate(zip(intervals, labels)))


def episode_page(episode: str, case: Dict[str, object], records: Dict[str, Dict[str, object]],
                 tempseg: Optional[Dict[str, object]], snap, match, full: bool,
                 video: Optional[str], combined: Optional[Dict[str, Dict[str, object]]] = None
                 ) -> Tuple[str, Dict[str, str]]:
    """Markdown of one episode, and its index entries (method -> F1 text).

    ``combined`` maps a name to one episode's ``combine.py score --segments`` file (C1-C3 and the
    model's gripper verdicts).
    """
    refs = [(r["start_sec"], r["end_sec"]) for r in case["references"]]
    ref_labels = [r["subtask"] for r in case["references"]]
    phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
    events = gripper_events(case)
    lines = [f"## {episode}", "",
             f"**Task:** {cell(str(case.get('instruction', '')))}  ",
             f"{case['family']}, {case['duration_sec']:.1f} s, {len(phases)} phases, {len(refs)} subtasks"
             + (f", [video]({video})" if video else ""), ""]

    # Phases: the captions of each run whose intervals are the phases.
    names = [n for n, rec in records.items()
             if len(rec["rounds"][0]) == len(phases)
             and all(abs(a - p["start_s"]) < 1e-6 and abs(b - p["end_s"]) < 1e-6
                     for (a, b), p in zip(phases, rec["rounds"][0]))]
    other = [n for n in records if n not in names]
    lines += ["| # | Phase (s) | Arm | Gripper | Reference subtask | " + " | ".join(names) + " |",
              "|---:|---|---|---|---|" + "---|" * len(names)]
    previous_ref = None
    for k, (a, b) in enumerate(phases):
        mid = (a + b) / 2
        j = next((j for j, (s, e) in enumerate(refs) if s <= mid < e), None)
        ref = "–" if j is None else ("▸ " if j != previous_ref else "") + ref_labels[j]
        previous_ref = j
        arm = case["predictions"][k]["arm"].lower()
        grip = ", ".join(f"{arm_[0]} {kind}" for arm_, kind, t0, _ in events if a <= t0 < b)
        first = records[names[0]]["rounds"][0][k] if names else None
        row = []
        for n in names:
            p = records[n]["rounds"][0][k]
            text = p["caption"] if p.get("status") == "complete" else f"*{p.get('status')}*"
            if not full and n != names[0] and normalise(p["caption"]) == normalise(first["caption"]):
                text = "="
            row.append(cell(text))
        lines.append(f"| {k} | {a:.1f}–{b:.1f} | {'–' if arm == 'stable' else arm} | {grip} | {cell(ref)} | "
                     + " | ".join(row) + " |")
    if other:
        lines += ["", f"(Not on these phases, segments only: {', '.join(other)}.)"]

    # Segmentations.
    lines += ["", "| Method | F1 (TP/pred/ref) | Segments |", "|---|---:|---|",
              f"| reference | – | {segment_list(refs, ref_labels, set())} |"]
    index: Dict[str, str] = {}

    def add(name: str, intervals: List[Interval], labels: List[str]) -> None:
        hits = matched(intervals, refs, snap, match)
        index[name] = f1_text(len(hits), len(intervals), len(refs))
        lines.append(f"| {name} | {index[name]} | {segment_list(intervals, labels, hits)} |")

    if tempseg is not None:
        segs = sorted((s["start_s"], s["end_s"]) for s in tempseg["segments"])
        add("TempSeg", segs, [""] * len(segs))
        split = split_at_grasps(segs, events, boundaries(phases))
        add("TempSeg + gripper split", split, [""] * len(split))
    for n, rec in records.items():
        for join, rule in JOINS.items():
            joined = rule(rec["rounds"][0])
            add(f"{n} · {join}", [(j["start_s"], j["end_s"]) for j in joined], [j["key"] for j in joined])
    for n, seg in (combined or {}).items():
        for key, title in (("C1", "model's subtasks"), ("C2", "C1 + confirmed gripper split"),
                           ("C3", "TempSeg + confirmed gripper split")):
            add(f"{n} · {key} {title}", [(x["start_s"], x["end_s"]) for x in seg[key]],
                [x.get("label", "") for x in seg[key]])
        verdicts = "<br>".join(f"phase {g.get('phase')} {g.get('hand', '')} {g.get('event')}: {g.get('is')}"
                               for g in seg.get("gripper") or [])
        lines.append(f"| {n} · gripper verdicts | – | {cell(verdicts) or '(none)'} |")
    return "\n".join(lines) + "\n", index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"), help="autolabel checkout (scorer)")
    ap.add_argument("--kinematic", required=True, help="autolabel WGO run with predictions/*/<episode>.json")
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=DIR caption run folders (caption_merge.py run)")
    ap.add_argument("--tempseg", help="segment.py output dir: also show TempSeg and TempSeg + gripper split")
    ap.add_argument("--episodes", nargs="+", help="only these episodes")
    ap.add_argument("--shared", action="store_true", help="only episodes every run has")
    ap.add_argument("--videos", help="folder of <episode>.mp4, linked from each episode")
    ap.add_argument("--full", action="store_true", help="print every caption (default: '=' where equal to the first run's)")
    ap.add_argument("--combined", nargs="+", default=[], help="NAME=DIR segment folders of combine.py score --segments")
    ap.add_argument("--out", required=True, help="markdown file to write")
    args = ap.parse_args(argv)
    if not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    snap, match = protocol(Path(args.autolabel))
    kin = {}
    for path in Path(args.kinematic).glob("predictions/*/*.json"):
        case = json.loads(path.read_text())
        kin[case["id"]] = case
    runs = dict(item.split("=", 1) for item in args.runs)
    have = {name: {p.stem for p in Path(folder).glob("*.json") if p.stem in kin} for name, folder in runs.items()}
    episodes = set.intersection(*have.values()) if args.shared else set.union(*have.values())
    if args.episodes:
        episodes &= set(args.episodes)
    out = Path(args.out)
    pages, rows = [], []
    for episode in sorted(episodes, key=lambda e: (kin[e]["family"] != "galaxea", kin[e]["family"], e)):
        records = {n: json.loads((Path(runs[n]) / f"{episode}.json").read_text()) for n in runs if episode in have[n]}
        tempseg_path = Path(args.tempseg) / f"{episode}.json" if args.tempseg else None
        tempseg = json.loads(tempseg_path.read_text()) if tempseg_path and tempseg_path.exists() else None
        video = None
        if args.videos and (Path(args.videos) / f"{episode}.mp4").exists():
            video = os.path.relpath(Path(args.videos) / f"{episode}.mp4", out.parent.resolve())
        combined = {n: json.loads((Path(d) / f"{episode}.json").read_text())
                    for n, d in (c.split("=", 1) for c in args.combined) if (Path(d) / f"{episode}.json").exists()}
        page, index = episode_page(episode, kin[episode], records, tempseg, snap, match, args.full, video, combined)
        pages.append(page)
        best = max(((v, k) for k, v in index.items() if "·" in k), key=lambda t: float(t[0].split()[0]),
                   default=("–", "–"))
        rows.append(f"| [{episode}](#{episode}) | {kin[episode]['family']} | {cell(str(kin[episode].get('instruction', '')))} | "
                    f"{len(kin[episode]['predictions'])} | {len(kin[episode]['references'])} | "
                    f"{index.get('TempSeg', '–')} | {index.get('TempSeg + gripper split', '–')} | {best[1]}: {best[0]} |")
    header = ["# Captioner examples", "",
              "Runs: " + ", ".join(f"{n} (`{os.path.relpath(Path(d).resolve(), out.parent.resolve())}`)"
                                   for n, d in runs.items()) + ".  ",
              f"Join rules: {', '.join(JOINS)}. F1@0.75 per episode, (TP/pred/ref) under the WGO protocol; "
              "✓ marks a segment matched to a reference. In the phase tables, ▸ marks the first phase of a "
              "reference subtask" + ("" if args.full else ", and = a caption equal to the first run's") + ".", "",
              "| Episode | Family | Task | Phases | Subtasks | TempSeg | TempSeg + gripper split | Best caption method |",
              "|---|---|---|---:|---:|---:|---:|---|"] + rows + [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n" + "\n".join(pages))
    print(f"wrote {out} ({len(pages)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
