"""Join kinematic phases with Flora-Captioner: phases captioned alike belong to one operation.

The Captioner writes one instruction per given interval, for the intended
operation and at the granularity of robot subtask annotation, with the
neighbouring intervals as context. Starting from the kinematic phases (or the
tiles between kinematic and TempSeg cuts, ``--cuts union``), each round

1. captions every interval that is new since the previous round,
2. joins consecutive intervals whose captions are equal after normalisation
   (lower case, punctuation and articles removed); an interval the Captioner
   cannot take (too few frames) joins its predecessor,

and stops when a round joins nothing (or after ``--rounds``). Every round's
intervals and captions are saved, so other joining rules can be evaluated
without the GPU.

    CUDA_VISIBLE_DEVICES=5,6 python kinseg/caption_merge.py run --kinematic <run> --videos <dir> \\
        --episodes galaxea_002 ... --out <dir> [--cuts union --tempseg <dir>] [--instruction]
    python kinseg/caption_merge.py score --autolabel <checkout> --kinematic <run> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARTICLES = {"a", "an", "the"}


def normalise(caption: str) -> str:
    """Lower case, punctuation and articles removed, single spaces."""
    words = re.sub(r"[^a-z0-9]+", " ", caption.lower()).split()
    return " ".join(w for w in words if w not in ARTICLES)


def join_equal(intervals: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Join consecutive intervals with equal normalised captions; uncaptioned ones join their predecessor."""
    out: List[Dict[str, object]] = []
    for item in intervals:
        key = normalise(item["caption"]) if item.get("status") == "complete" and item.get("caption") else None
        last = out[-1] if out else None
        if last is not None and (key is None or key == last["key"]):
            last["end_s"] = item["end_s"]
            last["members"] += item["members"]
            last["joined"] = True
            continue
        out.append(dict(start_s=item["start_s"], end_s=item["end_s"], key=key, caption=item.get("caption", ""),
                        status=item.get("status"), members=list(item["members"]), joined=False))
    return out


def tiles(cuts: List[float], start: float, end: float) -> List[Dict[str, object]]:
    edges = [start] + [c for c in sorted(set(cuts)) if start < c < end] + [end]
    return [dict(start_s=a, end_s=b, members=[k]) for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def initial_intervals(case: Dict[str, object], tempseg: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
    phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
    cuts = [e for _, e in phases[:-1]]
    if tempseg is not None:
        cuts += [t for s in tempseg["segments"] for t in (s["start_s"], s["end_s"])]
    return tiles(cuts, phases[0][0], phases[-1][1])


# ── GPU part ───────────────────────────────────────────────────────────


def caption_subset(pipeline, videos, segments, todo, profile, session, instruction) -> Dict[int, Dict[str, str]]:
    """FLORA's caption_single for ``segments[k]``, ``k`` in ``todo``, with context from the full list."""
    import hashlib
    import tempfile
    from flora_unified.unified_input import processor_inputs, prompt as native_prompt
    from transformers.video_utils import VideoMetadata

    plans, dispositions = pipeline.single_plans(videos, segments, profile)
    out = {d["event_index"]: dict(caption="", status=d["status"]) for d in dispositions if d["event_index"] in todo}
    plans = [p for p in plans if p["event_index"] in todo]
    if not plans:
        return out
    model, proc = session.load("captioner")
    with tempfile.TemporaryDirectory(prefix="kinseg-frames-") as temporary:
        root = Path(temporary)
        wanted = {i for plan in plans for block in plan["blocks"] for i in block["frame_indices"]}
        receipts = {}
        for i, image in pipeline.frames_for(videos, wanted).items():
            path = root / f"{i:012d}.png"
            image.save(path, format="PNG")
            receipts[i] = dict(path=path.name, png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               size=list(image.size))
        for plan in plans:
            sample = pipeline.a.caption_sample(plan, receipts, ["main"])
            if instruction.strip():
                from flora_unified.unified_input import load_frames
                frames, _ = load_frames(sample, root)
                text = proc.apply_chat_template([dict(role="user", content=native_prompt(sample, proc.video_token)
                                                      + pipeline.instruction_suffix(instruction))],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
                metadata = [VideoMetadata(total_num_frames=len(f), fps=float(videos["main"].rate),
                                          frames_indices=list(b["block_local_timestamp_indices"]))
                            for f, b in zip(frames, sample["blocks"], strict=True)]
                inputs = proc(text=[text], videos=frames, video_metadata=metadata, do_sample_frames=False,
                              return_metadata=True, return_tensors="pt")
            else:
                inputs, _, _ = processor_inputs(sample, root, proc)
            out[plan["event_index"]] = dict(caption=pipeline.generate(model, proc, inputs, 512), status="complete")
    return out


def run(args) -> int:
    import local_models
    from segment import PROFILES  # also puts tempseg/ on sys.path
    from batch_utils import pipeline
    from batch_utils.api import DataEnginePipeline
    from batch_utils.models import Session

    local_models.configure()
    kin = {json.loads(p.read_text())["id"]: p for p in Path(args.kinematic).glob("predictions/*/*.json")}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = Session()
    loaded = session.load("captioner")
    session.load = lambda kind: loaded  # one Captioner for every call
    engine = DataEnginePipeline()
    for episode in args.episodes:
        out = out_dir / f"{episode}.json"
        if out.exists() and not args.overwrite:
            print(f"skip {episode}")
            continue
        case = json.loads(kin[episode].read_text())
        tempseg = json.loads((Path(args.tempseg) / f"{episode}.json").read_text()) if args.cuts == "union" else None
        instruction = case.get("instruction", "") if args.instruction else ""
        videos, _ = engine._inputs(str(Path(args.videos) / f"{episode}.mp4"), PROFILES[args.profile], None, None,
                                   False, instruction)
        intervals = initial_intervals(case, tempseg)
        for item in intervals:
            item["status"], item["caption"] = None, ""
        rounds, started = [], time.monotonic()
        for r in range(args.rounds):
            todo = {k for k, item in enumerate(intervals) if item["status"] is None or item.get("joined")}
            rows = caption_subset(pipeline, videos, [dict(start_s=i["start_s"], end_s=i["end_s"]) for i in intervals],
                                  todo, PROFILES[args.profile], session, instruction)
            for k, row in rows.items():
                intervals[k].update(caption=row["caption"], status=row["status"], joined=False)
            rounds.append([dict(start_s=i["start_s"], end_s=i["end_s"], caption=i["caption"], status=i["status"],
                                members=i["members"]) for i in intervals])
            joined = join_equal(intervals)
            print(f"{episode} round {r + 1}: {len(todo)} captioned, {len(intervals)} -> {len(joined)} intervals",
                  flush=True)
            if len(joined) == len(intervals):
                break
            intervals = joined
        record = dict(id=episode, family=case["family"], cuts=args.cuts, instruction=instruction, rounds=rounds,
                      final=[dict(start_s=i["start_s"], end_s=i["end_s"], caption=i["caption"]) for i in
                             join_equal(intervals)],
                      elapsed_s=round(time.monotonic() - started, 1))
        out.write_text(json.dumps(record, indent=1) + "\n")
    return 0


# ── scores ─────────────────────────────────────────────────────────────


def score(args) -> int:
    from merge import scorer

    score_family = scorer(Path(args.autolabel))
    kin = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
           for p in Path(args.kinematic).glob("predictions/*/*.json")}
    fams: Dict[str, Dict[str, list]] = {}
    for path in sorted(Path(args.out).glob("*.json")):
        rec = json.loads(path.read_text())
        refs = kin[rec["id"]]["references"]
        fam = fams.setdefault(rec["family"], {})
        first = join_equal([dict(r, members=r["members"]) for r in rec["rounds"][0]])
        variants = {"round 1 join": first, "final": rec["final"]}
        for name, intervals in variants.items():
            fam.setdefault(name, []).append({"predictions": [{"start_sec": i["start_s"], "end_sec": i["end_s"]}
                                                             for i in intervals], "references": refs})
    lines = []
    for family, by in fams.items():
        for name, items in by.items():
            s = score_family(items)
            seg = s["segment@0.75"]
            lines.append(f"{family:8s} {name:13s} episodes {len(items):2d}  pred/ref {seg['n_pred']}/{seg['n_ref']}  "
                         f"F1@0.75 {seg['f1']:.3f}  F1@0.5 {s['segment@0.5']['f1']:.3f}  "
                         f"boundary recall 0.5 s {s['boundary_recall@0.5s']['recall']:.3f}")
    text = "\n".join(lines)
    print(text)
    (Path(args.out) / "score.md").write_text(text + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kinematic", required=True)
    r.add_argument("--videos", required=True, help="folder of <episode>.mp4")
    r.add_argument("--episodes", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cuts", choices=("ours", "union"), default="ours")
    r.add_argument("--tempseg", help="segment.py output dir, for --cuts union")
    r.add_argument("--profile", default="yam")
    r.add_argument("--instruction", action="store_true", help="give the episode's task instruction to the Captioner")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    s.add_argument("--kinematic", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "run" and args.cuts == "union" and not args.tempseg:
        ap.error("--cuts union needs --tempseg")
    if args.command == "score" and not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    return run(args) if args.command == "run" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
