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
from typing import Dict, List, Optional, Tuple

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


MIN_TILE_S = 0.05


PARTICLES = {"up", "down", "over", "out", "in", "off", "on", "back", "away", "into", "onto"}
LIMBS = {"hand", "hands", "arm", "arms", "gripper", "grippers"}
MOTION_VERBS = {"move", "raise", "lower", "lift", "hold", "withdraw", "retract", "return", "reach", "bring", "keep"}


def operation(caption: str) -> str:
    """Leading verb phrase of a caption ("pick up", "place", "push"), '' if empty."""
    words = normalise(caption).split()
    if not words:
        return ""
    return " ".join(words[:2]) if len(words) > 1 and words[1] in PARTICLES else words[0]


def limb_motion(caption: str) -> bool:
    """A caption that only moves a hand or arm (an approach or a retreat), e.g. "raise both hands"."""
    words = normalise(caption).split()
    return len(words) >= 2 and words[0] in MOTION_VERBS and any(w in LIMBS for w in words[1:4])


def operation_keys(intervals: List[Dict[str, object]]) -> List[str]:
    """Operation of each interval; limb-only motion takes the next operation's (the last, the previous one's)."""
    keys: List[str] = []
    for item in intervals:
        caption = item.get("caption") or ""
        keys.append("" if limb_motion(caption) else operation(caption))
    for k in range(len(keys) - 2, -1, -1):  # approaches take the next operation's key
        if keys[k] == "":
            keys[k] = keys[k + 1]
    for k in range(1, len(keys)):  # a trailing motion takes the previous one's
        if keys[k] == "":
            keys[k] = keys[k - 1]
    return keys


def split_by_operation(segments: List[Tuple[float, float]], intervals: List[Dict[str, object]]) -> List[Tuple[float, float]]:
    """Split each segment where the operation of the captioned intervals inside it changes."""
    keys = operation_keys(intervals)
    out: List[Tuple[float, float]] = []
    for a, b in segments:
        inside = [(item, key) for item, key in zip(intervals, keys) if a <= (item["start_s"] + item["end_s"]) / 2 < b]
        cuts = [item["start_s"] for (item, key), (_, prev) in zip(inside[1:], inside[:-1]) if key != prev]
        edges = [a] + [c for c in cuts if a < c < b] + [b]
        out += list(zip(edges[:-1], edges[1:]))
    return out


def join_operations(intervals: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Join consecutive intervals of one operation; limb-only motion joins the next operation."""
    keys = operation_keys(intervals)
    out: List[Dict[str, object]] = []
    for item, key in zip(intervals, keys):
        if out and key == out[-1]["key"]:
            out[-1]["end_s"] = item["end_s"]
            out[-1]["captions"].append(item.get("caption", ""))
        else:
            out.append(dict(start_s=item["start_s"], end_s=item["end_s"], key=key, captions=[item.get("caption", "")]))
    return out


def tiles(cuts: List[float], start: float, end: float) -> List[Dict[str, object]]:
    """Intervals between the cuts; cuts closer than MIN_TILE_S to the previous edge or to the end are dropped."""
    edges = [start]
    for c in sorted(set(cuts)):
        if c - edges[-1] >= MIN_TILE_S and end - c >= MIN_TILE_S:
            edges.append(c)
    edges.append(end)
    return [dict(start_s=a, end_s=b, members=[k]) for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def robot_state(case: Dict[str, object], start_s: float, end_s: float) -> str:
    """Gripper state at the start of [start_s, end_s), its gripper events and moving arms, in words.

    From the kinematic prediction: a gripper is closed after its last close before ``start_s`` (each
    gripper starts open; every recorded episode's first event is a close). A robot whose phases and
    events name only the right arm is described as one arm ("the gripper").
    """
    from merge import gripper_events

    events = gripper_events(case)
    phases = case["predictions"]
    bimanual = any(p["arm"] in ("LEFT", "BOTH") for p in phases) or any(e[0] == "LEFT" for e in events)
    arms = ("LEFT", "RIGHT") if bimanual else ("RIGHT",)

    def name(arm: str, part: str) -> str:
        return f"the {arm.lower()} {part}" if bimanual else f"the {part}"

    def listed(items: List[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    states = []
    for arm in arms:
        before = [kind for a, kind, t0, _ in events if a == arm and t0 < start_s]
        states.append(f"{name(arm, 'gripper')} is {'closed' if before and before[-1] == 'close' else 'open'}")
    during = [f"{name(a, 'gripper')} {'closes' if kind == 'close' else 'opens'}"
              for a, kind, t0, _ in events if a in arms and start_s <= t0 < end_s]
    moving = set()
    for p in phases:
        if p["start_sec"] < end_s and p["end_sec"] > start_s and p["arm"] != "STABLE":
            moving |= set(arms) if p["arm"] == "BOTH" else {p["arm"]} & set(arms)
    if not moving:
        motion = "Neither arm moves much." if bimanual else "The arm hardly moves."
    elif len(moving) == 2:
        motion = "Both arms move."
    else:
        motion = name(moving.pop(), "arm").capitalize() + " moves."
    return " ".join([f"At the start of TARGET {listed(states)}.",
                     f"During TARGET {listed(during)}." if during else "No gripper opens or closes during TARGET.",
                     motion])


def initial_intervals(case: Dict[str, object], tempseg: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
    phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
    cuts = [e for _, e in phases[:-1]]
    if tempseg is not None:
        cuts += [t for s in tempseg["segments"] for t in (s["start_s"], s["end_s"])]
    return tiles(cuts, phases[0][0], phases[-1][1])


# ── GPU part ───────────────────────────────────────────────────────────


CONTEXT_ROLES = {"both": ("BEFORE", "TARGET", "AFTER"), "before": ("BEFORE", "TARGET"),
                 "after": ("TARGET", "AFTER"), "none": ("TARGET",)}


def caption_subset(pipeline, videos, segments, todo, profile, session, text_for,
                   context: str = "both") -> Dict[int, Dict[str, str]]:
    """FLORA's caption_single for ``segments[k]``, ``k`` in ``todo``, with context from the full list.

    ``text_for(k, previous_caption, segment)`` gives the episode-instruction text of interval ``k`` (empty for
    none); intervals are captioned in order, so the previous caption is the one just written (or the
    stored one when ``k - 1`` is not captioned this round). ``context`` chooses which of FLORA's
    BEFORE/TARGET/AFTER video blocks are sent; FLORA's prompt lists exactly the blocks sent.
    """
    import hashlib
    import tempfile
    from flora_unified.unified_input import processor_inputs, prompt as native_prompt
    from transformers.video_utils import VideoMetadata

    roles = CONTEXT_ROLES[context]
    plans, dispositions = pipeline.single_plans(videos, segments, profile)
    out = {d["event_index"]: dict(caption="", status=d["status"]) for d in dispositions if d["event_index"] in todo}
    plans = [p for p in plans if p["event_index"] in todo]
    if not plans:
        return out
    model, proc = session.load("captioner")
    with tempfile.TemporaryDirectory(prefix="kinseg-frames-") as temporary:
        root = Path(temporary)
        wanted = {i for plan in plans for block in plan["blocks"] if block["role"] in roles
                  for i in block["frame_indices"]}
        receipts = {}
        for i, image in pipeline.frames_for(videos, wanted).items():
            path = root / f"{i:012d}.png"
            image.save(path, format="PNG")
            receipts[i] = dict(path=path.name, png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               size=list(image.size))
        for plan in sorted(plans, key=lambda p: p["event_index"]):
            k = plan["event_index"]
            previous = (out[k - 1]["caption"] if k - 1 in out else segments[k - 1].get("caption", "")) if k else ""
            text_instruction = text_for(k, previous, segments[k])
            plan = dict(plan, blocks=[b for b in plan["blocks"] if b["role"] in roles])
            sample = pipeline.a.caption_sample(plan, receipts, ["main"])
            if text_instruction.strip():
                from flora_unified.unified_input import load_frames
                frames, _ = load_frames(sample, root)
                text = proc.apply_chat_template([dict(role="user", content=native_prompt(sample, proc.video_token)
                                                      + pipeline.instruction_suffix(text_instruction))],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
                metadata = [VideoMetadata(total_num_frames=len(f), fps=float(videos["main"].rate),
                                          frames_indices=list(b["block_local_timestamp_indices"]))
                            for f, b in zip(frames, sample["blocks"], strict=True)]
                inputs = proc(text=[text], videos=frames, video_metadata=metadata, do_sample_frames=False,
                              return_metadata=True, return_tensors="pt")
            else:
                inputs, _, _ = processor_inputs(sample, root, proc)
            out[k] = dict(caption=pipeline.generate(model, proc, inputs, 512), status="complete",
                          instruction_text=text_instruction)
    return out


def episode_text(args, case: Dict[str, object], scene: Optional[Dict[str, object]]):
    """``text_for(k, previous, segment)`` for one episode, from --background / --instruction / --previous /
    --robot-state."""
    import background

    task = str(case.get("instruction", ""))
    if args.instruction:  # FLORA's plain episode instruction: the task sentence alone
        return lambda k, previous, segment: task
    def text_for(k: int, previous: str, segment: Dict[str, float]) -> str:
        robot = robot_state(case, segment["start_s"], segment["end_s"]) if args.robot_state else ""
        return background.background_text(args.background, task, scene, previous if args.previous else "", robot)
    return text_for


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
    loaded = local_models.load(session, "captioner")
    session.load = lambda kind: loaded  # one Captioner for every call
    engine = DataEnginePipeline()
    for episode in args.episodes:
        out = out_dir / f"{episode}.json"
        if out.exists() and not args.overwrite:
            print(f"skip {episode}")
            continue
        case = json.loads(kin[episode].read_text())
        tempseg = json.loads((Path(args.tempseg) / f"{episode}.json").read_text()) if args.cuts == "union" else None
        scene = (json.loads((Path(args.scenes) / f"{episode}.json").read_text())
                 if args.scenes and (Path(args.scenes) / f"{episode}.json").exists() else None)
        text_for = episode_text(args, case, scene)
        videos, _ = engine._inputs(str(Path(args.videos) / f"{episode}.mp4"), PROFILES[args.profile], None, None,
                                   False, "")
        intervals = initial_intervals(case, tempseg)
        for item in intervals:
            item["status"], item["caption"] = None, ""
        rounds, started = [], time.monotonic()
        for r in range(args.rounds):
            todo = {k for k, item in enumerate(intervals) if item["status"] is None or item.get("joined")}
            rows = caption_subset(pipeline, videos,
                                  [dict(start_s=i["start_s"], end_s=i["end_s"], caption=i["caption"]) for i in intervals],
                                  todo, PROFILES[args.profile], session, text_for, args.context)
            for k, row in rows.items():
                intervals[k].update(caption=row["caption"], status=row["status"], joined=False,
                                    instruction_text=row.get("instruction_text", ""))
            rounds.append([dict(start_s=i["start_s"], end_s=i["end_s"], caption=i["caption"], status=i["status"],
                                members=i["members"], instruction_text=i.get("instruction_text", ""))
                           for i in intervals])
            joined = join_equal(intervals)
            print(f"{episode} round {r + 1}: {len(todo)} captioned, {len(intervals)} -> {len(joined)} intervals",
                  flush=True)
            if len(joined) == len(intervals):
                break
            intervals = joined
        record = dict(id=episode, family=case["family"], cuts=args.cuts, task=case.get("instruction", ""),
                      plain_instruction=args.instruction, background=args.background, context=args.context,
                      previous=args.previous, robot_state=args.robot_state, rounds=rounds,
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
        first = rec["rounds"][0]
        variants = {"exact join": join_equal([dict(r, members=r["members"]) for r in first]),
                    "operation join": join_operations(first)}
        if len(rec["rounds"]) > 1:
            variants["exact, iterated"] = rec["final"]
        if args.tempseg:
            segs = sorted((s["start_s"], s["end_s"]) for s in
                          json.loads((Path(args.tempseg) / f"{rec['id']}.json").read_text())["segments"])
            variants["TempSeg"] = [dict(start_s=a, end_s=b) for a, b in segs]
            variants["TempSeg + op. split"] = [dict(start_s=a, end_s=b) for a, b in split_by_operation(segs, first)]
        for name, intervals in variants.items():
            fam.setdefault(name, []).append({"predictions": [{"start_sec": i["start_s"], "end_sec": i["end_s"]}
                                                             for i in intervals], "references": refs})
    lines = []
    for family, by in fams.items():
        for name, items in by.items():
            s = score_family(items)
            seg = s["segment@0.75"]
            lines.append(f"{family:8s} {name:16s} episodes {len(items):2d}  pred/ref {seg['n_pred']}/{seg['n_ref']}  "
                         f"F1@0.75 {seg['f1']:.3f}  F1@0.5 {s['segment@0.5']['f1']:.3f}  "
                         f"boundary recall 0.5 s {s['boundary_recall@0.5s']['recall']:.3f}")
    text = "\n".join(lines)
    print(text)
    (Path(args.out) / "score.md").write_text(text + "\n")
    return 0


def echoes(caption: str, task: str, threshold: float = 0.8) -> bool:
    """A caption that repeats the task brief: it contains at least ``threshold`` of the brief's words."""
    a, b = set(normalise(caption).split()), set(normalise(task).split())
    return bool(a) and len(b) >= 3 and len(a & b) / len(b) >= threshold


def compare(args) -> int:
    """One table for several caption runs on the episodes they share."""
    from merge import scorer

    score_family = scorer(Path(args.autolabel))
    kin = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
           for p in Path(args.kinematic).glob("predictions/*/*.json")}
    runs = dict(item.split("=", 1) for item in args.runs)
    common = set.intersection(*[{p.stem for p in Path(d).glob("*.json") if p.name != "score.json"}
                                for d in runs.values()])
    lines = [f"Episodes: {len(common)} shared by all runs.", "",
             "| Run | Family | Captions | Echo task | Same op. inside / across | Exact join F1 | "
             "Operation join F1 (pred) | TempSeg + op. split F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, folder in runs.items():
        fams: Dict[str, Dict[str, object]] = {}
        for episode in sorted(common):
            rec = json.loads((Path(folder) / f"{episode}.json").read_text())
            case = kin[episode]
            refs = case["references"]
            phases = rec["rounds"][0]
            keys = operation_keys(phases)
            ref_cuts = [r["end_sec"] for r in refs[:-1]]
            f = fams.setdefault(rec["family"], {"captions": 0, "echo": 0, "inside": [0, 0], "across": [0, 0],
                                                "exact": [], "op": [], "split": []})
            f["captions"] += len(phases)
            f["echo"] += sum(echoes(p["caption"], str(case.get("instruction", ""))) for p in phases)
            for k in range(len(phases) - 1):
                where = "across" if any(abs(phases[k]["end_s"] - r) <= 0.5 for r in ref_cuts) else "inside"
                f[where][0] += 1
                f[where][1] += keys[k] == keys[k + 1]
            items = lambda intervals: {"predictions": [{"start_sec": i["start_s"], "end_sec": i["end_s"]}  # noqa: E731
                                                       for i in intervals], "references": refs}
            f["exact"].append(items(join_equal([dict(p) for p in phases])))
            f["op"].append(items(join_operations(phases)))
            if args.tempseg:
                segs = sorted((s["start_s"], s["end_s"]) for s in
                              json.loads((Path(args.tempseg) / f"{episode}.json").read_text())["segments"])
                f["split"].append(items([dict(start_s=a, end_s=b) for a, b in split_by_operation(segs, phases)]))
        for family, f in sorted(fams.items()):
            def f1(cases):
                if not cases:
                    return "-"
                seg = score_family(cases)["segment@0.75"]
                return f"{seg['f1']:.3f}"
            op = score_family(f["op"])["segment@0.75"]
            rate = lambda pair: f"{pair[1] / pair[0]:.2f}" if pair[0] else "-"  # noqa: E731
            lines.append(f"| {name} | {family} | {f['captions']} | {f['echo'] / f['captions']:.2f} | "
                         f"{rate(f['inside'])} / {rate(f['across'])} | {f1(f['exact'])} | "
                         f"{op['f1']:.3f} ({op['n_pred']}/{op['n_ref']}) | {f1(f['split'])} |")
    if args.tempseg:  # the reference points on the same episodes
        from merge import boundaries, gripper_events, split_at_grasps

        base: Dict[str, Dict[str, list]] = {}
        for episode in sorted(common):
            case = kin[episode]
            segs = sorted((s["start_s"], s["end_s"]) for s in
                          json.loads((Path(args.tempseg) / f"{episode}.json").read_text())["segments"])
            cuts = boundaries([(p["start_sec"], p["end_sec"]) for p in case["predictions"]])
            b = base.setdefault(case["family"], {"TempSeg alone": [], "TempSeg + gripper split": []})
            for key, intervals in (("TempSeg alone", segs),
                                   ("TempSeg + gripper split", split_at_grasps(segs, gripper_events(case), cuts))):
                b[key].append({"predictions": [{"start_sec": s, "end_sec": e} for s, e in intervals],
                               "references": case["references"]})
        lines.append("")
        for family, b in sorted(base.items()):
            lines.append(f"{family} on these episodes: " + "; ".join(
                f"{key} F1 {score_family(items)['segment@0.75']['f1']:.3f}" for key, items in b.items()))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
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
    r.add_argument("--instruction", action="store_true",
                   help="FLORA's plain episode instruction: the task sentence alone (overrides --background)")
    r.add_argument("--background", choices=("none", "task_brief", "scene_context", "task_and_scene"),
                   default="none", help="background text in the workspace template (kinseg/background.py)")
    r.add_argument("--scenes", help="folder of <episode>.json with target_objects/other_objects/workspace")
    r.add_argument("--context", choices=("both", "before", "after", "none"), default="both",
                   help="which BEFORE/AFTER video blocks around TARGET are sent")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--previous", action="store_true",
                   help="add the previous interval's caption to the background, for consistent names only")
    r.add_argument("--robot-state", action="store_true",
                   help="add TARGET's gripper state and moving arms from the kinematics to the background")
    r.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    s.add_argument("--kinematic", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tempseg", help="segment.py output dir: also score TempSeg split where the operation changes")
    c = sub.add_parser("compare", help="one table for several runs on their shared episodes")
    c.add_argument("--runs", nargs="+", required=True, help="NAME=DIR caption run folders")
    c.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    c.add_argument("--kinematic", required=True)
    c.add_argument("--tempseg", help="segment.py output dir: also score TempSeg split by operation")
    c.add_argument("--out", help="write the table here")
    args = ap.parse_args(argv)
    if args.command == "run" and args.cuts == "union" and not args.tempseg:
        ap.error("--cuts union needs --tempseg")
    if args.command in ("score", "compare") and not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    return {"run": run, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
