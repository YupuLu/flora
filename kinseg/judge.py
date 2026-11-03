"""Ask a vision-language model whether two clips around a cut show the same operation.

Candidate cuts of an episode are the kinematic phase boundaries (an autolabel
``exp_wgo_bench`` run) together with TempSeg's cuts (``segment.py`` output).
For each cut ``c`` the model (Qwen3.5-9B base; TempSeg's adapter is not
loaded) sees clip A = ``[c - 3 s, c)`` and clip B = ``[c, c + 3 s)`` from the
main camera at 4 Hz, with the granularity of TempSeg's own prompt, asked in
one of three ways (``--variant``): ``pair``, whether the two clips show the
same operation; ``single``, one clip across the cut, whether a new operation
starts at the cut; ``describe``, name both operations, then answer Same or
New. The score of the cut is the probability of the "new operation" answer
against the other one, read at the answer token: high means a new operation
starts at ``c``.

    CUDA_VISIBLE_DEVICES=7 python kinseg/judge.py run --kinematic <run> --tempseg <dir> \\
        --episodes galaxea_002 robointer_droid_000001 --out <dir>
    python kinseg/judge.py score --autolabel <checkout> --kinematic <run> --out <dir>

``score`` labels each candidate (the nearest one within 0.5 s of a reference
cut is positive), reports the ROC AUC, and scores phases joined across cuts
below a range of thresholds with the WGO protocol, next to TempSeg alone and
the perfect-judge ceiling on the same candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CLIP_S = 3.0
SAMPLE_HZ = 4
MAX_PIXELS = 160_000  # per frame, before the processor
DEDUP_S = 0.05
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

POLICY = """An operation is one meaningful step such as picking up an object, placing it, opening or pushing something.
Acquisition and delivery are distinct operations; retain distinct directional handovers. Ordinary contacts,
adjustments and low-motion pauses within the same ongoing goal do not start a new operation."""

PAIR = """You see two consecutive clips of a robot manipulation video.{task}
Clip A (the {a:.1f} s just before a moment in the video):
<video>
Clip B (the {b:.1f} s just after that moment):
<video>
Is the robot performing the same operation in clip A and clip B, or does a new operation start between them?
""" + POLICY + """
Answer with exactly one word: Yes if it is the same operation, No if a new operation starts."""

SINGLE = """You see a {d:.1f} s clip of a robot manipulation video.{task}
<video>
Does the robot start a new operation at {m:.1f} seconds into this clip, or does the same operation continue?
""" + POLICY + """
Answer with exactly one word: Yes if a new operation starts at {m:.1f} seconds, No if the same operation continues."""

DESCRIBE = """You see two consecutive clips of a robot manipulation video.{task}
Clip A (the {a:.1f} s just before a moment in the video):
<video>
Clip B (the {b:.1f} s just after that moment):
<video>
""" + POLICY + """
First name the robot's operation in clip A, then in clip B, each in at most eight words; then decide whether
they are the same operation or a new operation starts in clip B. Use exactly this format:
A: <operation>
B: <operation>
Answer: Same or New"""

VARIANTS = ("pair", "single", "describe")


# ── candidates and frames ──────────────────────────────────────────────


def candidate_cuts(case: Dict[str, object], tempseg: Dict[str, object]) -> List[Dict[str, object]]:
    """Internal cuts of the phases and of TempSeg's segments, merged when closer than DEDUP_S."""
    phases = [(p["start_sec"], p["end_sec"]) for p in case["predictions"]]
    start, end = phases[0][0], phases[-1][1]
    tagged = [(e, "kinematic") for _, e in phases[:-1]]
    tagged += [(t, "tempseg") for s in tempseg["segments"] for t in (s["start_s"], s["end_s"]) if start < t < end]
    cuts: List[Dict[str, object]] = []
    for t, source in sorted(tagged):
        if cuts and t - cuts[-1]["t"] < DEDUP_S:
            if source not in cuts[-1]["sources"]:
                cuts[-1]["sources"].append(source)
            continue
        cuts.append({"t": round(t, 4), "sources": [source]})
    return cuts


def decode_frames(path: str, times: Sequence[float], max_pixels: int = MAX_PIXELS):
    """RGB frames (PIL) nearest to ``times`` (seconds from the first frame)."""
    import av

    wanted = sorted(set(times))
    frames, stamps = [], []
    with av.open(path) as container:
        stream = container.streams.video[0]
        zero = None
        for frame in container.decode(stream):
            zero = frame.pts if zero is None else zero
            stamps.append(float((frame.pts - zero) * stream.time_base))
            frames.append(frame)
            if stamps[-1] > wanted[-1] + 1.0:
                break
        picked = {}
        for t in wanted:
            k = min(range(len(stamps)), key=lambda i: abs(stamps[i] - t))
            image = frames[k].to_image().convert("RGB")
            scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
            if scale < 1.0:
                image = image.resize((max(32, int(image.width * scale)), max(32, int(image.height * scale))))
            picked[t] = image
    return picked


def _pairs(times: List[float], near_end: bool) -> List[float]:
    """An even number of frames (Qwen merges frames in temporal pairs), dropping the one farthest from the cut."""
    if len(times) < 2:
        return times * 2
    if len(times) % 2:
        times = times[1:] if near_end else times[:-1]
    return times


def clip_times(cut: float, duration: float) -> Tuple[List[float], List[float]]:
    """Sample times of clip A = [cut - CLIP_S, cut) and clip B = [cut, cut + CLIP_S), inside the video."""
    step, n = 1.0 / SAMPLE_HZ, int(CLIP_S * SAMPLE_HZ)
    before = [t for t in (cut - CLIP_S + i * step for i in range(n)) if t >= 0] or [0.0]
    after = [t for t in (cut + i * step for i in range(n)) if t < duration] or [max(0.0, duration - step)]
    return _pairs(before, near_end=True), _pairs(after, near_end=False)


# ── model ──────────────────────────────────────────────────────────────


def load_model():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    path = snapshot_download(BASE, revision=BASE_REVISION, allow_patterns=["*.safetensors", "*.json", "*.jinja"])
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda:0").eval()
    return model, AutoProcessor.from_pretrained(path)


def _inputs(proc, prompt: str, clips, answer_prefix: str = ""):
    import torch
    from transformers.video_utils import VideoMetadata

    text = proc.apply_chat_template([dict(role="user", content=prompt.replace("<video>", proc.video_token))],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    meta = [VideoMetadata(total_num_frames=len(c), fps=SAMPLE_HZ, frames_indices=list(range(len(c)))) for c in clips]
    inputs = proc(text=[text + answer_prefix], videos=list(clips), video_metadata=meta, do_sample_frames=False,
                  return_tensors="pt")
    return {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in inputs.items() if k != "video_metadata"}


def _two_way(model, inputs, tokenizer, positive: str, negative: str) -> Dict[str, float]:
    """P(positive) between two one-token answers at the next position, and their total mass."""
    import torch

    with torch.inference_mode():
        logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
    ids = []
    for word in (positive, negative):
        tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
        assert len(tokens) == 1, word
        ids.append(tokens[0])
    pair = torch.softmax(logits[ids], dim=-1)
    return {"p": float(pair[0]), "mass": float(torch.softmax(logits, dim=-1)[ids].sum())}


def ask(model, proc, clip_a, clip_b, variant: str = "pair", instruction: str = "") -> Dict[str, object]:
    """Probability that a new operation starts at the cut between ``clip_a`` and ``clip_b``."""
    import torch

    a, b = len(clip_a) / SAMPLE_HZ, len(clip_b) / SAMPLE_HZ
    task = f"\nThe robot's task in this episode: {instruction.strip()}" if instruction.strip() else ""
    if variant == "pair":
        inputs = _inputs(proc, PAIR.format(a=a, b=b, task=task), [clip_a, clip_b])
        res = _two_way(model, inputs, proc.tokenizer, "No", "Yes")
    elif variant == "single":
        inputs = _inputs(proc, SINGLE.format(d=a + b, m=a, task=task), [list(clip_a) + list(clip_b)])
        res = _two_way(model, inputs, proc.tokenizer, "Yes", "No")
    else:
        inputs = _inputs(proc, DESCRIBE.format(a=a, b=b, task=task), [clip_a, clip_b])
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        reply = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        head = reply.split("Answer:")[0].rstrip() + "\nAnswer:"
        res = _two_way(model, _inputs(proc, DESCRIBE.format(a=a, b=b, task=task), [clip_a, clip_b], head),
                       proc.tokenizer, " New", " Same")
        res["reply"] = reply.strip()
    out = {"p_new": round(res.pop("p"), 4), "answer_mass": round(res.pop("mass"), 4),
           "tokens": int(inputs["input_ids"].shape[1])}
    out.update(res)
    return out


def run(args) -> int:
    import local_models

    kin = {json.loads(p.read_text())["id"]: p for p in Path(args.kinematic).glob("predictions/*/*.json")}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_models.configure()
    model, proc = load_model()
    for episode in args.episodes:
        out = out_dir / f"{episode}.json"
        if out.exists() and not args.overwrite:
            print(f"skip {episode}")
            continue
        case = json.loads(kin[episode].read_text())
        tempseg = json.loads((Path(args.tempseg) / f"{episode}.json").read_text())
        cuts = candidate_cuts(case, tempseg)
        duration = tempseg["duration_s"]
        plans = [clip_times(c["t"], duration) for c in cuts]
        video = str(Path(args.videos) / f"{episode}.mp4") if args.videos else tempseg["cameras"]["main"]
        frames = decode_frames(video, [t for a, b in plans for t in a + b], args.max_pixels)
        instruction = case.get("instruction", "") if args.instruction else ""
        started = time.monotonic()
        for cut, (a, b) in zip(cuts, plans):
            cut.update(ask(model, proc, [frames[t] for t in a], [frames[t] for t in b], args.variant, instruction))
        record = dict(id=episode, family=case["family"], model=f"{BASE}@{BASE_REVISION}", variant=args.variant,
                      instruction=instruction, max_pixels=args.max_pixels,
                      clip_s=CLIP_S,
                      sample_hz=SAMPLE_HZ, cuts=cuts, elapsed_s=round(time.monotonic() - started, 1))
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"{episode}: {len(cuts)} cuts, {record['elapsed_s']} s", flush=True)
    return 0


# ── evaluation ─────────────────────────────────────────────────────────


def score(args) -> int:
    from beliefs import auc, labels, merge_phases
    from merge import scorer

    score_family = scorer(Path(args.autolabel))
    kin = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
           for p in Path(args.kinematic).glob("predictions/*/*.json")}
    fams: Dict[str, Dict[str, list]] = {}
    for path in sorted(Path(args.out).glob("*.json")):
        rec = json.loads(path.read_text())
        case = kin[rec["id"]]
        refs = [(r["start_sec"], r["end_sec"]) for r in case["references"]]
        start, end = case["predictions"][0]["start_sec"], case["predictions"][-1]["end_sec"]
        ts = [c["t"] for c in rec["cuts"]]
        tiles = list(zip([start] + ts, ts + [end]))
        y = labels(ts, refs)
        fam = fams.setdefault(rec["family"], {"p": [], "y": [], "src": [], "cases": []})
        fam["p"] += [c["p_new"] for c in rec["cuts"]]
        fam["y"] += y
        fam["src"] += [c["sources"] for c in rec["cuts"]]
        fam["cases"].append(dict(tiles=tiles, cuts=rec["cuts"], y=y, references=case["references"]))
    lines = []
    for family, fam in fams.items():
        kin_only = [(p, v) for p, v, s in zip(fam["p"], fam["y"], fam["src"]) if "kinematic" in s]
        lines.append(f"{family}: {len(fam['cases'])} episodes, {len(fam['y'])} cuts, {sum(fam['y'])} positive; "
                     f"AUC {auc(fam['p'], fam['y']):.3f} (kinematic cuts only "
                     f"{auc([p for p, _ in kin_only], [v for _, v in kin_only]):.3f})")

        def f1(select) -> str:
            items = []
            for c in fam["cases"]:
                keep = [cut["t"] for cut, y in zip(c["cuts"], c["y"]) if select(cut, y)]
                merged = merge_phases(c["tiles"], keep)
                items.append({"predictions": [{"start_sec": s, "end_sec": e} for s, e in merged],
                              "references": c["references"]})
            s = score_family(items)["segment@0.75"]
            return f"{s['f1']:.3f} ({s['n_pred']}/{s['n_ref']})"

        lines.append(f"  TempSeg cuts only: F1@0.75 {f1(lambda cut, y: 'tempseg' in cut['sources'])}")
        lines.append(f"  perfect judge on these candidates: {f1(lambda cut, y: y == 1)}")
        for th in THRESHOLDS:
            lines.append(f"  judge, keep p_new >= {th}: {f1(lambda cut, y, th=th: cut['p_new'] >= th)}")
    text = "\n".join(lines)
    print(text)
    (Path(args.out) / "score.md").write_text(text + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kinematic", required=True)
    r.add_argument("--tempseg", required=True, help="segment.py output dir (free run)")
    r.add_argument("--videos", help="folder of <episode>.mp4 (default: the path recorded by segment.py)")
    r.add_argument("--instruction", action="store_true", help="give the episode's task instruction in the prompt")
    r.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help="pixels per frame before the processor")
    r.add_argument("--episodes", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--variant", choices=VARIANTS, default="pair",
                   help="pair: two clips, same operation?; single: one clip across the cut, new operation at "
                        "the cut?; describe: name both operations, then Same or New")
    s = sub.add_parser("score")
    s.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    s.add_argument("--kinematic", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "score" and not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    return run(args) if args.command == "run" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
