"""Flora-TempSeg segmentation only, with the per-window model outputs kept.

Runs the released TempSeg windowing, prompts and continuation stitching
(``tempseg/batch_utils``) without the Captioner, loads the model once for
all jobs, and writes one JSON file per job::

    {"id", "profile", "duration_s", "cameras", "segments": [{start_s, end_s, metadata}],
     "windows": [{window, frame_plan, prompt_sha256, output}], "elapsed_s"}

``segments`` are the stitched episode-level intervals in seconds from the
first source frame; ``windows`` keeps each window's raw JSON reply in 20 Hz
local ticks, so other merge rules can reuse them.

With ``--kinematic <run>`` (an autolabel ``exp_wgo_bench`` run), decoding is
constrained: every ``s`` and ``e`` the model writes must be the local tick of
one of the episode's kinematic phase boundaries, or a window edge (valid or
core). The model still decides which segments exist, so it merges the
kinematic phases rather than placing cuts of its own; the windowing, prompts
and stitching are unchanged.

Usage (one GPU)::

    CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --jobs jobs.jsonl --output-dir out/
    CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --video a.mp4 --profile yam --output-dir out/
    CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --jobs jobs.jsonl --kinematic <run> --output-dir out/

A job row is ``{"id", "video", "profile", "left_wrist"?, "right_wrist"?, "instruction"?}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

TEMPSEG = Path(__file__).resolve().parents[1] / "tempseg"
sys.path.insert(0, str(TEMPSEG))

from batch_utils.api import DataEnginePipeline  # noqa: E402
from batch_utils.config import PROFILE_POOLS  # noqa: E402

PROFILES = {"human": "Human", "yam": "YAM", "dex": "Dex", "tianji": "Tianji"}
TICK_US = 50_000  # one 20 Hz output tick


def kinematic_cuts(run: Path) -> dict[str, list[float]]:
    """Phase boundaries in seconds, outer edges included, per episode of an exp_wgo_bench run."""
    cuts = {}
    for path in (run / "predictions").glob("*/*.json"):
        case = json.loads(path.read_text())
        cuts[case["id"]] = sorted({t for p in case["predictions"] for t in (p["start_sec"], p["end_sec"])})
    return cuts


def allowed_ticks(window, cuts: list[float]) -> list[int]:
    """Local ticks of the cuts inside the window's valid range, plus the window edges."""
    lo, hi = window.valid_start_tick, window.valid_end_tick
    ticks = {round((t * 1e6 - window.requested_view_start_us) / TICK_US) for t in cuts}
    edges = {lo, hi, window.core_start_tick, window.core_end_tick}
    return sorted({k for k in ticks if lo <= k <= hi} | edges)


class TickConstraint:
    """Logits processor: a number after ``"s":`` or ``"e":`` must be one of ``allowed``."""

    def __init__(self, tokenizer, prompt_len: int, allowed: list[int]):
        self.tok, self.n = tokenizer, prompt_len
        self.allowed = {str(k) for k in allowed}
        self.digits = {str(d): tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)}

    def __call__(self, input_ids, scores):
        text = self.tok.decode(input_ids[0, self.n:], skip_special_tokens=True)
        match = re.search(r'"[se]":(\d*)$', text)
        if match is None:
            return scores
        prefix = match.group(1)
        following = {a[len(prefix)] for a in self.allowed if a.startswith(prefix) and len(a) > len(prefix)}
        if prefix in self.allowed:  # the number may end here; only digits that lead to another value
            blocked = [i for d, i in self.digits.items() if d not in following]
            scores[:, blocked] = float("-inf")
            return scores
        keep = [self.digits[d] for d in following]
        constrained = scores.new_full(scores.shape, float("-inf"))
        constrained[:, keep] = scores[:, keep]
        return constrained


def constrain_decoding(pipeline, job: dict) -> list[dict]:
    """Make each window's generation obey TickConstraint for ``job["cuts"]``; returns the audit list."""
    from transformers import LogitsProcessorList

    audit: list[dict] = []
    current: dict = {}
    temporal_inputs, generate = pipeline.temporal_inputs, pipeline.generate

    def inputs_with_window(frames, window, *args, **kwargs):
        current["window"] = window
        return temporal_inputs(frames, window, *args, **kwargs)

    def constrained_generate(model, proc, inputs, budget):
        window = current["window"]
        ticks = allowed_ticks(window, job["cuts"])
        entry = dict(scope_id=window.scope_id, envelope=[window.valid_start_tick, window.valid_end_tick,
                                                         window.core_start_tick, window.core_end_tick],
                     allowed_ticks=ticks)
        audit.append(entry)
        processor = TickConstraint(proc.tokenizer, inputs["input_ids"].shape[1], ticks)
        inner = model.generate
        model.generate = lambda *a, **k: inner(*a, logits_processor=LogitsProcessorList([processor]), **k)
        try:
            entry["output"] = generate(model, proc, inputs, budget)
            return entry["output"]
        finally:
            model.generate = inner

    pipeline.temporal_inputs, pipeline.generate = inputs_with_window, constrained_generate
    return audit


def _jobs(args) -> list[dict]:
    if args.video:
        return [dict(id=args.id or Path(args.video).stem, video=args.video, profile=args.profile,
                     left_wrist=args.left_wrist, right_wrist=args.right_wrist, instruction=args.instruction)]
    rows = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    base = Path(args.jobs).resolve().parent
    for row in rows:
        for key in ("video", "left_wrist", "right_wrist"):
            if row.get(key) and not Path(row[key]).is_absolute():
                row[key] = str(base / row[key])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--jobs", help="JSONL of jobs")
    ap.add_argument("--id")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="yam")
    ap.add_argument("--left-wrist")
    ap.add_argument("--right-wrist")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--kinematic", help="autolabel exp_wgo_bench run: constrain cuts to its phase boundaries")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from batch_utils import pipeline
    from batch_utils.models import Session

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = Session()
    loaded = session.load("tempseg")
    session.load = lambda kind: loaded  # reuse one model for every job
    engine = DataEnginePipeline()
    quiet = lambda *a, **k: None  # noqa: E731
    cuts = kinematic_cuts(Path(args.kinematic)) if args.kinematic else None
    current: dict = {}
    audit = constrain_decoding(pipeline, current) if cuts is not None else None
    for job in _jobs(args):
        out = out_dir / f"{job['id']}.json"
        if out.exists() and not args.overwrite:
            print(f"skip {out}")
            continue
        if cuts is not None:
            if job["id"] not in cuts:
                raise SystemExit(f"{job['id']} has no kinematic phases in {args.kinematic}")
            current["cuts"] = cuts[job["id"]]
            audit.clear()
        profile = PROFILES[job.get("profile", args.profile)]
        videos, canonical = engine._inputs(job["video"], profile, job.get("left_wrist"), job.get("right_wrist"),
                                           False, job.get("instruction", ""))
        started = time.monotonic()
        try:
            segments, windows = pipeline.segment(videos, PROFILE_POOLS[canonical], session,
                                                 job.get("instruction", ""), quiet)
        except RuntimeError as exc:  # FLORA's output errors; keep the replies and go on
            error = dict(id=job["id"], error=f"{type(exc).__name__}: {exc}", cause=repr(exc.__cause__),
                         constraints=list(audit or []))
            (out_dir / f"{job['id']}.error.json").write_text(json.dumps(error, indent=1, default=str) + "\n")
            print(f"{job['id']}: FAILED {error['error']}", flush=True)
            continue
        record = dict(id=job["id"], profile=canonical, duration_s=videos["main"].duration_s,
                      cameras={role: str(v.path) for role, v in videos.items()},
                      segments=segments, windows=windows, elapsed_s=round(time.monotonic() - started, 2))
        if cuts is not None:
            record.update(kinematic_run=str(args.kinematic), kinematic_cuts=current["cuts"], constraints=list(audit))
        out.write_text(json.dumps(record, indent=1, default=str) + "\n")
        print(f"{job['id']}: {len(segments)} segments, {len(windows)} windows, {record['elapsed_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
