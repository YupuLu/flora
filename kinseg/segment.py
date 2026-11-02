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
constrained: every ``s`` and ``e`` the model writes is chosen whole, as the
most probable of the local ticks of the episode's kinematic phase boundaries
and the window edges (``kinseg/decoding.py``). The model still decides which
segments exist, so it merges the kinematic phases rather than placing cuts of
its own; the windowing, prompts and stitching are unchanged. Each window's
audit keeps every scored candidate with its log-probability.

With ``--probe``, decoding stays free (replies identical to TempSeg's) and, at
each ``s``/``e``, the log-probability of every value on TempSeg's 0.5 s grid
in the window is saved under ``probes``: where the model believes that start
or end could lie. ``beliefs.py`` turns these into scores for kinematic cuts.

Usage (one GPU)::

    CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --jobs jobs.jsonl --output-dir out/
    CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --video a.mp4 --profile yam --output-dir out/
    CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --jobs jobs.jsonl --kinematic <run> --output-dir out/

A job row is ``{"id", "video", "profile", "left_wrist"?, "right_wrist"?, "instruction"?}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TEMPSEG = Path(__file__).resolve().parents[1] / "tempseg"
sys.path.insert(0, str(TEMPSEG))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def candidates(ticks: list[int], key: str, row_starts: list[int], valid_end: int) -> list[int]:
    """Allowed values for the next number: a start not before the previous row's, an end after its start."""
    if key == "s":
        floor = row_starts[-1] if row_starts else ticks[0]
        values = [k for k in ticks if floor <= k < valid_end]
    else:
        values = [k for k in ticks if k > row_starts[-1]]
    return values or [valid_end]


def grid_ticks(window, step: int = 10) -> list[int]:
    """TempSeg's output grid in a window (multiples of ``step`` ticks, 0.5 s) plus the window edges."""
    lo, hi = window.valid_start_tick, window.valid_end_tick
    first = -(-lo // step) * step
    return sorted(set(range(first, hi + 1, step)) | {lo, hi, window.core_start_tick, window.core_end_tick})


def probe_decoding(pipeline) -> list[dict]:
    """Decode freely with kinseg.decoding, recording grid log-probabilities at each s/e; returns the audit."""
    import decoding

    audit: list[dict] = []
    current: dict = {}
    temporal_inputs = pipeline.temporal_inputs

    def inputs_with_window(frames, window, *args, **kwargs):
        current["window"] = window
        return temporal_inputs(frames, window, *args, **kwargs)

    def probed_generate(model, proc, inputs, budget):
        window = current["window"]
        grid = grid_ticks(window)
        entry = dict(scope_id=window.scope_id, view_start_us=window.requested_view_start_us,
                     envelope=[window.valid_start_tick, window.valid_end_tick,
                               window.core_start_tick, window.core_end_tick], beliefs=[])
        audit.append(entry)
        entry["output"] = decoding.generate(model, proc.tokenizer, inputs, budget,
                                            probe=lambda key: grid, record=entry["beliefs"])
        return entry["output"]

    pipeline.temporal_inputs, pipeline.generate = inputs_with_window, probed_generate
    return audit


def constrain_decoding(pipeline, job: dict) -> list[dict]:
    """Decode each window with kinseg.decoding, choosing s/e among ``job["cuts"]``; returns the audit list."""
    import decoding

    audit: list[dict] = []
    current: dict = {}
    temporal_inputs = pipeline.temporal_inputs

    def inputs_with_window(frames, window, *args, **kwargs):
        current["window"] = window
        return temporal_inputs(frames, window, *args, **kwargs)

    def constrained_generate(model, proc, inputs, budget):
        window = current["window"]
        ticks = allowed_ticks(window, job["cuts"])
        entry = dict(scope_id=window.scope_id, envelope=[window.valid_start_tick, window.valid_end_tick,
                                                         window.core_start_tick, window.core_end_tick],
                     allowed_ticks=ticks, choices=[])
        audit.append(entry)
        entry["output"] = decoding.generate(
            model, proc.tokenizer, inputs, budget,
            allowed=lambda key, starts: candidates(ticks, key, starts, window.valid_end_tick),
            record=entry["choices"])
        return entry["output"]

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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--kinematic", help="autolabel exp_wgo_bench run: constrain cuts to its phase boundaries")
    mode.add_argument("--probe", action="store_true",
                      help="decode freely and record grid log-probabilities at each s/e (as 'probes')")
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
    audit = constrain_decoding(pipeline, current) if cuts is not None else (
        probe_decoding(pipeline) if args.probe else None)
    for job in _jobs(args):
        out = out_dir / f"{job['id']}.json"
        if out.exists() and not args.overwrite:
            print(f"skip {out}")
            continue
        if cuts is not None:
            if job["id"] not in cuts:
                raise SystemExit(f"{job['id']} has no kinematic phases in {args.kinematic}")
            current["cuts"] = cuts[job["id"]]
        if audit is not None:
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
        elif args.probe:
            record.update(probes=list(audit))
        out.write_text(json.dumps(record, indent=1, default=str) + "\n")
        print(f"{job['id']}: {len(segments)} segments, {len(windows)} windows, {record['elapsed_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
