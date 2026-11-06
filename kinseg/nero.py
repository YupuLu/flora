"""Kinematic phases of Nero Dog episodes (our collected data), in the format of autolabel's WGO predictions.

Nero Dog episodes are LeRobot v2.1 at 15 Hz with a 33-value ``observation.state``: per arm the TCP pose
in base_link (xyz in metres, then SciPy ``xyz`` Euler angles) and the gripper (left 0-6, right 7-13),
the joints and the gripper again (left 14-21, right 22-29), and the base velocity ``[v_x, v_y, yaw rate]``
(30-32). Segmentation uses the TCP positions, the joints, the gripper opening (``nero_dog.toml``; brief
glitches to fully open removed, :func:`despike`) and the base pose integrated from the velocity; there is
no torso. ``action[6]`` / ``action[13]`` (gripper
targets) become autolabel's command convention (-1 close, +1 open), used only as diagnostics.

Usage (an environment with pandas; ``--autolabel`` or ``$AUTOLABEL_ROOT`` is an autolabel checkout)::

    python kinseg/nero.py --jobs <selection.json> --source <local copy root> --out <run dir> [--speed-mode 1]

writes ``<run dir>/predictions/<group>/<id>.json`` (as ``modules/exp_wgo_bench/run.py``, without
references) and prints phases and gripper events per group.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

FPS = 15.0
SETTINGS = Path(__file__).with_name("nero_dog.toml")
ARM = {"left": dict(position=slice(0, 3), gripper=6, joints=slice(14, 21)),
       "right": dict(position=slice(7, 10), gripper=13, joints=slice(22, 29))}
BASE_VELOCITY = slice(30, 33)


def despike(opening: np.ndarray, max_frames: int = 3, level: float = 0.99, drop: float = 0.4) -> np.ndarray:
    """``opening`` with brief jumps to fully open marked missing (NaN).

    The water and bag robots' gripper state sometimes reads 1.00 for 1-3 frames between closed or holding
    values (0.18 -> 1.00 x3 -> 0.18) while a real opening takes about 0.4 s; such a run, both of whose
    neighbours are lower by more than ``drop``, is set to NaN, which the event detector interpolates.
    """
    u = np.asarray(opening, dtype=np.float64).copy()
    high = u >= level
    t = 0
    while t < len(u):
        if not high[t]:
            t += 1
            continue
        end = t
        while end < len(u) and high[end]:
            end += 1
        if end - t <= max_frames and t > 0 and end < len(u) and max(u[t - 1], u[end]) < level - drop:
            u[t:end] = np.nan
        t = end
    return u


def load_trajectory(parquet: Path) -> Dict[str, Any]:
    """autolabel's trajectory dictionary for one episode file."""
    import pandas as pd
    from autolabel.datasets.galaxea import _build_base_trajectory_from_action

    df = pd.read_parquet(parquet)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    T = len(state)
    traj: Dict[str, Any] = {"timestamps": df["timestamp"].to_numpy(dtype=np.float64), "torso_trajectory": None,
                            "base_trajectory": _build_base_trajectory_from_action(state[:, BASE_VELOCITY], FPS)}
    for side, idx in ARM.items():
        opening = despike(state[:, idx["gripper"]])
        traj[f"trajectory_{side}"] = np.column_stack(
            [state[:, idx["position"]], np.full((T, 4), np.nan), opening, opening])
        traj[f"qpos_{side}"] = state[:, idx["joints"]]
        traj[f"action_gripper_{side}"] = np.where(action[:, idx["gripper"]] > 0.5, 1.0, -1.0)
    return traj


def settings(speed_mode=None):
    """``speed_mode`` and ``segmentation_overrides`` of ``nero_dog.toml``, as autolabel's dataset_config reads them."""
    from autolabel.settings import load
    s = load(SETTINGS)
    return SimpleNamespace(speed_mode=s.speed_mode if speed_mode is None else speed_mode,
                           segmentation_overrides=s.segmentation_overrides)


def predict(job: Dict[str, Any], source: Path, config) -> Dict[str, Any]:
    from autolabel.segmentation.pipeline import segment_episode

    dataset = source / job["dataset"].lstrip("/")
    traj = load_trajectory(dataset / "data" / "chunk-000" / f"episode_{job['episode_index']:06d}.parquet")
    result = segment_episode(traj, fps=FPS, config=config)
    T = len(traj["timestamps"])
    return {
        "id": job["id"], "family": job["group"], "speed_mode": config.speed_mode,
        "close_containment": config.close_containment, "instruction": job["task"], "fps": FPS,
        "duration_sec": T / FPS, "dataset": job["dataset"], "episode_index": job["episode_index"],
        "despiked_frames": {side: int(np.isnan(traj[f"trajectory_{side}"][:, 7]).sum()) for side in ARM},
        "predictions": [{
            "start_sec": iv.st_duration[0] / FPS, "end_sec": iv.st_duration[1] / FPS,
            "arm": iv.arm, "base": iv.base, "torso": iv.torso,
            "gripper_events": [{"arm": ev.arm, "kind": ev.kind, "start_sec": ev.start / FPS,
                                "end_sec": ev.end / FPS, "source": ev.source} for ev in iv.gripper_events],
        } for iv in result.intervals],
        "references": [],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, help="selection.json of the FLORA run (its 'jobs')")
    ap.add_argument("--source", required=True, help="local copy: <source>/<dataset path without leading />")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed-mode", type=int, help="over nero_dog.toml's")
    ap.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    args = ap.parse_args(argv)
    if not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    sys.path.insert(0, args.autolabel)
    from autolabel.segmentation.pipeline import dataset_config

    config = dataset_config(settings(args.speed_mode), FPS)
    jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    stats = defaultdict(lambda: defaultdict(list))
    for job in jobs:
        record = predict(job, Path(args.source), config)
        folder = Path(args.out) / "predictions" / job["group"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{job['id']}.json").write_text(json.dumps(record, indent=1) + "\n")
        g = stats[job["group"]]
        phases = record["predictions"]
        g["phases"].append(len(phases))
        g["per_min"].append(60 * len(phases) / record["duration_sec"])
        g["length"] += [p["end_sec"] - p["start_sec"] for p in phases]
        g["base"].append(sum(p["base"] != "STABLE" for p in phases))
        events = [e for p in phases for e in p["gripper_events"]]
        g["close"].append(sum(e["kind"] == "close" for e in events))
        g["open"].append(sum(e["kind"] == "open" for e in events))
        g["despiked"].append(sum(record["despiked_frames"].values()))
    print(f"speed mode {config.speed_mode}; per episode (median, range); phase length median s")
    print(f"{'group':12s} {'phases':>12s} {'per min':>8s} {'length':>7s} {'base':>7s} {'closes':>7s} {'opens':>7s} {'despiked':>9s}")
    for group, g in sorted(stats.items()):
        span = lambda v: f"{statistics.median(v):g} ({min(v)}-{max(v)})"
        print(f"{group:12s} {span(g['phases']):>12s} {statistics.median(g['per_min']):8.1f} "
              f"{statistics.median(g['length']):7.2f} {statistics.median(g['base']):7g} "
              f"{statistics.median(g['close']):7g} {statistics.median(g['open']):7g} {sum(g['despiked']):9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
