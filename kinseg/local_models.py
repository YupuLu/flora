"""Load FLORA models reliably: local weight folders, warmed files, free GPU memory.

When ``FLORA_LOCAL_MODELS`` names a module file with a
``configure_local_models()`` function (on the remote server,
``workspace/flora_local.py``), :func:`configure` calls it before any model is
loaded. It points FLORA's ``snapshot`` and ``huggingface_hub.snapshot_download``
at the local weight folders, offline, and keeps FLORA's integrity checks.
Without the variable nothing changes.

:func:`load` wraps ``session.load(kind)`` against the two ways remote runs
failed:

* **Warming.** FLORA's manifest check records each file's size and
  modification and change times, hashes the files, and refuses to load if any
  record changed meanwhile. On the remote's shared storage the first read of an
  idle file rewrites its times (the content stays the same; probably tiered
  storage bringing the data back). So, before FLORA's check, every file it
  checks is read once and the times are watched until they stay unchanged for
  ``SETTLE_S``. The rare failure that still slips through is retried; a wrong
  content hash is never retried.
* **Shared GPUs.** Other users can hold memory on the same GPUs. Before
  loading, :func:`wait_for_memory` waits (up to ``MEMORY_TIMEOUT_S``) until the
  visible GPUs have the memory the model needs, instead of failing half-way.
"""

from __future__ import annotations

import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WARMING_MESSAGE = "changed during its integrity check"
SETTLE_S = 30.0
SETTLE_ROUNDS = 10
READ_CHUNK = 64 * 1024 * 1024
# GPU memory (GiB) a model needs: TempSeg (and the 9B judge) on the first visible GPU; the
# Captioner (27B, bf16) spread over all visible GPUs.
NEED_GIB = {"tempseg": 22.0, "judge": 22.0, "captioner": 62.0}
SPREAD = {"captioner"}
MEMORY_TIMEOUT_S = 3600.0
MEMORY_POLL_S = 60.0

_PATHS: Dict[str, Path] = {}


def configure():
    """Apply the local-weights module named by ``FLORA_LOCAL_MODELS``, if any; returns its result."""
    path = os.environ.get("FLORA_LOCAL_MODELS")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("flora_local_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = module.configure_local_models()
    if isinstance(paths, dict):
        _PATHS.update({k: Path(v) for k, v in paths.items()})
    return paths


# ── warming ────────────────────────────────────────────────────────────


def checked_files(model_dir: Path) -> List[Path]:
    """The files FLORA's manifest check covers (SOURCE_MANIFEST.sha256), else every file in the folder."""
    manifest = model_dir / "SOURCE_MANIFEST.sha256"
    if manifest.is_file():
        names = [line.split(maxsplit=1)[1].lstrip("*") for line in manifest.read_text().splitlines() if line.strip()]
        return [model_dir / name for name in names] + [manifest]
    return sorted(p for p in model_dir.rglob("*") if p.is_file())


def fingerprints(files: List[Path]) -> List[Tuple[int, int, int]]:
    out = []
    for f in files:
        st = f.resolve().stat()
        out.append((st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    return out


def _read(path: Path) -> int:
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(READ_CHUNK):
            total += len(chunk)
    return total


def warm(model_dir: Path, settle_s: float = SETTLE_S, rounds: int = SETTLE_ROUNDS, threads: int = 8) -> None:
    """Read every checked file once, then wait until their sizes and times stay unchanged for ``settle_s``."""
    files = checked_files(model_dir)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        size = sum(pool.map(_read, files))
    print(f"warm {model_dir.name}: read {len(files)} files, {size / 2**30:.1f} GiB in "
          f"{time.monotonic() - started:.0f} s", flush=True)
    before = fingerprints(files)
    for _ in range(rounds):
        time.sleep(settle_s)
        after = fingerprints(files)
        if after == before:
            return
        changed = sum(a != b for a, b in zip(after, before))
        print(f"warm {model_dir.name}: {changed} files still changing; waiting", flush=True)
        before = after
    print(f"warm {model_dir.name}: files did not settle; loading anyway (the load is retried)", flush=True)


# ── GPU memory ─────────────────────────────────────────────────────────


def free_gib() -> List[float]:
    import torch

    return [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())]


def enough_memory(kind: str, free: List[float]) -> bool:
    need = NEED_GIB.get(kind, 0.0)
    if not free:
        return False
    if kind in SPREAD:
        return sum(free) >= need and min(free) >= need / len(free)
    return free[0] >= need


def wait_for_memory(kind: str, timeout_s: float = MEMORY_TIMEOUT_S, poll_s: float = MEMORY_POLL_S) -> None:
    """Wait until the visible GPUs have the memory ``kind`` needs; raise after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        free = free_gib()
        if enough_memory(kind, free):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{kind}: not enough free GPU memory after {timeout_s / 60:.0f} min "
                               f"(free GiB per GPU: {[round(f, 1) for f in free]}, need {NEED_GIB.get(kind)})")
        print(f"{kind}: waiting for GPU memory (free GiB per GPU: {[round(f, 1) for f in free]}, "
              f"need {NEED_GIB.get(kind)})", flush=True)
        time.sleep(poll_s)


# ── loading ────────────────────────────────────────────────────────────


def load(session, kind: str, attempts: int = 3, wait_s: float = 60.0, model_dir: Optional[Path] = None):
    """``session.load(kind)`` after warming its files and waiting for GPU memory.

    Only the timestamp failure of FLORA's integrity check is retried.
    """
    model_dir = model_dir or _PATHS.get(kind)
    if model_dir is not None:
        warm(Path(model_dir))
    wait_for_memory(kind)
    for attempt in range(1, attempts + 1):
        try:
            return session.load(kind)
        except RuntimeError as exc:
            if WARMING_MESSAGE not in str(exc) or attempt == attempts:
                raise
            print(f"{kind}: model files changed while being verified (attempt {attempt}); retrying in "
                  f"{wait_s:g} s", flush=True)
            time.sleep(wait_s)
            if model_dir is not None:
                warm(Path(model_dir))
