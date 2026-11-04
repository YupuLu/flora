"""Optionally read the model weights from local folders instead of the Hugging Face cache.

When ``FLORA_LOCAL_MODELS`` names a module file with a
``configure_local_models()`` function (on the remote server,
``workspace/flora_local.py``), that function is called before any model is
loaded. It points FLORA's ``snapshot`` and ``huggingface_hub.snapshot_download``
at the local weight folders, offline, and keeps FLORA's integrity checks.
Without the variable nothing changes.
"""

from __future__ import annotations

import importlib.util
import os
import time

# FLORA's manifest check compares each file's size and times before and after hashing it. On the
# remote's shared storage, the first read of an idle file rewrites its timestamps (the content stays
# the same), so a load can fail once with this message and succeed once the files are warm.
WARMING_MESSAGE = "changed during its integrity check"


def configure():
    """Apply the local-weights module named by ``FLORA_LOCAL_MODELS``, if any; returns its result."""
    path = os.environ.get("FLORA_LOCAL_MODELS")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("flora_local_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_local_models()


def load(session, kind: str, attempts: int = 3, wait_s: float = 60.0):
    """``session.load(kind)``, retried only when files changed during FLORA's integrity check."""
    for attempt in range(1, attempts + 1):
        try:
            return session.load(kind)
        except RuntimeError as exc:
            if WARMING_MESSAGE not in str(exc) or attempt == attempts:
                raise
            print(f"{kind}: model files changed while being verified (attempt {attempt}); retrying in "
                  f"{wait_s:g} s", flush=True)
            time.sleep(wait_s)
