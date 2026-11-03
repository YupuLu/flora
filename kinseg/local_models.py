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


def configure():
    """Apply the local-weights module named by ``FLORA_LOCAL_MODELS``, if any; returns its result."""
    path = os.environ.get("FLORA_LOCAL_MODELS")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("flora_local_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_local_models()
