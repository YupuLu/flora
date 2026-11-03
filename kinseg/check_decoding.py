"""Check on real windows that kinseg.decoding's greedy loop reproduces TempSeg's own generate.

For every window of one video, runs FLORA's ``generate`` and ``decoding.generate`` with no
candidate list on the same inputs and reports whether the replies are identical::

    CUDA_VISIBLE_DEVICES=7 python kinseg/check_decoding.py --video galaxea_002.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from segment import PROFILES  # noqa: E402  (also puts tempseg/ on sys.path)

from batch_utils.api import DataEnginePipeline  # noqa: E402
from batch_utils.config import PROFILE_POOLS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="yam")
    args = ap.parse_args(argv)

    import decoding
    import local_models
    from batch_utils import pipeline
    from batch_utils.models import Session

    local_models.configure()
    session = Session()
    loaded = session.load("tempseg")
    session.load = lambda kind: loaded
    flora_generate = pipeline.generate
    results = []

    def both(model, proc, inputs, budget):
        reference = flora_generate(model, proc, inputs, budget)
        ours = decoding.generate(model, proc.tokenizer, inputs, budget)
        results.append(reference == ours)
        print("same" if reference == ours else f"DIFFERENT\n  flora: {reference}\n  ours:  {ours}", flush=True)
        return reference

    pipeline.generate = both
    videos, canonical = DataEnginePipeline()._inputs(args.video, PROFILES[args.profile], None, None, False, "")
    pipeline.segment(videos, PROFILE_POOLS[canonical], session, "", lambda *a, **k: None)
    print(f"{sum(results)} of {len(results)} windows identical")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
