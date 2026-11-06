"""Where the reference cuts lie in our phases, and what placing them on phase boundaries costs.

For every internal reference cut: the phase boundary just before and just after it. With perfect
grouping, every reference cut is moved to the nearest boundary, to the one before it, or to the one
after it, and the result is scored with the WGO protocol; this bounds what any method that cuts only on
our phase boundaries can reach.

    python kinseg/cut_ceiling.py --autolabel <checkout> --kinematic <run>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    ap.add_argument("--kinematic", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.autolabel)))
    from modules.exp_wgo_bench.score import score_family

    dist = collections.defaultdict(list)
    cases = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(Path(args.kinematic).glob("predictions/*/*.json")):
        case = json.loads(path.read_text())
        fam = case["family"]
        refs = [(r["start_sec"], r["end_sec"]) for r in case["references"]]
        bounds = sorted({p["start_sec"] for p in case["predictions"]} | {case["predictions"][-1]["end_sec"]})
        moved = {"nearest": [], "before": [], "after": []}
        for r in case["references"][:-1]:
            c = r["end_sec"]
            before = max((b for b in bounds if b <= c), default=bounds[0])
            after = min((b for b in bounds if b >= c), default=bounds[-1])
            dist[fam].append((c - before, after - c, after - before))
            moved["nearest"].append(before if c - before <= after - c else after)
            moved["before"].append(before)
            moved["after"].append(after)
        for key, cuts in moved.items():
            edges = [refs[0][0]] + cuts + [refs[-1][1]]
            cases[fam][key].append({"predictions": [{"start_sec": a, "end_sec": b} for a, b in zip(edges[:-1], edges[1:])],
                                    "references": case["references"]})
    for fam, d in sorted(dist.items()):
        near = [min(a, b) for a, b, _ in d]
        print(f"{fam}: {len(d)} reference cuts; phase around a cut: median {statistics.median([w for _, _, w in d]):.2f} s; "
              f"nearest phase boundary: median {statistics.median(near):.2f} s, within 0.25 s {sum(x <= 0.25 for x in near)}, "
              f"within 0.5 s {sum(x <= 0.5 for x in near)}, over 1 s {sum(x > 1 for x in near)}")
        for key in ("nearest", "before", "after"):
            s = score_family(cases[fam][key])
            print(f"  perfect grouping, every cut on the {key} phase boundary: F1@0.75 {s['segment@0.75']['f1']:.3f}, "
                  f"F1@0.5 {s['segment@0.5']['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
