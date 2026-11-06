"""Where the reference cuts lie in our phases, and what placing them on phase boundaries costs.

For every internal reference cut: the phase boundary just before and just after it. With perfect
grouping, every reference cut is moved to the nearest boundary, to the one before it, or to the one
after it, and the result is scored with the WGO protocol; this bounds what any method that cuts only on
our phase boundaries can reach.

    python kinseg/cut_ceiling.py --autolabel <checkout> --kinematic <run> [--examples EPISODE:CUT ...]

``--examples`` draws, for chosen reference cuts (CUT counts from 0), a time line of the two subtasks
around the cut, our phase boundaries, and the two placements on the boundaries before and after it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path


WIDTH = 72


def tiou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if inter > 0 and union > 0 else 0.0


def timeline(case, j: int) -> str:
    """Text drawing of reference cut ``j``: subtasks, phase boundaries, and both placements."""
    refs = case["references"]
    r0, r1 = (refs[j]["start_sec"], refs[j]["end_sec"]), (refs[j + 1]["start_sec"], refs[j + 1]["end_sec"])
    lo, hi = r0[0], r1[1]
    scale = (hi - lo) / WIDTH
    col = lambda t: min(WIDTH, max(0, round((t - lo) / scale)))  # noqa: E731
    bounds = sorted({p["start_sec"] for p in case["predictions"]} | {case["predictions"][-1]["end_sec"]})
    c = r0[1]
    before = max(b for b in bounds if b <= c)
    after = min(b for b in bounds if b >= c)

    def bars(a, b, labels):
        line = [" "] * (WIDTH + 1)
        for idx, ((x, y), text) in enumerate(zip(((a[0], a[1]), (b[0], b[1])), labels)):
            x0, x1 = col(x), col(y)
            line[x0] = "[" if idx == 0 else "|"
            for k in range(x0 + 1, x1):
                line[k] = "="
            line[x1] = "]" if idx == 1 else "|"
            text = f" {text} "
            if x1 - x0 - 1 >= len(text):
                start = x0 + 1 + (x1 - x0 - 1 - len(text)) // 2
                line[start:start + len(text)] = list(text)
        return "".join(line)

    marks = [" "] * (WIDTH + 1)
    for b in bounds:
        if lo <= b <= hi:
            marks[col(b)] = "|"
    span = [" "] * (WIDTH + 1)
    for k in range(col(before), col(after) + 1):
        span[k] = "-"
    span[col(before)], span[col(after)], span[col(c)] = "<", ">", "^"
    name0, name1 = refs[j]["subtask"], refs[j + 1]["subtask"]
    rows = [
        f"{case['id']}, reference cut {j} at {c:.2f} s ({scale:.2f} s per character, {lo:.2f}-{hi:.2f} s)",
        "",
        f"reference      {bars(r0, r1, ['A', 'B'])}",
        f"phases         {''.join(marks)}",
        f"phase at cut   {''.join(span)}   {before:.2f}-{after:.2f} s; cut {c - before:.2f} s after its start",
        f"cut on before  {bars((r0[0], before), (before, r1[1]), ['A', 'B'])}   "
        f"tIoU A {tiou((r0[0], before), r0):.2f}, B {tiou((before, r1[1]), r1):.2f}",
        f"cut on after   {bars((r0[0], after), (after, r1[1]), ['A', 'B'])}   "
        f"tIoU A {tiou((r0[0], after), r0):.2f}, B {tiou((after, r1[1]), r1):.2f}",
        "",
        f"A = {name0} ({r0[0]:.2f}-{r0[1]:.2f} s); B = {name1} ({r1[0]:.2f}-{r1[1]:.2f} s); a match needs tIoU 0.75",
    ]
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    ap.add_argument("--kinematic", required=True)
    ap.add_argument("--examples", nargs="+", default=[], help="EPISODE:CUT reference cuts to draw")
    args = ap.parse_args(argv)
    if args.examples:
        kin = {}
        for path in Path(args.kinematic).glob("predictions/*/*.json"):
            case = json.loads(path.read_text())
            kin[case["id"]] = case
        for item in args.examples:
            episode, j = item.rsplit(":", 1)
            print(timeline(kin[episode], int(j)) + "\n")
        return 0
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
