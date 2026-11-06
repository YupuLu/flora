"""WGO-Bench semantic scores: label accuracy and semantic F1, with Macrodata's published LLM judge.

The judge prompt (``prompts/semantic/wgo_judge.txt``) is copied verbatim from Macrodata Labs,
"Segmenting Robot Video into Actionable Subtasks" (2026): it gets the gold label, the predicted label and
the episode instruction and answers ``{"match": true}`` or ``false``. Their judge is Gemini 3.5 Flash;
here any provider of autolabel's VL client can judge (``--provider``), and every verdict is cached
(``--cache``), so reruns and other methods reuse it.

Scores per family, for predicted segments that carry labels:
* temporal F1: the WGO protocol (outer edges snapped, tIoU >= threshold, one-to-one matching);
* label accuracy on temporal matches: the share of those matched pairs the judge accepts;
* semantic precision, recall and F1: a pair counts only if its tIoU passes and the judge accepts its
  label ("a predicted segment also has to pass the labeling test in order to be considered a match");
  the one-to-one matching is taken over those pairs.
* gold-window accuracy (``gold`` command): labels written for the reference windows, judged against the
  reference labels.

    python kinseg/semantic_eval.py score --autolabel <checkout> --kinematic <run> --provider <name> \\
        --cache judge.jsonl --segments NAME=DIR[:KEY] ... [--threshold 0.75 0.5] --out <file>

A segment folder holds ``<episode>.json`` with a list ``KEY`` (default ``C1``; ``segments`` for other
sources) of ``{"start_s", "end_s", "label"}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

JUDGE = Path(__file__).resolve().parent / "prompts" / "semantic" / "wgo_judge.txt"
Interval = Tuple[float, float]


def judge_prompt(gold: str, pred: str, instruction: str, template: Optional[str] = None) -> str:
    template = template if template is not None else JUDGE.read_text().rstrip("\n")
    return template.replace("{gt_label}", gold).replace("{pred_label}", pred).replace("{instruction}", instruction)


def parse_match(answer: Optional[str]) -> Optional[bool]:
    """The judge's verdict, or None when the answer has no readable ``match``."""
    m = re.search(r'"match"\s*:\s*(true|false)', answer or "", flags=re.IGNORECASE)
    return None if m is None else m.group(1).lower() == "true"


class Judge:
    """Cached judge calls through autolabel's VL client."""

    def __init__(self, autolabel: str, provider: str, cache: Path, model: Optional[str] = None, workers: int = 4):
        sys.path.insert(0, autolabel)
        from autolabel.describe.vl_client import VLClient, get_provider_api_key

        self.make = lambda: VLClient(model=model, provider=provider, api_key=self.key, max_tokens=2048, timeout=180)
        self.key = get_provider_api_key(provider)
        self.model = self.make().model
        self.cache_path = cache
        self.cache: Dict[str, dict] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        self.lock = threading.Lock()
        self.workers = workers

    def key_of(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def run(self, prompts: Sequence[str]) -> Dict[str, Optional[bool]]:
        todo = sorted({p for p in prompts if self.key_of(p) not in self.cache})

        def one(prompt: str) -> None:
            answer = self.make().analyze_video(prompt, frames=[], system_prompt="")
            row = dict(key=self.key_of(prompt), model=self.model, prompt=prompt, answer=answer,
                       match=parse_match(answer))
            with self.lock:
                self.cache[row["key"]] = row
                with self.cache_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(one, todo))
        return {p: self.cache[self.key_of(p)]["match"] for p in prompts}


def tiou(a: Interval, b: Interval) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if inter > 0 and union > 0 else 0.0


def max_matching(edges: List[List[int]]) -> List[Tuple[int, int]]:
    owner: Dict[int, int] = {}

    def augment(i: int, seen: set) -> bool:
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in range(len(edges)):
        augment(i, set())
    return sorted((i, j) for j, i in owner.items())


def load_kinematic(run: Path) -> Dict[str, dict]:
    out = {}
    for path in run.glob("predictions/*/*.json"):
        case = json.loads(path.read_text())
        out[case["id"]] = case
    return out


def score(args) -> int:
    sys.path.insert(0, args.autolabel)
    from modules.exp_wgo_bench.score import snap_outer_edges

    kin = load_kinematic(Path(args.kinematic))
    judge = Judge(args.autolabel, args.provider, Path(args.cache), args.model, args.workers)
    sources = {}
    for item in args.segments:
        name, rest = item.split("=", 1)
        folder, _, key = rest.partition(":")
        sources[name] = (Path(folder), key or "C1")
    episodes = sorted(set.intersection(*[{p.stem for p in d.glob("*.json")} for d, _ in sources.values()]) & set(kin))
    # Collect every pair that passes a threshold, judge them all at once, then score.
    plan = []
    for name, (folder, key) in sources.items():
        for episode in episodes:
            case = kin[episode]
            refs = case["references"]
            ref_iv = [(r["start_sec"], r["end_sec"]) for r in refs]
            segs = json.loads((folder / f"{episode}.json").read_text())[key]
            pred = [(s["start_s"], s["end_s"]) for s in segs]
            snapped = snap_outer_edges(pred, ref_iv)
            pairs = {(i, j): tiou(p, r) for i, p in enumerate(snapped) for j, r in enumerate(ref_iv)}
            prompts = {(i, j): judge_prompt(refs[j]["subtask"], str(segs[i].get("label") or ""),
                                            str(case.get("instruction", "")))
                       for (i, j), t in pairs.items() if t >= min(args.threshold)}
            plan.append((name, episode, case["family"], len(pred), len(refs), pairs, prompts))
    verdicts = judge.run([p for *_, prompts in plan for p in prompts.values()])
    counts: Dict[tuple, List[int]] = {}
    unreadable = 0
    for name, episode, fam, n_pred, n_ref, pairs, prompts in plan:
        for th in args.threshold:
            c = counts.setdefault((name, fam, th), [0, 0, 0, 0, 0])
            temporal = max_matching([[j for j in range(n_ref) if pairs[(i, j)] >= th] for i in range(n_pred)])
            accepted = [[j for j in range(n_ref) if pairs[(i, j)] >= th and verdicts[prompts[(i, j)]]]
                        for i in range(n_pred)]
            semantic = max_matching(accepted)
            c[0] += len(temporal)
            c[1] += sum(bool(verdicts[prompts[pair]]) for pair in temporal)
            c[2] += len(semantic)
            c[3] += n_pred
            c[4] += n_ref
        unreadable += sum(verdicts[p] is None for p in prompts.values())
    lines = [f"Judge: {judge.model} via {args.provider}, Macrodata's prompt ({JUDGE.name}); "
             f"{len(verdicts)} judged pairs, {unreadable} unreadable answers (counted as reject).", "",
             "| Source | Family | tIoU | temporal F1 | label accuracy on temporal matches | semantic P | semantic R | semantic F1 | pred/ref |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, fam, th), (tp, ok, sem, n_pred, n_ref) in sorted(counts.items(), key=lambda kv: (kv[0][1], -kv[0][2], kv[0][0])):
        p, r = sem / n_pred if n_pred else 0, sem / n_ref if n_ref else 0
        lines.append(f"| {name} | {fam} | {th:g} | {2 * tp / (n_pred + n_ref):.3f} | {ok / tp if tp else 0:.3f} ({ok}/{tp}) | "
                     f"{p:.3f} | {r:.3f} | {2 * sem / (n_pred + n_ref):.3f} | {n_pred}/{n_ref} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


def gold(args) -> int:
    """Gold-window accuracy: ``--labels`` holds ``<episode>.json`` with ``labels``, one per reference."""
    kin = load_kinematic(Path(args.kinematic))
    judge = Judge(args.autolabel, args.provider, Path(args.cache), args.model, args.workers)
    rows = []
    for path in sorted(Path(args.labels).glob("*.json")):
        case = kin[path.stem]
        labels = json.loads(path.read_text())["labels"]
        for r, label in zip(case["references"], labels):
            rows.append((case["family"], judge_prompt(r["subtask"], str(label or ""), str(case.get("instruction", "")))))
    verdicts = judge.run([p for _, p in rows])
    by: Dict[str, List[int]] = {}
    for fam, p in rows:
        by.setdefault(fam, [0, 0])
        by[fam][0] += 1
        by[fam][1] += bool(verdicts[p])
    text = "\n".join(f"{fam}: gold-window label accuracy {ok / n:.3f} ({ok}/{n})" for fam, (n, ok) in sorted(by.items()))
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("score", "gold"):
        p = sub.add_parser(name)
        p.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"), required=False)
        p.add_argument("--kinematic", required=True)
        p.add_argument("--provider", default="openlux-codex-gpt-1")
        p.add_argument("--model")
        p.add_argument("--cache", required=True, help="JSONL of judged pairs (appended)")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--out")
    sub.choices["score"].add_argument("--segments", nargs="+", required=True, help="NAME=DIR[:KEY]")
    sub.choices["score"].add_argument("--threshold", nargs="+", type=float, default=[0.75, 0.5])
    sub.choices["gold"].add_argument("--labels", required=True)
    args = ap.parse_args(argv)
    if not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    return {"score": score, "gold": gold}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
