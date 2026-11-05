"""Combine captioned phases into subtasks with a language model that reads a short status log.

After captioning (``caption_merge.py run``), each episode becomes a text log, one line per phase: its
time, the hands' motion and gripper changes measured from the robot, and the phase's caption. A
language model groups consecutive phases into subtasks (with instruction ``v1`` it also says, for every
gripper close and open, whether it grasped or released an object). The answer is checked and turned into
segments:

* C1: the model's subtasks;
* C2 (v1): C1, then split where a confirmed grasp is followed by a confirmed release of the same hand
  inside one subtask (``merge.split_at_grasps``, at our first cut after the close);
* C3 (v1): TempSeg's segments with the same split from confirmed events only (the gripper split of
  ``merge.py`` uses every event);
* C4: C1 with each endpoint moved to TempSeg's nearest cut within ``SNAP_S`` (the model's grouping
  with TempSeg's boundary placement, which is closer to the references than our phase cuts).

    python kinseg/combine.py prompts --captions <caption run> --kinematic <run> --out <dir>
    python kinseg/combine.py ask --prompts <dir> --provider openlux-codex-gpt-1 --autolabel <checkout> --out <dir>
    python kinseg/combine.py score --prompts <dir> --answers NAME=DIR ... --autolabel <checkout> \\
        --kinematic <run> --tempseg <dir> --out <file> [--segments <dir>]

``ask`` sends one text-only request per episode through autolabel's VL client (the key comes from the
provider's environment variable or ``pass`` entry) and saves the answer with its settings and usage.
Any other model can answer instead: an answer folder holds ``<episode>.json`` with an ``answer`` string.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

SNAP_S = 1.0
SYSTEM = "You combine short phases of a robot demonstration into subtasks. Answer in JSON only."
STEP = ("Combine consecutive phases into subtasks: one manipulation step that a person would instruct on its "
        "own (pick up an object, place it, pour, open or close something, push, wipe ...).")
INSTRUCTIONS = {
    # v1 (2026-11-05): conventions read off WGO-Bench's references, and gripper verdicts.
    "v1": STEP + """
- Reaching for an object belongs to the step that acts on it; carrying a held object belongs to the step that puts it down; moving away after a release belongs to the next step.
- Different words for one movement ("lift", "move", "put") do not start a new step by themselves.
- For each gripper close or open in the log, decide from the instructions whether it grasped an object, released one, or did neither.

Answer in JSON only:
{"gripper": [{"phase": <index>, "hand": "left" or "right", "event": "close" or "open", "is": "grasp", "release" or "none"}, ...],
 "subtasks": [{"phases": [<first index>, <last index>], "instruction": "<one instruction for the subtask>"}, ...]}
List every gripper close and open of the log. The subtasks cover every phase from 0 to the last, in order, each phase once.""",
    # v2: no task-specific conventions; captions can be wrong or discontinuous; kinematics support the
    # captions; subtasks only.
    "v2": STEP + """
- Some captions may be wrong, and the captions are sometimes discontinuous; read each phase together with its neighbours.
- Use the kinematic signals (hand motion, gripper closes and opens) to support what the captions describe.

Answer in JSON only:
{"subtasks": [{"phases": [<first index>, <last index>], "instruction": "<one instruction for the subtask>"}, ...]}
The subtasks cover every phase from 0 to the last, in order, each phase once.""",
    # v5: v2 with a rule to keep the basic manipulation actions separate.
    "v5": STEP + """
- Keep basic manipulation actions (pick up, place, pour, open, close, push, wipe ...) as separate subtasks.
- Some captions may be wrong, and the captions are sometimes discontinuous; read each phase together with its neighbours.
- Use the kinematic signals (hand motion, gripper closes and opens) to support what the captions describe.

Answer in JSON only:
{"subtasks": [{"phases": [<first index>, <last index>], "instruction": "<one instruction for the subtask>"}, ...]}
The subtasks cover every phase from 0 to the last, in order, each phase once.""",
    # v3: v2 with FLORA-TempSeg's own granularity policy (its prompt) in place of our conventions.
    "v3": STEP + """
- Acquisition and delivery are distinct when observable. Ordinary contacts, adjustments and low-motion pauses within the same ongoing goal do not create new subtasks.
- Some captions may be wrong, and the captions are sometimes discontinuous; read each phase together with its neighbours.
- Use the kinematic signals (hand motion, gripper closes and opens) to support what the captions describe.

Answer in JSON only:
{"subtasks": [{"phases": [<first index>, <last index>], "instruction": "<one instruction for the subtask>"}, ...]}
The subtasks cover every phase from 0 to the last, in order, each phase once.""",
}
INSTRUCTION = INSTRUCTIONS["v1"]


def hands(case: Dict[str, object]) -> Tuple[str, ...]:
    """The robot's hands as named in the kinematics: ('left', 'right'), or ('right',) for one arm."""
    from merge import gripper_events

    phases = case["predictions"]
    two = any(p["arm"] in ("LEFT", "BOTH") for p in phases) or any(e[0] == "LEFT" for e in gripper_events(case))
    return ("left", "right") if two else ("right",)


def status(case: Dict[str, object], k: int, robot_hands: Sequence[str]) -> str:
    """Motion and gripper changes of phase ``k``: 'still', 'right moves', 'both move, left closes' …"""
    from merge import gripper_events

    phase = case["predictions"][k]
    arm = phase["arm"]
    motion = {"STABLE": "still", "BOTH": "both move"}.get(arm, f"{arm.lower()} moves")
    if len(robot_hands) == 1 and arm != "STABLE":
        motion = "moves"
    changes = [(f"{a.lower()} " if len(robot_hands) > 1 else "") + ("closes" if kind == "close" else "opens")
               for a, kind, t0, _ in gripper_events(case) if phase["start_sec"] <= t0 < phase["end_sec"]]
    return ", ".join([motion] + changes)


def episode_log(case: Dict[str, object], phases: List[Dict[str, object]], version: str = "v1") -> str:
    """The user prompt for one episode: task, robot, phase log and instruction ``version``."""
    robot_hands = hands(case)
    lines = [f"Task: {str(case.get('instruction', '')).strip()}",
             "Robot: " + ("one arm (right hand)." if len(robot_hands) == 1 else "two arms (left and right hands)."),
             "",
             "Phases. Time in seconds; hand motion and gripper changes are measured from the robot; each "
             "instruction was written by a video captioner for that phase alone and can be wrong."]
    for k, p in enumerate(phases):
        caption = (p.get("caption") or "").strip() if p.get("status", "complete") == "complete" else ""
        lines.append(f"{k:<3d} {p['start_s']:.1f}-{p['end_s']:.1f}  {status(case, k, robot_hands):<24} | "
                     f"{caption or '(no caption)'}")
    return "\n".join(lines + ["", INSTRUCTIONS[version]])


def parse_answer(text: Optional[str]) -> Optional[dict]:
    """The JSON object in an answer (code fences and <think> blocks removed), or None."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def subtask_ranges(answer: dict, n: int) -> Tuple[List[Tuple[int, int, str]], List[str]]:
    """(first, last, instruction) ranges covering 0..n-1 in order from ``answer['subtasks']``, and repairs.

    Ranges are sorted by first phase and clipped to the episode; a gap joins the previous range and an
    overlap is cut at the next range's first phase.
    """
    repairs: List[str] = []
    ranges = []
    for item in answer.get("subtasks") or []:
        try:
            a, b = (int(v) for v in item["phases"])
        except (KeyError, TypeError, ValueError):
            repairs.append(f"unreadable subtask {item!r}")
            continue
        a, b = max(0, min(a, b)), min(n - 1, max(a, b))
        if a <= b:
            ranges.append((a, b, str(item.get("instruction", ""))))
    ranges.sort(key=lambda r: (r[0], r[1]))
    out: List[Tuple[int, int, str]] = []
    for a, b, label in ranges:
        if out and a <= out[-1][1]:
            if b <= out[-1][1]:
                repairs.append(f"dropped nested subtask {a}-{b}")
                continue
            repairs.append(f"overlap at {a}")
            a = out[-1][1] + 1
        if out and a > out[-1][1] + 1:
            repairs.append(f"gap {out[-1][1] + 1}-{a - 1} joined to the previous subtask")
            out[-1] = (out[-1][0], a - 1, out[-1][2])
        out.append((a, b, label))
    if not out:
        return [], repairs + ["no subtasks"]
    if out[0][0] > 0:
        repairs.append(f"phases 0-{out[0][0] - 1} joined to the first subtask")
        out[0] = (0, out[0][1], out[0][2])
    if out[-1][1] < n - 1:
        repairs.append(f"phases {out[-1][1] + 1}-{n - 1} joined to the last subtask")
        out[-1] = (out[-1][0], n - 1, out[-1][2])
    return out, repairs


def confirmed_events(answer: dict, case: Dict[str, object], phases: List[Dict[str, object]]):
    """Gripper events the answer confirms: closes judged 'grasp' and opens judged 'release'."""
    from merge import gripper_events

    robot_hands = hands(case)
    events = gripper_events(case)
    keep = set()
    for item in answer.get("gripper") or []:
        try:
            k, kind, verdict = int(item["phase"]), str(item["event"]).lower(), str(item["is"]).lower()
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= k < len(phases) or (kind, verdict) not in (("close", "grasp"), ("open", "release")):
            continue
        hand = str(item.get("hand", robot_hands[-1])).lower()
        for e in events:
            if (e[1] == kind and phases[k]["start_s"] <= e[2] < phases[k]["end_s"]
                    and (len(robot_hands) == 1 or e[0].lower() == hand)):
                keep.add(e)
    return sorted(keep, key=lambda e: e[2])


# ── commands ───────────────────────────────────────────────────────────


def load_kinematic(run: Path) -> Dict[str, Dict[str, object]]:
    out = {}
    for path in run.glob("predictions/*/*.json"):
        case = json.loads(path.read_text())
        out[case["id"]] = case
    return out


def prompts(args) -> int:
    kin = load_kinematic(Path(args.kinematic))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = sorted(p.stem for p in Path(args.captions).glob("*.json") if p.stem in kin)
    for episode in episodes:
        rec = json.loads((Path(args.captions) / f"{episode}.json").read_text())
        phases = [dict(start_s=p["start_s"], end_s=p["end_s"], caption=p.get("caption", ""), status=p.get("status"))
                  for p in rec["rounds"][0]]
        prompt = episode_log(kin[episode], phases, args.version)
        (out / f"{episode}.json").write_text(json.dumps(dict(id=episode, family=kin[episode]["family"],
                                                             captions=str(args.captions), system=SYSTEM,
                                                             version=args.version, prompt=prompt, phases=phases),
                                                        indent=1) + "\n")
        (out / f"{episode}.txt").write_text(f"SYSTEM: {SYSTEM}\n\n{prompt}\n")
    print(f"wrote {len(episodes)} prompts to {out}")
    return 0


def ask(args) -> int:
    sys.path.insert(0, str(Path(args.autolabel)))
    from autolabel.describe.vl_client import VLClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(Path(args.prompts).glob("*.json")) if args.overwrite or not (out / p.name).exists()]
    if args.episodes:
        todo = [p for p in todo if p.stem in set(args.episodes)]

    def one(path: Path) -> str:
        item = json.loads(path.read_text())
        client = VLClient(model=args.model, provider=args.provider, max_tokens=args.max_tokens,
                          timeout=args.timeout)
        started = time.monotonic()
        answer = client.analyze_video(item["prompt"], frames=[], system_prompt=item["system"])
        (out / path.name).write_text(json.dumps(dict(
            id=item["id"], provider=client.provider, model=client.model, temperature=client.temperature,
            max_tokens=client.max_tokens, reasoning_effort=client.reasoning_effort, answer=answer,
            error=client.last_error, usage=client.last_usage, elapsed_s=round(time.monotonic() - started, 1)),
            indent=1) + "\n")
        return f"{item['id']}: {'ok' if answer else client.last_error}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(one, todo):
            print(line, flush=True)
    return 0


def score(args) -> int:
    from caption_merge import join_operations, limb_motion, motion_only
    from merge import boundaries, gripper_events, scorer, snap, split_at_grasps

    score_family = scorer(Path(args.autolabel))
    kin = load_kinematic(Path(args.kinematic))
    items = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.prompts).glob("*.json"))}
    answers = dict(a.split("=", 1) for a in args.answers)
    seg_root = Path(args.segments) if args.segments else None

    def case_of(segments, refs):
        return {"predictions": [{"start_sec": a, "end_sec": b} for a, b in segments], "references": refs}

    table: Dict[Tuple[str, str], Dict[str, list]] = {}
    notes: List[str] = []
    for episode, item in items.items():
        case = kin[episode]
        refs = case["references"]
        phases = item["phases"]
        cuts = boundaries([(p["start_sec"], p["end_sec"]) for p in case["predictions"]])
        tempseg = sorted((s["start_s"], s["end_s"]) for s in
                         json.loads((Path(args.tempseg) / f"{episode}.json").read_text())["segments"])
        fam = item["family"]
        base = table.setdefault(("baselines", fam), {})
        base.setdefault("TempSeg", []).append(case_of(tempseg, refs))
        base.setdefault("TempSeg + gripper split (all events)", []).append(
            case_of(split_at_grasps(tempseg, gripper_events(case), cuts), refs))
        for rule, minor in (("operation join", limb_motion), ("motion absorbed", motion_only)):
            base.setdefault(f"captions, {rule}", []).append(
                case_of([(j["start_s"], j["end_s"]) for j in join_operations(phases, minor)], refs))
        for name, folder in answers.items():
            path = Path(folder) / f"{episode}.json"
            row = table.setdefault((name, fam), {})
            stats = row.setdefault("_stats", [])
            answer = parse_answer(json.loads(path.read_text()).get("answer")) if path.exists() else None
            if answer is None:
                ranges, repairs = [], ["no answer" if not path.exists() else "unparseable answer"]
            else:
                ranges, repairs = subtask_ranges(answer, len(phases))
            if ranges:
                c1 = [(phases[a]["start_s"], phases[b]["end_s"]) for a, b, _ in ranges]
                labels = [label for _, _, label in ranges]
            else:  # fallback: the operation join
                joined = join_operations(phases)
                c1, labels = [(j["start_s"], j["end_s"]) for j in joined], [j["key"] for j in joined]
            confirmed = confirmed_events(answer, case, phases) if answer else []
            c2 = split_at_grasps(c1, confirmed, cuts)
            c3 = split_at_grasps(tempseg, confirmed, cuts)
            c4 = snap(c1, sorted(set(boundaries(tempseg) + [tempseg[0][0], tempseg[-1][1]])), SNAP_S)
            row.setdefault("C1 model's subtasks", []).append(case_of(c1, refs))
            if item.get("version", "v1") == "v1":  # gripper verdicts asked for
                row.setdefault("C2 C1 + confirmed gripper split", []).append(case_of(c2, refs))
                row.setdefault("C3 TempSeg + confirmed gripper split", []).append(case_of(c3, refs))
            row.setdefault(f"C4 C1 snapped to TempSeg cuts ({SNAP_S:g} s)", []).append(case_of(c4, refs))
            stats.append(dict(answered=answer is not None, repairs=len(repairs), confirmed=len(confirmed),
                              events=len(gripper_events(case))))
            if repairs:
                notes.append(f"{name} {episode}: " + "; ".join(repairs))
            if seg_root is not None:
                (seg_root / name).mkdir(parents=True, exist_ok=True)
                (seg_root / name / f"{episode}.json").write_text(json.dumps(dict(
                    id=episode, C1=[dict(start_s=a, end_s=b, label=l) for (a, b), l in zip(c1, labels)],
                    C2=[dict(start_s=a, end_s=b) for a, b in c2], C3=[dict(start_s=a, end_s=b) for a, b in c3],
                    C4=[dict(start_s=a, end_s=b, label=l) for (a, b), l in zip(c4, labels)],
                    confirmed=[dict(arm=e[0], kind=e[1], start_sec=e[2]) for e in confirmed],
                    gripper=(answer or {}).get("gripper"), repairs=repairs), indent=1) + "\n")
    lines = ["| Source | Family | Method | F1@0.75 | pred/ref | Answers / repaired / events confirmed |",
             "|---|---|---|---:|---:|---|"]
    for (name, fam), row in sorted(table.items(), key=lambda kv: (kv[0][0] != "baselines", kv[0][0], kv[0][1])):
        stats = row.get("_stats", [])
        extra = (f"{sum(s['answered'] for s in stats)}/{len(stats)} / {sum(s['repairs'] > 0 for s in stats)} / "
                 f"{sum(s['confirmed'] for s in stats)} of {sum(s['events'] for s in stats)}") if stats else ""
        for method, cases in row.items():
            if method == "_stats":
                continue
            seg = score_family(cases)["segment@0.75"]
            lines.append(f"| {name} | {fam} | {method} | {seg['f1']:.3f} | {seg['n_pred']}/{seg['n_ref']} | {extra} |")
            extra = ""
    text = "\n".join(lines + ([""] + ["Repairs:"] + [f"- {n}" for n in notes] if notes else []))
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


def examples(args) -> int:
    """Markdown: per episode, the status log next to each model's subtasks, then the segments."""
    from caption_examples import cell, f1_text, matched, protocol, segment_list
    from merge import boundaries, snap

    snap_edges, match = protocol(Path(args.autolabel))
    kin = load_kinematic(Path(args.kinematic))
    answers = dict(a.split("=", 1) for a in args.answers)
    items = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.prompts).glob("*.json"))}
    episodes = [e for e in items if not args.episodes or e in args.episodes]
    episodes.sort(key=lambda e: (items[e]["family"] != "galaxea", e))
    out = Path(args.out)
    pages, index = [], []
    for episode in episodes:
        item, case = items[episode], kin[episode]
        phases, robot_hands = item["phases"], hands(case)
        refs = [(r["start_sec"], r["end_sec"]) for r in case["references"]]
        ref_labels = [r["subtask"] for r in case["references"]]
        tempseg = sorted((s["start_s"], s["end_s"]) for s in
                         json.loads((Path(args.tempseg) / f"{episode}.json").read_text())["segments"])
        tempseg_cuts = sorted(set(boundaries(tempseg) + [tempseg[0][0], tempseg[-1][1]]))
        columns: Dict[str, List[str]] = {}
        segment_rows: List[str] = []
        scores: Dict[str, str] = {}

        def add(name, intervals, labels):
            hits = matched(intervals, refs, snap_edges, match)
            scores[name] = f1_text(len(hits), len(intervals), len(refs))
            segment_rows.append(f"| {name} | {scores[name]} | {segment_list(intervals, labels, hits)} |")

        add("TempSeg", tempseg, [""] * len(tempseg))
        for name, folder in answers.items():
            path = Path(folder) / f"{episode}.json"
            answer = parse_answer(json.loads(path.read_text()).get("answer")) if path.exists() else None
            ranges = subtask_ranges(answer, len(phases))[0] if answer else []
            cells = ["(no answer)"] * len(phases)
            for s, (a, b, label) in enumerate(ranges):
                for k in range(a, b + 1):
                    cells[k] = f"**S{s + 1}** {label}" if k == a else "┃"
            for g in (answer or {}).get("gripper") or []:
                try:
                    k = int(g["phase"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= k < len(phases):
                    hand = f"{g.get('hand')} " if len(robot_hands) > 1 else ""
                    cells[k] += f" · *{hand}{g.get('event')}: {g.get('is')}*"
            columns[name] = cells
            c1 = [(phases[a]["start_s"], phases[b]["end_s"]) for a, b, _ in ranges]
            labels = [label for _, _, label in ranges]
            if c1:
                add(f"{name} · subtasks (C1)", c1, labels)
                add(f"{name} · snapped to TempSeg cuts (C4)", snap(c1, tempseg_cuts, SNAP_S), labels)
        names = list(columns)
        lines = [f"## {episode}", "", f"**Task:** {cell(str(case.get('instruction', '')))}  ",
                 f"{item['family']}, {len(phases)} phases, {len(refs)} reference subtasks", "",
                 "| # | Time (s) | Robot | Caption | Reference subtask | " + " | ".join(names) + " |",
                 "|---:|---|---|---|---|" + "---|" * len(names)]
        previous = None
        for k, p in enumerate(phases):
            mid = (p["start_s"] + p["end_s"]) / 2
            j = next((j for j, (s, e) in enumerate(refs) if s <= mid < e), None)
            ref = "–" if j is None else (f"▸ {ref_labels[j]}" if j != previous else "┃")
            previous = j
            lines.append(f"| {k} | {p['start_s']:.1f}–{p['end_s']:.1f} | {status(case, k, robot_hands)} | "
                         f"{cell(p.get('caption') or '')} | {cell(ref)} | "
                         + " | ".join(cell(columns[n][k]) for n in names) + " |")
        lines += ["", "| Method | F1 (TP/pred/ref) | Segments |", "|---|---:|---|",
                  f"| reference | – | {segment_list(refs, ref_labels, set())} |"] + segment_rows
        pages.append("\n".join(lines) + "\n")
        index.append(f"| [{episode}](#{episode}) | {item['family']} | {len(phases)} | {len(refs)} | "
                     f"{scores['TempSeg']} | " + " | ".join(
                         f"{scores.get(f'{n} · subtasks (C1)', '–')} / {scores.get(f'{n} · snapped to TempSeg cuts (C4)', '–')}"
                         for n in names) + " |")
    names = list(answers)
    header = ["# How the combination works: examples", "",
              "Each episode shows the status log the model reads (time, robot, caption; one line per phase), "
              "the reference subtasks (not shown to the model), and each model's answer: **S*n*** starts a "
              "subtask with its instruction, ┃ continues it, and *close: grasp* is the model's verdict on a "
              "gripper event. Below, the segments: the model's subtasks (C1) and the same with each endpoint "
              f"moved to TempSeg's nearest cut within {SNAP_S:g} s (C4); ✓ marks a segment matched to a "
              "reference (F1@0.75, WGO protocol).", "",
              "The instruction after the log, the same for every episode:", "", "```text",
              INSTRUCTIONS[next(iter(items.values())).get("version", "v1")], "```", "",
              "| Episode | Family | Phases | Subtasks | TempSeg | " + " | ".join(f"{n} C1 / C4" for n in names) + " |",
              "|---|---|---:|---:|---:|" + "---:|" * len(names)] + index + [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n" + "\n".join(pages))
    print(f"wrote {out} ({len(pages)} episodes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts", help="one status-log prompt per episode")
    p.add_argument("--captions", required=True, help="caption_merge.py run output folder")
    p.add_argument("--kinematic", required=True)
    p.add_argument("--version", choices=sorted(INSTRUCTIONS), default="v2", help="instruction after the log")
    p.add_argument("--out", required=True)
    a = sub.add_parser("ask", help="answers from a provider of autolabel's VL client")
    a.add_argument("--prompts", required=True)
    a.add_argument("--provider", required=True)
    a.add_argument("--model", help="default: the provider's")
    a.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"), required=False)
    a.add_argument("--max-tokens", type=int, default=8192)
    a.add_argument("--timeout", type=float, default=300.0)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--episodes", nargs="+", help="only these episodes")
    a.add_argument("--out", required=True)
    s = sub.add_parser("score", help="segments from the answers, scored with the baselines")
    s.add_argument("--prompts", required=True)
    s.add_argument("--answers", nargs="+", required=True, help="NAME=DIR answer folders")
    s.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    s.add_argument("--kinematic", required=True)
    s.add_argument("--tempseg", required=True)
    s.add_argument("--segments", help="also write each answer's segments here, per source and episode")
    s.add_argument("--out")
    e = sub.add_parser("examples", help="markdown: the log next to each model's subtasks, and the segments")
    e.add_argument("--prompts", required=True)
    e.add_argument("--answers", nargs="+", required=True, help="NAME=DIR answer folders")
    e.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    e.add_argument("--kinematic", required=True)
    e.add_argument("--tempseg", required=True)
    e.add_argument("--episodes", nargs="+", help="only these episodes")
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command in ("ask", "score", "examples") and not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    return {"prompts": prompts, "ask": ask, "score": score, "examples": examples}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
