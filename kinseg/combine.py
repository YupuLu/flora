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
  with TempSeg's boundary placement, which is closer to the references than our phase cuts);
* C5 (with verdicts): C1 with a cut within one phase of a confirmed grasp moved to the end of the
  grasp's phase (a confirmed grasp completes the pick-up; ``anchored``);
* C6: C5 snapped like C4;
* C7 (with transitions, v7): C1 with the model's transition phases (reach, lift, carry, retreat) re-attached
  by the dataset setting ``--attach`` (e.g. ``retreat=next,lift=previous``): a transition phase at the
  edge of a subtask moves to the neighbouring subtask on the set side (``reattach``);
* C8: C7, then the grasp anchors of C5. Neither uses TempSeg.

    python kinseg/combine.py prompts --captions <caption run> --kinematic <run> --out <dir>
    python kinseg/combine.py ask --prompts <dir> --provider openlux-codex-gpt-1 --autolabel <checkout> --out <dir>
    python kinseg/combine.py score --prompts <dir> --answers NAME=DIR ... --autolabel <checkout> \\
        --kinematic <run> --tempseg <dir> --out <file> [--segments <dir>]
    python kinseg/combine.py segments --prompts <dir> --answers <dir> --kinematic <run> --out <dir>

``ask`` sends one text-only request per episode through autolabel's VL client (the key comes from the
provider's environment variable or ``pass`` entry) and saves the answer with its settings and usage.
Any other model can answer instead: an answer folder holds ``<episode>.json`` with an ``answer`` string.
"""

from __future__ import annotations

import argparse
import hashlib
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
PROMPTS = Path(__file__).resolve().parent / "prompts" / "combine"


def load_instructions(folder: Path = PROMPTS) -> Dict[str, str]:
    """Instruction versions: every ``<version>.txt`` in ``folder`` except ``system.txt``.

    Each file is the text after the status log, verbatim; a new version is a new file. The versions so
    far (2026-11-05): v1 our conventions + gripper verdicts; v2 no conventions, subtasks only; v3 v2 +
    FLORA-TempSeg's granularity sentence; v5 v2 + basic actions kept separate; v6 v1 and v5 in general
    terms; v6-movement, v6-grouping, v6-previous, v6-noverdict, v6-boundary: v6 with one change each.
    """
    return {f.stem: f.read_text().rstrip("\n") for f in sorted(folder.glob("*.txt")) if f.stem != "system"}


SYSTEM = (PROMPTS / "system.txt").read_text().strip()
INSTRUCTIONS = load_instructions()
INSTRUCTION = INSTRUCTIONS["v1"]


def asks_gripper(item: Dict[str, object]) -> bool:
    """Whether a prompt item's instruction asks for gripper verdicts."""
    return '"gripper"' in str(item.get("instruction") or INSTRUCTIONS[str(item.get("version", "v1"))])


def hands(case: Dict[str, object]) -> Tuple[str, ...]:
    """The robot's hands as named in the kinematics: ('left', 'right'), or ('right',) for one arm."""
    from merge import gripper_events

    phases = case["predictions"]
    two = any(p["arm"] in ("LEFT", "BOTH") for p in phases) or any(e[0] == "LEFT" for e in gripper_events(case))
    return ("left", "right") if two else ("right",)


def mobile(case: Dict[str, object]) -> bool:
    """Whether the base moves in any phase (fixed-base episodes have only 'STABLE')."""
    return any(p.get("base", "STABLE") != "STABLE" for p in case["predictions"])


def status(case: Dict[str, object], k: int, robot_hands: Sequence[str]) -> str:
    """Motion and gripper changes of phase ``k``: 'still', 'right moves', 'both move, left closes',
    'base moves', 'right moves, base moves' …"""
    from merge import gripper_events

    phase = case["predictions"][k]
    arm = phase["arm"]
    motion = {"STABLE": "still", "BOTH": "both move"}.get(arm, f"{arm.lower()} moves")
    if len(robot_hands) == 1 and arm != "STABLE":
        motion = "moves"
    if phase.get("base", "STABLE") != "STABLE":
        motion = "base moves" if arm == "STABLE" else motion + ", base moves"
    changes = [(f"{a.lower()} " if len(robot_hands) > 1 else "") + ("closes" if kind == "close" else "opens")
               for a, kind, t0, _ in gripper_events(case) if phase["start_sec"] <= t0 < phase["end_sec"]]
    return ", ".join([motion] + changes)


def episode_log(case: Dict[str, object], phases: List[Dict[str, object]], version: str = "v1",
                instruction: Optional[str] = None) -> str:
    """The user prompt for one episode: task, robot, phase log and ``instruction`` (default: ``version``'s)."""
    robot_hands = hands(case)
    base = " on a mobile base" if mobile(case) else ""
    lines = [f"Task: {str(case.get('instruction', '')).strip()}",
             "Robot: " + ("one arm (right hand)" if len(robot_hands) == 1 else "two arms (left and right hands)")
             + base + ".",
             "",
             "Phases. Time in seconds; " + ("hand and base motion" if base else "hand motion")
             + " and gripper changes are measured from the robot; each "
             "instruction was written by a video captioner for that phase alone and can be wrong."]
    for k, p in enumerate(phases):
        caption = (p.get("caption") or "").strip() if p.get("status", "complete") == "complete" else ""
        lines.append(f"{k:<3d} {p['start_s']:.1f}-{p['end_s']:.1f}  {status(case, k, robot_hands):<24} | "
                     f"{caption or '(no caption)'}")
    return "\n".join(lines + ["", instruction if instruction is not None else INSTRUCTIONS[version]])


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


def verdicts(answer: dict, case: Dict[str, object], phases: List[Dict[str, object]]) -> List[Tuple[tuple, str]]:
    """(gripper event, verdict) for every verdict of the answer that names an event of the log."""
    from merge import gripper_events

    robot_hands = hands(case)
    events = gripper_events(case)
    out = []
    for item in answer.get("gripper") or []:
        try:
            k, kind, verdict = int(item["phase"]), str(item["event"]).lower(), str(item["is"]).lower()
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= k < len(phases):
            continue
        hand = str(item.get("hand", robot_hands[-1])).lower()
        out += [(e, verdict) for e in events
                if e[1] == kind and phases[k]["start_s"] <= e[2] < phases[k]["end_s"]
                and (len(robot_hands) == 1 or e[0].lower() == hand)]
    return out


def at_reference_cut(event: tuple, references: List[Dict[str, float]]) -> bool:
    """A reference cut lies between 0.5 s before and 2 s after the event's end."""
    return any(-0.5 <= r["end_sec"] - event[3] <= 2.0 for r in references[:-1])


ANCHOR_KINDS = ("close",)


def anchored(ranges: List[Tuple[int, int, str]], phases: List[Dict[str, object]], events,
             kinds: Sequence[str] = ANCHOR_KINDS) -> List[Tuple[float, float]]:
    """Segments of ``ranges`` after moving each cut that lies within one phase of a confirmed event of
    ``kinds`` to the end of that event's phase. Cuts are indices of the first phase of each subtask."""
    starts = [a for a, _, _ in ranges][1:]
    for e in events:
        if e[1] not in kinds:
            continue
        k = next((i for i, p in enumerate(phases) if p["start_s"] <= e[2] < p["end_s"]), None)
        if k is None or k + 1 >= len(phases):
            continue
        near = [b for b in starts if abs(b - (k + 1)) <= 1]
        if near:
            b = min(near, key=lambda x: abs(x - (k + 1)))
            starts[starts.index(b)] = k + 1
    edges = [0] + sorted(set(starts)) + [len(phases)]
    return [(phases[a]["start_s"], phases[b - 1]["end_s"]) for a, b in zip(edges[:-1], edges[1:]) if b > a]


TRANSITION_TYPES = ("reach", "lift", "carry", "retreat")


def parse_attach(text: str) -> Dict[str, str]:
    """``retreat=next,lift=previous`` -> {"retreat": "next", "lift": "previous"} (other types stay put)."""
    out = {}
    for part in filter(None, (t.strip() for t in text.split(","))):
        kind, side = (x.strip() for x in part.split("="))
        if kind not in TRANSITION_TYPES or side not in ("next", "previous"):
            raise ValueError(f"bad --attach item {part!r}")
        out[kind] = side
    return out


def transition_types(answer: dict, n: int) -> Dict[int, str]:
    """Phase index -> transition type from the answer's ``transitions`` list."""
    out = {}
    for item in answer.get("transitions") or []:
        try:
            k, kind = int(item["phase"]), str(item["type"]).lower()
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= k < n and kind in TRANSITION_TYPES:
            out[k] = kind
    return out


def reattach(ranges: List[Tuple[int, int, str]], transitions: Dict[int, str],
             attach: Dict[str, str]) -> List[Tuple[int, int, str]]:
    """Move transition phases across subtask boundaries to the side ``attach`` sets for their type.

    At each boundary, transition phases at the start of the later subtask whose type is set to
    "previous" join the earlier subtask, and transition phases at the end of the earlier subtask whose
    type is set to "next" join the later one; every subtask keeps at least one phase.
    """
    out = [list(r) for r in ranges]
    for s in range(1, len(out)):
        prev, cur = out[s - 1], out[s]
        while cur[0] < cur[1] and attach.get(transitions.get(cur[0], "")) == "previous":
            cur[0] += 1
            prev[1] += 1
        while prev[0] < prev[1] and attach.get(transitions.get(prev[1], "")) == "next":
            prev[1] -= 1
            cur[0] -= 1
    return [tuple(r) for r in out]


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
    if args.instruction:
        version, instruction = Path(args.instruction).stem, Path(args.instruction).read_text().rstrip("\n")
    else:
        version, instruction = args.version, INSTRUCTIONS[args.version]
    for episode in episodes:
        rec = json.loads((Path(args.captions) / f"{episode}.json").read_text())
        phases = [dict(start_s=p["start_s"], end_s=p["end_s"], caption=p.get("caption", ""), status=p.get("status"))
                  for p in rec["rounds"][0]]
        prompt = episode_log(kin[episode], phases, version, instruction)
        (out / f"{episode}.json").write_text(json.dumps(dict(id=episode, family=kin[episode]["family"],
                                                             captions=str(args.captions), system=SYSTEM,
                                                             version=version, instruction=instruction,
                                                             instruction_sha256=hashlib.sha256(
                                                                 instruction.encode()).hexdigest(),
                                                             prompt=prompt, phases=phases),
                                                        indent=1) + "\n")
        (out / f"{episode}.txt").write_text(f"SYSTEM: {SYSTEM}\n\n{prompt}\n")
    print(f"wrote {len(episodes)} prompts to {out}")
    return 0


def ask(args) -> int:
    sys.path.insert(0, str(Path(args.autolabel)))
    from autolabel.describe.vl_client import VLClient, get_provider_api_key

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(Path(args.prompts).glob("*.json")) if args.overwrite or not (out / p.name).exists()]
    if args.episodes:
        todo = [p for p in todo if p.stem in set(args.episodes)]
    # One key lookup per run: concurrent `pass show` calls can fail.
    key = get_provider_api_key(args.provider) if todo else None

    def one(path: Path) -> str:
        item = json.loads(path.read_text())
        client = VLClient(model=args.model, provider=args.provider, api_key=key, max_tokens=args.max_tokens,
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
    attach = parse_attach(args.attach)
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
            c5 = c6 = None
            if asks_gripper(item):
                row.setdefault("C2 C1 + confirmed gripper split", []).append(case_of(c2, refs))
                row.setdefault("C3 TempSeg + confirmed gripper split", []).append(case_of(c3, refs))
                c5 = anchored(ranges, phases, confirmed) if ranges else c1
                c6 = snap(c5, sorted(set(boundaries(tempseg) + [tempseg[0][0], tempseg[-1][1]])), SNAP_S)
                row.setdefault("C5 C1, cuts anchored at confirmed grasps", []).append(case_of(c5, refs))
                row.setdefault(f"C6 C5 snapped to TempSeg cuts ({SNAP_S:g} s)", []).append(case_of(c6, refs))
            row.setdefault(f"C4 C1 snapped to TempSeg cuts ({SNAP_S:g} s)", []).append(case_of(c4, refs))
            c7 = c8 = None
            if answer is not None and "transitions" in answer and ranges:
                moved = reattach(ranges, transition_types(answer, len(phases)), attach)
                c7 = [(phases[a]["start_s"], phases[b]["end_s"]) for a, b, _ in moved]
                c8 = anchored(moved, phases, confirmed)
                tag = ",".join(f"{k}={v}" for k, v in sorted(attach.items())) or "none"
                row.setdefault(f"C7 C1, transitions attached ({tag})", []).append(case_of(c7, refs))
                row.setdefault("C8 C7 + grasp anchors", []).append(case_of(c8, refs))
            stats.append(dict(answered=answer is not None, repairs=len(repairs), confirmed=len(confirmed),
                              events=len(gripper_events(case)),
                              verdicts=[(v, at_reference_cut(e, refs)) for e, v in
                                        (verdicts(answer, case, phases) if answer else [])]))
            if repairs:
                notes.append(f"{name} {episode}: " + "; ".join(repairs))
            if seg_root is not None:
                (seg_root / name).mkdir(parents=True, exist_ok=True)
                (seg_root / name / f"{episode}.json").write_text(json.dumps(dict(
                    id=episode, C1=[dict(start_s=a, end_s=b, label=l) for (a, b), l in zip(c1, labels)],
                    C2=[dict(start_s=a, end_s=b) for a, b in c2], C3=[dict(start_s=a, end_s=b) for a, b in c3],
                    C4=[dict(start_s=a, end_s=b, label=l) for (a, b), l in zip(c4, labels)],
                    **({} if c5 is None else dict(C5=[dict(start_s=a, end_s=b) for a, b in c5],
                                                  C6=[dict(start_s=a, end_s=b) for a, b in c6])),
                    **({} if c7 is None else dict(C7=[dict(start_s=a, end_s=b) for a, b in c7],
                                                  C8=[dict(start_s=a, end_s=b) for a, b in c8],
                                                  transitions=(answer or {}).get("transitions"))),
                    confirmed=[dict(arm=e[0], kind=e[1], start_sec=e[2]) for e in confirmed],
                    gripper=(answer or {}).get("gripper"), repairs=repairs), indent=1) + "\n")
    lines = ["| Source | Family | Method | F1@0.75 | F1@0.5 | pred/ref | Answers / repaired / events confirmed |",
             "|---|---|---|---:|---:|---:|---|"]
    verdict_lines = []
    for (name, fam), row in sorted(table.items(), key=lambda kv: (kv[0][0] != "baselines", kv[0][0], kv[0][1])):
        stats = row.get("_stats", [])
        extra = (f"{sum(s['answered'] for s in stats)}/{len(stats)} / {sum(s['repairs'] > 0 for s in stats)} / "
                 f"{sum(s['confirmed'] for s in stats)} of {sum(s['events'] for s in stats)}") if stats else ""
        pairs = [v for s in stats for v in s.get("verdicts", [])]
        if pairs:
            by = {}
            for verdict, at in pairs:
                by.setdefault(verdict, [0, 0])
                by[verdict][0] += 1
                by[verdict][1] += at
            verdict_lines.append(f"| {name} | {fam} | " + " | ".join(
                f"{v}: {n} ({a} at a reference cut)" for v, (n, a) in sorted(by.items())) + " |")
        for method, cases in row.items():
            if method == "_stats":
                continue
            scored = score_family(cases)
            seg, half = scored["segment@0.75"], scored["segment@0.5"]
            lines.append(f"| {name} | {fam} | {method} | {seg['f1']:.3f} | {half['f1']:.3f} | "
                         f"{seg['n_pred']}/{seg['n_ref']} | {extra} |")
            extra = ""
    if verdict_lines:
        lines += ["", "Gripper verdicts (an event is at a reference cut when one lies 0.5 s before to 2 s after its end):",
                  ""] + verdict_lines
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
    first = next(iter(items.values()))
    header = ["# How the combination works: examples", "",
              "Each episode shows the status log the model reads (time, robot, caption; one line per phase), "
              "the reference subtasks (not shown to the model), and each model's answer: **S*n*** starts a "
              "subtask with its instruction, ┃ continues it, and *close: grasp* is the model's verdict on a "
              "gripper event. Below, the segments: the model's subtasks (C1) and the same with each endpoint "
              f"moved to TempSeg's nearest cut within {SNAP_S:g} s (C4); ✓ marks a segment matched to a "
              "reference (F1@0.75, WGO protocol).", "",
              "The instruction after the log, the same for every episode:", "", "```text",
              str(first.get("instruction") or INSTRUCTIONS[first.get("version", "v1")]), "```", "",
              "| Episode | Family | Phases | Subtasks | TempSeg | " + " | ".join(f"{n} C1 / C4" for n in names) + " |",
              "|---|---|---:|---:|---:|" + "---:|" * len(names)] + index + [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n" + "\n".join(pages))
    print(f"wrote {out} ({len(pages)} episodes)")
    return 0


def labelled(segments: List[Tuple[float, float]], ranges: List[Tuple[int, int, str]],
             phases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Segments with the instruction of the model's subtask they overlap most."""
    def overlap(seg, r):
        return min(seg[1], phases[r[1]]["end_s"]) - max(seg[0], phases[r[0]]["start_s"])
    return [dict(start_s=a, end_s=b, instruction=max(ranges, key=lambda r: overlap((a, b), r))[2] if ranges else "")
            for a, b in segments]


def segments(args) -> int:
    """C1 and C5 per episode with the model's instructions, for data without references (no scores)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kin = load_kinematic(Path(args.kinematic))
    for path in sorted(Path(args.prompts).glob("*.json")):
        item = json.loads(path.read_text())
        episode, phases = item["id"], item["phases"]
        case = kin[episode]
        apath = Path(args.answers) / f"{episode}.json"
        answer = parse_answer(json.loads(apath.read_text()).get("answer")) if apath.exists() else None
        ranges, repairs = subtask_ranges(answer, len(phases)) if answer else ([], ["no answer"])
        confirmed = confirmed_events(answer, case, phases) if answer else []
        c1 = [(phases[a]["start_s"], phases[b]["end_s"]) for a, b, _ in ranges]
        c5 = anchored(ranges, phases, confirmed) if ranges and asks_gripper(item) else c1
        record = dict(id=episode, family=item["family"], task=case.get("instruction", ""),
                      answers=str(args.answers), repairs=repairs,
                      subtasks=[dict(phases=[a, b], instruction=label) for a, b, label in ranges],
                      confirmed=[dict(arm=e[0], kind=e[1], start_sec=e[2], end_sec=e[3]) for e in confirmed],
                      C1=labelled(c1, ranges, phases), C5=labelled(c5, ranges, phases))
        (out / f"{episode}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"{episode}: {len(c1)} subtasks, {len(confirmed)} confirmed events" + (f"; {repairs}" if repairs else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts", help="one status-log prompt per episode")
    p.add_argument("--captions", required=True, help="caption_merge.py run output folder")
    p.add_argument("--kinematic", required=True)
    p.add_argument("--version", choices=sorted(INSTRUCTIONS), default="v6",
                   help=f"instruction after the log, from {PROMPTS}")
    p.add_argument("--instruction", help="an instruction file instead of --version (its name is the version)")
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
    s.add_argument("--attach", default="", help="C7 setting, e.g. retreat=next,lift=previous (types: "
                                               + ", ".join(TRANSITION_TYPES) + ")")
    s.add_argument("--out")
    e = sub.add_parser("examples", help="markdown: the log next to each model's subtasks, and the segments")
    e.add_argument("--prompts", required=True)
    e.add_argument("--answers", nargs="+", required=True, help="NAME=DIR answer folders")
    e.add_argument("--autolabel", default=os.environ.get("AUTOLABEL_ROOT"))
    e.add_argument("--kinematic", required=True)
    e.add_argument("--tempseg", required=True)
    e.add_argument("--episodes", nargs="+", help="only these episodes")
    e.add_argument("--out", required=True)
    g = sub.add_parser("segments", help="C1 and C5 with the model's instructions, without scores")
    g.add_argument("--prompts", required=True)
    g.add_argument("--answers", required=True, help="answer folder")
    g.add_argument("--kinematic", required=True)
    g.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command in ("ask", "score", "examples") and not args.autolabel:
        ap.error("give --autolabel or set AUTOLABEL_ROOT")
    return {"prompts": prompts, "ask": ask, "score": score, "examples": examples,
            "segments": segments}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
