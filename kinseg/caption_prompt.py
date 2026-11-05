"""Our prompt for Flora-Captioner, used only by kinseg's captioning calls (FLORA's files are unchanged).

FLORA's prompt is a camera heading (three cameras only), the video blocks and a fixed instruction; an
episode instruction, when given, follows as a JSON line. Ours keeps the heading, the blocks and the
instruction, with two changes:

* episode information is written as plain lines between the video blocks and the instruction: the task
  brief, the objects, the robot kinematics during TARGET and the previous interval's instruction, and,
  with ``notes``, one short note per item on how to use it;
* the instruction's context sentence names the blocks actually sent (FLORA's always names BEFORE and
  AFTER); with TARGET alone it is dropped.

Example (three blocks, every item, with notes)::

    BEFORE:
    <video>

    TARGET:
    <video>

    AFTER:
    <video>

    Episode information:
    Task brief: Remove the pen from the mug and put it on the table
    Objects: pen; mug; table
    Robot kinematics during TARGET: Right hand moves; its gripper is open at the start, then closes.
    Previous interval: "Move the right hand toward the pen in the red cup."
    Notes: use this information to understand TARGET.
    - The task brief is the goal of the whole episode; describe only the step in TARGET.
    - Name objects with the names in the object list.
    - Use the robot kinematics to confirm the action.
    - Stay consistent with the previous interval, or continue from it.

    Write one concise, self-contained English instruction for the local operation being attempted in TARGET.
    ... (FLORA's instruction, ending with its output line)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

FLORA_CONTEXT = "Use BEFORE and AFTER only to disambiguate TARGET, not to add operations occurring outside it."
CONTEXT_SENTENCE = {
    ("BEFORE", "TARGET", "AFTER"): FLORA_CONTEXT,
    ("BEFORE", "TARGET"): "Use BEFORE only to disambiguate TARGET, not to add operations occurring outside it.",
    ("TARGET", "AFTER"): "Use AFTER only to disambiguate TARGET, not to add operations occurring outside it.",
    ("TARGET",): "",
}
NOTES_HEADING = "Notes: use this information to understand TARGET."
NOTES = {
    "brief": "The task brief is the goal of the whole episode; describe only the step in TARGET.",
    "objects": "Name objects with the names in the object list.",
    "kinematics": "Use the robot kinematics to confirm the action.",
    "previous": "Stay consistent with the previous interval, or continue from it.",
}
LABELS = {"brief": "Task brief: ", "objects": "Objects: ", "kinematics": "Robot kinematics during TARGET: ",
          "previous": "Previous interval: "}


def flora_instruction() -> str:
    """FLORA's fixed caption instruction (from its checksum-verified vendored code)."""
    import segment  # noqa: F401  (puts FLORA's batch_utils on sys.path)
    from batch_utils import bootstrap  # noqa: F401  (puts the vendored modules on sys.path)
    from flora_unified.unified_input import INTENT_INSTRUCTION
    return INTENT_INSTRUCTION


def instruction(roles: Sequence[str], base: Optional[str] = None) -> str:
    """FLORA's instruction with the context sentence for the blocks ``roles``."""
    base = base if base is not None else flora_instruction()
    if FLORA_CONTEXT not in base:
        raise RuntimeError("FLORA's caption instruction changed; update caption_prompt.FLORA_CONTEXT")
    sentence = CONTEXT_SENTENCE[tuple(roles)]
    return base.replace(FLORA_CONTEXT + " ", sentence + " " if sentence else "")


def information(items: Dict[str, str], notes: bool) -> str:
    """The episode-information lines for the non-empty ``items`` (keys of ``LABELS``), '' if none."""
    present = [key for key in LABELS if items.get(key, "").strip()]
    if not present:
        return ""
    lines = ["Episode information:"]
    for key in present:
        value = items[key].strip()
        lines.append(LABELS[key] + (f'"{value}"' if key == "previous" else value))
    if notes:
        lines += [NOTES_HEADING] + [f"- {NOTES[key]}" for key in present]
    return "\n".join(lines)


def build(sample: Dict[str, object], video_token: str, items: Dict[str, str], notes: bool,
          base: Optional[str] = None) -> str:
    """Our user message for a FLORA caption sample: FLORA's heading and blocks, our information, our instruction."""
    from flora_unified.unified_input import prompt as native_prompt

    roles = [b["role"] for b in sample["blocks"]]
    native = native_prompt(sample, video_token)  # validates the sample; heading + blocks + FLORA's instruction
    base = base if base is not None else flora_instruction()
    if not native.endswith(base):
        raise RuntimeError("FLORA's prompt layout changed; update caption_prompt.build")
    extra = information(items, notes)
    return native[: len(native) - len(base)] + (extra + "\n\n" if extra else "") + instruction(roles, base)


def objects_line(scene: Optional[Dict[str, object]]) -> str:
    """Objects of a scene file (``target_objects`` then ``other_objects``), '' without one."""
    if not scene:
        return ""
    names: List[str] = list(scene.get("target_objects") or []) + list(scene.get("other_objects") or [])
    return "; ".join(str(n) for n in names)
