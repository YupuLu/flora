"""Our prompt for Flora-Captioner, used only by kinseg's captioning calls (FLORA's files are unchanged).

FLORA's prompt is a camera heading (three cameras only), the video blocks and a fixed instruction; an
episode instruction, when given, follows as a JSON line. Ours keeps the heading, the blocks and the
instruction, with two changes:

* the instruction's context sentence names the blocks actually sent (FLORA's always names BEFORE and
  AFTER); with TARGET alone it is dropped;
* episode information is written as plain lines after the instruction: the task brief, the objects,
  the robot kinematics during TARGET and the previous interval's instruction. With ``rules`` the model
  is asked to build an understanding of TARGET from the video and this information first, with one rule
  per item on how to use it. A last line repeats the output format.

Example (three blocks, every item, with rules)::

    BEFORE:
    <video>

    TARGET:
    <video>

    AFTER:
    <video>

    Write one concise, self-contained English instruction for the local operation being attempted in TARGET.
    ... (FLORA's instruction) ...
    Output only the instruction, on one line, without timestamps, execution-quality commentary or the block names.

    Episode information:
    Task brief: Remove the pen from the mug and put it on the table
    Objects: pen; red mug; table
    Robot kinematics during TARGET: Right hand moves; its gripper is open at the start, then closes.
    Previous interval: "Move the right hand toward the pen in the red cup."

    Before writing, build an understanding of TARGET from the video and the information above:
    - Task brief: ...
    Output only the instruction for TARGET, on one line.
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
RULES_HEADING = "Before writing, build an understanding of TARGET from the video and the information above:"
RULES = {
    "brief": ("Task brief: the goal of the whole episode, not proof of what happened. Decide which step of it "
              "TARGET shows, and describe only that step."),
    "objects": ("Objects: call the objects you see by these names; do not mention a listed object that TARGET "
                "does not show."),
    "kinematics": ("Robot kinematics: measured from the robot. The moving hand is the working hand; a gripper "
                   "that closes usually starts holding an object and one that opens ends it, but a gripper can "
                   "close or open on nothing, so the video decides."),
    "previous": ("Previous interval: the step just before TARGET. Keep its object names; if TARGET continues "
                 "the same operation, keep its instruction, otherwise describe the step that follows it."),
}
LABELS = {"brief": "Task brief: ", "objects": "Objects: ", "kinematics": "Robot kinematics during TARGET: ",
          "previous": "Previous interval: "}
OUTPUT_LINE = "Output only the instruction for TARGET, on one line."


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


def information(items: Dict[str, str], rules: bool) -> str:
    """The episode-information lines for the non-empty ``items`` (keys of ``LABELS``), '' if none."""
    present = [key for key in LABELS if items.get(key, "").strip()]
    if not present:
        return ""
    lines = ["Episode information:"]
    for key in present:
        value = items[key].strip()
        lines.append(LABELS[key] + (f'"{value}"' if key == "previous" else value))
    if rules:
        lines += ["", RULES_HEADING] + [f"- {RULES[key]}" for key in present]
    return "\n".join(lines + [OUTPUT_LINE])


def build(sample: Dict[str, object], video_token: str, items: Dict[str, str], rules: bool,
          base: Optional[str] = None) -> str:
    """Our user message for a FLORA caption sample: FLORA's heading and blocks, our instruction and information."""
    from flora_unified.unified_input import prompt as native_prompt

    roles = [b["role"] for b in sample["blocks"]]
    native = native_prompt(sample, video_token)  # validates the sample; heading + blocks + FLORA's instruction
    base = base if base is not None else flora_instruction()
    if not native.endswith(base):
        raise RuntimeError("FLORA's prompt layout changed; update caption_prompt.build")
    text = native[: len(native) - len(base)] + instruction(roles, base)
    extra = information(items, rules)
    return text + ("\n\n" + extra if extra else "")


def objects_line(scene: Optional[Dict[str, object]]) -> str:
    """Objects of a scene file (``target_objects`` then ``other_objects``), '' without one."""
    if not scene:
        return ""
    names: List[str] = list(scene.get("target_objects") or []) + list(scene.get("other_objects") or [])
    return "; ".join(str(n) for n in names)
