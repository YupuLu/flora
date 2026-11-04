"""Background text for Flora-Captioner's optional episode-instruction field (our version).

Based on the background conditions of the remote workspace
(``workspace/benchmark/caption_controls.py`` and ``scene_background.py``),
whose wording is copied unchanged so results stay comparable with its A0 runs:

* ``none``: no text (FLORA's default workflow, condition A);
* ``task_brief``: ``Task brief: ...`` (condition B);
* ``scene_context``: target objects, other objects and workspace (condition C);
* ``task_and_scene``: both (condition D).

Each non-empty background ends with the workspace's guidance, which tells the
model that the background is intent, not observation, and only disambiguates
the operation visible in TARGET. Our additions:

* ``robot``: the arm motion and gripper state of TARGET measured from the
  robot's kinematics (``caption_merge.robot_state``), with a guidance line that
  a gripper can close without grasping (as autolabel's VL prompt warns that a
  gripper event is not a grasp or a release);
* ``previous``: the caption of the previous interval, with a guidance line that
  limits it to keeping object and action names consistent (it does not ask the
  model to repeat it).

The Captioner's own prompt and the BEFORE/TARGET/AFTER video blocks are not
changed here; which video blocks are sent is chosen separately.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Copied from workspace/benchmark/caption_controls.py (BACKGROUND_GUIDANCE).
BACKGROUND_GUIDANCE = (
    'Background describes the intended task and available objects, not proven execution. '
    'Use it only to disambiguate the visible local operation. '
    'Keep the native one-line action instruction format; do not assume success, '
    'invent invisible actions, or describe operations outside TARGET.'
)
ROBOT_GUIDANCE = (
    'Robot state comes from joint and gripper readings: a gripper can close without grasping '
    'and open without releasing anything, so use the video to decide what is held.'
)
PREVIOUS_GUIDANCE = (
    'The previous description only keeps object and action names consistent; '
    'describe only the operation in TARGET.'
)
MODES = ("none", "task_brief", "scene_context", "task_and_scene")
LIMIT = 2000  # FLORA's limit for the episode-instruction field


def background_text(mode: str, task_brief: str = "", scene: Optional[Dict[str, object]] = None,
                    previous: str = "", robot: str = "") -> str:
    """The episode-instruction text for ``mode``, optionally with the robot state and the previous caption."""
    if mode not in MODES:
        raise ValueError(f"unknown background mode {mode!r}; expected one of {MODES}")
    lines: List[str] = []
    if mode in ("task_brief", "task_and_scene"):
        if not task_brief.strip():
            raise ValueError(f"{mode} needs a task brief")
        lines.append("Task brief: " + task_brief.strip())
    if mode in ("scene_context", "task_and_scene"):
        scene = scene or {}
        targets, others = scene.get("target_objects"), scene.get("other_objects", [])
        workspace = scene.get("workspace", "")
        if not targets or not isinstance(targets, list) or not str(workspace).strip():
            raise ValueError(f"{mode} needs target_objects and workspace")
        lines += ["Target objects: " + "; ".join(targets) + ".",
                  "Other objects: " + ("; ".join(others) if others else "none specified") + ".",
                  "Workspace: " + str(workspace).strip() + "."]
    if robot.strip():
        lines.append("Robot state from kinematics: " + robot.strip())
    if previous.strip():
        lines.append(f'Previous interval was described as: "{previous.strip()}"')
    if not lines:
        return ""
    text = "\n".join(["Background:"] + lines + ["", BACKGROUND_GUIDANCE]
                     + ([ROBOT_GUIDANCE] if robot.strip() else [])
                     + ([PREVIOUS_GUIDANCE] if previous.strip() else []))
    if len(text) > LIMIT:
        raise ValueError(f"background exceeds FLORA's {LIMIT}-character limit")
    return text
