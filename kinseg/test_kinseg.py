"""Checks of the merge rules and the tick constraint (CPU; the tokenizer test needs the TempSeg files)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOKENIZER = Path.home() / ".cache" / "flora" / "models" / "tempseg"


def test_group_and_snap():
    from merge import boundaries, group, snap

    phases = [(0, 1), (1, 2.2), (2.2, 3), (3, 5), (5, 6)]
    segments = [(0.1, 2.9), (3.2, 6)]
    assert group(phases, segments) == [(0, 3), (3, 6)]
    assert snap(segments, boundaries(phases), 0.5) == [(0, 3), (3, 6)]
    assert snap(segments, boundaries(phases), 0.05) == segments
    assert group(phases, []) == [(0, 6)]


def test_split_at_grasps():
    from merge import split_at_grasps

    cuts = [0, 2, 4.5, 6, 8, 10]
    events = [("RIGHT", "close", 4.0, 4.2), ("RIGHT", "open", 9.0, 9.1)]
    assert split_at_grasps([(0, 10)], events, cuts, "first") == [(0, 4.5), (4.5, 10)]
    assert split_at_grasps([(0, 10)], events, cuts, "mid") == [(0, 6), (6, 10)]
    # Close and open in different segments, or of different arms: unchanged.
    assert split_at_grasps([(0, 6), (6, 10)], events, cuts) == [(0, 6), (6, 10)]
    other = [("RIGHT", "close", 4.0, 4.2), ("LEFT", "open", 9.0, 9.1)]
    assert split_at_grasps([(0, 10)], other, cuts) == [(0, 10)]


def test_allowed_ticks():
    from segment import allowed_ticks

    window = SimpleNamespace(valid_start_tick=0, valid_end_tick=320, core_start_tick=40, core_end_tick=280,
                             requested_view_start_us=10_000_000)
    # 10 s window origin: 12.0 s -> tick 40, 13.37 s -> 67; 9 s and 30 s lie outside.
    assert allowed_ticks(window, [9.0, 12.0, 13.37, 30.0]) == [0, 40, 67, 280, 320]


def test_beliefs_labels_and_merge():
    from beliefs import auc, boundary_beliefs, cut_belief, labels, merge_phases

    record = {"probes": [{"view_start_us": 10_000_000, "envelope": [0, 320, 40, 280],
                          "beliefs": [{"key": "e", "values": [0, 60, 70, 320], "log_probs": [0.0, 0.0, -10.0, 0.0]}]}]}
    beliefs = boundary_beliefs(record)          # edges 0 and 320 dropped; 60 -> 13.0 s, 70 -> 13.5 s
    assert [round(t, 2) for t, _ in beliefs] == [13.0, 13.5]
    assert cut_belief(13.1, beliefs) > 0.3 and cut_belief(14.0, beliefs) == 0.0
    assert labels([1.0, 2.0, 2.3, 5.0], [(0, 2.2), (2.2, 6)]) == [0, 0, 1, 0]
    assert auc([0.9, 0.1, 0.2], [1, 0, 0]) == 1.0
    assert merge_phases([(0, 1), (1, 2), (2, 3)], [2]) == [(0, 2), (2, 3)]


def test_caption_join():
    from caption_merge import initial_intervals, join_equal, normalise

    assert normalise("Pick up the black bowl.") == normalise("pick up  black bowl") == "pick up black bowl"
    items = [dict(start_s=0, end_s=1, caption="Pick up the bowl.", status="complete", members=[0]),
             dict(start_s=1, end_s=2, caption="pick up bowl", status="complete", members=[1]),
             dict(start_s=2, end_s=2.1, caption="", status="unsupported_target_frames", members=[2]),
             dict(start_s=2.1, end_s=4, caption="Place the bowl on the plate.", status="complete", members=[3])]
    joined = join_equal(items)
    assert [(j["start_s"], j["end_s"], j["members"], j["joined"]) for j in joined] == [
        (0, 2.1, [0, 1, 2], True), (2.1, 4, [3], False)]
    case = {"predictions": [{"start_sec": 0, "end_sec": 1}, {"start_sec": 1, "end_sec": 3}]}
    assert [(i["start_s"], i["end_s"]) for i in initial_intervals(case, None)] == [(0, 1), (1, 3)]
    tempseg = {"segments": [{"start_s": 0.0, "end_s": 2.0}, {"start_s": 2.0, "end_s": 2.9999999}]}
    # The near-duplicate end (2.9999999 against 3) gives no zero-length interval.
    assert [(i["start_s"], i["end_s"]) for i in initial_intervals(case, tempseg)] == [(0, 1), (1, 2.0), (2.0, 3)]


def test_warm_reads_the_checked_files(tmp_path):
    import local_models

    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "b.bin").write_bytes(b"y" * 20)
    (tmp_path / "SOURCE_MANIFEST.sha256").write_text("00 *a.bin\n11 b.bin\n")
    assert [p.name for p in local_models.checked_files(tmp_path)] == ["a.bin", "b.bin", "SOURCE_MANIFEST.sha256"]
    local_models.warm(tmp_path, settle_s=0.01, rounds=2)  # stable files: returns after one check


def test_memory_rule():
    from local_models import enough_memory

    assert enough_memory("tempseg", [30.0, 0.0]) and not enough_memory("tempseg", [20.0, 60.0])
    assert enough_memory("captioner", [35.0, 35.0]) and not enough_memory("captioner", [60.0, 23.5])
    assert not enough_memory("captioner", [])


def test_load_retries_only_the_warming_failure(monkeypatch, tmp_path):
    import functools
    import local_models

    monkeypatch.setattr(local_models, "wait_for_memory", lambda kind: None)
    monkeypatch.setattr(local_models, "free_gib", lambda: [80.0, 80.0])
    monkeypatch.setattr(local_models, "load", functools.partial(local_models.load, lock_file=tmp_path / "lock"))

    class Session:
        def __init__(self, failures, message):
            self.failures, self.message, self.calls = failures, message, 0

        def load(self, kind):
            self.calls += 1
            if self.calls <= self.failures:
                raise RuntimeError(self.message)
            return ("model", "proc")

    warming = Session(1, "A model asset changed during its integrity check")
    assert local_models.load(warming, "captioner", wait_s=0) == ("model", "proc") and warming.calls == 2
    corrupt = Session(1, "A model asset failed its integrity check")
    with pytest.raises(RuntimeError, match="failed its integrity check"):
        local_models.load(corrupt, "captioner", wait_s=0)
    assert corrupt.calls == 1


def test_load_checks_memory_again_under_the_lock(monkeypatch, tmp_path):
    import local_models

    waits, free = [], iter([[10.0, 10.0], [80.0, 80.0]])   # taken by another load, then free again
    monkeypatch.setattr(local_models, "wait_for_memory", lambda kind: waits.append(kind))
    monkeypatch.setattr(local_models, "free_gib", lambda: next(free))

    class Session:
        def load(self, kind):
            return ("model", "proc")

    assert local_models.load(Session(), "captioner", lock_file=tmp_path / "lock") == ("model", "proc")
    assert waits == ["captioner", "captioner"]


def test_one_load_at_a_time(tmp_path):
    import threading
    from local_models import one_load_at_a_time

    order = []

    def second():
        with one_load_at_a_time(tmp_path / "lock"):
            order.append("second")

    with one_load_at_a_time(tmp_path / "lock"):
        thread = threading.Thread(target=second)
        thread.start()
        thread.join(0.3)
        order.append("first done")
    thread.join(5)
    assert order == ["first done", "second"]


def test_background_text():
    from background import BACKGROUND_GUIDANCE, PREVIOUS_GUIDANCE, background_text

    assert background_text("none") == ""
    brief = background_text("task_brief", "Move the bowl to the right")
    assert brief == "Background:\nTask brief: Move the bowl to the right\n\n" + BACKGROUND_GUIDANCE
    with_previous = background_text("task_brief", "Move the bowl", previous="Pick up the black bowl.")
    assert 'Previous interval was described as: "Pick up the black bowl."' in with_previous
    assert with_previous.endswith(PREVIOUS_GUIDANCE) and "repeat" not in with_previous
    scene = {"target_objects": ["black bowl"], "other_objects": [], "workspace": "tabletop"}
    full = background_text("task_and_scene", "Move the bowl", scene)
    assert "Target objects: black bowl.\nOther objects: none specified.\nWorkspace: tabletop." in full
    with pytest.raises(ValueError):
        background_text("task_brief", "")
    with pytest.raises(ValueError):
        background_text("task_brief", "x" * 2000)


def test_robot_state():
    from background import ROBOT_GUIDANCE, background_text
    from caption_merge import robot_state

    def event(arm, kind, t):
        return dict(arm=arm, kind=kind, start_sec=t, end_sec=t + 0.2, source="measured")

    two_arms = {"predictions": [
        dict(start_sec=0, end_sec=2, arm="RIGHT", gripper_events=[event("RIGHT", "close", 1.5)]),
        dict(start_sec=2, end_sec=4, arm="BOTH", gripper_events=[event("LEFT", "close", 3.0)]),
        dict(start_sec=4, end_sec=6, arm="STABLE", gripper_events=[event("RIGHT", "open", 5.0)])]}
    assert robot_state(two_arms, 0, 2) == ("At the start of TARGET the left gripper is open and the right gripper "
                                           "is open. During TARGET the right gripper closes. The right arm moves.")
    assert robot_state(two_arms, 2, 4).endswith("During TARGET the left gripper closes. Both arms move.")
    assert robot_state(two_arms, 4, 6) == ("At the start of TARGET the left gripper is closed and the right gripper "
                                           "is closed. During TARGET the right gripper opens. Neither arm moves much.")
    one_arm = {"predictions": [dict(start_sec=0, end_sec=2, arm="RIGHT", gripper_events=[event("RIGHT", "close", 1.0)]),
                               dict(start_sec=2, end_sec=3, arm="STABLE", gripper_events=[])]}
    assert robot_state(one_arm, 2, 3) == ("At the start of TARGET the gripper is closed. "
                                          "No gripper opens or closes during TARGET. The arm hardly moves.")
    text = background_text("task_brief", "Put the cup on the tray", robot=robot_state(one_arm, 2, 3))
    assert "\nRobot state from kinematics: At the start of TARGET the gripper is closed." in text
    assert text.endswith(ROBOT_GUIDANCE)


def test_caption_prompt_template():
    from caption_prompt import FLORA_CONTEXT, NOTES_HEADING, information, instruction
    from caption_merge import kinematics

    base = "Infer the local goal. " + FLORA_CONTEXT + " A failed attempt keeps its instruction."
    assert instruction(["BEFORE", "TARGET", "AFTER"], base) == base
    assert instruction(["TARGET", "AFTER"], base) == ("Infer the local goal. Use AFTER only to disambiguate TARGET, "
                                                      "not to add operations occurring outside it. A failed attempt "
                                                      "keeps its instruction.")
    assert instruction(["TARGET"], base) == "Infer the local goal. A failed attempt keeps its instruction."
    assert information({}, notes=True) == ""
    text = information(dict(brief="Move the cup", previous="Reach for the cup.", objects=""), notes=True)
    assert text == ('Episode information:\nTask brief: Move the cup\nPrevious interval: "Reach for the cup."\n'
                    + NOTES_HEADING + "\n- The task brief is the goal of the whole episode; describe only the step "
                    "in TARGET.\n- Stay consistent with the previous interval, or continue from it.")
    assert NOTES_HEADING not in information(dict(brief="Move the cup"), notes=False)

    def event(arm, kind, t):
        return dict(arm=arm, kind=kind, start_sec=t, end_sec=t + 0.2, source="measured")

    two_arms = {"predictions": [
        dict(start_sec=0, end_sec=2, arm="RIGHT", gripper_events=[event("RIGHT", "close", 1.5)]),
        dict(start_sec=2, end_sec=4, arm="STABLE", gripper_events=[event("RIGHT", "open", 3.0)]),
        dict(start_sec=4, end_sec=6, arm="LEFT", gripper_events=[])]}
    assert kinematics(two_arms, 0, 2) == ("Left hand is still; its gripper stays open. "
                                          "Right hand moves; its gripper is open at the start, then closes.")
    assert kinematics(two_arms, 2, 4) == ("Left hand is still; its gripper stays open. "
                                          "Right hand is still; its gripper is closed at the start, then opens.")
    one_arm = {"predictions": [dict(start_sec=0, end_sec=2, arm="RIGHT", gripper_events=[event("RIGHT", "close", 1.0)])]}
    assert kinematics(one_arm, 0, 2) == "Right hand moves; its gripper is open at the start, then closes."


def test_echoes():
    from caption_merge import echoes

    assert echoes("Move the bowl to the right with the right hand.", "Move the bowl to the right")
    assert not echoes("Pick up the black bowl with the right hand.", "Move the bowl to the right")
    assert not echoes("place it", "go")  # briefs under three words are not judged


def test_context_roles_are_valid_flora_layouts():
    from caption_merge import CONTEXT_ROLES

    # FLORA's sample validator accepts exactly these role orders.
    allowed = [["TARGET"], ["BEFORE", "TARGET"], ["TARGET", "AFTER"], ["BEFORE", "TARGET", "AFTER"]]
    assert all(list(roles) in allowed for roles in CONTEXT_ROLES.values())


def test_operation_join():
    from caption_merge import join_operations, limb_motion, operation

    assert operation("Pick up the pink test tube.") == "pick up" and operation("place the marker") == "place"
    assert limb_motion("move the left hand back.") and limb_motion("Raise both hands.")
    assert not limb_motion("move the pink stick to the right")
    caps = ["raise both hands.", "move the left hand toward the pink stick", "pick up the pink cylindrical object",
            "pick up the pink test tube", "place the pink marker vertically", "place the pink marker below",
            "move the left hand back."]
    items = [dict(start_s=float(k), end_s=float(k + 1), caption=c) for k, c in enumerate(caps)]
    joined = join_operations(items)
    assert [(j["start_s"], j["end_s"], j["key"]) for j in joined] == [(0.0, 4.0, "pick up"), (4.0, 7.0, "place")]
    from caption_merge import split_by_operation
    # One TempSeg segment over pick-up and place phases is split where the operation changes.
    assert split_by_operation([(0.2, 7.0)], items) == [(0.2, 4.0), (4.0, 7.0)]
    assert split_by_operation([(0.0, 3.5), (3.5, 7.0)], items) == [(0.0, 3.5), (3.5, 4.0), (4.0, 7.0)]
    from caption_merge import motion_only
    # Moving a held object also joins the next operation: pick up | lift, move, put | (retract).
    caps = ["pick up the pen from the red cup", "lift the pen out of the red cup", "move the pen away from the cup",
            "put the pen on the table", "move the right hand away from the pen"]
    items = [dict(start_s=float(k), end_s=float(k + 1), caption=c) for k, c in enumerate(caps)]
    assert [(j["start_s"], j["end_s"]) for j in join_operations(items)] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0),
                                                                          (3.0, 5.0)]
    assert [(j["start_s"], j["end_s"], j["key"]) for j in join_operations(items, motion_only)] == [
        (0.0, 1.0, "pick up"), (1.0, 5.0, "put")]


def test_caption_examples_page():
    from caption_examples import episode_page

    case = {"family": "droid", "duration_sec": 3.0, "instruction": "Move the cup",
            "references": [dict(start_sec=0.0, end_sec=1.5, subtask="pick up the cup"),
                           dict(start_sec=1.5, end_sec=3.0, subtask="place the cup")],
            "predictions": [dict(start_sec=0.0, end_sec=1.5, arm="RIGHT", gripper_events=[
                                dict(arm="RIGHT", kind="close", start_sec=1.0, end_sec=1.2)]),
                            dict(start_sec=1.5, end_sec=3.0, arm="STABLE", gripper_events=[])]}
    rounds = lambda *caps: {"rounds": [[dict(start_s=a, end_s=b, caption=c, status="complete")  # noqa: E731
                                        for (a, b), c in zip([(0.0, 1.5), (1.5, 3.0)], caps)]]}
    records = {"A": rounds("Pick up the cup.", "Place the cup."), "B": rounds("pick up the cup", "Push the cup.")}
    exact = lambda pred, ref, threshold: [(i, j) for i, p in enumerate(pred)  # noqa: E731
                                          for j, r in enumerate(ref) if p == r]
    page, index = episode_page("ep", case, records, None, lambda pred, ref: list(pred), exact, False, None)
    assert "| 0 | 0.0–1.5 | right | R close | ▸ pick up the cup | Pick up the cup. | = |" in page
    assert "| 1 | 1.5–3.0 | – |  | ▸ place the cup | Place the cup. | Push the cup. |" in page
    assert "`0.0–1.5` pick up ✓<br>`1.5–3.0` place ✓" in page
    assert index["A · operation join"] == "1.00 (2/2/2)"


def test_combine_log_answer_and_events():
    from combine import confirmed_events, episode_log, parse_answer, subtask_ranges

    case = {"instruction": "Move the cup", "references": [],
            "predictions": [dict(start_sec=0.0, end_sec=1.0, arm="RIGHT", gripper_events=[
                                dict(arm="RIGHT", kind="close", start_sec=0.5, end_sec=0.7)]),
                            dict(start_sec=1.0, end_sec=2.0, arm="STABLE", gripper_events=[]),
                            dict(start_sec=2.0, end_sec=3.0, arm="RIGHT", gripper_events=[
                                dict(arm="RIGHT", kind="open", start_sec=2.5, end_sec=2.7)])]}
    phases = [dict(start_s=0.0, end_s=1.0, caption="Pick up the cup."), dict(start_s=1.0, end_s=2.0, caption="Hold it."),
              dict(start_s=2.0, end_s=3.0, caption="Put the cup down.")]
    log = episode_log(case, phases)
    assert "Robot: one arm (right hand)." in log
    assert "0   0.0-1.0  moves, closes" in log and "1   1.0-2.0  still" in log and "| Put the cup down." in log
    answer = parse_answer('<think>x</think>```json\n{"gripper": [{"phase": 0, "event": "close", "is": "grasp"},'
                          ' {"phase": 2, "event": "open", "is": "none"}],'
                          ' "subtasks": [{"phases": [0, 0], "instruction": "Pick up"}, {"phases": [2, 2]}]}\n```')
    ranges, repairs = subtask_ranges(answer, 3)
    assert ranges == [(0, 1, "Pick up"), (2, 2, "")] and repairs == ["gap 1-1 joined to the previous subtask"]
    assert [e[1] for e in confirmed_events(answer, case, phases)] == ["close"]  # the open did nothing
    assert parse_answer("no json here") is None
    assert subtask_ranges({"subtasks": [{"phases": [1, 5]}, {"phases": [0, 2]}]}, 4)[0] == [(0, 2, ""), (3, 3, "")]


def test_anchored_cuts():
    from combine import anchored

    phases = [dict(start_s=float(k), end_s=float(k + 1)) for k in range(6)]
    close = ("RIGHT", "close", 1.5, 1.7)  # in phase 1
    # The model cut after phase 2 (subtasks 0-2, 3-5); the grasp is in phase 1, so the cut moves to 2.0.
    assert anchored([(0, 2, ""), (3, 5, "")], phases, [close]) == [(0.0, 2.0), (2.0, 6.0)]
    # A cut two phases away is left alone, and opens are not anchors by default.
    assert anchored([(0, 3, ""), (4, 5, "")], phases, [close]) == [(0.0, 4.0), (4.0, 6.0)]
    assert anchored([(0, 2, ""), (3, 5, "")], phases, [("RIGHT", "open", 1.5, 1.7)]) == [(0.0, 3.0), (3.0, 6.0)]


def test_transition_regions_and_forgiving():
    from transition_eval import forgive, regions

    case = {"references": [dict(start_sec=0.0, end_sec=4.2), dict(start_sec=4.2, end_sec=8.0)],
            "predictions": [dict(start_sec=0.0, end_sec=3.0, arm="RIGHT", gripper_events=[
                                dict(arm="RIGHT", kind="open", start_sec=2.5, end_sec=2.8)]),
                            dict(start_sec=3.0, end_sec=5.0, arm="RIGHT", gripper_events=[]),
                            dict(start_sec=5.0, end_sec=8.0, arm="RIGHT", gripper_events=[
                                dict(arm="RIGHT", kind="close", start_sec=6.0, end_sec=6.2)])]}
    # The cut at 4.2 s lies in phase 3.0-5.0, which has no gripper event; with one neighbour each side the
    # region still stops at the open (2.8 s) and the close (6.0 s).
    assert regions(case) == [(4.2, (3.0, 5.0))]
    assert regions(case, neighbours=1) == [(4.2, (2.8, 6.0))]
    assert forgive([(0.0, 3.0), (3.0, 8.0)], regions(case)) == [(0.0, 4.2), (4.2, 8.0)]
    assert forgive([(0.0, 5.5), (5.5, 8.0)], regions(case)) == [(0.0, 5.5), (5.5, 8.0)]  # outside: unchanged


def test_reattach_transitions():
    from combine import parse_attach, reattach

    # pick up 0-2 (phase 2 a lift) | place 3-5 (phase 3 a retreat? no: carry) | pick up 6-8 (phase 6 a retreat)
    ranges = [(0, 2, "pick"), (3, 5, "place"), (6, 8, "pick")]
    kinds = {2: "lift", 3: "carry", 6: "retreat"}
    assert reattach(ranges, kinds, {}) == ranges
    assert reattach(ranges, kinds, parse_attach("lift=next")) == [(0, 1, "pick"), (2, 5, "place"), (6, 8, "pick")]
    assert reattach(ranges, kinds, parse_attach("retreat=previous")) == [(0, 2, "pick"), (3, 6, "place"),
                                                                        (7, 8, "pick")]
    assert reattach([(0, 0, "a"), (1, 1, "b")], {0: "lift"}, {"lift": "next"}) == [(0, 0, "a"), (1, 1, "b")]


def test_grid_ticks():
    from segment import grid_ticks

    interior = SimpleNamespace(valid_start_tick=0, valid_end_tick=320, core_start_tick=40, core_end_tick=280)
    assert grid_ticks(interior) == list(range(0, 321, 10))
    last = SimpleNamespace(valid_start_tick=0, valid_end_tick=197, core_start_tick=40, core_end_tick=197)
    assert grid_ticks(last)[-3:] == [180, 190, 197]


def test_candidates_keep_rows_valid():
    from segment import candidates

    ticks = [0, 40, 67, 120, 280, 320]
    assert candidates(ticks, "s", [], 320) == [0, 40, 67, 120, 280]       # a start before the window end
    assert candidates(ticks, "s", [40], 320) == [40, 67, 120, 280]        # not before the previous row
    assert candidates(ticks, "e", [40, 67], 320) == [120, 280, 320]       # after this row's start
    assert candidates(ticks, "e", [320], 320) == [320]                    # never empty


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="TempSeg tokenizer not downloaded")
def test_number_detection():
    from transformers import AutoTokenizer
    from decoding import NUMBER_START, ROW_START, _digit_ids

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    assert len(_digit_ids(tok)) == 10
    text = '{"segments":[{"s":40,"e":160,"lo":false,"ro":false},{"s":'
    assert NUMBER_START.search(text).group(1) == "s"
    assert [int(s) for s in ROW_START.findall(text)] == [40]
    assert NUMBER_START.search('{"segments":') is None and NUMBER_START.search('"lo":') is None


def test_nero_despike():
    import numpy as np
    from nero import despike

    u = np.array([0.18, 0.18, 1.0, 1.0, 1.0, 0.18, 0.2, 0.42, 1.0, 1.0, 0.92, 1.0, 0.75, 0.23, 1.0, 0.2])
    out = despike(u)
    assert np.isnan(out[2:5]).all() and np.isnan(out[14])        # 1-3 frame jumps between low values
    assert np.isfinite(out[8:12]).all()                           # a real opening: next value 0.92
    assert np.isfinite(despike(np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))).all()   # four frames: kept
    assert np.isfinite(despike(np.array([1.0, 1.0, 0.0]))).all()  # at the start: no neighbour before


def test_caption_prompt_three_cameras():
    from caption_prompt import FLORA_CONTEXT, build_three

    heading = "Each frame is a camera mosaic.\n\n"
    base = "Infer the local goal. " + FLORA_CONTEXT + " A failed attempt keeps its instruction."
    block = lambda role, ticks: dict(role=role, pts_ticks=ticks)   # noqa: E731
    plan = dict(time_base="1/15", blocks=[block("BEFORE", [0, 15]), block("TARGET", [30, 45]), block("AFTER", [60])])
    flora = (heading + "BEFORE:\nFrames are in this order; native source timestamps in seconds: "
             "[0.000000000, 1.000000000].\n<V>TARGET:\nFrames are in this order; native source timestamps in "
             "seconds: [2.000000000, 3.000000000].\n<V>AFTER:\nFrames are in this order; native source "
             "timestamps in seconds: [4.000000000].\n<V>")
    assert build_three(plan, "<V>", {}, False, heading, base) == flora + base   # FLORA's own text
    text = build_three(plan, "<V>", dict(brief="Move the cup"), False, heading, base)
    assert text == flora + "Episode information:\nTask brief: Move the cup\n\n" + base
    two = dict(plan, blocks=plan["blocks"][1:])
    assert "Use AFTER only to disambiguate TARGET" in build_three(two, "<V>", {}, False, heading, base)
