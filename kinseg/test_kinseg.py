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
