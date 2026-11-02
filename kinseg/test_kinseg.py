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


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="TempSeg tokenizer not downloaded")
def test_constraint_forces_allowed_numbers():
    import torch
    from transformers import AutoTokenizer
    from segment import TickConstraint

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    prompt = tok("prompt", add_special_tokens=False)["input_ids"]
    vocab = len(tok)

    def greedy(prefer: str, allowed):
        """Decode with logits that always prefer the characters of ``prefer`` in order."""
        ids = list(prompt)
        target = tok(prefer, add_special_tokens=False)["input_ids"]
        constraint = TickConstraint(tok, len(prompt), allowed)
        for step in range(len(target) + 4):
            scores = torch.zeros(1, vocab)
            want = target[step] if step < len(target) else tok.convert_tokens_to_ids(",")
            scores[0, want] = 10.0
            for d in range(10):  # mild preference order among digits
                scores[0, tok.convert_tokens_to_ids(str(d))] += 0.1 * (9 - d)
            ids.append(int(constraint(torch.tensor([ids]), scores).argmax()))
            text = tok.decode(ids[len(prompt):])
            if text.endswith(","):
                return text
        return tok.decode(ids[len(prompt):])

    # The model wants 160; only 137 and 158 are allowed.
    out = greedy('{"s":160', [137, 158])
    assert out.startswith('{"s":1') and out.rstrip(",")[-3:] in {"137", "158"}
    # An allowed value passes through unchanged.
    assert greedy('{"s":40', [40, 158]).startswith('{"s":40,')
