"""Greedy TempSeg decoding in which each ``s``/``e`` value is chosen whole from a candidate list.

TempSeg writes its reply as JSON, one token at a time, numbers digit by digit.
This loop decodes greedily like ``model.generate`` and, whenever the model is
about to write the number after ``"s":`` or ``"e":``, scores every allowed
value as a whole instead of letting it write digits:

    log P(v) = sum of log P(digit) over the digits of v + log P(the number ends after v)

where "ends" is the probability of any non-digit token. All candidates are
scored in one batched forward pass from a copy of the cache; the best value is
written, a digit right after it is forbidden, and greedy decoding resumes.
Every scored candidate list is returned, so the probabilities can also be
read as evidence for or against a cut at each candidate.

With ``allowed=None`` the loop is plain greedy decoding and reproduces
``model.generate`` token for token, which is how it is checked. ``probe``
keeps decoding free and only records, at each number, the log-probability of
a given list of values (TempSeg's 0.5 s grid), i.e. where the model believes
that start or end could lie.
"""

from __future__ import annotations

import copy
import re
from typing import Callable, Dict, List, Optional, Sequence

import torch

NUMBER_START = re.compile(r'"([se])":$')
ROW_START = re.compile(r'"s":(\d+)')
WRITTEN = re.compile(r'"[se]":(\d+)')


def _digit_ids(tokenizer) -> List[int]:
    ids = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
    assert len(set(ids)) == 10, "digits must be single tokens"
    return ids


def score_values(model, cache, logits: torch.Tensor, values: Sequence[int], digit_ids: Sequence[int]) -> List[float]:
    """Log-probability of writing each value whole, followed by a non-digit token."""
    seqs = [[digit_ids[int(c)] for c in str(v)] for v in values]
    width = max(len(s) for s in seqs)
    batch = torch.tensor([s + [s[-1]] * (width - len(s)) for s in seqs], device=logits.device)
    branch = copy.deepcopy(cache)
    branch.reorder_cache(torch.zeros(len(values), dtype=torch.long, device=logits.device))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids=batch, past_key_values=branch, use_cache=True)
    first = torch.log_softmax(logits.float(), dim=-1)                  # (vocab,) before the first digit
    later = torch.log_softmax(out.logits.float(), dim=-1)              # (n, width, vocab) after each digit
    digits = torch.tensor(digit_ids, device=logits.device)
    scores = []
    for i, seq in enumerate(seqs):
        total = first[seq[0]].item()
        for j in range(1, len(seq)):
            total += later[i, j - 1, seq[j]].item()
        p_digit = later[i, len(seq) - 1, digits].exp().sum().clamp(max=1 - 1e-12)
        scores.append(total + torch.log1p(-p_digit).item())
    return scores


def generate(model, tokenizer, inputs: Dict[str, torch.Tensor], budget: int,
             allowed: Optional[Callable[[str, List[int]], List[int]]] = None,
             record: Optional[list] = None,
             probe: Optional[Callable[[str], List[int]]] = None) -> str:
    """Greedy decoding; ``allowed(key, row_starts)`` gives the candidate values for ``key`` in ``"se"``.

    With ``probe(key)`` instead, decoding stays free (the reply is unchanged) and, at each
    number, the log-probability of every value in ``probe(key)`` is recorded together with
    the value the model then writes.
    """
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items() if k != "video_metadata"}
    eos = tokenizer.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    digit_ids = _digit_ids(tokenizer)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**inputs, use_cache=True, logits_to_keep=1)
    cache, logits = out.past_key_values, out.logits[0, -1]
    del out
    generated: List[int] = []
    block_digits = False

    def step(token_ids: List[int]):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = model(input_ids=torch.tensor([token_ids], device=device), past_key_values=cache, use_cache=True)
        return res.logits[0, -1]

    probed: List[dict] = []
    while len(generated) < budget:
        text = tokenizer.decode(generated, skip_special_tokens=True)
        match = NUMBER_START.search(text) if (allowed is not None or probe is not None) else None
        if match and allowed is None:
            values = probe(match.group(1))
            scores = score_values(model, cache, logits, values, digit_ids)
            probed.append(dict(key=match.group(1), at=len(generated), values=list(values),
                               log_probs=[round(s, 4) for s in scores]))
            match = None
        if match:
            starts = [int(s) for s in ROW_START.findall(text)]
            values = allowed(match.group(1), starts)
            scores = score_values(model, cache, logits, values, digit_ids)
            best = max(range(len(values)), key=scores.__getitem__)
            if record is not None:
                record.append(dict(key=match.group(1), at=len(generated), values=list(values),
                                   log_probs=[round(s, 4) for s in scores], chosen=values[best]))
            tokens = [digit_ids[int(c)] for c in str(values[best])]
            generated += tokens
            logits = step(tokens)
            block_digits = True
            continue
        scores = logits.float().clone()
        if block_digits:
            scores[digit_ids] = float("-inf")
            block_digits = False
        token = int(scores.argmax())
        generated.append(token)
        if token in eos:
            break
        logits = step([token])
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if probed:
        written = [int(v) for v in WRITTEN.findall(text)]
        for entry, value in zip(probed, written):
            entry["written"] = value
        if record is not None:
            record.extend(probed)
    return text
