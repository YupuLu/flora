# Kinematic segmentation with Flora-TempSeg

Exploration of combining autolabel's kinematic activity phases
(`manip/prep/autolabel`) with Flora-TempSeg, scored on the robot episodes of
WGO-Bench with autolabel's scorer (`modules/exp_wgo_bench/score.py`, the
benchmark's protocol: snapped outer edges, one-to-one matching, F1 at
tIoU ≥ 0.75).

The kinematic phases cut where the robot's motion changes. They contain most
reference boundaries (merge recall@0.75 0.805 on Galaxea, 0.940 on DROID) but
are 3–6 times finer than WGO's subtasks. TempSeg groups the video into
subtasks; its cuts lie on a 0.5 s grid in practice.

| File | Purpose |
|---|---|
| `segment.py` | TempSeg segmentation only (no Captioner), model loaded once, raw window replies kept; `--kinematic <run>` chooses each `s`/`e` among the episode's kinematic boundary ticks |
| `decoding.py` | Greedy decoding with whole-value choice of `s`/`e` and their log-probabilities |
| `check_decoding.py` | GPU check that `decoding.py` without candidates reproduces TempSeg's `generate` |
| `merge.py` | Post-hoc combination rules and their scores |
| `test_kinseg.py` | CPU checks of the rules and the candidate lists |

## Setup

TempSeg needs the Qwen3.5-9B base (19.3 GB) and the adapter (0.35 GB); the
Captioner (about 108 GB) is not used. Both repositories are public.

```bash
mamba create -n flora python=3.12 -y
conda activate flora
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r tempseg/batch_utils/requirements.txt pytest
```

The weights download on first use (base into the Hugging Face cache, adapter
into `~/.cache/flora/models/tempseg`, or `$FLORA_MODEL_CACHE`). One episode
needs 18.4 GB of GPU memory; without the optional `flash-linear-attention`
kernels a 16 s window takes 5–25 s on an H20.

## Usage

```bash
# autolabel: kinematic phases of WGO-Bench (writes outputs/exp_wgo_bench/<run>/)
python modules/exp_wgo_bench/run.py --label kinseg-base
# TempSeg, free and constrained
CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --jobs wgo_jobs.jsonl --output-dir out/tempseg
CUDA_VISIBLE_DEVICES=7 python kinseg/segment.py --jobs wgo_jobs.jsonl --kinematic <run> --output-dir out/constrained
# scores of every rule
python kinseg/merge.py --autolabel <autolabel checkout> --kinematic <run> --flora out/tempseg --out out/merge
```

A job row is `{"id", "video", "profile": "yam"}`; WGO videos are single-view,
so the robot profile runs on the main view only.

## Results (2026-11-02, WGO-Bench robot episodes, F1@0.75)

| Rule | Galaxea (25) | DROID (50) |
|---|---:|---:|
| kinematic phases alone | 0.017 | 0.000 |
| TempSeg alone (published Flora: 0.708 / 0.561) | 0.687 | 0.562 |
| `group`: phases joined by the TempSeg segment overlapping each most | 0.593 | 0.560 |
| `snap@0.5`: TempSeg endpoints moved to the nearest cut within 0.5 s | 0.571 | 0.555 |
| `snap@1` | 0.556 | 0.577 |
| `split-first`: segments holding a close and a later open split at the first cut after the close | 0.690 | 0.581 |
| `split-mid`: the same, at the cut nearest the middle | 0.690 | 0.524 |
| constrained TempSeg (`--kinematic`, whole values; see below) | 0.498 | 0.317 |

Bounds from the references (not a method): moving each TempSeg endpoint to the
reference cut itself within 1 s gives 0.788 / 0.555; to the best kinematic
cut within 1 s, 0.757 / 0.584. Boundary precision is therefore worth at most
about 0.07–0.10 on Galaxea and nothing on DROID, where TempSeg's errors are
structural: it under-segments (124 predictions for 150 references, merge
recall 0.52), often returning one segment for a pick-and-place.

### Constrained decoding (`--kinematic`)

TempSeg writes each tick digit by digit. A first version masked digits one at a
time and broke: on `galaxea_002` window 2 the model began a start with `3`,
every continuation it preferred was blocked, and the only allowed completion
was `320`, giving the invalid row `{"s":320,"e":161}`.

`decoding.py` instead chooses whole values. It decodes greedily itself (checked
by `check_decoding.py`: identical to TempSeg's `generate` on all five windows
of `galaxea_002`), and at each `"s":`/`"e":` scores every allowed value in one
batched forward pass from a copy of the cache,

    log P(v) = sum of log P(digit) over the digits of v + log P(a non-digit follows v),

writes the best and resumes. Allowed values are the window-local ticks of the
episode's phase boundaries plus the window edges; a start may not precede the
previous row's start and an end must follow its row's start. Every scored list
is kept in the output (`constraints[*].choices`), so the probabilities are also
per-boundary evidence for or against a cut. Episodes that still fail are
written as `<id>.error.json` with each window's envelope, allowed ticks and
reply.

Result (all 75 episodes ran): constrained TempSeg is much worse, F1@0.75 0.498
on Galaxea and 0.317 on DROID (free: 0.687 / 0.562), with fewer segments
(110 / 96 against 136 / 124) and larger boundary offsets (median 0.53 s /
1.89 s against 0.27 s / 0.99 s); `split-first` on top reaches 0.514 / 0.487.
The cause is the model's output grid: 88 % of the ticks it writes freely are
multiples of 10, i.e. on a 0.5 s grid. Token probabilities of off-grid values
are low and say nothing about numeric distance, so the most probable allowed
value was a window edge (0, 40, 280 or 320, all on the grid) in 63 % of the
choices, whatever the video showed. Whole-value constrained decoding is
therefore not a way to read TempSeg's judgement at our cuts; its beliefs have
to be read on its own 0.5 s grid.
