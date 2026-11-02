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
| `segment.py` | TempSeg segmentation only (no Captioner), model loaded once, raw window replies kept; `--kinematic <run>` constrains each `s`/`e` to the episode's kinematic boundary ticks |
| `merge.py` | Post-hoc combination rules and their scores |
| `test_kinseg.py` | CPU checks of the rules and the decoding constraint |

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

Bounds from the references (not a method): moving each TempSeg endpoint to the
reference cut itself within 1 s gives 0.788 / 0.555; to the best kinematic
cut within 1 s, 0.757 / 0.584. Boundary precision is therefore worth at most
about 0.07–0.10 on Galaxea and nothing on DROID, where TempSeg's errors are
structural: it under-segments (124 predictions for 150 references, merge
recall 0.52), often returning one segment for a pick-and-place.

### Constrained decoding (`--kinematic`): not usable yet

Masking digits one at a time ignores the magnitude of the number. On
`galaxea_002` window 2 the model began a start tick with `3`; every
continuation it preferred was blocked, so the only allowed completion was
`320`, giving the invalid row `{"s":320,"e":161}`, which FLORA rejects and its
bounded retry cannot repair. A usable version must choose a whole allowed
value: either stop when a number ends off the cut list, replace it with the
nearest cut and continue (one extra prefill per correction; equivalent to
in-generation snapping), or score every allowed value by its likelihood and
take the best (about ten times slower). Failed episodes are written as
`<id>.error.json` with each window's envelope, allowed ticks and reply.
