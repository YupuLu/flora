# Kinematic segmentation with Flora-TempSeg

For the overall picture (components, everything tried, diagnosis and open
questions) see [SYSTEM.md](SYSTEM.md); this file holds the details.

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
| `beliefs.py` | TempSeg's grid beliefs (from `--probe`) as scores of our cuts |
| `judge.py` | A vision-language same-or-new judge of candidate cuts and its scores |
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

### Flora-Captioner on our phases (`caption_merge.py`), all 75 episodes

Every kinematic phase was captioned once by Flora-Captioner (27B, on the
remote server), with its neighbours as context, with and without the episode
instruction; phases were then joined offline.

| F1@0.75 | Galaxea | DROID |
|---|---:|---:|
| TempSeg alone | 0.687 | 0.562 |
| join phases with identical (normalised) captions | 0.029 | 0.000 |
| join phases with the same operation (leading verb; hand-only motion joins the next) | 0.303 | 0.120 |
| the same, captions with the instruction | 0.311 | 0.093 |
| TempSeg, split where the operation changes inside a segment | 0.320 | 0.128 |

The Captioner is trained on whole subtasks; on 0.5–2 s phases it describes the
fragment ("hold", "lift", "move toward …"), and without the instruction it
misreads DROID's objects (a bowl becomes a pen, a cable, a mouse). Its
operations carry some signal but too little: neighbouring phases inside one
reference subtask share the operation in 68 % (Galaxea) and 55 % (DROID) of
cases, phases across a reference cut in 40 % and 29 %.

### Captioner inputs (2026-11-04, 26 episodes)

`caption_merge.py run` varies what the Captioner receives per phase:
`--context both|before|after|none` (which BEFORE/AFTER video blocks are sent),
`--background none|task_brief|scene_context|task_and_scene` (the remote
workspace's background template, `background.py`), `--robot-state` (gripper
state, gripper events and moving arms of the phase, from the kinematics) and
`--previous` (the previous phase's caption, for consistent names).
`caption_merge.py compare --runs NAME=DIR ...` scores runs on their shared
episodes. Eight variants on 13 Galaxea and 13 DROID episodes, best operation
joins:

| F1@0.75 on these episodes | Galaxea | DROID |
|---|---:|---:|
| TempSeg alone | 0.676 | 0.530 |
| TempSeg + gripper split | 0.681 | 0.587 |
| FLORA's default inputs, operation join | 0.282 | 0.118 |
| task brief, operation join with motion absorbed | 0.362 | 0.182 |
| task brief + robot state + previous caption (TARGET, AFTER), motion absorbed | 0.250 | 0.306 |

Design, examples and discussion: [SYSTEM.md](SYSTEM.md), section 5.

`--prompt template` (`caption_prompt.py`) writes the information into the
prompt itself, before FLORA's instruction: `--brief`, `--objects` (from
`--scenes`), `--kinematics`, `--previous`, and `--notes` (one line per item on
how to use it). On the same episodes the previous interval gave the best
separation (DROID: same operation in 52 % of pairs inside a subtask, 14 %
across a cut) and a motion-absorbed join of 0.300; SYSTEM.md section 5.6.

```bash
CUDA_VISIBLE_DEVICES=0,1 python kinseg/caption_merge.py run --kinematic <run> --videos <dir> \
    --episodes galaxea_002 ... --rounds 1 --background task_brief --robot-state --out <dir>
python kinseg/caption_merge.py compare --autolabel <checkout> --kinematic <run> --tempseg out/tempseg \
    --runs A=<dir> B=<dir> --out compare.md
# examples: per episode, every run's caption of each phase and every method's segments (✓ = matched)
python kinseg/caption_examples.py --autolabel <checkout> --kinematic <run> --tempseg out/tempseg \
    --videos <dir> --runs A=<dir> B=<dir> [--shared] --out examples.md
```

Join rules are listed in `JOINS` at the top of `caption_examples.py`; a new
rule added there appears for every run.

### Nero Dog, our collected data (`nero.py`, 2026-11-06)

`nero.py` writes kinematic phases of Nero Dog LeRobot episodes in the same
`predictions/<group>/<id>.json` format (no references), with the settings of
`nero_dog.toml` (autolabel's `full_body` reference: speed mode 1, gripper
opening 0 closed to 1 open, no torso). Jobs are the FLORA run's
`selection.json`; parquet files are read from a local copy.

```bash
python kinseg/nero.py --autolabel <checkout> --jobs <run>/selection.json \
    --source outputs/kinseg/nero/source --out outputs/kinseg/nero/kinematic_mode1
# Captioner per phase, three cameras (jobs: FLORA annotation JSONL with left_wrist / right_wrist), template T4
CUDA_VISIBLE_DEVICES=6,7 python kinseg/caption_merge.py run --kinematic <kinematic> --jobs jobs3.jsonl \
    --episodes <ids> --profile yam --prompt template --brief --objects --previous --notes \
    --scenes <scenes> --out <captions>
# combination by gpt-6.1-sol (v6), then C1 / C5 with the model's instructions
python kinseg/combine.py prompts --captions <captions> --kinematic <kinematic> --version v6 --out <c>/prompts
python kinseg/combine.py ask --prompts <c>/prompts --provider openlux-codex-gpt-1 --autolabel <checkout> \
    --out <c>/answers/sol_run1
python kinseg/combine.py segments --prompts <c>/prompts --answers <c>/answers/sol_run1 \
    --kinematic <kinematic> --out <c>/segments_sol_run1
# review videos and page next to FLORA's run (on the machine with the videos; ffmpeg with libass)
python kinseg/nero_review.py --jobs jobs3.jsonl --kinematic <kinematic> --captions <captions> \
    --segments <c>/segments_sol_run1 --flora <FLORA run> --method C1 --out <review>
```

The first three demos of each task (21, 16.5 min) took 342 Captioner calls,
61 min on one GPU pair; results and observations are in SYSTEM.md §5.11.

The 70 episodes of the FLORA pilot (7 tasks, 53 min), per episode (median):

| Task | Speed mode 0: phases (length) | Mode 1 | Mode 2 | Gripper closes / opens |
|---|---:|---:|---:|---:|
| door_enter | 9 (2.9 s) | 9 (2.6 s) | 11.5 (2.0 s) | 0 / 0 |
| door_exit | 13 (2.9 s) | 16.5 (2.1 s) | 22 (1.3 s) | 0 / 0 |
| give_water | 3 (2.4 s) | 3 (1.9 s) | 5 (1.1 s) | 1 / 1 |
| grasp_water | 8.5 (3.3 s) | 10.5 (2.0 s) | 15 (1.3 s) | 4 / 2 |
| move_water | 17.5 (3.5 s) | 22.5 (2.3 s) | 36.5 (1.3 s) | 2 / 2 |
| place_water | 7 (3.1 s) | 8.5 (2.5 s) | 13 (1.3 s) | 2 / 2 |
| put_bag | 13.5 (3.0 s) | 15.5 (2.7 s) | 18 (2.3 s) | 1 / 1 |

Gripper events do not depend on the mode. The water and bag robots' gripper
state sometimes reads 1.00 for 1-3 frames between closed values; `despike`
removes these (343 frames in 70 episodes), which otherwise add an
open-close pair each. The door robot's gripper reads 0-0.024 (in the one
enter episode checked, the fingers stay together in the wrist video
throughout), so door episodes have no gripper events.

## Remote server

`kinseg/remote/flora-remote` drives a remote development server whose account
is shared, so it never deletes anything there and keeps no credentials there:
the remote checkout pulls this public fork, and private code and data are
copied with tar over ssh. Configuration is in `kinseg/remote/remote.env`
(untracked; see `remote.env.example`).

```bash
kinseg/remote/flora-remote sync      # push this branch, fast-forward the remote checkout
kinseg/remote/flora-remote verify    # the remote checkout is unchanged since the last sync
kinseg/remote/flora-remote push --parallel 8 --update <local dir> <remote rel>   # resumable copy
kinseg/remote/flora-remote run --gpus 0 --bg <name> -- python kinseg/judge.py run ...
kinseg/remote/flora-remote log <name>
kinseg/remote/flora-remote status    # GPU memory and ticli workers
kinseg/remote/flora-remote launch --log <name> --cpu 8 --gpu 1 --mem 64 -- <command>   # ticli worker
```

A worker needs the notebook's Worker feature enabled (the TIONE API refuses
the launch otherwise); `FLORA_REMOTE_WORKER_FLAGS` in `remote.env` holds the
fixed ticli flags (GPU type, notebook ID, worker key). Model loads on one
machine run one at a time (`local_models.py`), since concurrent reads change
the weight files' timestamps and fail FLORA's integrity check.

`sync`, `run` and `launch` refuse to use a remote checkout with edited
tracked files, untracked files under `kinseg/` or commits missing from
GitHub. On the remote, `FLORA_REMOTE_LOCAL_MODELS` points `kinseg` at the
local weight folders through `kinseg/local_models.py`. The link limits each
connection (about 44 KB/s), so `push --parallel` splits files over several
streams; public datasets can also be downloaded on the remote through its
proxy. `check_decoding.py` gives identical replies on the remote's RTX PRO
5000 GPUs (all six windows of `galaxea_002`), where a window takes 3–6 s, but
TempSeg used six windows there against five on our H20, so remote and local
runs need not be bit-identical.

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

### TempSeg's grid beliefs as a judge of our cuts (`--probe`, `beliefs.py`)

Decoding stays free (segments identical to TempSeg's) while the probability of
every grid value is recorded at each start and end. A kinematic cut scores the
largest probability any prediction puts on its nearest grid point. Ends are
confident (typically above 90 % on one grid point), so the beliefs say little
away from TempSeg's own cuts.

| | Galaxea | DROID |
|---|---:|---:|
| AUC of the belief, cuts nearest a reference boundary against the rest | 0.735 | 0.726 |
| AUC of the distance to TempSeg's nearest cut | 0.796 | 0.694 |
| Best F1@0.75 merging our phases across cuts below a belief threshold | 0.416 | 0.356 |
| Ceiling: a perfect judge keeping exactly the cuts nearest reference boundaries | 0.728 | 0.729 |

A same-or-different judge on our cuts is worth having (the ceiling beats
TempSeg alone, by 0.17 on DROID), but TempSeg's probabilities are not that
judge: they add nothing to the position of its own cuts.

The same ceiling per candidate set (for each reference cut, keep the nearest
candidate within 0.5 s; join across the rest):

| Candidate cuts | Galaxea: ref. cuts covered / F1@0.75 | DROID: ref. cuts covered / F1@0.75 |
|---|---:|---:|
| ours | 80/98, 0.728 | 69/100, 0.729 |
| TempSeg's | 76/98, 0.741 | 30/100, 0.374 |
| ours and TempSeg's | 94/98, 0.917 | 80/100, 0.850 |

The two sets are complementary: each holds reference cuts the other misses,
so a judge choosing among both could go well beyond either. The 0.5 s rule is
conservative: TempSeg alone reaches 0.562 on DROID, above its 0.374 ceiling,
because on DROID's long segments a cut more than 0.5 s off can still reach
tIoU 0.75.

### A 9B vision-language judge (`judge.py`), small batch

Qwen3.5-9B base (TempSeg's adapter not loaded) judged every candidate cut
(ours and TempSeg's) from 3 s of main-camera video before and after it at
4 Hz, using TempSeg's granularity wording. Six episodes (galaxea_002, 011,
045; DROID 000003, 000008, 000047; 173 cuts, 36 nearest a reference cut):

| Variant | AUC Galaxea | AUC DROID |
|---|---:|---:|
| `pair`: two clips, same operation? | 0.468 | 0.445 |
| `single`: one 6 s clip, new operation at 3.0 s? | 0.581 | 0.561 |
| `describe`: name both operations, then Same or New | 0.550 | 0.464 |

All are near chance, and merging by any threshold scores far below TempSeg
alone (0.500 / 0.261 on these episodes; perfect judge 0.857 / 0.870). The
descriptions show why: on DROID 000003 ("pick up the black bowl | push the
black bowl") the model calls the bowl "chopsticks" and "the black cable", and
it counts an approach followed by a grasp as two operations. At this frame
size (about 160 k pixels per frame, single view) and without the task
instruction, the model misreads both the objects and the granularity.
