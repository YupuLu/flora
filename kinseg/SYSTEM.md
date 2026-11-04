# Kinematic segmentation with FLORA: system overview

Status 2026-11-04, branch `kinseg-merge`. This document describes the parts of
the system, how they connect, what has been tried and what remains to decide.
Measurements and commands are in [README.md](README.md).

## 1. Goal

Segment robot demonstrations into subtasks (and, later, describe each subtask)
by combining autolabel's kinematic segmentation with FLORA's video models. The
benchmark is the robot part of WGO-Bench: 25 Galaxea episodes (head camera,
123 subtasks) and 50 DROID episodes (exterior camera, 150 subtasks), scored by
segment F1 at tIoU ≥ 0.75 under one-to-one matching with snapped outer edges.

## 2. Components

```text
robot episode: video (1 or 3 cameras) + kinematics (end-effector, joints, gripper) + task instruction
  │                                            │
  │                             autolabel kinematic segmentation
  │                                            ▼
  │                   activity phases + gripper events = candidate cuts
  │                   (precise; finer than subtasks)
  ▼
Flora-TempSeg (9B): 16 s windows, 0.5 s output grid, stitched across windows
  → subtask segments (good grouping, coarse boundaries)
  │
  ▼
[decision: which candidate cuts are subtask boundaries?]      <- the open problem (section 4)
  │
  ▼
subtask segments → descriptions (Flora-Captioner 27B, or autolabel's VL describe + aggregation)
  │
  ▼
WGO-Bench scoring (autolabel modules/exp_wgo_bench/score.py)
```

| Component | Input | Output | Strength | Limitation |
|---|---|---|---|---|
| **autolabel kinematic segmentation** (speed mode 2 for WGO) | kinematics | activity phases, gripper events | boundaries at motion changes, close to reference cuts (median offset 0.29 s Galaxea, 0.32 s DROID) | 3-6× finer than subtasks (median phase 0.93 s / 1.20 s; subtask 2.92 s / 7.94 s) |
| **Flora-TempSeg** | video | subtask intervals | groups motion into subtasks well (F1 0.687 / 0.562) | writes times on a 0.5 s grid; under-segments DROID (124 predictions for 150 subtasks) |
| **Flora-Captioner** | video interval with neighbours as context, optional episode instruction | one instruction per interval | trained on robot subtasks | on 0.5-2 s fragments it describes the fragment; misreads small objects without the instruction |
| **Flora-Rewarder** | video, task | success and quality scores | — | not used yet |
| **autolabel VL describe + aggregation** | phases, frames, a GPT-class VL model | merged rows with descriptions | already joins phases semantically | never scored on WGO-Bench |

## 3. What has been tried (all 75 episodes unless noted)

| # | Approach | Galaxea F1 | DROID F1 | Why it fell short |
|---|---|---:|---:|---|
| 0 | TempSeg alone (published Flora: 0.708 / 0.561) | 0.687 | 0.562 | baseline |
| 1 | our phases alone | 0.017 | 0.000 | far too fine |
| 2 | phases grouped by the TempSeg segment overlapping each most | 0.593 | 0.560 | inherits TempSeg's grouping errors |
| 3 | TempSeg cuts snapped to our nearest cut (0.5 s / 1 s) | 0.571 / 0.556 | 0.555 / 0.577 | our cuts are dense; the nearest is often wrong |
| 4 | **TempSeg + gripper split** (close then open of one arm inside a segment) | **0.690** | **0.581** | best so far; failed grasps and pushes break the rule |
| 5 | TempSeg constrained to write our cut ticks | 0.498 | 0.317 | it writes on a 0.5 s grid; off-grid values have no meaningful probability |
| 6 | TempSeg's grid probabilities as a score of our cuts | 0.416 | 0.356 | only informative next to its own cuts |
| 7 | Qwen3.5-9B judging "same operation?" per cut (3 prompts, ± instruction, 2 frame sizes; 6 episodes) | AUC 0.44-0.58 | AUC 0.39-0.61 | chance: misreads objects, wrong granularity |
| 8 | Captioner per phase, join identical captions | 0.029 | 0.000 | every fragment captioned differently |
| 9 | Captioner per phase, join by operation (leading verb) | 0.303 | 0.120 | verbs drift within a subtask (same verb in 68 % / 55 % of pairs inside a subtask, 40 % / 29 % across a cut) |
| 10 | TempSeg split where the Captioner's operation changes | 0.320 | 0.128 | the same noise, over-splitting |

Upper bounds (from the references, not methods): keeping exactly the cuts
nearest each reference cut within 0.5 s gives 0.728 / 0.729 with our cuts,
0.741 / 0.374 with TempSeg's, and **0.917 / 0.850 with both**. The boundaries
are available; choosing them is the problem.

## 4. Diagnosis

1. The candidate cuts are good: together, ours and TempSeg's contain almost
   every reference boundary (94 of 98 on Galaxea, 80 of 100 on DROID, within
   0.5 s).
2. The decision of which cuts to keep is the bottleneck. TempSeg makes it well
   at its own granularity; nothing tried so far makes it better.
3. The released FLORA models, and an untrained 9B model, are reliable on
   subtask-sized intervals but not on 0.5-2 s fragments: approaches 5-10 all
   ask them about fragments.

## 5. Open questions

- **Q1 Product.** Subtask segments only, or segments with instructions? For
  WGO-Bench only, or also Galaxea Open-World and our own three-camera,
  whole-body datasets?
- **Q2 Unit of decision.** (a) fragments, as so far; (b) subtask-sized
  proposals: TempSeg's segments as the base, our cuts for precise boundaries and
  as split candidates, so that the models are asked about what they were
  trained on.
- **Q3 Who decides.** (a) TempSeg's grouping plus targeted checks, e.g. caption
  the two halves of a segment holding a grasp and a later release, around our
  cut, and split only when they differ; (b) a judge trained with keep/drop
  labels made by the perfect-judge rule on other annotated datasets (not
  WGO-Bench); (c) a person answering yes/no per cut, about 20 questions per
  episode, near the 0.92 / 0.85 ceiling.
- **Q4 Baseline.** Score autolabel's own VL describe + aggregation on
  WGO-Bench (costs VL API calls) as the semantic baseline.
- **Q5 Evaluation.** WGO-Bench has 75 episodes; anything tuned on it needs a
  held-out split (the Galaxea task-family split of the speed-mode work).
- **Q6 Descriptions.** Once segments are fixed, the Captioner (in its trained
  setting) or autolabel's VL pipeline writes the instructions; how are they
  scored (WGO's caption protocol is not implemented yet)?

## 6. Infrastructure

- **Local**: H20 GPUs, conda env `flora`; TempSeg weights in
  `~/.cache/flora/models` and the Hugging Face cache.
- **Remote** (shared root account, `kinseg/remote/flora-remote`): 8 × RTX PRO
  5000 72 GB; FLORA's TempSeg, Captioner and Rewarder weights already on
  `/cfs`; our data and outputs in `/cfs/users/yplu/kinseg`. The script never
  deletes anything there, keeps no credentials there and refuses to run a
  checkout someone changed.
- **Throughput**: TempSeg 3-6 s per 16 s window on the remote; Captioner about
  3.6 s per caption (about 85 min for all 1,422 phases on one worker).
- **Known issues**: the remote GPUs are shared and can fill up during a run;
  `/cfs` is Lustre and the first read of an idle weight file rewrites its
  timestamps (probably tiered storage), which trips FLORA's integrity check
  once (`local_models.load` retries that failure only); the link to the server
  limits each connection to about 44 KB/s (`push --parallel`).

## 7. Files

| File | Role |
|---|---|
| `segment.py` | TempSeg runner (free, `--kinematic` constrained, `--probe` beliefs) |
| `decoding.py`, `check_decoding.py` | greedy decoding with whole-value choices; equivalence check |
| `merge.py`, `beliefs.py` | post-hoc combinations; TempSeg beliefs at our cuts |
| `judge.py` | vision-language same-or-new judge |
| `caption_merge.py` | Captioner on our phases; join rules; TempSeg split by operation |
| `local_models.py` | local weight folders and the load retry |
| `remote/flora-remote` | remote server control |
| `test_kinseg.py` | CPU tests |
