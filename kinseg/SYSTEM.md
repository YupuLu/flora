# Kinematic segmentation with FLORA: system overview

Status 2026-11-04, branch `kinseg-merge`. This document records the technical
route of combining autolabel's kinematic segmentation with FLORA: the parts
(section 2), the route taken, with each direction's hypothesis, evidence and
conclusion (section 3), how each method works (section 4), the Captioner
pipeline as it works now, with its design and a comparison of prompt variants
(section 5), the results (section 6), and the decisions ahead (section 8).
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
[decision: which candidate cuts are subtask boundaries?]      <- the open problem (sections 3 and 7)
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

## 3. Technical route

### 3.1 Starting point and framing

The two inputs are complementary. The kinematic phases cut where the robot's
motion changes, precisely (median 0.3 s from the reference cuts) but 3-6 times
too often. TempSeg groups the video into subtask-sized segments, at the right
granularity but with boundaries on a 0.5 s grid and with merged subtasks on
DROID. Joining the right consecutive phases would give most reference subtasks
(merge recall@0.75 0.805 on Galaxea, 0.940 on DROID).

So the problem is framed as: **a segmentation is a subset of candidate cuts,
chosen by a judge.** The perfect-judge ceiling (section 4.6) measures what a
candidate set allows: 0.728 / 0.729 with our cuts, 0.917 / 0.850 with ours and
TempSeg's together, against 0.687 / 0.562 for TempSeg alone. The route below is
the search for that judge, along two directions that differ in who decides.

### 3.2 Direction 1: TempSeg decides, our phases refine ("TempSeg on phases")

**Hypothesis.** TempSeg's grouping is right and only its boundaries are
imprecise, so moving its boundaries onto our cuts, or letting it choose among
our cuts, should raise tIoU above 0.75 for more segments.

1. **Combine outputs after TempSeg** (#2 group, #3 snap). Both lowered Galaxea
   (0.687 → 0.59 / 0.57). Our cuts come every 0.9 s on Galaxea, so the nearest
   one is often the wrong one. A bound settled the question: moving each
   TempSeg endpoint to the reference cut itself (within 1 s) reaches only
   0.788 on Galaxea (0.757 through our cuts) and leaves DROID unchanged.
   **Boundary precision is worth at most 0.07-0.10 on Galaxea and nothing on
   DROID; TempSeg's main errors are structural**: on DROID it
   returns 124 segments for 150 subtasks, often one segment for a
   pick-and-place.
2. **Repair the structure with kinematic evidence** (#4 gripper split). A close
   followed by an open of the same arm inside one segment marks an acquisition
   and a delivery; splitting there at our cut gave the best result so far,
   0.690 / 0.581. It is a rule: a failed grasp, a push with a closed gripper or
   a tool use breaks it.
3. **Let TempSeg choose among our cuts** (#5 constrained decoding). TempSeg
   kept deciding how many segments there are, but each endpoint had to be one
   of our cuts, chosen by its own probability. It fell to 0.498 / 0.317. The
   cause is TempSeg's output vocabulary: 88 % of the ticks it writes are
   multiples of 10 (a 0.5 s grid), so an off-grid cut such as tick 137 has no
   meaningful probability, and window edges (all on the grid) won 63 % of the
   choices.
4. **Read TempSeg's judgement at our cuts instead of forcing it** (#6 grid
   beliefs). With decoding left free, its probability over the grid is
   concentrated on its own choice (for example 92 % on one grid point, 6 % on
   the next), so it says nothing about cuts it did not consider; merging by
   these beliefs reached 0.416 / 0.356.

**Conclusion.** TempSeg cannot be made to judge our cuts without retraining.
Its value is its grouping; our cuts and gripper events can repair it only
through targeted rules.

### 3.3 Direction 2: our phases first, a model judges each join ("Captioner per phase")

**Hypothesis.** A model that understands manipulation can tell, for two
neighbouring phases, whether they belong to the same subtask (the "give it two
segments and ask whether they do the same thing" idea). Our phases then
provide precise boundaries and the model the grouping; the ceiling says a good
judge would reach about 0.73 on our cuts and 0.92 / 0.85 with TempSeg's.

1. **A general 9B vision-language judge** (#7). Asked "same operation?" about
   the 3 s before and after each cut, in three phrasings, with and without the
   task instruction, at two frame sizes: AUC 0.39-0.61, chance. The model
   misread objects (DROID's bowl became "chopsticks" and "a cable") and counted
   an approach and the grasp that follows it as two operations.
2. **A robot-trained describer, Flora-Captioner, on every phase** (#8, #9).
   Joining identical captions joined almost nothing (421 segments for 123
   subtasks on Galaxea). Joining by operation (the leading verb) reached
   0.303 / 0.120. The captions do carry the signal: neighbouring phases share
   an operation in 68 % (Galaxea) and 55 % (DROID) of the pairs inside one
   subtask and in 40 % / 29 % of the pairs across a reference cut. But the
   signal is too noisy. The Captioner is trained on whole subtasks, and on
   0.5-2 s phases it describes the fragment ("hold", "lift", "move toward");
   without the instruction it misreads DROID's small objects, and with the
   instruction it repeats the task sentence for most phases.
3. **The Captioner's operations as a repair of TempSeg** (#10): splitting
   TempSeg's segments where the operation changes over-split them (0.320 /
   0.128); the same noise.
4. **Better inputs for the Captioner** (#11, section 5). The Captioner's
   optional text was rebuilt on the remote workspace's background template
   (task brief plus guidance that it is intent, not observation), with three
   additions tested on 26 episodes: which neighbouring video blocks are sent,
   the previous phase's caption for consistent names, and the robot state of
   the phase from the kinematics (gripper state and events, moving arms, an
   idea from autolabel's VL prompt). The brief fixed DROID's object names and
   the robot state separated subtasks best, but the best caption-only
   segmentation stayed at 0.36 / 0.31 against TempSeg's 0.68 / 0.53 on the
   same episodes.

**Conclusion so far.** These models judge reliably at the scale they were
trained on, whole subtasks, and not on fragments; asking about fragments was
the common failure of direction 2 and of steps 3-4 of direction 1. Better
inputs (step 4) shift the Captioner's agreement inside and across subtasks by
up to about 0.15, not enough to change that.

### 3.4 Lessons that shape the next step

1. The candidate cuts are not the bottleneck; the judge is.
2. TempSeg's grouping is the strongest judge available; it errs mainly by
   merging (DROID) and slightly by splitting (Galaxea, 136 for 123).
3. A model should be asked about **subtask-sized** spans, the scale it was
   trained on, and our cuts used for where to cut, not as units to describe.
4. Kinematic evidence (gripper events) points to where TempSeg merged
   subtasks, but alone it is a rule; it needs a model to confirm it.

These point to routes such as: keep TempSeg's segments, use our cuts and
gripper events to propose splits (and neighbouring segments to propose
merges), and confirm each with the Captioner on the subtask-sized halves or
unions; or train a judge with keep/drop labels from other annotated datasets;
or let a person answer the keep/drop questions. Section 8 lists them as
decisions.

## 4. Method details

### 4.1 Terms

- **Phase**: an activity phase of autolabel's kinematic segmentation, the
  interval between two consecutive changes of robot motion (arm direction,
  arm activity, base or torso activity). Phases tile the episode.
- **Our cuts**: the internal phase boundaries. **TempSeg cuts**: the start and
  end times of TempSeg's stitched segments. **Candidate cuts**: either or both.
- **Gripper events**: closes and opens detected by autolabel, from the
  measured finger opening (Galaxea) or the binary command (DROID).
- **Window ticks**: TempSeg sees 16 s windows (12 s core plus 2 s guards) at
  4 Hz and writes each segment as integer ticks of 0.05 s from the window start,
  in practice multiples of 10 ticks (a 0.5 s grid).
- **Joining across a cut**: removing the cut, so the phases on both sides
  become one interval. Every method below ends by choosing which candidate cuts
  to keep; the intervals between kept cuts are the prediction.

### 4.2 Where each method connects the two

```text
kinematics ─► phases, our cuts, gripper events ───────────────┐
                                                              ├─► A. after TempSeg: combine outputs (#2-#4)
video ─► TempSeg windows ─► replies ─► stitching ─► segments ─┘
              ▲               │
              │               └─► B2. read TempSeg's grid probabilities at our cuts (#6)
              └─ B1. TempSeg may only write our cut ticks (#5)

candidate cuts (ours ∪ TempSeg's) ─► C. another model judges each cut (#7-#11) ─► kept cuts ─► segments
```

### 4.3 A. Combining outputs after TempSeg (`merge.py`)

TempSeg runs unchanged; its segments and our phases are combined
geometrically.

- **#2 group.** Each phase is assigned to the TempSeg segment it overlaps
  most; a phase overlapping none takes its predecessor's segment. Consecutive
  phases with the same segment are joined. Result: TempSeg's grouping with our
  boundaries.
- **#3 snap@d.** Each TempSeg endpoint moves to our nearest cut (or episode
  edge) when one lies within `d` seconds (0.5 s or 1 s); an endpoint without
  one stays. Result: TempSeg's segments with our boundary precision.
- **#4 gripper split.** Inside each TempSeg segment, look for a close of one
  arm followed by an open of the same arm (an acquisition and a delivery, which
  TempSeg's own policy keeps distinct). If found, split the segment at our
  first cut after the close (`split-first`) or at our cut nearest the middle
  between close and open (`split-mid`), then repeat on the remainder. Result:
  pick-and-places that TempSeg returned as one segment are split.

### 4.4 B. Changing or reading TempSeg's decoding (`segment.py`, `decoding.py`, `beliefs.py`)

TempSeg writes each segment as JSON (`{"s":40,"e":160,"lo":false,"ro":false}`),
one token at a time, numbers digit by digit. Our own greedy decoding loop
reproduces TempSeg's replies token for token (`check_decoding.py`), so changes
to it affect nothing else; windows, prompts and stitching stay FLORA's.

- **#5 constrained decoding.** For each window, the allowed values are the
  window ticks of our cuts plus the window's edges. At every `"s":` or `"e":`
  the loop scores each allowed value as a whole number, log P(digits) plus the
  log-probability that the number ends there, in one batched forward pass from
  a copy of the model's cache, and writes the best (a start not before the
  previous row's start, an end after its start). TempSeg still decides how many
  segments there are; only their endpoints are forced onto our cuts. (A first
  version chose digits one at a time and could complete a number far from the
  one the model began, e.g. `"s":320` with `"e":161`.)
- **#6 grid beliefs.** Decoding stays free (replies unchanged); at every
  `"s":`/`"e":` the loop records the probability of every value on TempSeg's
  0.5 s grid in the window. Each of our cuts gets the largest probability any
  prediction puts on its nearest grid point (within 0.25 s), excluding the
  window's visible edges, which mean "continues beyond this window". Cuts at or
  above a threshold are kept and the phases are joined across the rest.

### 4.5 C. Another model judges each cut (`judge.py`, `caption_merge.py`)

- **#7 9B judge.** Candidates are our cuts and TempSeg's, merged when closer
  than 0.05 s. For each cut `c`, Qwen3.5-9B (TempSeg's base model, without its
  adapter) sees the main camera at 4 Hz and is asked, with the granularity
  wording of TempSeg's prompt, in one of three ways: two 3 s clips before and
  after `c`, "same operation?" (`pair`); one 6 s clip, "does a new operation
  start at 3.0 s?" (`single`); or name both operations, then "Same or New"
  (`describe`). The score is the probability of the "new operation" answer
  token; cuts at or above a threshold are kept. Variants added the task
  instruction and larger frames (400 k instead of 160 k pixels).
- **#8, #9 Captioner on phases.** Every phase is captioned by Flora-Captioner
  as its own target interval, with the neighbouring phases as before/after
  context (FLORA's caption sampling: target 8-40 frames at 8 fps, context 4-12
  frames), optionally with the episode instruction. Consecutive phases are then
  joined when their captions are identical after normalisation (#8: lower
  case, no punctuation or articles), or when they share an operation (#9: the
  leading verb phrase, e.g. "pick up", "place"; a caption that only moves a hand
  or arm, an approach or retreat, takes the next phase's operation).
- **#10 TempSeg + operation split.** TempSeg's segments are kept, and each is
  split at our cuts where the Captioner's operation of the phases inside it
  changes.
- **#11 Captioner inputs.** The text and video the Captioner receives per
  phase are varied (background, video context, previous caption, robot
  state); section 5 describes the pipeline and the comparison. (The first
  "consistency captioning" design, which asked the model to repeat the
  previous caption exactly, was replaced: it duplicated the BEFORE video
  block and invited copying.)

### 4.6 D. The perfect-judge ceiling (from the references, not a method)

For each reference cut, the candidate cut nearest to it within 0.5 s is kept;
all others are dropped, and the intervals between kept cuts are scored. It
measures how good a candidate set is, independent of the judge, and is the
target the judges in C aim at.

## 5. Captioner per phase: how it works now

This section describes direction 2's Captioner pipeline as it now stands
(`caption_merge.py`, `background.py`): what happens to an episode, what the
model receives, why each input is there, and what the input variants measured
on 26 episodes (2026-11-04).

### 5.1 Pipeline

```text
episode: main-camera video + kinematic prediction (phases, gripper events) + task sentence
  │
  1  intervals: our phases (or the tiles between our cuts and TempSeg's, --cuts union)
  2  FLORA's caption plan per interval: TARGET = the interval; BEFORE / AFTER = the nearest
  │  non-overlapping neighbouring intervals; 8 fps, TARGET 8-40 frames, context 4-12 frames
  3  video context (--context both | before | after | none): which of BEFORE / AFTER are sent
  4  text (FLORA's episode-instruction field): background [+ robot state] [+ previous caption]
  5  Flora-Captioner 27B, greedy: one instruction per interval, in time order
  6  joining: equal captions, or equal operations (leading verb phrase) ─► segments ─► F1@0.75
```

1. **Intervals.** The kinematic phases tile the episode; each phase is one
   interval to describe. With `--cuts union` the intervals are the tiles
   between our cuts and TempSeg's (tiles shorter than 0.05 s are dropped).
2. **Caption plan.** The same function FLORA uses for its own TempSeg
   segments (`pipeline.single_plans`) turns each interval into video blocks:
   TARGET, the interval, sampled at 8 fps (8-40 frames), and BEFORE and AFTER,
   the neighbouring intervals (4-12 frames each). For an interval FLORA cannot
   sample, the status is not `complete`, the caption is empty, and the interval
   joins its predecessor.
3. **Video context.** `--context` keeps a subset of the blocks. FLORA's
   validator accepts exactly the four layouts `[TARGET]`, `[BEFORE, TARGET]`,
   `[TARGET, AFTER]` and `[BEFORE, TARGET, AFTER]`, and its prompt lists the
   blocks that are sent, so the model is never told about a block it does not
   see:

   ```text
   both:  BEFORE:\n<video>\n\nTARGET:\n<video>\n\nAFTER:\n<video>\n\n<instruction>
   none:  TARGET:\n<video>\n\n<instruction>
   ```

   where `<instruction>` is FLORA's fixed prompt, unchanged: "Write one
   concise, self-contained English instruction for the local operation being
   attempted in TARGET. Use a natural imperative, concrete objects and the
   working hand; … Use BEFORE and AFTER only to disambiguate TARGET, not to add
   operations occurring outside it. A failed or repeated attempt retains its
   established instruction; do not replace the goal with an accident, a
   success claim or a description of mechanical adjustments. Do not invent an
   object, hand role or hidden goal. Output only the instruction, on one line,
   …"
4. **Text.** The Captioner's only optional text input is the episode
   instruction, which FLORA appends to the prompt as
   `\n{"episode_instruction":"…"}` (at most 2,000 characters). Our text for it
   is built by `background.background_text` from up to three parts, each with
   its own guidance line at the end. An example (DROID episode 000005, phase
   3.3-7.2 s, variant B_robot_prev):

   ```text
   Background:
   Task brief: Remove the pen from the mug and put it on the table
   Robot state from kinematics: At the start of TARGET the gripper is open. During TARGET the gripper closes. The arm moves.
   Previous interval was described as: "Move the gripper toward the pen in the red cup."

   Background describes the intended task and available objects, not proven execution. Use it only to
   disambiguate the visible local operation. Keep the native one-line action instruction format; do not
   assume success, invent invisible actions, or describe operations outside TARGET.
   Robot state comes from joint and gripper readings: a gripper can close without grasping and open
   without releasing anything, so use the video to decide what is held.
   The previous description only keeps object and action names consistent; describe only the operation
   in TARGET.
   ```

   The Captioner answered "Grasp the pen in the red cup with the gripper."
   (FLORA's default answered "Move the right hand toward the blue pen in the
   red cup.", B_both "Move the right hand toward the pen in the red cup.")

   - *Background* (`--background`): `task_brief` gives the task sentence as
     `Task brief:`; `scene_context` gives target objects, other objects and
     workspace; `task_and_scene` both; `none` nothing. The wording, including
     the first guidance paragraph, is copied unchanged from the workspace's
     background conditions (`workspace/benchmark/caption_controls.py`), whose
     A0 runs compared the same conditions on human video.
   - *Robot state* (`--robot-state`, `caption_merge.robot_state`): from the
     kinematic prediction, the gripper state at the start of TARGET (closed
     after its last close, open after its last open or before any event), the
     closes and opens during TARGET, and the moving arms (phases whose arm
     label is not `STABLE`). A robot whose phases and events only name the
     right arm is described as one arm ("the gripper").
   - *Previous caption* (`--previous`): the caption just written for the
     previous interval.
   - `--instruction` instead sends the bare task sentence (FLORA's own use of
     the field, the earlier "plain" runs).
5. **Captioning.** FLORA's frame extraction, processor and greedy generation
   (at most 512 tokens), one call per interval, in time order so that the
   previous caption exists. Every caption is saved with the exact text it was
   given (`instruction_text`).
6. **Joining and scoring** (`score`, `compare`, CPU only, from the saved
   captions):
   - *exact join*: consecutive equal captions (lower case, no punctuation or
     articles);
   - *operation join*: consecutive equal operations, the leading verb phrase
     ("pick up", "place"); a caption that only moves a hand ("move the left
     hand back", an approach or a retreat) takes the next operation;
   - *operation join, motion absorbed*: also any caption led by a motion verb
     (move, lift, raise, lower, hold, bring, …; "lift the pen", "move the cup
     away") takes the next operation, following autolabel's VL pipeline, whose
     post-process lets low-importance intervals (reach, transport, retract)
     join the neighbouring action;
   - *TempSeg + operation split*: TempSeg's segments, split at our cuts where
     the operation changes.

   Besides F1@0.75, `compare` reports the share of captions that repeat the
   task (at least 80 % of the task's words, for tasks of three words or more)
   and the **operation agreement**: the share of neighbouring phase pairs with
   the same operation, inside one reference subtask and across a reference
   cut (pair boundary within 0.5 s of it). A good judge agrees inside and
   disagrees across; the gap between the two is the signal a join can use.

### 5.2 Design logic

- **Why a background, not the bare task sentence.** The episode instruction
  describes the whole episode, while the Captioner is asked about one phase of
  it. Given the bare sentence, the model sometimes returned it as the phase's
  instruction (7 % of DROID captions repeat the task): it took task-level
  context for the label of the fragment. The workspace template presents the
  sentence as a *task brief* and adds guidance that it is intent, not proven
  execution, and may only disambiguate what TARGET shows. Copying its wording
  keeps the results comparable with the A0 background study.
- **Why only the task brief.** `scene_context` needs per-episode object lists
  (target objects, other objects, workspace). WGO-Bench provides only the task
  sentence; the lists would have to be written by hand or by a VL model
  (question Q7).
- **Why a video-context switch.** FLORA's BEFORE and AFTER were designed for
  subtask-sized neighbours. Around a 0.5-2 s phase they are other fragments of
  the same subtask, often showing more of the operation than TARGET does. They
  may help (they say which operation the fragment belongs to) or leak (the
  model names the neighbour's operation). Sending TARGET alone tests which.
- **Why the previous caption is limited to names.** The first consistency
  design asked the model to repeat the previous caption exactly when the
  operation continued. That duplicated the BEFORE block, which already shows
  the previous interval as video, and invited copying. Now the previous
  caption only keeps object and action names consistent, and it is used with
  `--context after`: the previous interval appears as text instead of as
  video.
- **Robot state: what we took from autolabel's VL pipeline (v3).** Autolabel's
  per-interval prompt gives the VL model a SEG INFO line (the interval's motion
  class: ARM, NAV, STABLE), a SCENE AT start line (what each hand holds,
  carried over from the previous answer) and the previous answer, and warns
  that a gripper close or open is not a grasp or a release. We adopted the
  motion and hand state, computed deterministically from the kinematics rather
  than carried over from the model's own answers, and the warning. Its
  post-process rule that low-importance intervals join their neighbours
  became the *motion absorbed* join. We did not adopt the structured JSON
  answer (per-arm action, target, contact, acquire/release, scene graph) or
  the multi-pass post-process: the Captioner is trained to write one
  instruction, and asking it for fields would need a different model, which is
  autolabel's pipeline itself (Q4).

### 5.3 Experiment

26 WGO-Bench episodes: every second Galaxea episode (13; 250 phases, 67
subtasks) and every fourth DROID episode (13; 260 phases, 43 subtasks); our
phases as intervals, one captioning round. A and plain are the earlier full
runs restricted to these episodes.

| Variant | Text | Video |
|---|---|---|
| A | none (FLORA's default) | BEFORE, TARGET, AFTER |
| plain | the bare task sentence | BEFORE, TARGET, AFTER |
| A_none | none | TARGET |
| B_both | task brief | BEFORE, TARGET, AFTER |
| B_none | task brief | TARGET |
| B_prev | task brief + previous caption | TARGET, AFTER |
| B_robot | task brief + robot state | BEFORE, TARGET, AFTER |
| B_robot_prev | task brief + robot state + previous caption | TARGET, AFTER |

On these episodes TempSeg alone scores **0.676 / 0.530** (Galaxea / DROID)
and TempSeg + gripper split **0.681 / 0.587**.

| Variant | DROID: repeats task / names a task object | Same operation inside / across, Galaxea | DROID | Operation join F1 (pred.), Galaxea | DROID | Motion absorbed, Galaxea | DROID |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.00 / 0.52 | 0.71 / 0.47 | 0.50 / 0.19 | 0.282 (96) | 0.118 (143) | 0.311 (81) | 0.200 (37) |
| plain | 0.07 / 0.82 | 0.69 / 0.48 | 0.43 / 0.24 | 0.289 (99) | 0.100 (157) | 0.318 (84) | 0.208 (34) |
| A_none | 0.00 / 0.48 | 0.76 / 0.77 | 0.65 / 0.43 | 0.162 (69) | 0.122 (105) | 0.169 (51) | 0.063 (20) |
| B_both | 0.05 / 0.78 | 0.71 / 0.45 | 0.46 / 0.29 | **0.305** (97) | 0.135 (149) | **0.362** (82) | 0.182 (34) |
| B_none | 0.06 / 0.82 | 0.78 / 0.79 | 0.57 / 0.33 | 0.212 (65) | 0.072 (124) | 0.235 (52) | 0.060 (24) |
| B_prev | 0.04 / 0.78 | 0.65 / 0.32 | 0.49 / 0.29 | 0.207 (117) | 0.075 (144) | 0.239 (92) | 0.175 (37) |
| B_robot | 0.04 / 0.75 | **0.74 / 0.39** | 0.52 / 0.24 | 0.280 (97) | 0.144 (138) | 0.331 (84) | 0.207 (44) |
| B_robot_prev | 0.03 / 0.77 | 0.68 / 0.37 | **0.59 / 0.24** | 0.240 (108) | **0.182** (122) | 0.250 (93) | **0.306** (42) |

(Galaxea captions never repeat their task sentences, which are long. "Names a
task object" is the share of captions sharing a content word with the task
sentence; it is uninformative on Galaxea, whose tasks are worded generally.)
TempSeg + operation split stayed below TempSeg alone in every variant (best
0.452 on Galaxea, B_none with motion absorbed; 0.495 on DROID, A_none with
motion absorbed). Full table: `outputs/kinseg/caption_merge/v2/compare.md`.

### 5.4 Examples

Galaxea episode 013 ("place the food in the right basket with the right
gripper, and place the non food items in the left basket with the left
gripper"), its first two subtasks:

| Phase (s) | Arm | Gripper | Reference subtask | A | A_none | B_robot | B_prev |
|---|---|---|---|---|---|---|---|
| 0.0-0.7 | – | | pick up the blue cup | Move the left hand back | Move the left hand back to the original position. | move both hands back | move the left hand back |
| 0.7-1.5 | left | | pick up the blue cup | move the left hand back | Move the left hand back. | move the left hand forward | move the left hand forward |
| 1.5-3.4 | left | | pick up the blue cup | move the left hand toward the blue cup on the green tray | (same as A) | (same as A) | pick up the cup from the green tray with the left hand |
| 3.4-4.1 | – | L close | pick up the blue cup | pick up the blue cup from the green tray with the left hand | Lift the blue cup from the green tray with the left hand. | (same as A) | hold the cup with the left hand |
| 4.1-4.9 | left | | drop the blue cup into the black basket | pick up the blue mug from the green tray with the left hand | move the left hand back | (same as A) | move the left hand upward while holding the cup |
| 4.9-5.6 | left | | drop the blue cup into the black basket | place the blue mug into the left basket on the top of the surface with the left hand | move the left hand back | place the blue mug into the left basket with the left hand | place the cup into the left basket with the left hand |
| 5.6-6.2 | – | L open | drop the blue cup into the black basket | place the blue bottle into the left basket on the top left of the table with the left hand | place the green bottle into the left basket on the white surface with the left hand | place the blue mug in the left basket with the left hand | place the cup into the left basket with the left hand |

With the neighbours (A, B_robot), the lift and the transport (4.1-5.6 s) are
named by operations (pick up, place; the lift is given to the pick-up, which
the reference ends at the grasp); with TARGET alone (A_none) they become
"move the left hand back", and the cup becomes a "green bottle" (A: "blue
bottle"; B_robot keeps "blue mug"). With the
previous caption (B_prev), the model keeps the name "the cup" but describes
the hold and the lift as separate steps and anticipates the pick-up already
in the approach (1.5-3.4 s).

DROID episode 000005 ("Remove the pen from the mug and put it on the table"):

| Phase (s) | Gripper | Reference subtask | A | A_none | B_both | B_robot_prev |
|---|---|---|---|---|---|---|
| 1.4-2.7 | | pick up the pen from the black-red cup | Move the right hand toward the black holder. | Move the right hand toward the black box. | Move the right hand toward the pen in the red cup. | Move the gripper toward the pen in the red cup. |
| 3.3-7.2 | close | pick up the pen from the black-red cup | Move the right hand toward the blue pen in the red cup. | Move the robotic hand toward the scissors on the black box. | Move the right hand toward the pen in the red cup. | Grasp the pen in the red cup with the gripper. |
| 7.2-8.1 | | pick up the pen from the black-red cup (to 7.8 s) | Pick up the blue pen from the red cup with the right hand. | Lift the toothbrush out of the red cup with the right hand. | Pick up the pen from the red cup with the right hand. | Lift the pen out of the red cup with the right hand. |
| 8.1-8.6 | | place the pen on the table left next to the cup | Lift the blue toothbrush out of the red cup with the right hand. | Lift the toothbrush away from the red cup with the right hand. | Lift the pen out of the red cup with the right hand. | Move the pen away from the red cup with the right hand. |
| 8.6-9.5 | | place the pen on the table left next to the cup | Move the pen away from the red cup with the right hand. | Move the marker away from the red cup with the right hand. | (same as A) | (same as A) |
| 9.5-11.4 | | place the pen on the table left next to the cup (to 11.7 s) | Put the pen on the table beside the red cup with the right hand. | Put the pen on the table beside the cup with the right hand. | (same as A) | (same as A) |

Without the brief, the pen becomes a toothbrush, scissors or a marker (A
also says "blue toothbrush" once); with it the object is named consistently.
The place subtask is still described as lift, move and put: B_both's
operation join cuts it into three segments, and the motion absorbed join
gives 0-8.1 s and 8.1-12.6 s, both matching the references (0-7.8 s,
7.8-11.7 s) at tIoU ≥ 0.75. The
robot state's "the gripper closes" turned the approach into a grasp, and its
single-arm wording ("the gripper") was copied into the captions.

### 5.5 Discussion

1. **Video context is needed.** Without BEFORE and AFTER the model mostly
   writes the limb motion of a short fragment, so the operations stop telling
   subtasks apart: on Galaxea the agreement across a cut rises to the
   agreement inside one (inside 0.76, across 0.77 for A_none; 0.78 and 0.79
   for B_none), and operation joins fall to 0.16-0.21. The neighbours are what let the
   Captioner name the operation a fragment belongs to; FLORA's default layout
   stays.
2. **The task brief works as intended but does not decide boundaries.** It
   fixes the objects on DROID (52 % → 78 % of captions name a task object;
   pen instead of toothbrush, scissors or marker), with fewer repetitions of
   the task than the bare sentence (5 % against 7 %), and is the best
   Galaxea variant (0.305; 0.362 with motion absorbed). Object names, however,
   rarely change at a subtask boundary, so the joins gain little.
3. **The previous caption acts as a prior on the next step, not as a name
   list.** It lowers the agreement across cuts (Galaxea 0.45 → 0.32) but also
   inside subtasks (0.71 → 0.65) and over-splits (117 segments for 67
   subtasks): after "pick up" the model tends to write the next step. The
   guidance line does not stop that.
4. **The robot state gives the clearest separation.** Inside minus across is
   0.35 / 0.28 for B_robot and 0.31 / 0.35 for B_robot_prev, against 0.26 /
   0.17 for B_both; B_robot_prev with motion absorbed is the best caption-only
   DROID result (0.306, 42 segments for 43 subtasks). It changes 43 % of the
   captions (lift → move, pick up → grasp when a close falls inside TARGET).
   Its costs: the single-arm wording "the gripper" enters the captions, 8 of
   510 captions are gripper mechanics ("open the right hand"; 1 for B_both),
   and Galaxea's
   operation join drops slightly (0.305 → 0.280). As in direction 1, where the
   gripper split was the best combination, the kinematic gripper events carry
   the boundary information; here they reach the decision only through the
   Captioner's wording.
5. **Motion absorption helps where transports are parts of a subtask and
   hurts where they are the subtask.** It raises Galaxea joins by 0.01-0.06
   and DROID joins with context by 0.05-0.12, but DROID tasks such as "Move the
   bowl to the right" make "move" an operation of its own, and without context
   it merges almost everything (A_none: 20 segments for 43 subtasks).
6. **Overall.** No input variant comes near TempSeg (0.676 / 0.530) or TempSeg
   + gripper split (0.681 / 0.587); the best caption-only segmentations are
   0.362 (Galaxea) and 0.306 (DROID), and splitting TempSeg by operation never
   helps. The inputs move the agreement across cuts by up to 0.16 (Galaxea
   0.48 → 0.32) but the agreement inside subtasks with it, and the gap between
   the two stays at 0.35 or below: a description of a 1 s fragment does not
   say whether the next fragment starts a new subtask. With 13
   episodes per family, differences below about 0.03 are within noise.
7. **What carries forward.** For any later use of the Captioner, the input
   recipe is FLORA's full context plus the task brief, with the robot state
   where gripper wording is acceptable. The decision about cuts should be
   asked at subtask scale (Q2 (b), Q3 (a)), and the kinematic events should
   enter as proposals, as in the gripper split, rather than only as text.

### 5.6 Our prompt template (2026-11-05)

Sections 5.1-5.5 kept FLORA's prompt and put our text inside its
episode-instruction JSON line. The template (`caption_prompt.py`,
`caption_merge.py run --prompt template`) instead writes the information into
the prompt itself, between the video blocks and FLORA's instruction, and is
used only by our captioning calls (FLORA's files are unchanged; the
instruction is read from FLORA's vendored code):

```text
BEFORE:
<video>

TARGET:
<video>

AFTER:
<video>

Episode information:
Task brief: Remove the pen from the mug and put it on the table                --brief
Objects: pen; mug; table                                                       --objects
Robot kinematics during TARGET: Right hand moves; its gripper is open at the start, then closes.   --kinematics
Previous interval: "Move the right hand toward the pen in the red cup."        --previous
Notes: use this information to understand TARGET.                              --notes
- The task brief is the goal of the whole episode; describe only the step in TARGET.
- Name objects with the names in the object list.
- Use the robot kinematics to confirm the action.
- Stay consistent with the previous interval, or continue from it.

Write one concise, self-contained English instruction … (FLORA's instruction, ending with its output line)
```

- **Information first, instruction last.** The model reads what it knows
  before it reads the request, and FLORA's output line stays the last line.
- **Notes, one line per item, only for the items given.** The key idea is to
  ask the model to build an understanding of TARGET from the video and the
  information, not only to see the information.
- **The context sentence follows the blocks sent** ("Use AFTER only …" with
  `--context after`; dropped with TARGET alone).
- **Kinematics in FLORA's words** ("Right hand moves; its gripper …"; a
  one-armed robot is its right hand), since "the gripper" of section 5's robot
  state entered the captions.
- **Objects** are the nouns of the task sentence, extracted once per episode
  by a Claude Haiku agent (`outputs/kinseg/scenes_from_task/`; substances such
  as liquid or powder left out). WGO-Bench has no object lists of its own.

Five runs on the same 26 episodes, FLORA's full context, one round. Each adds
one item to the previous run; kinematics, which matter least, come last.

| Run | Information | Same op. inside / across, Galaxea | DROID | Op. join F1 (pred), Galaxea | DROID | Motion absorbed, Galaxea | DROID |
|---|---|---:|---:|---:|---:|---:|---:|
| B_both (field) | task brief + guidance | 0.71 / 0.45 | 0.46 / 0.29 | 0.305 (97) | 0.135 (149) | 0.362 (82) | 0.182 (34) |
| T1 | brief | 0.70 / 0.45 | 0.42 / 0.33 | 0.287 (100) | 0.100 (157) | 0.331 (84) | 0.182 (34) |
| T2 | + notes | 0.71 / 0.39 | 0.45 / 0.33 | 0.308 (102) | 0.113 (152) | 0.333 (89) | 0.189 (31) |
| T3 | + objects | 0.70 / 0.37 | 0.45 / 0.33 | 0.292 (104) | 0.113 (152) | 0.316 (91) | 0.187 (32) |
| T4 | + previous | 0.74 / 0.34 | **0.52 / 0.14** | 0.287 (100) | 0.143 (139) | 0.327 (92) | **0.300** (37) |
| T5 | + kinematics | **0.75 / 0.32** | 0.64 / 0.33 | 0.301 (99) | 0.159 (108) | 0.346 (89) | 0.198 (38) |

(TempSeg on these episodes: 0.676 / 0.530; TempSeg + gripper split 0.681 /
0.587. Task objects are named in 80-81 % of DROID captions in every T run, 78 %
for B_both.)

- **Template versus field** (T1 against B_both, same information): no gain;
  the position of the text matters less than its content.
- **Notes** lower the agreement across Galaxea cuts (0.45 → 0.39) without
  changing the joins much.
- **Objects** from the task sentence add nothing beyond the brief, which
  already contains them.
- **The previous interval** is the item that helps. With full video context
  and "stay consistent … or continue from it", DROID's agreement across cuts
  falls from 0.33 to 0.14, the best separation of any run (inside − across
  0.38), and DROID's motion-absorbed join reaches 0.300. It rewrites 70 % of
  the captions, mostly towards consistent wording: in DROID episode 000033 the
  retreat after the place is "Move the right hand away from the pink and
  silver measuring spoon on the kitchen counter" five times in T4, where T3
  alternates put / pick up (once "with the left hand"), and the
  motion-absorbed join matches both subtasks (T3: one). Section 5's B_prev,
  which sent the previous caption without BEFORE and only for names,
  over-split instead.
- **Kinematics** give Galaxea's best separation (0.75 / 0.32) but raise
  DROID's agreement across cuts again (0.14 → 0.33).
- **The joins stay the limit.** The best separations (0.40-0.43 Galaxea, 0.38
  DROID) still give operation joins of 0.29-0.35 and 0.14-0.30. The errors of
  T4's operation join: on Galaxea 34 of 54 reference cuts are found, and 15 of
  the 21 missed ones are a pick-up whose lift or first transport is still
  called "pick up"; on DROID 18 of 30 are found, with 108 false cuts in 226
  pairs inside subtasks, mostly between hold, lift, move and scoop. The next
  step is the join rule (section 8, Q8).

Table: `outputs/kinseg/caption_merge/v3/compare.md`; examples of all nine
runs: `outputs/kinseg/caption_merge/v3/examples.md`.

### 5.7 A language model combines the captions (2026-11-05)

The joins of 5.1-5.6 compare neighbouring captions word by word. Here a
language model reads the whole episode as a short status log and combines the
phases itself (`combine.py`). It sees no video, only text.

```text
Task: Remove the pen from the mug and put it on the table
Robot: one arm (right hand).

Phases. Time in seconds; hand motion and gripper changes are measured from the robot; each instruction
was written by a video captioner for that phase alone and can be wrong.
0   0.0-1.4   still           | Keep both hands clear of the mug.
3   3.3-7.2   moves, closes   | Move the right hand toward the pen in the red cup.
4   7.2-8.1   moves           | Pick up the pen from the red cup with the right hand.
5   8.1-8.6   moves           | Lift the pen out of the red cup with the right hand.
…
8   11.4-12.6 moves, opens    | Move the right hand away from the pen and red cup.

Combine consecutive phases into subtasks: one manipulation step that a person would instruct on its own
(pick up an object, place it, pour, open or close something, push, wipe ...).
- Reaching for an object belongs to the step that acts on it; carrying a held object belongs to the step
  that puts it down; moving away after a release belongs to the next step.
- Different words for one movement ("lift", "move", "put") do not start a new step by themselves.
- For each gripper close or open in the log, decide from the instructions whether it grasped an object,
  released one, or did neither.
Answer in JSON only: {"gripper": [...], "subtasks": [{"phases": [first, last], "instruction": ...}, ...]}
```

The answer is checked (phases in order, each once; nothing needed repair in
78 answers) and scored four ways: C1 the model's subtasks; C2 C1 split where a
confirmed grasp and a later confirmed release of the same hand share a
subtask; C3 TempSeg with that split; C4 C1 with each endpoint moved to
TempSeg's nearest cut within 1 s. Captions: T4 (5.6). Models: GPT-6.1-sol
(`openlux-codex-gpt-1`, about 8 s per episode) and GPT-6-luna
(`openlux-azure-gpt-2`, 12 s, about 1,000 reasoning tokens) through
autolabel's VL client, and Claude Haiku as a subagent.

| F1@0.75 (pred/ref) | Galaxea | DROID |
|---|---:|---:|
| TempSeg | 0.676 (72/67) | 0.530 (40/43) |
| TempSeg + gripper split | 0.681 (74/67) | 0.587 (49/43) |
| T4 captions, motion-absorbed join | 0.327 (92/67) | 0.300 (37/43) |
| sol C1, the model's subtasks | 0.356 (79/67) | **0.574** (51/43) |
| sol C4, C1 snapped to TempSeg cuts | 0.521 (79/67) | **0.596** (51/43) |
| luna C1 / C4 | 0.248 / 0.353 | 0.247 / 0.272 |
| haiku C1 / C4 | 0.155 / 0.211 | 0.145 / 0.145 |

- **GPT-6.1-sol groups the phases far better than any join rule.** On DROID
  it matches TempSeg + gripper split from the captions alone (C1 0.574), and
  with TempSeg's boundary placement it is the best DROID result so far (C4
  0.596). Luna, despite its reasoning, and Haiku group much worse.
- **On Galaxea the grouping is mostly right and the boundaries are not.** Of
  sol's 79 Galaxea segments, 26 match and 32 more overlap a reference with
  tIoU 0.5-0.75: the cut sits on our phase boundary, often about 0.5 s from
  the reference (pick-ups end during the lift, after our cut at the grasp).
  Snapping to TempSeg's cuts recovers part of it (0.356 → 0.521). An extra
  segment at an episode's start or end ("Return the right hand to its
  original position") costs about one match in several episodes.
- **The gripper verdicts add nothing yet.** Sol confirms 58 of 58 Galaxea and
  40 of 42 DROID events, so C2 equals C1 and C3 equals the gripper split up to
  two events. The two it rejected (DROID 000037, a closed gripper pushing a
  drawer) were real subtask boundaries although nothing was grasped.
- **Caveat.** The prompt's conventions and the 1 s snap were chosen with these
  26 episodes in view; the remaining 49 episodes are the check (Q5).

**Instruction ablation** (sol; the log is unchanged, only the text after it):

| Instruction | Galaxea C1 / C4 | DROID C1 / C4 |
|---|---:|---:|
| v1: our conventions (reach, carry and retreat; word variants) + gripper verdicts | 0.356 / 0.521 | 0.574 / 0.596 |
| v2: no conventions; "some captions may be wrong … discontinuous"; kinematics support the captions; subtasks only | 0.277 / 0.453 | 0.505 / 0.465 |
| v3: v2 + FLORA-TempSeg's own granularity policy ("Acquisition and delivery are distinct when observable. Ordinary contacts, adjustments and low-motion pauses within the same ongoing goal do not create new subtasks.") | 0.331 / 0.510 | 0.556 / 0.556 |
| v5: v2 + "Keep basic manipulation actions (pick up, place, pour, open, close, push, wipe ...) as separate subtasks." | 0.345 / **0.538** | 0.547 / 0.505 |

v2 mostly keeps v1's grouping (equal matches in 24 of 26 episodes) and drops
the extra edge segments, but without a granularity rule it chooses its own
granularity: in Galaxea 002 each pick-and-place becomes one "move the pink
cylinder from … to …" (11 segments for 17 subtasks, no match; v1 7), and in
DROID 000061 caption noise splits two scoops into 14 steps. v3 takes the
granularity sentence from FLORA's TempSeg prompt rather than from WGO's
references and recovers most of v1 (Galaxea 002: 19 segments; DROID 000061:
6). v5 states the same granularity in plain words and does about as well
(Galaxea 002: 21 segments, 4 matched; DROID 000061: 10 segments); its DROID
cuts sit later (median +0.27 s from the reference, v1 -0.28 s), and snapping
them to TempSeg's cuts lowers DROID (0.547 → 0.505). Single runs; sol's own repeatability on autolabel's describe task was
about 86 %, so differences of a few hundredths are noise. GPT-6-luna and
Haiku under v2: 0.189 / 0.284 and 0.101 / 0.203 (Galaxea), 0.241 / 0.241 and
0.128 / 0.156 (DROID).

Answers, segments, table and examples: `outputs/kinseg/combine/T4/`
(`examples.md` shows each model's subtasks and verdicts per episode).

**v6: v1 and v5 merged in general terms** (2026-11-05):

```text
Combine consecutive phases into subtasks: one manipulation step that a person would instruct on its own.
- Keep basic manipulation actions (pick up, place, pour, open, close, push, wipe ...) as separate subtasks,
  but group very short phases and phases with similar verbs into the action they belong to.
- Arm movement that only brings the hand or a held object to the next action (reaching, carrying, moving away)
  belongs to that action; after the last action, it belongs to the last action.
- Some captions may be wrong or discontinuous; read each phase together with its neighbours.
- Use the kinematic signals to support the captions. For each gripper close and open, decide whether it
  grasped an object, released one, or did neither.
Answer in JSON only: {"gripper": [...], "subtasks": [...]}
List every gripper close and open of the log. The subtasks cover every phase from 0 to the last, in order, each phase once.
```

The movement line restates v1's first rule as one principle (movement between
actions belongs to the action it leads into), limited to arm movement that
only brings the hand or a held object to the next action, so that moving an
object as the action itself ("Move the bowl to the right") stays an action.
The grouping clause restates v1's second rule; the gripper verdicts return.

| sol, 26 episodes | Galaxea C1 (F1@0.5) | Galaxea C4 | DROID C1 (F1@0.5) | DROID C4 |
|---|---:|---:|---:|---:|
| TempSeg | 0.676 (0.835) | | 0.530 (0.747) | |
| TempSeg + gripper split | 0.681 (0.851) | | 0.587 (0.804) | |
| v1, runs 1 / 2 | 0.356 / 0.372 (0.795 / 0.772) | 0.521 / 0.538 | 0.574 / 0.630 (0.702 / 0.739) | 0.596 / 0.652 |
| v5, runs 1 / 2 | 0.345 / 0.315 (0.745 / 0.740) | 0.538 / 0.493 | 0.547 / 0.526 (0.716 / 0.632) | 0.505 / 0.505 |
| **v6, runs 1 / 2** | **0.511 / 0.526** (0.917 / 0.917) | **0.632 / 0.647** | **0.578 / 0.622** (0.778 / 0.844) | **0.600 / 0.644** |
| v6 without the movement line | 0.373 (0.760) | 0.520 | 0.531 (0.735) | 0.551 |
| v6 without the grouping clause | 0.522 (0.910) | 0.627 | 0.600 (0.778) | 0.622 |

- **v6 is the best instruction on both families**, and it beats TempSeg +
  gripper split on DROID in both runs (C4 0.600 / 0.644 against 0.587). At
  tIoU 0.5 it beats TempSeg on both families (Galaxea 0.917 against 0.835).
- **The movement line carries the gain** (without it, Galaxea 0.373, DROID
  0.531); the grouping clause adds nothing measurable. The line also removes
  the extra end segments: in run 1 no segment starts after the last
  reference ends.
- **Galaxea's granularity is now right** (run 1: 66 segments for 67 subtasks); of the
  32 unmatched, 27 overlap a reference with tIoU 0.5-0.75. What is left on
  Galaxea is boundary placement, which TempSeg's cuts improve (C4) but do not
  close (0.647 against 0.676).
- **Run-to-run noise** with sol: up to 0.056 on DROID (v1 0.574 / 0.630) and
  0.03 on Galaxea, so only differences above about 0.05 count.
- **Gripper verdicts** are almost the event kinds: every run calls 29 of 29
  Galaxea closes grasps and 29 opens releases; on DROID 21 grasps, 19 releases
  and the same 2 "none" (DROID 000037's drawer push, one at a reference cut).
  So C2 equals C1, and C3 (TempSeg split at confirmed events) is the gripper
  split with those two events removed (DROID 0.549 against 0.587).

Example pages: `outputs/kinseg/combine/v6_examples.md` (v6 runs 1 and 2, v6
without the movement line, v1, v5, per episode).

**Annotation conventions: is moving away part of the next action?** Where
reference cuts sit after the last gripper event (within 3 s), measured on the
references themselves:

| References | Cut after an open (release) | Cut after a close (grasp) |
|---|---|---|
| WGO Galaxea (25 episodes) | median 0.04 s; 16 of 18 within 0.5 s | median 0.44 s |
| WGO DROID (50) | median 0.52 s (middle half 0.30-1.08 s) | median 1.70 s (command events) |
| Galaxea Open-World subtasks (415, autolabel's 2026-10-15 run) | median -0.07 s; 97 of 134 within 0.5 s | median 1.33 s |
| B1K skills (500) | median 1.47 s (middle half 0.87-2.17 s) | median 2.00 s |

Galaxea's annotations (WGO and Open-World) end a place at the release, so
moving away begins the next subtask; B1K's skills end well after the release,
so moving away stays with the previous skill. Datasets therefore differ, as
expected, and where a pick-up ends differs too (WGO Galaxea 0.4 s after the
grasp, Open-World 1.3 s). On WGO, v6 with the opposite attachment ("moving
away after an action belongs to the action it follows") falls from 0.511 /
0.526 to 0.421 on Galaxea and to 0.556 on DROID (0.578 / 0.622): about 0.1 on
Galaxea for a wrong convention. The attachment should be a per-dataset
setting, chosen from a few annotated episodes; the measurement above is the
way to choose it.

**Galaxea boundary placement.** For every internal reference cut, the signed
offset of the nearest cut (v6 run 1):

| Galaxea, reference cut after | model's cut | our nearest phase cut | TempSeg's nearest cut |
|---|---|---|---|
| a close (end of a pick-up), 31 cuts | median +0.26 s, 9 within 0.25 s | -0.25 s, 14 within 0.25 s | +0.29 s, 12 |
| an open (end of a place), 14 cuts | -0.45 s, 2 | +0.05 s, 6 | +0.26 s, 4 |

The candidates are there; the model picks a neighbouring phase boundary:
after a grasp it ends the pick-up one phase late, after a release one phase
early. A kinematic anchor fixes the first: **C5** moves a model cut that lies
within one phase of a confirmed grasp to the end of the grasp's phase (the
grasp completes the pick-up), and **C6** snaps C5 to TempSeg's cuts.

| v6, sol | Galaxea C1 / C4 / C5 / **C6** | DROID C1 / C4 / C5 / **C6** |
|---|---:|---:|
| run 1 | 0.511 / 0.632 / 0.571 / **0.677** | 0.578 / 0.600 / 0.578 / **0.600** |
| run 2 | 0.526 / 0.647 / 0.571 / **0.677** | 0.622 / 0.644 / 0.622 / **0.644** |

With C6, Galaxea equals TempSeg (0.676) in both runs and DROID stays above
TempSeg + gripper split (0.587). Anchoring at releases instead hurts (Galaxea
0.42-0.45), and with v1's answers the grasp anchor helps Galaxea but costs
DROID 0.04. The anchor encodes the WGO Galaxea convention (a pick-up ends
with the grasp), so it belongs with the per-dataset settings.

**Gripper verdicts, redesigned.** Without the verdict question (v6-noverdict)
the grouping stays within noise (Galaxea 0.478, DROID 0.667); asking instead
whether a subtask ends or begins at each event (v6-boundary) gives Galaxea
0.541, DROID 0.533, and its answers do not locate cuts: on Galaxea 21 of 22
closes called "neither" sit next to a reference cut, and on DROID 11 of 16
closes called "ends" do not. The grasp/release verdicts stay: they are what
C5 anchors on, and they reject the two DROID events where nothing was grasped.

## 6. Results (all 75 episodes unless noted)

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
| 11 | Captioner inputs varied (8 variants, 26 episodes; section 5.3); best operation join | 0.362 (TempSeg 0.676) | 0.306 (TempSeg 0.530) | inputs shift agreement inside and across subtasks together |
| 12 | Our prompt template, 5 runs (26 episodes; section 5.6); best operation join | 0.346 | 0.300 | best separation so far (previous interval), but the join rule wastes it |
| 13 | GPT-6.1-sol combines T4's captions from a status log (26 episodes; 5.7), snapped to TempSeg cuts | 0.521 (TempSeg 0.676) | **0.596** (TempSeg 0.530, + gripper split 0.587) | Galaxea: right grouping, boundaries on our cuts ~0.5 s off |
| 14 | the same with instruction v6 (5.7), two runs, snapped | 0.632 / 0.647 | **0.600 / 0.644** | Galaxea: boundary placement (F1@0.5 0.917, above TempSeg's 0.835) |

Upper bounds (from the references, not methods): keeping exactly the cuts
nearest each reference cut within 0.5 s gives 0.728 / 0.729 with our cuts,
0.741 / 0.374 with TempSeg's, and **0.917 / 0.850 with both**. The boundaries
are available; choosing them is the problem.

## 7. Where we stand

- **Best combination**: TempSeg + gripper split, 0.690 / 0.581 (TempSeg alone
  0.687 / 0.562).
- **Room**: the perfect judge on ours and TempSeg's cuts, 0.917 / 0.850; the
  two candidate sets together contain almost every reference cut (94 of 98 on
  Galaxea, 80 of 100 on DROID, within 0.5 s).
- **Closed**: both directions as tried (section 3): TempSeg cannot be made to
  judge our cuts, and the released models do not judge fragments.
- **Measured 2026-11-04/05**: the Captioner's inputs (section 5) and our
  prompt template (5.6). Best recipe: FLORA's full video context, the task
  brief and the previous interval with notes in our template; still far below
  TempSeg at fragment scale, now limited by the join rule (Q8).
- **Nothing is running.** Next steps wait for the decisions in section 8.

## 8. Open questions

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
- **Q7 Object lists.** The background's scene part (target objects, other
  objects, workspace) needs per-episode lists that WGO-Bench lacks. The nouns
  of the task sentence (section 5.6) added nothing; lists seen in the video
  (by a VL model) would be the next option.
- **Q8 Join rule.** With the better captions of section 5.6, which rule turns
  them into subtasks: verb classes instead of verbs, objects, gripper events
  as anchors for the pick-up/place boundary, or a subtask-scale check by the
  Captioner?

## 9. Infrastructure

- **Local**: H20 GPUs, conda env `flora`; TempSeg weights in
  `~/.cache/flora/models` and the Hugging Face cache.
- **Remote** (shared root account, `kinseg/remote/flora-remote`): 8 × RTX PRO
  5000 72 GB; FLORA's TempSeg, Captioner and Rewarder weights already on
  `/cfs`; our data and outputs in `/cfs/users/yplu/kinseg`. The script never
  deletes anything there, keeps no credentials there and refuses to run a
  checkout someone changed.
- **Throughput**: TempSeg 3-6 s per 16 s window on the remote; Captioner about
  3.6 s per caption (about 85 min for all 1,422 phases on one worker); a
  26-episode variant (510 captions) took 20-60 min on two GPUs, less with
  TARGET only, and a second process on the same output folder with the
  episodes reversed halves that (each skips what the other has written).
- **Known issues and their handling**:
  - `/cfs` is Lustre, and the first read of an idle weight file rewrites its
    timestamps (probably tiered storage), which trips FLORA's integrity check.
    `local_models.load` first reads every checked file and waits until the
    timestamps stay unchanged for 30 s (101.9 GiB of Captioner in 43 s), then
    loads; the timestamp failure alone is retried as a backstop.
  - The timestamps also change while another process reads the same files: on
    2026-11-04, of three Captioner runs started three minutes apart, two
    failed every attempt (a 10 GB shard's modification time moved from
    19:45:31 to 19:52:26 while the others were loading). Model loads now run
    one at a time per machine (`flock` on `/tmp/kinseg-model-load.lock`); the
    next seven loads, five of which waited for another, all passed on the
    first attempt.
  - The remote GPUs are shared with other users. `flora-remote run --gpus
    auto:N[:GiB]` picks the N GPUs with the most free memory, and the job waits
    up to an hour for the memory it needs (22 GiB for TempSeg or the 9B judge,
    62 GiB spread over the GPUs for the Captioner) instead of failing half-way.
    Dedicated GPUs need a ticli worker (`flora-remote launch --log NAME
    --cpu C --gpu G --mem M -- CMD`; fixed flags in `FLORA_REMOTE_WORKER_FLAGS`:
    GPU type `HCC-N50p-72G-vr`, the notebook ID, and a worker key in
    `/cfs/users/yplu/kinseg/.worker_ssh/`). A worker inherits the notebook's
    resource group. The launch reaches the TIONE API, which refuses it until
    the notebook's Worker feature is enabled (a notebook setting, or the API's
    ModifyNotebook). ticli must run without the proxy.
  - Runs skip episodes whose results exist, so a relaunch resumes.
  - The link to the server limits each connection to about 44 KB/s
    (`push --parallel`).

## 10. Files

| File | Role |
|---|---|
| `segment.py` | TempSeg runner (free, `--kinematic` constrained, `--probe` beliefs) |
| `decoding.py`, `check_decoding.py` | greedy decoding with whole-value choices; equivalence check |
| `merge.py`, `beliefs.py` | post-hoc combinations; TempSeg beliefs at our cuts |
| `judge.py` | vision-language same-or-new judge |
| `caption_merge.py` | Captioner on our phases (video context, background, robot state, previous caption); join rules; `compare` |
| `background.py` | the Captioner's background text, after the workspace template |
| `caption_prompt.py` | our Captioner prompt template (information and notes before FLORA's instruction) |
| `combine.py` | a language model combines captioned phases from a status log; scores C1-C4 |
| `caption_examples.py` | markdown pages: each run's captions per phase and each method's segments per episode |
| `local_models.py` | local weight folders, warming, one load at a time, the load retry |
| `remote/flora-remote` | remote server control |
| `test_kinseg.py` | CPU tests |
