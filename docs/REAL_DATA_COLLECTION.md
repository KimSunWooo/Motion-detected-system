# Real Pose Data Collection Guide

This project evaluates **helmet-removal ACTION** from pose sequences.
It does **not** learn from RGB pixels. Real video is converted to COCO-17 pose
and then discarded from the training loop.

```
Real Video → YOLO Pose → Pose Sequence → Synthetic-trained model (zero-shot)
```

Stage 4 of this repo is **ready and waiting for data**. Do not commit raw video.

## Privacy

- Do **not** git-commit real video or processed pose dumps of real people.
- Store raw video outside the repository after evaluation if possible.
- Use anonymized subject IDs only: `P001`, `P002`, … never names.
- Folder names like `person_01` are converted to `P001` by the extractor.

`data/real/raw/` and `data/real/processed/` are gitignored except `.gitkeep`
and the empty capture folders.

## Pilot protocol (first 180 clips)

This data is **not used for training**. The only goal of the first pilot is
**zero-shot measurement of the synthetic-trained model**.

| | Count |
|---|---|
| Subjects | 3 (`P001`, `P002`, `P003`) |
| Actions per subject | HELMET_REMOVE 15, HELMET_ADJUST 15, HEAD_SCRATCH 15, HEAD_TOUCH 15 |
| Total clips | 3 × 4 × 15 = **180** |

Do not expand the label set for this pilot. Additional actions
(HELMET_PUT_ON, WIPE_SWEAT, RAISE_ARMS, PHONE_NEAR_HEAD) can wait until
after the 180-clip freeze.

## Capture conditions

Camera: high-angle CCTV-like, pitch **30–70°**.

Distance (spread across takes): `near` / `medium` / `far`.

Speed (spread across takes): `slow` / `normal` / `fast`.

Handedness: left- and right-dominant variation.

HELMET_REMOVE takes should include, if possible:

- brim grasp
- lateral lift (left / right)
- one-hand then two-hand

A real safety helmet or a similar cap is fine. Capture only in an environment
that does not put the wearer at risk.

## Folder layout

```
data/real/raw/
  P001/
    helmet_remove/
    helmet_adjust/
    head_scratch/
    head_touch/
  P002/
    ...
  P003/
    ...
```

Optional camera folder between subject and action:

```
data/real/raw/P001/near/helmet_remove/take_01.mp4
```

## Hand-written manifest

Copy `docs/real_dataset_manifest_template.csv` and fill one row per clip
while shooting (do not rely on memory later):

```
subject_id,video_file,label,camera_pitch,camera_distance,dominant_hand,speed,occlusion,notes
```

Allowed `label` values:

- HELMET_REMOVE
- HELMET_PUT_ON
- HELMET_ADJUST
- HEAD_SCRATCH
- HEAD_TOUCH
- WIPE_SWEAT
- RAISE_ARMS
- PHONE_NEAR_HEAD

## Extract / validate / zero-shot

```bash
export PYTHONPATH=src
python scripts/extract_real_pose.py \
  --input data/real/raw \
  --output data/real/processed \
  --pose-model yolo11n-pose.pt

python scripts/validate_real_dataset.py --data data/real/processed

python scripts/evaluate_real_pose.py \
  --data data/real/processed \
  --model models/action_classifier.joblib
```

Zero-shot means the **synthetic-trained** sklearn model is frozen.
Real sequences are never used to fit weights in this stage.

When no videos are present the evaluator prints `REAL DATASET: NOT AVAILABLE`
and does **not** write dummy numeric metrics.

Subject IDs are stored on every sequence so a later leave-one-subject
fine-tune cannot leak the same person into both splits.

## Zero-shot report schema (once 180 clips exist)

Overall: Accuracy, Macro F1

HELMET_REMOVE: Precision, Recall, F1, FNR, FPR

Decision: ALERT / WATCH / UNKNOWN / SAFE rates

Safety: False Safe Rate, False Alarm Rate

Pose: PoseQuality mean/std, missing wrist ratio, missing ear ratio

Per subject: P001, P002, P003

Per condition: near / medium / far and slow / normal / fast

## What this will not tell you

Synthetic scores are **not** construction-site CCTV performance.
Pose alone cannot decide static helmet presence. Helmet State stays
UNKNOWN without a helmet-presence detector.
