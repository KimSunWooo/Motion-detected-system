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

`data/real/raw/` and `data/real/processed/` are gitignored except `.gitkeep`.

## Recommended capture

People: **3–5 or more** (`P001` …).

Camera: high-angle CCTV-like, pitch about **30–70°**.

Distance: `near`, `medium`, `far` (separate folders if you can).

Actions (folder names under each person):

| Folder | Meaning |
|---|---|
| `helmet_remove` | take the helmet / hard-hat off |
| `helmet_put_on` | put it on |
| `helmet_adjust` | tug / straighten, do not remove |
| `head_scratch` | one-hand scratch |
| `head_touch` | rest a hand on the head |
| `wipe_sweat` | brow wipe |
| `raise_arms` | stretch / raise without grasping a helmet |
| `phone_near_head` | phone to ear |

Each action: **10–20 takes per person**.

Variations (spread them across takes):

- fast / slow
- left-hand first / right-hand first / both hands
- one hand then two
- frontal / side
- head turned
- partial occlusion (a pole, another worker, a sleeve)

A real safety helmet or a similar cap is fine. Capture only in an environment
that does not put the wearer at risk.

## Folder layout

```
data/real/raw/
  P001/
    helmet_remove/
    helmet_put_on/
    helmet_adjust/
    head_scratch/
    head_touch/
    wipe_sweat/
    raise_arms/
    phone_near_head/
  P002/
    ...
```

Optional camera folder between subject and action:

```
data/real/raw/P001/near/helmet_remove/take_01.mp4
```

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

Subject IDs are stored on every sequence so a later leave-one-subject
fine-tune cannot leak the same person into both splits.

## What this will not tell you

Synthetic scores are **not** construction-site CCTV performance.
Pose alone cannot decide static helmet presence. Helmet State stays
UNKNOWN without a helmet-presence detector.
