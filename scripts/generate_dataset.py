#!/usr/bin/env python3
"""Generate a reproducible synthetic pose-action dataset.

python scripts/generate_dataset.py --samples 5000 --output data/synthetic
python scripts/generate_dataset.py --samples 500 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.config import load_config
from helmet_action.synthetic.generator import GENERATOR_VERSION, generate_one, scenario_mix


def _split_counts(n: int) -> dict[str, int]:
    n_test = max(1, int(round(n * 0.20)))
    n_val = max(1, int(round(n * 0.16)))
    n_train = max(1, n - n_test - n_val)
    return {"train": n_train, "validation": n_val, "test": n_test}


def _pad(seq: np.ndarray, conf: np.ndarray, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    t = seq.shape[0]
    k = np.zeros((t_max, 17, 2), dtype=np.float32)
    c = np.zeros((t_max, 17), dtype=np.float32)
    t_use = min(t, t_max)
    k[:t_use] = seq[:t_use]
    c[:t_use] = conf[:t_use]
    return k, c


def generate_split(n: int, split: str, bag: list[str], seed0: int, t_max: int) -> tuple[dict, list[dict]]:
    keypoints, confs, labels, scenarios, lengths, seeds = [], [], [], [], [], []
    metas: list[dict] = []
    for i in range(n):
        scenario = bag[i % len(bag)]
        seed = seed0 + i * 17 + (0 if split == "train" else 10_000 if split == "validation" else 80_000)
        seq, conf, meta = generate_one(seed=seed, scenario=scenario, split=split)
        k, c = _pad(seq, conf, t_max)
        keypoints.append(k)
        confs.append(c)
        labels.append(meta.label)
        scenarios.append(meta.scenario)
        lengths.append(meta.n_frames)
        seeds.append(meta.seed)
        metas.append(
            {
                "seed": meta.seed,
                "split": meta.split,
                "scenario": meta.scenario,
                "label": meta.label,
                "generator_version": meta.generator_version,
                "camera": meta.camera,
                "body": meta.body,
                "action": meta.action,
                "noise": meta.noise,
                "n_frames": meta.n_frames,
            }
        )
    return {
        "keypoints": np.stack(keypoints),
        "confidences": np.stack(confs),
        "labels": np.array(labels),
        "scenarios": np.array(scenarios),
        "lengths": np.array(lengths, dtype=np.int32),
        "seeds": np.array(seeds, dtype=np.int64),
    }, metas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20240916)
    args = parser.parse_args(argv)

    n = 500 if args.quick and args.samples == 2000 else args.samples
    if args.quick:
        n = min(n, 500)
    cfg = load_config()
    t_max = int(cfg.get("window.max_frames", 80))
    bag = scenario_mix(quick=args.quick)
    counts = _split_counts(n)
    out: Path = args.output
    out.mkdir(parents=True, exist_ok=True)

    all_meta: dict[str, list] = {}
    for split, count in counts.items():
        arrays, metas = generate_split(count, split, bag, args.seed, t_max)
        np.savez_compressed(out / f"{split}.npz", **arrays)
        all_meta[split] = metas
        print(f"[{split}] {count} sequences → {out / (split + '.npz')}")

    payload = {
        "random_seed": args.seed,
        "generator_version": cfg.get("synthetic.generator_version", GENERATOR_VERSION),
        "samples": n,
        "quick": bool(args.quick),
        "counts": counts,
        "scenarios": bag if args.quick else sorted(set(bag)),
        "split_policy": {
            "train_val": "in-distribution camera/body/noise ranges",
            "test": "OOD: wider pitch/distance, higher keypoint noise, different speed and body proportion",
        },
        "records": all_meta,
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"wrote {out / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
