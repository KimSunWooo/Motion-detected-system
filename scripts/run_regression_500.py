#!/usr/bin/env python3
"""Re-run the 500-sample seed-42 regression without clobbering baseline artifacts.

PYTHONPATH=src python scripts/run_regression_500.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_v1_v2 import train_eval_version  # noqa: E402
import generate_dataset  # noqa: E402
import numpy as np


def main() -> int:
    data = ROOT / "outputs" / "evaluation" / "regression" / "data_seed_42"
    if not (data / "train.npz").exists():
        generate_dataset.main(["--samples", "500", "--output", str(data), "--seed", "42"])
    npz_tr = np.load(data / "train.npz", allow_pickle=True)
    npz_te = np.load(data / "test.npz", allow_pickle=True)
    v1 = train_eval_version(npz_tr, npz_te, "v1", 42, data.parent / "model_seed_42_v1.joblib")
    v2 = train_eval_version(npz_tr, npz_te, "v2", 42, data.parent / "model_seed_42_v2.joblib")
    payload = {
        "name": "regression_500_stage4",
        "seed": 42,
        "samples": 500,
        "note": "Does not overwrite regression_500_summary.json from Stage 3.",
        "v1": v1,
        "v2": v2,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    dest = ROOT / "outputs" / "evaluation" / "regression_500_stage4.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"V1 acc={v1['accuracy']:.3f} FNR={v1['helmet_remove_fnr']:.3f}")
    print(f"V2 acc={v2['accuracy']:.3f} FNR={v2['helmet_remove_fnr']:.3f}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
