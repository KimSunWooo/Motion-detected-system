#!/usr/bin/env python3
"""Zero-shot real pose evaluation. Never retrains on real data.

PYTHONPATH=src python scripts/evaluate_real_pose.py \\
  --data data/real/processed --model models/action_classifier.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.real.dataset import default_processed_root  # noqa: E402
from helmet_action.real.evaluate import NOT_AVAILABLE, evaluate_real_zero_shot  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_processed_root())
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args(argv)

    if not Path(args.model).exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    report = evaluate_real_zero_shot(args.data, args.model, feature_version=args.feature_version)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "real_zero_shot.json"
    csv_path = out / "real_zero_shot.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not report.get("dataset_available"):
        print(NOT_AVAILABLE)
        print("REAL METRICS NOT MEASURED")
        # Do not write a numeric CSV that looks like a measured result.
        csv_path.write_text("status,dataset_available\nREAL DATASET: NOT AVAILABLE,false\n", encoding="utf-8")
        print(f"wrote {json_path} (no metrics)")
        return 0

    metrics = report.get("metrics") or {}
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "value"])
        w.writeheader()
        for k, v in metrics.items():
            if not isinstance(v, (list, dict)):
                w.writerow({"metric": k, "value": v})
    print(json.dumps({k: report[k] for k in report if k not in ("per_sequence", "validation")}, indent=2))
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
