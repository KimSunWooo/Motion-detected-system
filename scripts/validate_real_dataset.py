#!/usr/bin/env python3
"""Validate processed real pose sequences.

PYTHONPATH=src python scripts/validate_real_dataset.py --data data/real/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.real.dataset import default_processed_root  # noqa: E402
from helmet_action.real.validate import validate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_processed_root())
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = validate_dataset(args.data)
    slim = {k: v for k, v in report.items() if k != "sequences"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    if not report.get("available"):
        print("REAL DATASET: NOT AVAILABLE")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report.get("n_invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
