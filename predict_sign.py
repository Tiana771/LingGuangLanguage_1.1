"""Run offline prediction on one .npy sequence using a trained baseline bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from sign_baseline import predict_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="one .npy file with shape (60, 126)")
    parser.add_argument("--model", type=Path, required=True, help="path to model.joblib")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model is not ready: file not found: {args.model}")
    if not args.sequence.is_file():
        raise SystemExit(f"Sequence file not found: {args.sequence}")
    bundle = joblib.load(args.model)
    result = predict_sequence(np.load(args.sequence, allow_pickle=False), bundle, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
