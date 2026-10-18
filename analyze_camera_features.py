from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sign_baseline import EXPECTED_FEATURES, EXPECTED_FRAMES, pooled_sequence_features, validate_sequence


EPS = 1e-8

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def inspect_sample(path: Path) -> dict[str, object]:
    array = validate_sequence(np.load(path, allow_pickle=False))
    left = np.max(np.abs(array[:, :63]), axis=1) > EPS
    right = np.max(np.abs(array[:, 63:]), axis=1) > EPS
    both = left & right
    any_valid = left | right
    selected = array[any_valid]
    features = pooled_sequence_features(array, min_selected_frames=1)
    if features is None:
        raise ValueError(f"No valid hand frames: {path}")
    return {
        "path": path,
        "array": array,
        "features": features,
        "left": left,
        "right": right,
        "both": both,
        "any_valid": any_valid,
        "selected": selected,
    }


def _range(values: np.ndarray) -> str:
    return f"min={float(values.min()):.6g}, max={float(values.max()):.6g}, mean={float(values.mean()):.6g}, std={float(values.std()):.6g}"


def print_sample(sample: dict[str, object]) -> None:
    path = sample["path"]
    array = sample["array"]
    features = sample["features"]
    left = sample["left"]
    right = sample["right"]
    both = sample["both"]
    any_valid = sample["any_valid"]
    selected = sample["selected"]
    assert isinstance(path, Path)
    assert isinstance(array, np.ndarray) and array.shape == (EXPECTED_FRAMES, EXPECTED_FEATURES)
    assert isinstance(features, np.ndarray) and features.shape == (252,)
    assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
    assert isinstance(both, np.ndarray) and isinstance(any_valid, np.ndarray)
    assert isinstance(selected, np.ndarray)

    any_count = int(any_valid.sum())
    left_count = int(left.sum())
    right_count = int(right.sum())
    both_count = int(both.sum())
    print(f"\n[{path}]")
    print(f"shape={array.shape} dtype={array.dtype}")
    print(
        f"frames: any_valid={any_count}/60 ({any_count / 60:.1%}), both={both_count}/60, "
        f"left_slot={left_count}/60 ({left_count / 60:.1%}), "
        f"right_slot={right_count}/60 ({right_count / 60:.1%}), "
        f"left_only={int((left & ~right).sum())}, right_only={int((right & ~left).sum())}, "
        f"zero={int((~any_valid).sum())}"
    )
    print(f"raw any_valid values: {_range(selected)}")
    print(f"raw left slot: {_range(selected[:, :63]) if left_count else 'no selected values'}")
    print(f"raw right slot: {_range(selected[:, 63:]) if right_count else 'no selected values'}")
    means, stds = features[:126], features[126:]
    print(f"252-D mean part: {_range(means)}")
    print(f"252-D std part:  {_range(stds)}")
    print("feature first 10 (mean):", np.array2string(means[:10], precision=6, separator=", "))
    print("feature last 10 (std): ", np.array2string(stds[-10:], precision=6, separator=", "))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a saved webcam request with an offline (60,126) training sample using scheme-B features."
    )
    parser.add_argument(
        "offline_sample",
        nargs="?",
        type=Path,
        default=Path("sign_data_a/0_你好_A_0.npy"),
        help="offline training sample path",
    )
    parser.add_argument("--debug-dir", type=Path, default=Path("debug_frames"))
    parser.add_argument("--camera-sample", type=Path, default=None, help="specific debug .npy; defaults to newest")
    args = parser.parse_args()

    if args.camera_sample:
        camera_path = args.camera_sample
    else:
        candidates = list(args.debug_dir.glob("debug_*.npy")) if args.debug_dir.is_dir() else []
        if not candidates:
            parser.error(f"no debug_*.npy files found in {args.debug_dir}; perform a camera prediction first")
        camera_path = max(candidates, key=lambda item: item.stat().st_mtime_ns)

    for path in (args.offline_sample, camera_path):
        if not path.is_file():
            parser.error(f"sample does not exist: {path}")

    offline = inspect_sample(args.offline_sample)
    camera = inspect_sample(camera_path)
    print("方案 B: each sample's any_valid frames -> mean(126) + std(126), total 252 values")
    print_sample(offline)
    print_sample(camera)

    offline_count = int(np.asarray(offline["any_valid"]).sum())
    camera_count = int(np.asarray(camera["any_valid"]).sum())
    offline_lr = np.asarray([np.asarray(offline["left"]).sum(), np.asarray(offline["right"]).sum()], dtype=float)
    camera_lr = np.asarray([np.asarray(camera["left"]).sum(), np.asarray(camera["right"]).sum()], dtype=float)
    offline_slot_rates = offline_lr / max(offline_count, 1)
    camera_slot_rates = camera_lr / max(camera_count, 1)
    print("\n[comparison]")
    print(f"any_valid frame difference: camera - offline = {camera_count - offline_count}")
    print(f"left/right slot frame-count difference: camera - offline = {camera_lr - offline_lr}")
    print(f"left/right slot share among valid frames: offline={offline_slot_rates}, camera={camera_slot_rates}")
    if offline_count >= 30 and camera_count < 30:
        print("assessment: camera has fewer than the original 30-frame calibration gate; its score is outside the calibrated quality range.")
    elif np.max(np.abs(camera_slot_rates - offline_slot_rates)) > 0.10:
        print("assessment: hand-slot share differs; confirm the actual performed sign and MediaPipe handedness mapping before attributing it to a left/right swap.")
    else:
        print("assessment: hand-slot shares are aligned. Different absolute counts reflect different valid-frame totals; this pair alone cannot establish overall preprocessing equivalence.")
    print("Limitation: this reports numeric/slot differences for one pair; it cannot identify semantic correctness or prove camera-vs-collector MediaPipe equivalence by itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
