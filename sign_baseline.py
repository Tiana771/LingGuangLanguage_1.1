"""Utilities shared by the isolated-sign baseline trainer and prediction API."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_FRAMES = 60
EXPECTED_FEATURES = 126
FILENAME_PATTERN = re.compile(r"^\d+_(?P<label>.+?)(?:_[AB])?_(?P<sample_id>\d+)\.npy$")


@dataclass
class SampleRecord:
    path: str
    label: str | None
    shape: list[int] | None
    dtype: str | None
    finite: bool | None
    all_zero: bool | None
    status: str
    detail: str | None = None


def parse_label(path: Path) -> str:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"filename does not match '<numeric-class>_<label>_<numeric-id>.npy': {path.name}"
        )
    return match.group("label")


def scan_dataset(data_dir: Path) -> tuple[list[SampleRecord], dict[str, Any]]:
    paths = sorted(data_dir.glob("*.npy"))
    records: list[SampleRecord] = []
    for path in paths:
        try:
            label = parse_label(path)
        except ValueError as exc:
            records.append(SampleRecord(str(path), None, None, None, None, None, "invalid", str(exc)))
            continue
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            records.append(SampleRecord(str(path), label, None, None, None, None, "invalid", str(exc)))
            continue

        all_zero = bool(array.size > 0 and np.count_nonzero(array) == 0)
        shape = list(array.shape)
        dtype = str(array.dtype)
        if not np.issubdtype(array.dtype, np.number):
            finite = False
            status, detail = "invalid", f"expected numeric dtype, got {array.dtype}"
        else:
            finite = bool(np.isfinite(array).all())
            if array.shape != (EXPECTED_FRAMES, EXPECTED_FEATURES):
                status, detail = "invalid", f"expected (60, 126), got {tuple(array.shape)}"
            elif not finite:
                status, detail = "invalid", "contains NaN or Inf"
            elif all_zero:
                status, detail = "invalid", "all-zero sample"
            else:
                status, detail = "ok", None
        records.append(SampleRecord(str(path), label, shape, dtype, finite, all_zero, status, detail))

    valid = [record for record in records if record.status == "ok"]
    counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = {}
    for record in valid:
        assert record.label is not None
        counts[record.label] = counts.get(record.label, 0) + 1
    for record in records:
        if record.shape is not None:
            shape_key = str(tuple(record.shape))
            shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        if record.dtype is not None:
            dtype_counts[record.dtype] = dtype_counts.get(record.dtype, 0) + 1
    summary = {
        "directory": str(data_dir),
        "files_found": len(paths),
        "valid_samples": len(valid),
        "invalid_samples": len(records) - len(valid),
        "class_counts": dict(sorted(counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "non_finite_samples": sum(record.finite is False for record in records),
        "all_zero_samples": sum(record.all_zero is True for record in records),
        "abnormal_shape_samples": sum(
            record.shape is not None and tuple(record.shape) != (EXPECTED_FRAMES, EXPECTED_FEATURES)
            for record in records
        ),
        "records": [asdict(record) for record in records],
    }
    return records, summary


def sequence_features(sequence: np.ndarray, temporal_bins: int = 10) -> np.ndarray:
    """Average each channel inside fixed temporal bins; preserve whole-sequence samples."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape != (EXPECTED_FRAMES, EXPECTED_FEATURES):
        raise ValueError(f"expected sequence shape (60, 126), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("sequence contains NaN or Inf")
    if temporal_bins < 1 or EXPECTED_FRAMES % temporal_bins:
        raise ValueError("temporal_bins must be a positive divisor of 60")
    return array.reshape(temporal_bins, EXPECTED_FRAMES // temporal_bins, EXPECTED_FEATURES).mean(axis=1).reshape(-1)


def validate_sequence(sequence: Any) -> np.ndarray:
    try:
        array = np.asarray(sequence, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("sequence must be a numeric array") from exc
    if array.shape != (EXPECTED_FRAMES, EXPECTED_FEATURES):
        raise ValueError(f"expected sequence shape (60, 126), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("sequence contains NaN or Inf")
    return array


def pooled_sequence_features(sequence: Any, min_selected_frames: int = 30) -> np.ndarray | None:
    """Return scheme-B any-valid mean+std features, or None below the frame gate."""
    array = validate_sequence(sequence)
    left_on = np.max(np.abs(array[:, :63]), axis=1) > 1e-8
    right_on = np.max(np.abs(array[:, 63:]), axis=1) > 1e-8
    selected = left_on | right_on
    if int(selected.sum()) < min_selected_frames:
        return None
    rows = array[selected].astype(np.float32)
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)]).astype(np.float32)


def predict_sequence(sequence: Any, bundle: dict[str, Any], threshold: float = 0.55,
                     margin_threshold: float = 0.0,
                     min_selected_frames: int | None = None) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if margin_threshold < 0:
        raise ValueError("margin_threshold must be non-negative")
    array = validate_sequence(sequence)
    feature_config = bundle.get("feature_config", {})
    mode = feature_config.get("mode")
    selected_frames = None
    if mode == "B":
        left_on = np.max(np.abs(array[:, :63]), axis=1) > 1e-8
        right_on = np.max(np.abs(array[:, 63:]), axis=1) > 1e-8
        selected_frames = int((left_on | right_on).sum())
        minimum = int(min_selected_frames if min_selected_frames is not None
                      else feature_config.get("min_selected_frames", 30))
        if minimum < 1:
            raise ValueError("min_selected_frames must be positive")
        if selected_frames < minimum:
            return {
                "status": "ok", "candidate_label": None, "label": None,
                "confidence": 0.0, "accepted": False,
                "message": "暂未识别，请重试", "valid_frame_count": selected_frames,
            }
        features = pooled_sequence_features(array, minimum).reshape(1, -1)
    else:
        features = sequence_features(array).reshape(1, -1)
    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    predicted_label = model.predict(features)[0]
    try:
        probability_labels = list(getattr(model, "classes_", bundle["labels"]))
        if mode == "B":
            predicted_index = int(np.argmax(probabilities))
            predicted_label = probability_labels[predicted_index]
        else:
            predicted_index = probability_labels.index(predicted_label)
    except ValueError as exc:
        raise ValueError(f"model predicted an unknown label: {predicted_label}") from exc
    confidence = float(probabilities[predicted_index])
    sorted_probabilities = np.sort(np.asarray(probabilities, dtype=np.float64))
    top2_margin = float(sorted_probabilities[-1] - sorted_probabilities[-2]) if len(sorted_probabilities) > 1 else 1.0
    accepted = confidence >= threshold and top2_margin >= margin_threshold
    label = str(predicted_label) if accepted else None
    return {
        "status": "ok",
        "candidate_label": str(predicted_label),
        "label": label,
        "confidence": confidence,
        "top2_margin": top2_margin,
        "accepted": accepted,
        "message": str(predicted_label) if accepted else "暂未识别，请重试",
        **({"valid_frame_count": selected_frames} if selected_frames is not None else {}),
    }
