"""Train a small, single-recorder isolated-sign baseline from .npy sequences."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sign_baseline import SampleRecord, scan_dataset, sequence_features


def load_valid_samples(records: list[SampleRecord]) -> tuple[list[np.ndarray], list[str], list[str]]:
    samples: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    for record in records:
        if record.status != "ok" or record.label is None:
            continue
        samples.append(np.load(record.path, allow_pickle=False).astype(np.float32))
        labels.append(record.label)
        paths.append(record.path)
    return samples, labels, paths


def stratified_subset_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in sorted(set(labels.tolist())):
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) < 3:
            raise ValueError(
                f"class {label!r} has {len(class_indices)} samples; at least 3 are needed "
                "to create a subset with train and validation examples"
            )
        count = min(len(class_indices), max(3, int(round(len(class_indices) * fraction))))
        chosen.extend(rng.choice(class_indices, size=count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=int)


def make_output_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = parent / f"run-{stamp}"
    suffix = 1
    while output.exists():
        output = parent / f"run-{stamp}-{suffix}"
        suffix += 1
    output.mkdir()
    return output


def train(args: argparse.Namespace) -> Path:
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be > 0 and <= 1")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be > 0 and < 1")

    records, scan = scan_dataset(args.data_dir)
    samples, labels_list, paths = load_valid_samples(records)
    if not samples:
        raise RuntimeError(f"No valid samples found under {args.data_dir}")
    labels = np.asarray(labels_list)
    selected = stratified_subset_indices(labels, args.fraction, args.seed)
    selected_labels = labels[selected]
    selected_paths = np.asarray(paths)[selected]

    class_counts = {label: int(np.sum(selected_labels == label)) for label in sorted(set(selected_labels))}
    if any(count < 2 for count in class_counts.values()):
        raise ValueError(f"Selected subset cannot support stratification: {class_counts}")
    train_idx, validation_idx = train_test_split(
        np.arange(len(selected)),
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=selected_labels,
    )

    selected_samples = [samples[index] for index in selected]
    features = np.stack([sequence_features(sample) for sample in selected_samples])
    y = selected_labels
    x_train, x_validation = features[train_idx], features[validation_idx]
    y_train, y_validation = y[train_idx], y[validation_idx]

    model = make_pipeline(
        StandardScaler(),
        SVC(C=args.c, kernel="rbf", class_weight="balanced", probability=True, random_state=args.seed),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    validation_probabilities = model.predict_proba(x_validation)
    classes = model.named_steps["svc"].classes_.tolist()
    class_indices = {label: index for index, label in enumerate(classes)}
    validation_confidences = np.asarray(
        [validation_probabilities[row, class_indices[label]] for row, label in enumerate(predictions)]
    )
    accepted = validation_confidences >= args.threshold
    report = classification_report(
        y_validation, predictions, labels=classes, target_names=classes, zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(y_validation, predictions, labels=classes)
    output_dir = make_output_dir(args.output_dir)
    bundle = {
        "model": model,
        "labels": classes,
        "feature_config": {"method": "mean_pool", "temporal_bins": 10, "input_shape": [60, 126]},
        "confidence_note": "SVC probability is an experimental score; tune rejection threshold on validation data.",
        "training": {
            "seed": args.seed,
            "fraction": args.fraction,
            "validation_fraction": args.validation_fraction,
            "c": args.c,
            "recorder_count": 1,
            "evaluation_scope": "single-recorder workflow validation only; no unseen-recorder generalization claim",
        },
    }
    joblib.dump(bundle, output_dir / "model.joblib")

    validation_counts = {label: int(np.sum(y_validation == label)) for label in classes}
    train_counts = {label: int(np.sum(y_train == label)) for label in classes}
    metrics = {
        "scope": "单人数据上的流程验证；只代表本次随机划分，不代表陌生录制者泛化。",
        "all_valid_samples": len(samples),
        "selected_samples": len(selected),
        "train_samples": len(train_idx),
        "validation_samples": len(validation_idx),
        "class_labels": classes,
        "selected_per_class": class_counts,
        "train_per_class": train_counts,
        "validation_per_class": validation_counts,
        "parameters": {
            "model": "StandardScaler + RBF SVC(probability=True, class_weight='balanced')",
            "C": args.c,
            "temporal_bins": 10,
            "seed": args.seed,
            "subset_fraction": args.fraction,
            "validation_fraction": args.validation_fraction,
            "rejection_threshold_default": args.threshold,
            "threshold_note": "Provisional only; calibrate with held-out data. Not a validated safety threshold.",
        },
        "classification_report": report,
        "rejection_evaluation": {
            "threshold": args.threshold,
            "accepted_samples": int(accepted.sum()),
            "rejected_samples": int((~accepted).sum()),
            "coverage": float(accepted.mean()),
            "accuracy_on_accepted": (
                float(np.mean(predictions[accepted] == y_validation[accepted])) if accepted.any() else None
            ),
            "note": "Small single-recorder validation subset only; confidence threshold is not calibrated for real use.",
        },
        "confusion_matrix_labels": classes,
        "confusion_matrix": matrix.tolist(),
        "invalid_samples": [record.__dict__ for record in records if record.status != "ok"],
        "split": {
            "training_files": selected_paths[train_idx].tolist(),
            "validation_files": selected_paths[validation_idx].tolist(),
            "unused_files": [paths[index] for index in range(len(paths)) if index not in set(selected.tolist())],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "scan_report.json").write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *classes])
        writer.writerows([[label, *row] for label, row in zip(classes, matrix.tolist())])
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_validation, predictions, labels=classes, target_names=classes, zero_division=0),
        encoding="utf-8",
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("sign_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_output"))
    parser.add_argument("--fraction", type=float, default=0.6, help="per-class fraction selected for this run")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.35, help="provisional rejection threshold; calibrate on more held-out data")
    return parser


if __name__ == "__main__":
    try:
        output = train(build_parser().parse_args())
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
    print(f"Training completed. Artifacts: {output}")
