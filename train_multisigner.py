"""Inventory and evaluate the ten-word sign baseline with recorder-held-out folds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sign_baseline import EXPECTED_FEATURES, EXPECTED_FRAMES, parse_label, sequence_features


LABELS = ["你好", "谢谢", "请问", "医院", "厕所", "欢迎", "帮助", "报警", "吃饭", "再见"]
DATA_DIRS = {"sign_data": "本人", "sign_data_a": "同学A", "sign_data_b": "同学B"}
MIN_EFFECTIVE_FRAMES = 5
FILENAME_RECORDER = re.compile(r"_([AB])_\d+\.npy$")


def inventory(data_roots: dict[str, Path]) -> tuple[list[dict], dict]:
    records: list[dict] = []
    by_directory: dict[str, dict] = {}
    for directory, recorder in DATA_DIRS.items():
        root = data_roots[directory]
        entries = []
        for path in sorted(root.glob("*.npy")):
            reason = None
            try:
                label = parse_label(path)
                if label not in LABELS:
                    raise ValueError(f"标签不在限定词表中: {label}")
            except ValueError as exc:
                label, reason = None, str(exc)
            filename_recorder = FILENAME_RECORDER.search(path.name)
            suffix_recorder = filename_recorder.group(1) if filename_recorder else None
            sample = {
                "path": str(path.resolve()), "directory": directory, "recorder_id": recorder,
                "label": label, "filename_recorder_suffix": suffix_recorder,
                "shape": None, "dtype": None, "nan_count": None, "inf_count": None,
                "all_zero_sample": None, "zero_frame_count": None, "effective_frame_count": None,
                "status": "invalid" if reason else "ok", "detail": reason,
            }
            try:
                array = np.load(path, allow_pickle=False)
                sample["shape"], sample["dtype"] = list(array.shape), str(array.dtype)
                if not np.issubdtype(array.dtype, np.number):
                    raise ValueError(f"非数值 dtype: {array.dtype}")
                if array.ndim == 2:
                    finite = np.isfinite(array)
                    sample["nan_count"] = int(np.isnan(array).sum())
                    sample["inf_count"] = int(np.isinf(array).sum())
                    zero_rows = np.all(array == 0, axis=1)
                    sample["zero_frame_count"] = int(zero_rows.sum())
                    sample["effective_frame_count"] = int(array.shape[0] - zero_rows.sum())
                    sample["all_zero_sample"] = bool(array.size > 0 and np.all(array == 0))
                    if not finite.all():
                        sample["status"], sample["detail"] = "invalid", "包含 NaN/Inf"
                    elif array.shape != (EXPECTED_FRAMES, EXPECTED_FEATURES):
                        sample["status"], sample["detail"] = "invalid", f"期望(60,126)，实际{tuple(array.shape)}"
                    elif sample["all_zero_sample"]:
                        sample["status"], sample["detail"] = "invalid", "全零样本"
                    elif sample["effective_frame_count"] < MIN_EFFECTIVE_FRAMES:
                        sample["status"], sample["detail"] = "too_few_action_frames", f"有效动作帧少于{MIN_EFFECTIVE_FRAMES}"
                elif sample["status"] == "ok":
                    sample["status"], sample["detail"] = "invalid", "数组维度不是二维"
            except Exception as exc:
                sample["status"], sample["detail"] = "invalid", str(exc)
            records.append(sample)
            entries.append(sample)
        class_counts = Counter(s["label"] for s in entries if s["label"])
        effective_counts = [s["effective_frame_count"] for s in entries if s["effective_frame_count"] is not None]
        by_directory[directory] = {
            "recorder_id": recorder, "total_files": len(entries),
            "parsed_class_counts": dict(sorted(class_counts.items())),
            "shape_counts": dict(sorted(Counter(str(s["shape"]) for s in entries).items())),
            "dtype_counts": dict(sorted(Counter(s["dtype"] for s in entries).items())),
            "nan_samples": sum((s["nan_count"] or 0) > 0 for s in entries),
            "inf_samples": sum((s["inf_count"] or 0) > 0 for s in entries),
            "all_zero_samples": sum(s["all_zero_sample"] is True for s in entries),
            "total_zero_frames": sum(s["zero_frame_count"] or 0 for s in entries),
            "total_effective_frames": sum(s["effective_frame_count"] or 0 for s in entries),
            "effective_frames_per_sample": {
                "min": int(min(effective_counts)) if effective_counts else None,
                "p10": float(np.percentile(effective_counts, 10)) if effective_counts else None,
                "median": float(np.median(effective_counts)) if effective_counts else None,
                "max": int(max(effective_counts)) if effective_counts else None,
                "below_5_samples": sum(count < MIN_EFFECTIVE_FRAMES for count in effective_counts),
            },
            "file_label_parse_errors": sum(s["label"] is None for s in entries),
            "non_trainable_samples": sum(s["status"] != "ok" for s in entries),
        }
    overall = {
        "total_files": len(records), "per_directory": by_directory,
        "parsed_class_counts": dict(sorted(Counter(s["label"] for s in records if s["label"]).items())),
        "status_counts": dict(Counter(s["status"] for s in records)),
        "file_label_parse_errors": [s for s in records if s["label"] is None],
        "zero_frame_count": sum(s["zero_frame_count"] or 0 for s in records),
        "effective_frame_count": sum(s["effective_frame_count"] or 0 for s in records),
        "zero_frame_ratio": (
            sum(s["zero_frame_count"] or 0 for s in records) /
            max(1, sum((s["shape"] or [0])[0] for s in records))
        ),
        "min_effective_frames_for_training": MIN_EFFECTIVE_FRAMES,
    }
    return records, overall


def validation_split(train_pool: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    """Reserve whole samples, keeping at least one training sample per available class."""
    rng = np.random.default_rng(seed)
    train, validation = [], []
    class_plan = {}
    for label in LABELS:
        label_rows = [row for row in train_pool if row["label"] == label]
        by_recorder: dict[str, list[dict]] = defaultdict(list)
        for row in label_rows:
            by_recorder[row["recorder_id"]].append(row)
        selected = []
        for recorder in sorted(by_recorder):
            rows = by_recorder[recorder]
            if len(rows) >= 2:
                selected.append(rows[int(rng.integers(len(rows)))])
        if not selected and len(label_rows) >= 2:
            selected.append(label_rows[int(rng.integers(len(label_rows)))])
        selected_paths = {row["path"] for row in selected}
        validation.extend(selected)
        train.extend(row for row in label_rows if row["path"] not in selected_paths)
        class_plan[label] = {
            "train": len(label_rows) - len(selected), "validation": len(selected),
            "validation_recorders": sorted({row["recorder_id"] for row in selected}),
        }
    return train, validation, class_plan


def threshold_from_validation(probabilities: np.ndarray, predictions: np.ndarray, truth: np.ndarray) -> tuple[float, dict]:
    confidence = probabilities.max(axis=1)
    candidates = sorted(set([0.0, *confidence.tolist(), 1.000001]))
    viable = []
    for threshold in candidates:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage >= 0.5:
            accepted_accuracy = float(np.mean(predictions[accepted] == truth[accepted]))
            viable.append((accepted_accuracy, coverage, threshold))
    if viable:
        best = max(viable, key=lambda item: (item[0], item[1], -item[2]))
        threshold = float(best[2])
    else:
        threshold = 0.0
    accepted = confidence >= threshold
    return threshold, {
        "method": "validation confidence threshold maximizing accepted accuracy subject to >=50% validation coverage; ties prefer higher coverage",
        "validation_coverage": float(accepted.mean()),
        "validation_accepted_accuracy": float(np.mean(predictions[accepted] == truth[accepted])) if accepted.any() else None,
        "validation_samples": int(len(truth)),
    }


def fit_model(rows: list[dict], args: argparse.Namespace):
    x = np.stack([sequence_features(np.load(row["path"], allow_pickle=False).astype(np.float32)) for row in rows])
    y = np.asarray([row["label"] for row in rows])
    model = make_pipeline(StandardScaler(), SVC(C=args.c, kernel="rbf", class_weight="balanced", probability=True, random_state=args.seed))
    model.fit(x, y)
    return model


def row_counts(rows: list[dict]) -> dict:
    return {label: sum(row["label"] == label for row in rows) for label in LABELS}


def validate_fold_isolation(train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict], held_out: str) -> None:
    train_paths = {row["path"] for row in train_rows}
    validation_paths = {row["path"] for row in validation_rows}
    test_paths = {row["path"] for row in test_rows}
    if train_paths & validation_paths or train_paths & test_paths or validation_paths & test_paths:
        raise RuntimeError("样本级划分发生重叠")
    if any(row["directory"] == held_out for row in train_rows + validation_rows):
        raise RuntimeError("留出录制者泄漏到训练/验证集")
    if any(row["directory"] != held_out for row in test_rows):
        raise RuntimeError("测试集中混入非留出录制者")


def make_output_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = parent / f"multisigner_run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output, suffix = base, 1
    while output.exists():
        output = parent / f"{base.name}-{suffix}"
        suffix += 1
    output.mkdir()
    return output


def train(args: argparse.Namespace) -> Path:
    data_roots = {name: args.data_root / name for name in DATA_DIRS}
    records, inventory_summary = inventory(data_roots)
    usable = [row for row in records if row["status"] == "ok"]
    missing = {label: sum(row["label"] == label for row in usable) for label in LABELS}
    if any(count < 3 for count in missing.values()):
        raise RuntimeError(f"每類至少需要3條可訓練樣本，当前可用数: {missing}")
    output = make_output_dir(args.output_root)
    folds, all_predictions = [], []
    total_matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)

    for fold_index, (held_dir, held_recorder) in enumerate(DATA_DIRS.items()):
        test_rows = [row for row in usable if row["directory"] == held_dir]
        pool = [row for row in usable if row["directory"] != held_dir]
        train_rows, val_rows, val_plan = validation_split(pool, args.seed + fold_index)
        validate_fold_isolation(train_rows, val_rows, test_rows, held_dir)
        if not train_rows or not val_rows:
            raise RuntimeError(f"{held_recorder} 折无法构造训练/验证集")
        model = fit_model(train_rows, args)
        x_val = np.stack([sequence_features(np.load(row["path"], allow_pickle=False)) for row in val_rows])
        y_val = np.asarray([row["label"] for row in val_rows])
        val_prob = model.predict_proba(x_val)
        val_pred = model.predict(x_val)
        threshold, threshold_info = threshold_from_validation(val_prob, val_pred, y_val)

        x_test = np.stack([sequence_features(np.load(row["path"], allow_pickle=False)) for row in test_rows])
        y_test = np.asarray([row["label"] for row in test_rows])
        probabilities = model.predict_proba(x_test)
        predictions = model.predict(x_test)
        confidence = probabilities.max(axis=1)
        accepted = confidence >= threshold
        matrix = confusion_matrix(y_test, predictions, labels=LABELS)
        total_matrix += matrix
        report = classification_report(y_test, predictions, labels=LABELS, output_dict=True, zero_division=0)
        fold_dir = output / f"held_out_{held_dir}"
        fold_dir.mkdir()
        bundle = {
            "model": model, "labels": LABELS, "threshold": threshold,
            "feature_config": {"method": "mean_pool", "temporal_bins": 10, "input_shape": [60, 126]},
            "training": {"held_out_recorder": held_recorder, "training_recorders": sorted({r["recorder_id"] for r in train_rows}), "threshold_source": "training-pool validation only"},
        }
        joblib.dump(bundle, fold_dir / "model.joblib")
        fold = {
            "held_out_recorder": held_recorder, "held_out_directory": held_dir,
            "train_count": len(train_rows), "validation_count": len(val_rows), "test_count": len(test_rows),
            "train_per_class": row_counts(train_rows), "validation_per_class": row_counts(val_rows), "test_per_class": row_counts(test_rows),
            "validation_split_plan": val_plan, "test_recorders": sorted({r["recorder_id"] for r in test_rows}),
            "accuracy": float(accuracy_score(y_test, predictions)), "macro_f1": float(f1_score(y_test, predictions, labels=LABELS, average="macro", zero_division=0)),
            "classification_report": report, "confusion_matrix_labels": LABELS, "confusion_matrix": matrix.tolist(),
            "rejection": {"threshold": threshold, **threshold_info, "test_coverage": float(accepted.mean()), "accepted_samples": int(accepted.sum()), "rejected_samples": int((~accepted).sum()), "accepted_accuracy": float(np.mean(predictions[accepted] == y_test[accepted])) if accepted.any() else None,
                          "note": "未提供真实无手或词表外负样本，拒识性能未在真实负样本上验证。"},
            "split_files": {"train": [r["path"] for r in train_rows], "validation": [r["path"] for r in val_rows], "test": [r["path"] for r in test_rows]},
        }
        folds.append(fold)
        for row, truth, pred, conf, accept in zip(test_rows, y_test, predictions, confidence, accepted):
            all_predictions.append({"held_out_recorder": held_recorder, "path": row["path"], "label": truth, "prediction": pred, "confidence": float(conf), "accepted": bool(accept)})

    final_model = fit_model(usable, args)
    final_threshold = float(np.median([fold["rejection"]["threshold"] for fold in folds]))
    joblib.dump({"model": final_model, "labels": LABELS, "threshold": final_threshold,
                 "feature_config": {"method": "mean_pool", "temporal_bins": 10, "input_shape": [60, 126]},
                 "training": {"purpose": "camera demonstration only", "sample_count": len(usable), "threshold_note": "median of three development-fold validation thresholds; not an independent test estimate"}}, output / "final_demo_model.joblib")

    y_all = [item["label"] for item in all_predictions]
    pred_all = [item["prediction"] for item in all_predictions]
    metrics = {
        "created_at": datetime.now().astimezone().isoformat(), "scope": "三轮留一录制者测试；样本级切分，帧不拆分。验证集仅来自其余两位录制者。",
        "limitations": ["无录制场次元数据，未执行按场次隔离。", "拒识未使用专门无手或词表外负样本验证。", "最终全量模型仅用于演示，不代表独立测试成绩。"],
        "camera_preprocessing_review": {
            "collector": "collect_data.py flips the OpenCV frame horizontally before MediaPipe; MediaPipe Left is written to the right-hand 63 values, MediaPipe Right to the left-hand 63 values; each hand uses wrist-relative xyz and divides by its own max absolute coordinate plus 1e-6; features concatenate left then right; missing hands/frames are all-zero and each sample is 60 consecutive frames.",
            "browser": "The page mirrors the preview with CSS but has no browser MediaPipe feature extraction. Camera frames are not sent to inference. Exact feature parity is therefore not implemented or claimed.",
            "state": "recognition feature module not connected; preview only",
        },
        "parameters": {"model": "10 temporal-bin mean features + StandardScaler + RBF SVC(probability=True, class_weight=balanced)", "C": args.c, "temporal_bins": 10, "random_seed": args.seed,
                        "minimum_effective_action_frames": MIN_EFFECTIVE_FRAMES, "threshold_selection": "per-fold validation only; maximize accepted accuracy with at least 50% validation coverage"},
        "inventory": inventory_summary, "usable_training_samples": len(usable), "folds": folds,
        "pooled_test": {"accuracy": float(accuracy_score(y_all, pred_all)), "macro_f1": float(f1_score(y_all, pred_all, labels=LABELS, average="macro", zero_division=0)),
                        "classification_report": classification_report(y_all, pred_all, labels=LABELS, output_dict=True, zero_division=0), "confusion_matrix_labels": LABELS, "confusion_matrix": total_matrix.tolist(),
                        "test_samples_per_class": {label: sum(value == label for value in y_all) for label in LABELS},
                        "rejection_coverage": float(np.mean([item["accepted"] for item in all_predictions])),
                        "accepted_accuracy": float(np.mean([item["prediction"] == item["label"] for item in all_predictions if item["accepted"]])) if any(item["accepted"] for item in all_predictions) else None},
        "final_demo_model": {"path": "final_demo_model.joblib", "sample_count": len(usable), "threshold": final_threshold, "purpose": "演示/API加载模型"},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "inventory.json").write_text(json.dumps(inventory_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "sample_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)
    with (output / "pooled_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["actual\\predicted", *LABELS]); writer.writerows([[label, *row] for label, row in zip(LABELS, total_matrix.tolist())])
    (output / "report.txt").write_text(render_report(metrics), encoding="utf-8")
    return output


def render_report(metrics: dict) -> str:
    lines = ["限定词表手语识别：三录制者留一评估", "", f"样本总数：{metrics['inventory']['total_files']}", f"可训练样本：{metrics['usable_training_samples']}",
             f"全数据零帧比例：{metrics['inventory']['zero_frame_ratio']:.2%}", "", "数据质量（按录制者）："]
    for directory, summary in metrics["inventory"]["per_directory"].items():
        ef = summary["effective_frames_per_sample"]
        lines.append(f"- {summary['recorder_id']} ({directory}): {summary['total_files']}条，零帧{summary['total_zero_frames']}，有效帧{summary['total_effective_frames']}，单样本有效帧 min/median/max={ef['min']}/{ef['median']}/{ef['max']}，少于5帧={ef['below_5_samples']}，全零样本{summary['all_zero_samples']}，不可训练{summary['non_trainable_samples']}")
    lines.extend(["", "三轮留一录制者测试："])
    for fold in metrics["folds"]:
        rejection = fold["rejection"]
        lines.append(f"- 留出{fold['held_out_recorder']}: n={fold['test_count']} accuracy={fold['accuracy']:.4f} macro-F1={fold['macro_f1']:.4f}; 拒识阈值={rejection['threshold']:.4f}, coverage={rejection['test_coverage']:.2%}, 接受准确率={rejection['accepted_accuracy']}")
    pooled = metrics["pooled_test"]
    lines.extend(["", f"汇总: accuracy={pooled['accuracy']:.4f}, macro-F1={pooled['macro_f1']:.4f}, coverage={pooled['rejection_coverage']:.2%}, accepted accuracy={pooled['accepted_accuracy']}",
                  "", "限制：未提供录制场次元数据；无手/词表外动作负样本缺失，拒识性能未在真实负样本上验证。每条 .npy 为一个整体样本，未拆帧。", "过少有效动作帧定义为少于5/60帧；仅从训练/验证/测试建模样本中排除，不修改原始文件。"])
    lines.append("摄像头：采集端采用镜像画面、MediaPipe Left写入右手槽位、腕点相对xyz、每手独立max-abs归一化、左后右拼接和连续60帧；网页端仅镜像预览，尚无MediaPipe特征提取，因此未声称与采集端一致，也不输出实时预测。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("model_output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    output = train(args)
    print(output.resolve())


if __name__ == "__main__":
    main()
