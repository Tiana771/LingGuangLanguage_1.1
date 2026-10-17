# -*- coding: utf-8 -*-
"""Phase 2 offline feature comparison and recorder-held-out evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ["你好", "谢谢", "请问", "医院", "厕所", "欢迎", "帮助", "报警", "吃饭", "再见"]
DATA_DIRS = {"sign_data": "本人", "sign_data_a": "同学A", "sign_data_b": "同学B"}
EXPECTED_SHAPE = (60, 126)
EPS = 1e-8
MIN_VALIDATION_COVERAGE = 0.80
FILENAME_PATTERN = re.compile(r"^\d+_(?P<label>.+?)(?:_[AB])?_(?P<sample_id>\d+)\.npy$")


def parse_label(path: Path) -> str:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"filename label parse failed: {path.name}")
    return match.group("label")


def masks(array: np.ndarray) -> dict[str, np.ndarray]:
    left_on = np.max(np.abs(array[:, :63]), axis=1) > EPS
    right_on = np.max(np.abs(array[:, 63:]), axis=1) > EPS
    both = left_on & right_on
    any_valid = left_on | right_on
    return {"left_on": left_on, "right_on": right_on, "both": both, "any_valid": any_valid,
            "left_only": left_on & ~right_on, "right_only": right_on & ~left_on}


def temporal_features(array: np.ndarray) -> np.ndarray:
    return array.reshape(10, 6, 126).mean(axis=1).reshape(-1).astype(np.float32)


def pooled_features(array: np.ndarray, mode: str) -> tuple[np.ndarray | None, int]:
    state = masks(array)
    selected = state[mode]
    count = int(selected.sum())
    if count < 30:
        return None, count
    rows = array[selected].astype(np.float32)
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)]).astype(np.float32), count


def record_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for directory, recorder in DATA_DIRS.items():
        for path in sorted((root / directory).glob("*.npy")):
            row: dict[str, Any] = {"path": str(path.resolve()), "filename": path.name, "directory": directory,
                                   "recorder_id": recorder, "label": None, "status": "ok", "detail": None,
                                   "strict_valid": None, "any_valid": None, "left_only": None, "right_only": None,
                                   "zero_ratio": None, "one_hand_ratio": None, "longest_zero_run": None,
                                   "motion_amp": None}
            try:
                row["label"] = parse_label(path)
                array = np.load(path, allow_pickle=False)
                if array.shape != EXPECTED_SHAPE:
                    raise ValueError(f"expected {EXPECTED_SHAPE}, got {array.shape}")
                if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                    raise ValueError("non-numeric or NaN/Inf array")
                state = masks(array)
                both = state["both"]
                any_valid = state["any_valid"]
                zero = ~any_valid
                row["strict_valid"] = int(both.sum())
                row["any_valid"] = int(any_valid.sum())
                row["left_only"] = int(state["left_only"].sum())
                row["right_only"] = int(state["right_only"].sum())
                row["zero_ratio"] = round(float(zero.mean()), 4)
                row["one_hand_ratio"] = round(float((state["left_only"] | state["right_only"]).mean()), 4)
                longest = cur = 0
                for flag in zero:
                    cur = cur + 1 if flag else 0
                    longest = max(longest, cur)
                row["longest_zero_run"] = int(longest)
                if both.any(): row["motion_amp"] = float(array[both].astype(np.float32).std(axis=0).mean())
                else: row["motion_amp"] = 0.0
                if row["label"] not in LABELS:
                    raise ValueError(f"label outside target vocabulary: {row['label']}")
            except Exception as exc:
                row["status"] = "invalid"; row["detail"] = str(exc)
            records.append(row)
    return records


def split_pool(rows: list[dict[str, Any]], held_directory: str, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    test = [r for r in rows if r["directory"] == held_directory and r["status"] == "ok"]
    pool = [r for r in rows if r["directory"] != held_directory and r["status"] == "ok"]
    rng = np.random.default_rng(seed)
    validation, train = [], []
    for label in LABELS:
        by_rec = {rec: [r for r in pool if r["label"] == label and r["recorder_id"] == rec] for rec in ("本人", "同学A", "同学B")}
        selected = []
        for rec, candidates in by_rec.items():
            if candidates:
                selected.append(candidates[int(rng.integers(len(candidates)))])
        selected_paths = {r["path"] for r in selected}
        validation.extend(selected)
        train.extend(r for r in pool if r["label"] == label and r["path"] not in selected_paths)
    return train, validation, test


def vectorize(rows: list[dict], mode: str) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    features, labels, used, filtered = [], [], [], []
    for row in rows:
        array = np.load(row["path"], allow_pickle=False).astype(np.float32)
        if mode == "A": vec, count = temporal_features(array), 60
        else: vec, count = pooled_features(array, "any_valid" if mode == "B" else "both")
        if vec is None:
            filtered.append({"path": row["path"], "label": row["label"], "mode": mode, "valid_frame_count": count, "reason": "fewer than 30 selected frames"})
            continue
        features.append(vec); labels.append(row["label"]); used.append(row)
    if not features:
        return np.empty((0, 1260 if mode == "A" else 252), dtype=np.float32), np.empty((0,), dtype=str), used, filtered
    return np.stack(features), np.asarray(labels), used, filtered


def threshold_from_validation(prob: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> tuple[float, dict[str, Any]]:
    confidence = prob.max(axis=1)
    candidates = sorted(set([0.0, *confidence.tolist(), 1.000001]))
    choices = []
    for threshold in candidates:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage >= MIN_VALIDATION_COVERAGE:
            choices.append((float(np.mean(pred[accepted] == truth[accepted])), coverage, float(threshold)))
    # Prefer a higher threshold when validation accuracy ties, while retaining
    # an 80% validation coverage floor so the demo does not reject nearly all
    # cross-recorder windows.
    selected = max(choices, key=lambda x: (x[0], x[2], x[1])) if choices else (0.0, 0.0, 0.0)
    return selected[2], {"validation_coverage": selected[1], "validation_accepted_accuracy": selected[0], "method": "maximize validation accepted accuracy with validation coverage >=80%"}


def evaluate_fold(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], mode: str, args: argparse.Namespace, fold_dir: Path) -> dict:
    x_train, y_train, train_used, train_filtered = vectorize(train_rows, mode)
    x_val, y_val, val_used, val_filtered = vectorize(val_rows, mode)
    x_test, y_test, test_used, test_filtered = vectorize(test_rows, mode)
    if len(set(y_train)) < len(LABELS) or len(y_val) == 0 or len(y_test) == 0 or set(y_test) != set(LABELS):
        return {"mode": mode, "status": "unavailable", "train_count": len(y_train), "validation_count": len(y_val), "test_count": len(y_test), "filtered": train_filtered + val_filtered + test_filtered, "reason": "feature filtering removed a required split/class"}
    model = make_pipeline(StandardScaler(), SVC(C=args.c, kernel="rbf", class_weight="balanced", probability=True, random_state=args.seed))
    model.fit(x_train, y_train)
    val_prob, val_pred = model.predict_proba(x_val), model.predict(x_val)
    threshold, threshold_info = threshold_from_validation(val_prob, val_pred, y_val)
    test_prob, test_pred = model.predict_proba(x_test), model.predict(x_test)
    confidence = test_prob.max(axis=1); accepted = confidence >= threshold
    labels = LABELS
    report = classification_report(y_test, test_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, test_pred, labels=labels).tolist()
    per_class_counts = {label: {"test": int(np.sum(y_test == label)), "accepted": int(np.sum((y_test == label) & accepted)), "rejected": int(np.sum((y_test == label) & ~accepted))} for label in LABELS}
    fold = {"mode": mode, "status": "ok", "train_count": int(len(y_train)), "validation_count": int(len(y_val)), "test_count": int(len(y_test)),
            "train_recorders": sorted({r["recorder_id"] for r in train_used}), "test_recorders": sorted({r["recorder_id"] for r in test_used}),
            "accuracy": float(accuracy_score(y_test, test_pred)), "macro_f1": float(f1_score(y_test, test_pred, labels=labels, average="macro", zero_division=0)),
            "classification_report": report, "confusion_matrix": matrix, "labels": labels, "per_class_counts": per_class_counts, "threshold": threshold, **threshold_info,
            "test_coverage": float(accepted.mean()), "accepted_count": int(accepted.sum()), "rejected_count": int((~accepted).sum()),
            "accepted_accuracy": float(np.mean(test_pred[accepted] == y_test[accepted])) if accepted.any() else None,
            "filtered_count": len(train_filtered) + len(val_filtered) + len(test_filtered)}
    predictions = []
    for row, truth, pred, conf, ok in zip(test_used, y_test, test_pred, confidence, accepted):
        predictions.append({"mode": mode, "path": row["path"], "recorder_id": row["recorder_id"], "truth": str(truth), "prediction": str(pred), "confidence": float(conf), "accepted": bool(ok), "threshold": float(threshold)})
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "metrics.json").write_text(json.dumps(fold, ensure_ascii=False, indent=2), encoding="utf-8")
    (fold_dir / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (fold_dir / "confusion_matrix.json").write_text(json.dumps({"labels": labels, "matrix": matrix}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**fold, "predictions": predictions, "filtered": train_filtered + val_filtered + test_filtered, "model": model}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: path.write_text("", encoding="utf-8-sig"); return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=Path("model_output")); parser.add_argument("--c", type=float, default=1.0); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--min-candidate", type=int, default=0); args = parser.parse_args()
    root = Path(__file__).resolve().parent; stamp = datetime.now().strftime("%Y%m%d-%H%M%S"); output = root / args.output_root / f"phase2_train-{stamp}"; output.mkdir(parents=True, exist_ok=False)
    records = record_inventory(root)
    valid = [r for r in records if r["status"] == "ok"]
    write_csv(output / "all_candidates.csv", records)
    review = [r for r in valid if (r["any_valid"] or 0) < 30 or (r["longest_zero_run"] or 0) >= 20 or (r["one_hand_ratio"] or 0) > 0.2]
    write_csv(output / "review_candidates.csv", review)
    write_csv(output / "confirmed_training_candidates.csv", valid)
    all_metrics, all_predictions, filtered = [], [], []
    all_reports, all_matrices, all_thresholds = {}, {}, {}
    mode_summaries = {}
    for mode in ("A", "B", "C"):
        mode_dir = output / f"model_{mode}"; folds = []
        for held_index, held_directory in enumerate(DATA_DIRS):
            train_rows, val_rows, test_rows = split_pool(valid, held_directory, args.seed + held_index)
            fold = evaluate_fold(train_rows, val_rows, test_rows, mode, args, mode_dir / f"held_out_{held_directory}")
            fold["held_out_directory"] = held_directory; fold["held_out_recorder"] = DATA_DIRS[held_directory]
            split_dir = output / "fold_splits" / mode / f"held_out_{held_directory}"
            split_dir.mkdir(parents=True, exist_ok=True)
            for name, split_rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
                write_csv(split_dir / f"{name}.csv", [{"path": r["path"], "label": r["label"], "recorder_id": r["recorder_id"]} for r in split_rows])
            key = f"{mode}/{held_directory}"
            all_reports[key] = fold.get("classification_report")
            all_matrices[key] = {"labels": fold.get("labels", LABELS), "matrix": fold.get("confusion_matrix")}
            all_thresholds[key] = {"threshold": fold.get("threshold"), "validation_coverage": fold.get("validation_coverage"), "validation_accepted_accuracy": fold.get("validation_accepted_accuracy"), "test_coverage": fold.get("test_coverage"), "accepted_accuracy": fold.get("accepted_accuracy")}
            all_metrics.append({k: v for k, v in fold.items() if k not in ("classification_report", "confusion_matrix", "predictions", "model", "filtered")})
            all_predictions.extend(fold.get("predictions", [])); filtered.extend(fold.get("filtered", [])); folds.append(fold)
        available = [f for f in folds if f["status"] == "ok"]
        mode_summaries[mode] = {"folds": [{k:v for k,v in f.items() if k not in ("model", "predictions", "filtered", "classification_report", "confusion_matrix")} for f in folds], "mean_accuracy": float(np.mean([f["accuracy"] for f in available])) if available else None, "mean_macro_f1": float(np.mean([f["macro_f1"] for f in available])) if available else None, "mean_coverage": float(np.mean([f["test_coverage"] for f in available])) if available else None, "mean_accepted_accuracy": float(np.mean([f["accepted_accuracy"] for f in available if f["accepted_accuracy"] is not None])) if available else None, "available_folds": len(available)}
    # Select by mean macro-F1, then coverage, while requiring all folds.
    selectable = [(s["mean_macro_f1"], s["mean_coverage"], mode) for mode,s in mode_summaries.items() if s["available_folds"] == 3]
    selected_mode = max(selectable)[2] if selectable else "A"
    final_x, final_y, final_used, final_filtered = vectorize(valid, selected_mode)
    if len(set(final_y)) != len(LABELS):
        raise RuntimeError(f"Selected feature mode {selected_mode} does not retain all ten classes for final model")
    final_model = make_pipeline(StandardScaler(), SVC(C=args.c, kernel="rbf", class_weight="balanced", probability=True, random_state=args.seed))
    final_model.fit(final_x, final_y)
    fold_thresholds = [f["threshold"] for f in mode_summaries[selected_mode]["folds"] if f["status"] == "ok"]
    final_threshold = float(np.median(fold_thresholds)) if fold_thresholds else 0.55
    bundle = {"model": final_model, "labels": LABELS, "threshold": final_threshold, "feature_config": {"mode": selected_mode, "input_shape": list(EXPECTED_SHAPE), "model_feature_dimension": int(final_x.shape[1]), "min_selected_frames": 30 if selected_mode != "A" else 0}, "training": {"purpose": "camera demonstration only; not an independent test estimate", "sample_count": int(len(final_y)), "candidate_count": len(valid), "threshold_source": "median of three training-pool validation thresholds", "source_run": stamp}}
    joblib.dump(bundle, output / "final_demo_model.joblib")
    (output / "final_demo_metadata.json").write_text(json.dumps({k:v for k,v in bundle.items() if k != "model"}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "feature_filtered_samples.csv", filtered + final_filtered)
    write_csv(output / "fold_metrics.csv", all_metrics); write_csv(output / "predictions.csv", all_predictions)
    write_csv(output / "feature_comparison.csv", [{"mode": mode, "feature_dimension": 1260 if mode == "A" else 252, **{key: value for key,value in stats.items() if key.startswith("mean_")}, "available_folds": stats["available_folds"]} for mode,stats in mode_summaries.items()])
    (output / "classification_reports.json").write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "confusion_matrices.json").write_text(json.dumps(all_matrices, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "thresholds.json").write_text(json.dumps(all_thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "final_demo_manifest.csv", [{"path": r["path"], "label": r["label"], "recorder_id": r["recorder_id"], "feature_mode": selected_mode} for r in final_used])
    (output / "summary.json").write_text(json.dumps({"total_files": len(records), "valid_files": len(valid), "invalid_files": len(records)-len(valid), "review_count": len(review), "modes": mode_summaries, "selected_mode": selected_mode, "final_model_feature_dimension": int(final_x.shape[1]), "final_model_sample_count": int(len(final_y)), "final_threshold": final_threshold, "generated_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["第二阶段离线训练报告", f"总文件：{len(records)}；有效候选：{len(valid)}；复核候选：{len(review)}", "", "方案比较："]
    for mode, s in mode_summaries.items(): lines.append(f"- {mode}: folds={s['available_folds']}, mean accuracy={s['mean_accuracy']}, mean macro-F1={s['mean_macro_f1']}, mean coverage={s['mean_coverage']}, mean accepted accuracy={s['mean_accepted_accuracy']}")
    lines += ["", f"选择方案：{selected_mode}", f"最终演示模型样本：{len(final_y)}，模型特征维度：{final_x.shape[1]}，阈值：{final_threshold}", "", "说明：三轮测试模型和最终演示模型分开保存；最终模型训练成绩不作为测试成绩。", "低质量样本未物理删除；review_candidates.csv 仅为复核清单。", "本阶段没有修改 Flask、前端、视频、词条或原始数据。"]
    (output / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"phase2_output={output}"); print(f"files={len(records)} valid={len(valid)} review={len(review)}")
    for mode, s in mode_summaries.items(): print(f"{mode}: folds={s['available_folds']} mean_accuracy={s['mean_accuracy']} mean_macro_f1={s['mean_macro_f1']} mean_coverage={s['mean_coverage']}")
    print(f"selected_mode={selected_mode} final_model={output/'final_demo_model.joblib'}")
    print("PHASE2 OFFLINE TRAINING COMPLETE")


if __name__ == "__main__":
    main()
