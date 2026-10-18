# -*- coding: utf-8 -*-
"""Calibrate phase-2 B rejection rules using recorder-grouped OOF only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from phase2_train import DATA_DIRS, EXPECTED_SHAPE, LABELS, masks, parse_label

MIN_SELECTED_FRAMES = 30
CONFIDENCE_THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 1.0, 0.01)]
MARGIN_THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
QUALITY_GATES = {
    "Q0_any30": {"min_any_valid": 30},
    "Q1_zero80": {"min_any_valid": 30, "max_zero_ratio": 0.80},
    "Q2_zero60": {"min_any_valid": 30, "max_zero_ratio": 0.60},
    "Q3_onehand50": {"min_any_valid": 30, "max_one_hand_ratio": 0.50},
    "Q4_zero80_onehand50": {"min_any_valid": 30, "max_zero_ratio": 0.80, "max_one_hand_ratio": 0.50},
}
TARGET_COVERAGE = (0.50, 0.60)
MIN_ACCEPTED_ACCURACY = 0.95
MIN_ACCEPTED_MACRO_F1 = 0.85


def make_model(seed: int):
    return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed))


def longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for directory, recorder in DATA_DIRS.items():
        for path in sorted((root / directory).glob("*.npy")):
            try:
                label = parse_label(path)
                if label not in LABELS:
                    raise ValueError(f"label outside vocabulary: {label}")
                array = np.load(path, allow_pickle=False)
                if array.shape != EXPECTED_SHAPE or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                    raise ValueError(f"invalid array: shape={array.shape}, dtype={array.dtype}")
                state = masks(array)
                selected = state["any_valid"]
                count = int(selected.sum())
                if count < MIN_SELECTED_FRAMES:
                    continue
                rows = array[selected].astype(np.float32)
                feature = np.concatenate([rows.mean(axis=0), rows.std(axis=0)]).astype(np.float32)
                records.append({
                    "path": str(path.resolve()), "filename": path.name, "directory": directory,
                    "recorder_id": recorder, "label": label, "feature": feature,
                    "any_valid": count, "any_valid_ratio": float(selected.mean()),
                    "zero_ratio": float((~selected).mean()),
                    "both_ratio": float(state["both"].mean()),
                    "left_only_ratio": float(state["left_only"].mean()),
                    "right_only_ratio": float(state["right_only"].mean()),
                    "one_hand_ratio": float((state["left_only"] | state["right_only"]).mean()),
                    "selected_frames": count,
                })
            except Exception as exc:
                print(f"WARNING skipped {path}: {exc}", file=sys.stderr)
    return records


def fit_predict(train: list[dict], scored: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    classes = {row["label"] for row in train}
    if classes != set(LABELS):
        raise RuntimeError(f"grouped training split missing classes: {sorted(set(LABELS) - classes)}")
    x_train = np.stack([r["feature"] for r in train])
    y_train = np.asarray([r["label"] for r in train])
    x_scored = np.stack([r["feature"] for r in scored])
    model = make_model(seed)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_scored)
    # Align SVC probability columns to the fixed vocabulary order.
    aligned = np.zeros((len(scored), len(LABELS)), dtype=np.float64)
    for source, label in enumerate(model.classes_):
        aligned[:, LABELS.index(label)] = probabilities[:, source]
    return aligned, np.asarray(LABELS)[aligned.argmax(axis=1)]


def enrich_predictions(rows: list[dict], probabilities: np.ndarray, predictions: np.ndarray, scope: str, fold: str) -> list[dict]:
    result = []
    for row, prob, prediction in zip(rows, probabilities, predictions):
        order = np.argsort(prob)
        item = {key: value for key, value in row.items() if key not in ("feature", "directory")}
        item.update({"scope": scope, "outer_fold": fold, "prediction": str(prediction),
                     "confidence": float(prob.max()), "top2_label": LABELS[int(order[-2])],
                     "top2_probability": float(prob[order[-2]]),
                     "top2_margin": float(prob[order[-1]] - prob[order[-2]]),
                     "probabilities_json": json.dumps({label: float(prob[i]) for i, label in enumerate(LABELS)}, ensure_ascii=False)})
        result.append(item)
    return result


def accepted_metrics(rows: list[dict], accepted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray([r["label"] for r in rows])
    pred = np.asarray([r["prediction"] for r in rows])
    n = len(rows)
    selected = truth[accepted]
    selected_pred = pred[accepted]
    coverage = float(accepted.mean()) if n else 0.0
    return {
        "sample_count": n, "accepted_count": int(accepted.sum()), "rejected_count": int(n - accepted.sum()),
        "coverage": coverage,
        "accepted_accuracy": float(accuracy_score(selected, selected_pred)) if len(selected) else None,
        "accepted_macro_f1": float(f1_score(selected, selected_pred, labels=LABELS, average="macro", zero_division=0)) if len(selected) else None,
        "overall_accuracy": float(accuracy_score(truth, pred)) if n else None,
        "overall_macro_f1": float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)) if n else None,
    }


def quality_mask(rows: list[dict], gate_name: str) -> np.ndarray:
    spec = QUALITY_GATES[gate_name]
    passed = np.asarray([r["any_valid"] >= spec["min_any_valid"] for r in rows], dtype=bool)
    if "max_zero_ratio" in spec:
        passed &= np.asarray([r["zero_ratio"] <= spec["max_zero_ratio"] for r in rows])
    if "max_one_hand_ratio" in spec:
        passed &= np.asarray([r["one_hand_ratio"] <= spec["max_one_hand_ratio"] for r in rows])
    return passed


def score_rule(rows: list[dict], confidence_threshold: float, gate_name: str, margin_threshold: float) -> dict[str, Any]:
    confidence = np.asarray([r["confidence"] for r in rows])
    margin = np.asarray([r["top2_margin"] for r in rows])
    accepted = (confidence >= confidence_threshold) & (margin >= margin_threshold) & quality_mask(rows, gate_name)
    return {"confidence_threshold": float(confidence_threshold), "quality_gate": gate_name,
            "quality_gate_spec": QUALITY_GATES[gate_name], "top2_margin_threshold": float(margin_threshold),
            **accepted_metrics(rows, accepted)}


def objective_key(candidate: dict[str, Any]) -> tuple:
    coverage = candidate["coverage"]
    acc = candidate["accepted_accuracy"]
    macro = candidate["accepted_macro_f1"]
    valid_quality = acc is not None and macro is not None and acc >= MIN_ACCEPTED_ACCURACY and macro >= MIN_ACCEPTED_MACRO_F1
    in_target = TARGET_COVERAGE[0] <= coverage <= TARGET_COVERAGE[1]
    if valid_quality and in_target:
        return (3, -abs(coverage - 0.55), macro, acc)
    if valid_quality:
        return (2, coverage, macro, acc)
    # Diagnostic fallback only: keep the selected rule useful for analysis, but mark it non-deployable.
    return (1, macro if macro is not None else -1.0, coverage, acc if acc is not None else -1.0)


def sweep_candidates(rows: list[dict], fold: str, threshold_rows: list[dict], quality_rows: list[dict], margin_rows: list[dict]) -> dict[str, dict[str, Any]]:
    evaluated = []
    for gate_name in QUALITY_GATES:
        for margin in MARGIN_THRESHOLDS:
            for threshold in CONFIDENCE_THRESHOLDS:
                candidate = score_rule(rows, threshold, gate_name, margin)
                candidate["outer_fold"] = fold
                evaluated.append(candidate)
                if gate_name == "Q0_any30" and margin == 0:
                    threshold_rows.append(dict(candidate))
                if margin == 0:
                    quality_rows.append(dict(candidate))
                if gate_name == "Q0_any30":
                    margin_rows.append(dict(candidate))
    def best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return max(candidates, key=objective_key)
    rule_a = best([r for r in evaluated if r["quality_gate"] == "Q0_any30" and r["top2_margin_threshold"] == 0])
    rule_b = best([r for r in evaluated if r["top2_margin_threshold"] == 0])
    rule_c = best([r for r in evaluated if r["quality_gate"] == "Q0_any30"])
    rule_d = best(evaluated)
    return {"A_global_confidence": rule_a, "B_confidence_quality": rule_b,
            "C_confidence_margin": rule_c, "D_confidence_quality_margin": rule_d}


def annotate_acceptance(rows: list[dict], rule: dict[str, Any]) -> list[dict]:
    accepted = ((np.asarray([r["confidence"] for r in rows]) >= rule["confidence_threshold"]) &
                (np.asarray([r["top2_margin"] for r in rows]) >= rule["top2_margin_threshold"]) &
                quality_mask(rows, rule["quality_gate"]))
    output = []
    for row, ok in zip(rows, accepted):
        output.append({**row, "accepted": bool(ok), "reject_reason": "" if ok else (
            "quality_gate" if not quality_mask([row], rule["quality_gate"])[0] else
            "low_confidence" if row["confidence"] < rule["confidence_threshold"] else "low_top2_margin")})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value) for key, value in row.items()})


def evaluate_predictions(rows: list[dict], rule: dict[str, Any]) -> dict[str, Any]:
    scored = annotate_acceptance(rows, rule)
    accepted = np.asarray([r["accepted"] for r in scored])
    metrics = accepted_metrics(scored, accepted)
    truth = [r["label"] for r in scored]
    pred = [r["prediction"] for r in scored]
    metrics["classification_report_top1"] = classification_report(truth, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["confusion_matrix_top1"] = confusion_matrix(truth, pred, labels=LABELS).tolist()
    metrics["per_class_test_counts"] = {label: {"total": int(sum(t == label for t in truth)),
                                                 "accepted": int(sum(t == label and ok for t, ok in zip(truth, accepted)))} for label in LABELS}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("model_output"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output_parent = root / args.output_root
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = output_parent / f"rejection_calibration-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    all_records = load_records(root)
    if not all_records:
        raise RuntimeError("No valid B-feature samples with any_valid >= 30")
    total_npy_files = sum(1 for directory in DATA_DIRS for _ in (root / directory).glob("*.npy"))
    threshold_rows: list[dict] = []
    quality_rows: list[dict] = []
    margin_rows: list[dict] = []
    oof_rows: list[dict] = []
    outer_test_rows: list[dict] = []
    fold_calibration_counts = []
    fold_summaries = []
    fold_selected_rules = {}

    for outer_index, (held_directory, held_recorder) in enumerate(DATA_DIRS.items()):
        pool = [r for r in all_records if r["directory"] != held_directory]
        test = [r for r in all_records if r["directory"] == held_directory]
        pool_recorders = sorted({r["recorder_id"] for r in pool})
        fold = f"held_out_{held_recorder}"
        fold_oof = []
        for inner_index, oof_recorder in enumerate(pool_recorders):
            inner_train = [r for r in pool if r["recorder_id"] != oof_recorder]
            inner_valid = [r for r in pool if r["recorder_id"] == oof_recorder]
            if not inner_valid:
                continue
            prob, pred = fit_predict(inner_train, inner_valid, args.seed + outer_index * 10 + inner_index)
            fold_oof.extend(enrich_predictions(inner_valid, prob, pred, "training_pool_oof", fold))
            fold_calibration_counts.append({"outer_fold": fold, "oof_recorder": oof_recorder,
                                            "train_recorders": ";".join(sorted({r["recorder_id"] for r in inner_train})),
                                            "train_count": len(inner_train), "oof_count": len(inner_valid),
                                            "train_classes": len({r["label"] for r in inner_train}),
                                            "oof_classes": len({r["label"] for r in inner_valid})})
        if not fold_oof or {r["label"] for r in fold_oof} != set(LABELS):
            raise RuntimeError(f"Incomplete grouped OOF coverage/classes for {fold}")
        rules = sweep_candidates(fold_oof, fold, threshold_rows, quality_rows, margin_rows)
        fold_selected_rules[fold] = rules
        oof_rows.extend(fold_oof)

        # The outer test recorder is scored only after all rules are frozen from training-pool OOF.
        outer_train = pool
        test_prob, test_pred = fit_predict(outer_train, test, args.seed + 100 + outer_index)
        scored_test = enrich_predictions(test, test_prob, test_pred, "outer_test_exploratory", fold)
        outer_test_rows.extend(scored_test)
        strategies = {}
        for strategy, rule in rules.items():
            rule_eval = evaluate_predictions(scored_test, rule)
            rule_eval["frozen_rule"] = {k: rule[k] for k in ("confidence_threshold", "quality_gate", "quality_gate_spec", "top2_margin_threshold")}
            strategies[strategy] = rule_eval
        fold_summaries.append({"outer_fold": fold, "held_out_recorder": held_recorder,
                               "training_pool_recorders": pool_recorders,
                               "oof_count": len(fold_oof), "test_count": len(test),
                               "oof_top1_accuracy": float(accuracy_score([r["label"] for r in fold_oof], [r["prediction"] for r in fold_oof])),
                               "oof_top1_macro_f1": float(f1_score([r["label"] for r in fold_oof], [r["prediction"] for r in fold_oof], labels=LABELS, average="macro", zero_division=0)),
                               "selected_rules": rules, "outer_test_by_strategy": strategies})

    # A separate deployment-candidate calibration uses grouped OOF over all three recorders.
    global_oof = []
    global_counts = []
    recorders = sorted({r["recorder_id"] for r in all_records})
    for inner_index, oof_recorder in enumerate(recorders):
        inner_train = [r for r in all_records if r["recorder_id"] != oof_recorder]
        inner_valid = [r for r in all_records if r["recorder_id"] == oof_recorder]
        prob, pred = fit_predict(inner_train, inner_valid, args.seed + 300 + inner_index)
        global_oof.extend(enrich_predictions(inner_valid, prob, pred, "all_data_grouped_oof", "final_demo_calibration"))
        global_counts.append({"outer_fold": "final_demo_calibration", "oof_recorder": oof_recorder,
                              "train_recorders": ";".join(sorted({r["recorder_id"] for r in inner_train})),
                              "train_count": len(inner_train), "oof_count": len(inner_valid),
                              "train_classes": len({r["label"] for r in inner_train}),
                              "oof_classes": len({r["label"] for r in inner_valid})})
    global_threshold_rows: list[dict] = []
    global_quality_rows: list[dict] = []
    global_margin_rows: list[dict] = []
    global_rules = sweep_candidates(global_oof, "final_demo_calibration", global_threshold_rows, global_quality_rows, global_margin_rows)
    global_selected = global_rules["D_confidence_quality_margin"]
    global_selected["selection_status"] = "meets_targets" if (global_selected["coverage"] >= TARGET_COVERAGE[0] and global_selected["coverage"] <= TARGET_COVERAGE[1] and global_selected["accepted_accuracy"] is not None and global_selected["accepted_accuracy"] >= MIN_ACCEPTED_ACCURACY and global_selected["accepted_macro_f1"] is not None and global_selected["accepted_macro_f1"] >= MIN_ACCEPTED_MACRO_F1) else "targets_not_met_on_oof"

    write_csv(output / "calibration_predictions.csv", oof_rows + outer_test_rows + global_oof)
    write_csv(output / "threshold_sweep.csv", threshold_rows + global_threshold_rows)
    write_csv(output / "quality_gate_sweep.csv", quality_rows + global_quality_rows)
    write_csv(output / "margin_sweep.csv", margin_rows + global_margin_rows)
    write_csv(output / "fold_calibration_counts.csv", fold_calibration_counts + global_counts)

    frozen = {"schema_version": 1, "feature": {"mode": "B", "input_shape": list(EXPECTED_SHAPE), "feature_dimension": 252,
                                                   "selected_frames": "any_valid", "min_selected_frames": MIN_SELECTED_FRAMES,
                                                   "feature_definition": "concatenate mean(126) and std(126) over any_valid frames"},
               "calibration_method": "recorder-grouped OOF; no held-out outer test used for rule selection",
               "selection_targets": {"coverage_range": list(TARGET_COVERAGE), "accepted_accuracy_min": MIN_ACCEPTED_ACCURACY,
                                     "accepted_macro_f1_min": MIN_ACCEPTED_MACRO_F1},
               "outer_fold_rules": fold_selected_rules,
               "final_demo_candidate": {"rule": global_selected, "all_strategy_rules": global_rules,
                                        "oof_count": len(global_oof), "oof_recorders": recorders}}
    (output / "frozen_rejection_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = ["B 方案拒识校准审计", f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
                    f"B 特征：any_valid 帧 (60,126) 序列中至少 {MIN_SELECTED_FRAMES} 帧，mean+std 得到 252 维。",
                    "阈值、质量门控和 top-2 margin 仅使用对应训练池的按录制者分组 OOF 预测选择。",
                    "测试录制者在规则冻结后只评估一次；本报告测试结果用于探索性评估，不用于调参。",
                    "当前无专门无手/静止/OOV 负样本，因此拒识性能未在真实负样本上验证。", "",
                    "目标仅作观察目标：coverage 50%-60%、accepted accuracy >=95%、accepted macro-F1 >=0.85。未达标如实报告。",
                    "候选选择顺序：满足三项目标且 coverage 在 50%-60% 时选最接近 55%；否则满足准确率/F1下限时选最高 coverage；否则按 accepted macro-F1、coverage 排序作为诊断规则，并标记不达标。",
                    "质量门控 Q0 固定要求 any_valid>=30；其他门控只另加零帧比例/单手帧比例上限。比例不进入 252 维分类特征。", ""]
    for fold in fold_summaries:
        report_lines.extend([f"{fold['outer_fold']}（测试者：{fold['held_out_recorder']}）",
                             f"训练池 OOF={fold['oof_count']}，封存测试={fold['test_count']}，OOF top-1 accuracy={fold['oof_top1_accuracy']:.4f}，macro-F1={fold['oof_top1_macro_f1']:.4f}"])
        for strategy, rule in fold["selected_rules"].items():
            test_metrics = fold["outer_test_by_strategy"][strategy]
            report_lines.append(f"  {strategy}: OOF coverage={rule['coverage']:.3f}, accepted accuracy={rule['accepted_accuracy']}, accepted macro-F1={rule['accepted_macro_f1']}; frozen threshold={rule['confidence_threshold']:.2f}, gate={rule['quality_gate']}, margin={rule['top2_margin_threshold']:.2f}; exploratory test coverage={test_metrics['coverage']:.3f}, accepted accuracy={test_metrics['accepted_accuracy']}, accepted macro-F1={test_metrics['accepted_macro_f1']}")
        report_lines.append("")
    report_lines.extend(["最终演示候选模型的全数据 grouped OOF D 规则", json.dumps(global_selected, ensure_ascii=False, indent=2),
                         "", "限制与解释", "- 早先用 outer-test 指标比较 A/B/C 并选择 B，因此此前 B 的 91.24% accuracy / 89.79% macro-F1 是探索性方案选择结果，不是完全无偏的最终泛化估计。",
                         "- 本次每个外层折只有两位训练录制者，内部 OOF 为按录制者留一；该设置避免样本泄漏，但 OOF 训练规模较小。",
                         "- 接受准确率只对被接受的已知词样本计算，低覆盖率会产生选择性偏差；需同时看 coverage 和 accepted macro-F1。",
                         "- 若最终演示 OOF 规则未达到目标，不应通过查看外层测试结果调阈值；可先做受控演示并清楚标注限制，或补采跨录制者/负样本。"])
    (output / "calibration_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    summary = {"total_npy_files": total_npy_files, "valid_b_samples": len(all_records), "excluded_or_invalid_samples": total_npy_files - len(all_records),
               "outer_folds": fold_summaries, "final_demo_oof_count": len(global_oof),
               "final_demo_oof_top1_accuracy": float(accuracy_score([r["label"] for r in global_oof], [r["prediction"] for r in global_oof])),
               "final_demo_oof_top1_macro_f1": float(f1_score([r["label"] for r in global_oof], [r["prediction"] for r in global_oof], labels=LABELS, average="macro", zero_division=0)),
               "final_demo_rules": global_rules, "generated_at": datetime.now().isoformat(timespec="seconds")}
    (output / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"calibration_output={output}")
    print(f"total_npy_files={total_npy_files} valid_b_samples={len(all_records)} excluded_or_invalid={total_npy_files-len(all_records)}")
    for fold in fold_summaries:
        print(f"{fold['outer_fold']}: test={fold['test_count']}")
        for strategy, result in fold["outer_test_by_strategy"].items():
            print(f"  {strategy}: coverage={result['coverage']:.3f} accepted_accuracy={result['accepted_accuracy']} accepted_macro_f1={result['accepted_macro_f1']}")
    print(f"final_demo_candidate={global_selected['selection_status']} coverage={global_selected['coverage']:.3f} accepted_accuracy={global_selected['accepted_accuracy']} accepted_macro_f1={global_selected['accepted_macro_f1']}")
    print("PHASE2_REJECTION_CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
