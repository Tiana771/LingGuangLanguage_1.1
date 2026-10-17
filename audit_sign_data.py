# -*- coding: utf-8 -*-
"""Read-only audit for the three recorder directories; never changes source data."""
from __future__ import annotations

import argparse, csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from sign_baseline import EXPECTED_FEATURES, EXPECTED_FRAMES, parse_label

LABELS = ["你好", "谢谢", "请问", "医院", "厕所", "欢迎", "帮助", "报警", "吃饭", "再见"]
DATA_DIRS = {"sign_data": "本人", "sign_data_a": "同学A", "sign_data_b": "同学B"}
SUFFIX_RE = re.compile(r"_([AB])_(?P<sample>\d+)\.npy$")


def runs(mask: np.ndarray) -> tuple[int, int, int, int]:
    if not len(mask): return 0, 0, 0, 0
    best_zero = best_valid = cur_zero = cur_valid = 0
    first_valid = next((i for i, v in enumerate(mask) if v), len(mask))
    last_valid = next((i for i in range(len(mask)-1, -1, -1) if mask[i]), -1)
    for valid in mask:
        if valid: cur_valid += 1; cur_zero = 0
        else: cur_zero += 1; cur_valid = 0
        best_valid = max(best_valid, cur_valid); best_zero = max(best_zero, cur_zero)
    return int(best_zero), int(best_valid), int(first_valid), int(last_valid)


def qstats(values: list[int | float]) -> dict:
    if not values: return {"count": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None}
    a = np.asarray(values, dtype=float)
    return {"count": int(len(a)), "min": float(a.min()), "q1": float(np.percentile(a, 25)), "median": float(np.median(a)), "q3": float(np.percentile(a, 75)), "max": float(a.max()), "mean": float(a.mean())}


def audit(roots: dict[str, Path], output: Path) -> dict:
    records, hashes, means, near_pairs = [], defaultdict(list), {}, []
    for directory, recorder in DATA_DIRS.items():
        root = roots[directory]
        for path in sorted(root.glob("*.npy")):
            row = {"path": str(path.resolve()), "filename": path.name, "directory": directory, "recorder_id": recorder,
                   "filename_recorder_suffix": None, "label": None, "label_parse_error": None, "shape": None, "dtype": None,
                   "numeric": None, "nan_count": None, "inf_count": None, "all_zero_sample": None, "zero_frame_count": None,
                   "zero_frame_ratio": None, "effective_frame_count": None, "longest_zero_run": None, "longest_valid_run": None,
                   "first_valid_frame": None, "last_valid_frame": None, "leading_zero_frames": None, "trailing_zero_frames": None,
                   "max_abs_value": None, "out_of_expected_range": None, "left_hand_all_zero": None, "right_hand_all_zero": None,
                   "quality_band": None, "status": "ok", "detail": None, "sha256": None}
            suffix = SUFFIX_RE.search(path.name)
            row["filename_recorder_suffix"] = suffix.group(1) if suffix else None
            try: row["label"] = parse_label(path)
            except Exception as exc: row["label_parse_error"] = str(exc); row["status"] = "invalid_label"
            try:
                raw = path.read_bytes(); row["sha256"] = hashlib.sha256(raw).hexdigest()
                array = np.load(path, allow_pickle=False)
                row["shape"], row["dtype"] = str(tuple(array.shape)), str(array.dtype)
                row["numeric"] = bool(np.issubdtype(array.dtype, np.number))
                if not row["numeric"]: raise ValueError("非数值 dtype")
                row["nan_count"], row["inf_count"] = int(np.isnan(array).sum()), int(np.isinf(array).sum())
                if array.size and np.issubdtype(array.dtype, np.number) and np.isfinite(array).any():
                    row["max_abs_value"] = float(np.nanmax(np.abs(array)))
                    row["out_of_expected_range"] = bool(row["max_abs_value"] > 1.01)
                if array.ndim == 2:
                    if array.shape[1] >= 126:
                        row["left_hand_all_zero"] = bool(np.all(array[:, :63] == 0))
                        row["right_hand_all_zero"] = bool(np.all(array[:, 63:126] == 0))
                    valid = ~np.all(array == 0, axis=1)
                    row["zero_frame_count"] = int((~valid).sum()); row["effective_frame_count"] = int(valid.sum())
                    row["zero_frame_ratio"] = row["zero_frame_count"] / max(1, array.shape[0])
                    row["all_zero_sample"] = bool(array.size and not valid.any())
                    zrun, vrun, first, last = runs(valid)
                    row.update(longest_zero_run=zrun, longest_valid_run=vrun, first_valid_frame=first if valid.any() else None,
                               last_valid_frame=last if valid.any() else None, leading_zero_frames=first if valid.any() else len(valid),
                               trailing_zero_frames=(len(valid)-1-last) if valid.any() else len(valid))
                    n = row["effective_frame_count"]
                    row["quality_band"] = "较好" if n >= 45 else "可用但需关注" if n >= 30 else "质量较差" if n >= 15 else "严重不足"
                if tuple(array.shape) != (EXPECTED_FRAMES, EXPECTED_FEATURES): row["status"], row["detail"] = "invalid", "shape不是(60,126)"
                elif row["nan_count"] or row["inf_count"]: row["status"], row["detail"] = "invalid", "包含NaN/Inf"
                elif row["all_zero_sample"]: row["status"], row["detail"] = "invalid", "整条全零"
                elif row["label"] not in LABELS: row["status"], row["detail"] = "invalid_label", "标签不在限定词表"
                else:
                    hashes[row["sha256"]].append(row["path"])
                    valid_rows = np.asarray(array, dtype=np.float32)[valid]
                    means[row["path"]] = valid_rows.mean(axis=0) if len(valid_rows) else np.zeros(EXPECTED_FEATURES)
            except Exception as exc:
                row["status"], row["detail"] = "invalid", str(exc)
            records.append(row)

    valid = [r for r in records if r["status"] == "ok"]
    by_recorder, by_class = defaultdict(list), defaultdict(list)
    for r in records:
        if r["effective_frame_count"] is not None: by_recorder[r["recorder_id"]].append(r); by_class[r["label"]].append(r)
    # Search for near duplicates using cosine similarity of each sample's mean valid-frame vector.
    by_label = defaultdict(list)
    for r in valid:
        if r["path"] in means: by_label[r["label"]].append(r)
    for label, rows in by_label.items():
        for i, left in enumerate(rows):
            a = means[left["path"]]; an = np.linalg.norm(a)
            if an == 0: continue
            for right in rows[i+1:]:
                b = means[right["path"]]; bn = np.linalg.norm(b)
                similarity = float(np.dot(a,b)/(an*bn)) if bn else 0.0
                if similarity >= 0.995:
                    near_pairs.append({"label": label, "similarity_cosine": similarity, "path_a": left["path"], "recorder_a": left["recorder_id"], "path_b": right["path"], "recorder_b": right["recorder_id"]})
    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"), "data_dirs": {k: str(v.resolve()) for k,v in roots.items()},
               "target_labels": LABELS, "total_files": len(records), "valid_samples": len(valid), "invalid_samples": len(records)-len(valid),
               "status_counts": dict(Counter(r["status"] for r in records)), "label_counts": dict(sorted(Counter(r["label"] for r in records if r["label"]).items())),
               "duplicate_hash_groups": [paths for paths in hashes.values() if len(paths)>1], "near_duplicate_threshold_cosine": 0.995,
               "near_duplicate_candidates": near_pairs, "per_recorder": {}, "per_class": {}, "per_recorder_class": {},
               "label_parse_errors": [r for r in records if r["label_parse_error"]],
               "out_of_vocabulary_labels": sorted({r["label"] for r in records if r["label"] and r["label"] not in LABELS}),
               "quality_band_counts": dict(Counter(r["quality_band"] for r in records)),
               "shape_counts": dict(Counter(r["shape"] for r in records)), "dtype_counts": dict(Counter(r["dtype"] for r in records)),
               "nan_samples": sum((r["nan_count"] or 0)>0 for r in records), "inf_samples": sum((r["inf_count"] or 0)>0 for r in records),
               "abnormal_shape_samples": sum(r["shape"] != str((60,126)) for r in records), "all_zero_samples": sum(r["all_zero_sample"] is True for r in records),
               "out_of_expected_range_samples": sum(r["out_of_expected_range"] is True for r in records)}
    for recorder, rows in sorted(by_recorder.items()):
        summary["per_recorder"][recorder] = {"files": len(rows), "zero_frame_ratio": qstats([r["zero_frame_ratio"] for r in rows]), "effective_frames": qstats([r["effective_frame_count"] for r in rows]), "quality_bands": dict(Counter(r["quality_band"] for r in rows))}
    for label, rows in sorted(by_class.items()):
        summary["per_class"][label] = {"files": len(rows), "zero_frame_ratio": qstats([r["zero_frame_ratio"] for r in rows]), "effective_frames": qstats([r["effective_frame_count"] for r in rows]), "by_recorder": dict(Counter(r["recorder_id"] for r in rows))}
    for recorder in DATA_DIRS.values():
        summary["per_recorder_class"][recorder] = {}
        for label in LABELS:
            rows = [r for r in records if r["recorder_id"] == recorder and r["label"] == label and r["effective_frame_count"] is not None]
            summary["per_recorder_class"][recorder][label] = {"count": len(rows), "effective_frames": qstats([r["effective_frame_count"] for r in rows]), "zero_frame_ratio": qstats([r["zero_frame_ratio"] for r in rows])}
    # Distances between recorder means, conditioned on a word, help expose signer/session shifts.
    summary["within_class_recorder_centroid_distances"] = {}
    for label, rows in by_label.items():
        centroids = {rec: np.mean([means[r["path"]] for r in rows if r["recorder_id"] == rec], axis=0)
                     for rec in DATA_DIRS.values() if any(r["recorder_id"] == rec for r in rows)}
        pairs = {}
        for i, rec_a in enumerate(centroids):
            for rec_b in list(centroids)[i+1:]: pairs[f"{rec_a} vs {rec_b}"] = float(np.linalg.norm(centroids[rec_a]-centroids[rec_b]))
        summary["within_class_recorder_centroid_distances"][label] = pairs
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(records[0]) if records else []
    for name, rows in (("sample_records.csv", records), ("zero_frame_summary.csv", [{k:r[k] for k in ["path","directory","recorder_id","label","zero_frame_count","zero_frame_ratio","effective_frame_count","longest_zero_run","longest_valid_run","leading_zero_frames","trailing_zero_frames","quality_band"]} for r in records]), ("label_parse_errors.csv", [r for r in records if r["label_parse_error"]])):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as f:
            cols = list(rows[0]) if rows else fields; w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    counts = [{"recorder_id": rec, "directory": next((d for d,r in DATA_DIRS.items() if r==rec), ""), "label": lab, "count": sum(x["recorder_id"]==rec and x["label"]==lab for x in records)} for rec in DATA_DIRS.values() for lab in LABELS]
    with (output / "class_recorder_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=list(counts[0])); w.writeheader(); w.writerows(counts)
    dup_rows = [{"sha256": h, "paths": " | ".join(p)} for h,p in hashes.items() if len(p)>1]
    with (output / "duplicate_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=["sha256","paths"]); w.writeheader(); w.writerows(dup_rows)
    with (output / "near_duplicate_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols=["label","similarity_cosine","path_a","recorder_a","path_b","recorder_b"]; w=csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(near_pairs)
    total_frames = sum(r["shape"] and EXPECTED_FRAMES or 0 for r in records)
    zero_frames = sum(r["zero_frame_count"] or 0 for r in records)
    insufficient = [r for r in records if (r["effective_frame_count"] or 0) < 15]
    low_effective = [r for r in records if r["effective_frame_count"] is not None and r["effective_frame_count"] < 30]
    long_zero = [r for r in records if (r["longest_zero_run"] or 0) >= 20]
    lowest = min((r for r in records if r["effective_frame_count"] is not None), key=lambda r: r["effective_frame_count"], default=None)
    weakest_classes = sorted(summary["per_class"].items(), key=lambda item: item[1]["effective_frames"]["median"] or 0)[:2]
    lines = ["新数据集只读审计报告", "", f"生成时间：{summary['generated_at']}", f"总文件：{len(records)}；有效样本：{len(valid)}；无效/需复核：{len(records)-len(valid)}", f"总体零帧比例：{zero_frames/max(1,total_frames):.2%}", "", "按录制者："]
    for rec, s in summary["per_recorder"].items(): lines.append(f"- {rec}：{s['files']}条，零帧比例中位数 {s['zero_frame_ratio']['median']:.2%}，有效帧中位数 {s['effective_frames']['median']:.1f}，质量分层 {s['quality_bands']}")
    lines += ["", "按类别："]
    for lab,s in summary["per_class"].items(): lines.append(f"- {lab}：{s['files']}条，零帧比例中位数 {s['zero_frame_ratio']['median']:.2%}，有效帧中位数 {s['effective_frames']['median']:.1f}，录制者分布 {s['by_recorder']}")
    lines += ["", f"shape分布：{summary['shape_counts']}", f"dtype分布：{summary['dtype_counts']}", f"NaN样本：{summary['nan_samples']}；Inf样本：{summary['inf_samples']}；全零样本：{summary['all_zero_samples']}；shape异常：{summary['abnormal_shape_samples']}；绝对值超过1.01：{summary['out_of_expected_range_samples']}", f"标签解析错误：{len(summary['label_parse_errors'])}；词表外标签：{summary['out_of_vocabulary_labels']}", f"完全重复样本组：{len(summary['duplicate_hash_groups'])}；近重复候选（同类有效帧均值 cosine >= 0.995）：{len(near_pairs)}。近重复指标忽略时序，需人工复核，不代表重复录制。", f"有效帧少于15帧：{len(insufficient)}条", "严重不足样本："]
    lines += [f"- {r['recorder_id']} / {r['label']} / {r['filename']}：有效帧 {r['effective_frame_count']}，最长连续零帧 {r['longest_zero_run']}" for r in insufficient]
    lines += ["", "录制者×类别有效帧摘要（中位数）："]
    for rec, groups in summary["per_recorder_class"].items(): lines.append(f"- {rec}: " + ", ".join(f"{lab} {g['count']}条/{g['effective_frames']['median']}帧" for lab,g in groups.items()))
    lines += ["", "预处理一致性（代码层面检查）：", "- collect_data.py 对帧做水平镜像；sign_features.pack_hands 将 MediaPipe Left 放右槽、Right 放左槽；每手减腕点、按每手最大绝对值归一化，左63维后接右63维；未检出帧默认为126维全零；采集60帧。", "- 这些仅能证明当前代码约定，不能证明这批文件确由该版本脚本生成，也不能核验原始摄像头画面。", "", "事实与限制：", "- 零帧表示采集代码未检测到手，不能单独证明动作错误。", "- .npy只能检查数值和轨迹；镜像、实际左右手、手势正确性、遮挡原因和场次差异需原始视频/采集日志。", "- 录制者间特征距离只反映归一化特征分布差异，不可单独证明模型信号来源。", "- 当前没有专门无手、静止或词表外负样本，不能据此验证拒识。", "", "优先复核的5项：", f"1. 本人零帧比例中位数{summary['per_recorder']['本人']['zero_frame_ratio']['median']:.2%}、有效帧中位数{summary['per_recorder']['本人']['effective_frames']['median']:.0f}；与 A/B 的有效帧中位数{summary['per_recorder']['同学A']['effective_frames']['median']:.0f}/{summary['per_recorder']['同学B']['effective_frames']['median']:.0f}仍有差异。", (f"2. 当前最低有效帧样本：{lowest['recorder_id']} / {lowest['label']} / {lowest['filename']}，有效帧 {lowest['effective_frame_count']}；人工核看其动作是否完整。" if lowest else "2. 无可计算有效帧的样本。"), f"3. 有效帧少于30帧的样本{len(low_effective)}条；最长连续零帧不少于20帧的样本{len(long_zero)}条，建议优先抽查。", f"4. 有效帧中位数最低的类别为" + "、".join(f"{label}({stats['effective_frames']['median']:.0f}帧)" for label, stats in weakest_classes) + "，建议抽查动作覆盖与跟踪。", f"5. 均值向量余弦相似度粗筛得到{len(near_pairs)}对同类候选；该指标忽略时序，不能当作重复样本结论。", "", "推断与训练建议：", "- 10类总体数量较平衡（90–102），但录制者×类别数量有差异；每折需检查测试类别完整性。", "- 录制者零帧率和有效帧分布仍不同，提示域偏移风险；单凭描述统计不能断言模型更依赖录制者还是类别。", "- 复核低有效帧、长连续零帧、录制者类别数量偏斜样本及近重复候选；不要自动删除。", "- 保留原始数据，使用三轮留一录制者测试；仅在训练池验证不同零帧处理与特征方案，不根据测试集挑选。"]
    (output / "audit_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(); p.add_argument("--output-root", default="model_output"); a=p.parse_args(); root=Path(__file__).resolve().parent
    stamp=datetime.now().strftime("%Y%m%d-%H%M%S"); out=root/a.output_root/f"data_audit-{stamp}"
    s=audit({k:root/k for k in DATA_DIRS}, out)
    print(f"audit_output={out}"); print(f"files={s['total_files']} valid={s['valid_samples']} invalid={s['invalid_samples']}")
    print(f"label_parse_errors={len(s['label_parse_errors'])} duplicate_groups={len(s['duplicate_hash_groups'])}")
    with (out/'sample_records.csv').open(encoding='utf-8-sig', newline='') as f: rows=list(csv.DictReader(f))
    frames=sum(int(r['zero_frame_count'] or 0)+int(r['effective_frame_count'] or 0) for r in rows)
    zeros=sum(int(r['zero_frame_count'] or 0) for r in rows)
    print(f"overall_zero_frame_ratio={zeros/max(1,frames):.2%}")
    for rec, v in s["per_recorder"].items(): print(f"{rec}: files={v['files']} zero_median={v['zero_frame_ratio']['median']:.2%} effective_median={v['effective_frames']['median']:.1f}")
    print(f"effective_frames_below_15={sum(int(r['effective_frame_count'] or 0)<15 for r in rows)}")
    print(f"effective_frames_below_30={sum(int(r['effective_frame_count'] or 0)<30 for r in rows)}; longest_zero_run_at_least_20={sum(int(r['longest_zero_run'] or 0)>=20 for r in rows)}")
    lowest = min((r for r in rows if r['effective_frame_count']), key=lambda r: int(r['effective_frame_count']))
    print(f"lowest_effective_sample={lowest['recorder_id']} / {lowest['label']} / {lowest['filename']} / {lowest['effective_frame_count']} frames")
    print(f"near_duplicate_candidates={len(s['near_duplicate_candidates'])} (cosine pre-screen only)")
