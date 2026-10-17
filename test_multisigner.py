from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sign_baseline import parse_label
from train_multisigner import DATA_DIRS, LABELS, inventory, validate_fold_isolation, validation_split


class MultiSignerTests(unittest.TestCase):
    def test_recorder_suffix_is_not_a_class_label(self):
        self.assertEqual(parse_label(Path("1_谢谢_A_6.npy")), "谢谢")
        self.assertEqual(parse_label(Path("1_谢谢_B_2.npy")), "谢谢")
        self.assertEqual(parse_label(Path("1_谢谢_0.npy")), "谢谢")

    def test_inventory_reports_effective_and_zero_frames_with_directory_recorder(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = {}
            for directory in DATA_DIRS:
                root = Path(temp) / directory
                root.mkdir()
                roots[directory] = root
            sample = np.ones((60, 126), dtype=np.float64)
            sample[:7] = 0
            np.save(roots["sign_data_a"] / "1_谢谢_A_0.npy", sample)
            records, summary = inventory(roots)
        record = records[0]
        self.assertEqual(record["label"], "谢谢")
        self.assertEqual(record["recorder_id"], "同学A")
        self.assertEqual(record["filename_recorder_suffix"], "A")
        self.assertEqual(record["zero_frame_count"], 7)
        self.assertEqual(record["effective_frame_count"], 53)
        self.assertEqual(summary["per_directory"]["sign_data_a"]["file_label_parse_errors"], 0)

    def test_validation_is_sample_level_and_keeps_each_recorder_in_pool(self):
        rows = [
            {"path": f"{recorder}-{label}-{index}", "label": label, "recorder_id": recorder}
            for recorder in ("本人", "同学A")
            for label in LABELS
            for index in range(3)
        ]
        train, validation, plan = validation_split(rows, 42)
        self.assertFalse({row["path"] for row in train} & {row["path"] for row in validation})
        self.assertEqual({row["recorder_id"] for row in validation}, {"本人", "同学A"})
        self.assertTrue(all(plan[label]["validation"] == 2 for label in LABELS))
        self.assertTrue(all(count > 0 for count in {label: sum(row["label"] == label for row in train) for label in LABELS}.values()))

    def test_fold_isolation_rejects_recorder_leakage_and_sample_overlap(self):
        train = [{"path": "a", "directory": "sign_data_a"}]
        validation = [{"path": "b", "directory": "sign_data_b"}]
        test = [{"path": "c", "directory": "sign_data"}]
        validate_fold_isolation(train, validation, test, "sign_data")
        with self.assertRaises(RuntimeError):
            validate_fold_isolation(train, validation, test + [{"path": "a", "directory": "sign_data"}], "sign_data")
        with self.assertRaises(RuntimeError):
            validate_fold_isolation(train + [{"path": "leak", "directory": "sign_data"}], validation, test, "sign_data")


if __name__ == "__main__":
    unittest.main()
