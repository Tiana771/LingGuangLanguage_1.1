from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sign_features import pack_hands


class CameraFeatureTests(unittest.TestCase):
    def test_tasks_handedness_is_adapted_to_training_label_convention(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is not installed; handedness mapping check skipped")
        module_url = Path(__file__).with_name("camera_features.mjs").as_uri()
        js = (
            f"import {{ taskHandednessToTrainingLabel }} from {json.dumps(module_url)}; "
            "console.log(JSON.stringify(['Left','Right','Unknown'].map(taskHandednessToTrainingLabel)));"
        )
        completed = subprocess.run([node, "--input-type=module", "-e", js],
                                   check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(completed.stdout), ["Right", "Left", "Unknown"])

    def test_python_and_browser_feature_packing_match(self):
        points_left = [
            {"x": 0.1 + i * 0.003, "y": 0.2 - i * 0.002, "z": -i * 0.001}
            for i in range(21)
        ]
        points_right = [
            {"x": 0.7 - i * 0.004, "y": 0.4 + i * 0.001, "z": i * 0.002}
            for i in range(21)
        ]
        detections = [
            {"landmarks": points_left, "handedness": "Left", "score": 0.91},
            {"landmarks": points_right, "handedness": "Right", "score": 0.87},
        ]
        python_points = lambda values: [SimpleNamespace(**point) for point in values]
        expected = pack_hands([
            (python_points(points_left), "Left", 0.91),
            (python_points(points_right), "Right", 0.87),
        ])

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is not installed; browser helper parity check skipped")
        module_url = Path(__file__).with_name("camera_features.mjs").as_uri()
        js = (
            f"import {{ frameFeatures }} from {json.dumps(module_url)}; "
            "const input=JSON.parse(process.env.CAMERA_FIXTURE); "
            "console.log(JSON.stringify(frameFeatures(input).features));"
        )
        completed = subprocess.run(
            [node, "--input-type=module", "-e", js],
            check=True, capture_output=True, text=True,
            env={**__import__("os").environ, "CAMERA_FIXTURE": json.dumps(detections)},
        )
        browser_features = np.asarray(json.loads(completed.stdout), dtype=np.float64)
        self.assertEqual(browser_features.shape, (126,))
        np.testing.assert_allclose(browser_features, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
