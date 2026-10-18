"""Compare Python and browser scheme-B features on a real recorded sample."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from sign_baseline import pooled_sequence_features, validate_sequence


ROOT = Path(__file__).resolve().parent


def main() -> None:
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node.js is required for Python/JS feature parity verification")
    sample_path = next((ROOT / "sign_data").glob("*.npy"), None)
    if sample_path is None:
        raise SystemExit("No real .npy sample found under sign_data")
    sequence = validate_sequence(np.load(sample_path, allow_pickle=False))
    python_features = pooled_sequence_features(sequence, min_selected_frames=30)
    if python_features is None:
        raise SystemExit(f"Selected sample has fewer than 30 any_valid frames: {sample_path}")
    module_url = (ROOT / "camera_features.mjs").as_uri()
    js = (
        "import fs from 'node:fs'; "
        f"import {{ sequenceFeatures }} from {json.dumps(module_url)}; "
        "const seq=JSON.parse(fs.readFileSync(process.argv[1], 'utf8')); "
        "console.log(JSON.stringify(sequenceFeatures(seq, 30)));"
    )
    with tempfile.TemporaryDirectory() as temp:
        sequence_file = Path(temp) / "sequence.json"
        sequence_file.write_text(json.dumps(sequence.tolist(), separators=(",", ":")), encoding="utf-8")
        completed = subprocess.run([node, "--input-type=module", "-e", js, str(sequence_file)],
                                   check=True, capture_output=True, text=True)
    js_features = np.asarray(json.loads(completed.stdout), dtype=np.float32)
    if js_features.shape != (252,):
        raise SystemExit(f"Expected 252 JS features, got {js_features.shape}")
    error = float(np.max(np.abs(python_features - js_features)))
    print(f"sample={sample_path.name}")
    print(f"python_features={python_features.tolist()}")
    print(f"javascript_features={js_features.tolist()}")
    print(f"max_absolute_error={error:.10g}")
    print(f"parity={'PASS' if error < 1e-5 else 'FAIL'}")
    if error >= 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
