"""Per-frame landmark normalization shared with the browser inference path."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def normalize_hand(landmarks: Iterable) -> np.ndarray:
    points = list(landmarks)
    if len(points) != 21:
        raise ValueError("a hand must contain 21 landmarks")
    wrist = points[0]
    values = np.asarray(
        [[point.x - wrist.x, point.y - wrist.y, point.z - wrist.z] for point in points],
        dtype=np.float64,
    ).reshape(-1)
    return values / (np.max(np.abs(values)) + 1e-6)


def pack_hands(hands: Iterable[tuple[Iterable, str]]) -> np.ndarray:
    left = np.zeros(63, dtype=np.float64)
    right = np.zeros(63, dtype=np.float64)
    scores = {"Left": -1.0, "Right": -1.0}
    for landmarks, label, *score_value in hands:
        if label not in scores:
            continue
        score = float(score_value[0]) if score_value else 1.0
        if score < scores[label]:
            continue
        features = normalize_hand(landmarks)
        if label == "Left":
            right = features
        else:
            left = features
        scores[label] = score
    return np.concatenate([left, right])
