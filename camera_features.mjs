export const FRAME_COUNT = 60;
export const FEATURE_COUNT = 126;

const ZERO_HAND = () => new Array(63).fill(0);

export function normalizeHand(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) {
    throw new Error("A hand must contain 21 landmarks");
  }
  const wrist = landmarks[0];
  const values = [];
  for (const point of landmarks) {
    values.push(point.x - wrist.x, point.y - wrist.y, point.z - wrist.z);
  }
  const scale = Math.max(...values.map((value) => Math.abs(value))) + 1e-6;
  return values.map((value) => value / scale);
}

export function frameFeatures(detections) {
  const left = ZERO_HAND();
  const right = ZERO_HAND();
  const confidence = { Left: -1, Right: -1 };
  let handCount = 0;

  for (const detection of detections) {
    const label = detection.handedness;
    if (label !== "Left" && label !== "Right") continue;
    const score = Number.isFinite(detection.score) ? detection.score : 1;
    if (score < confidence[label]) continue;
    const values = normalizeHand(detection.landmarks);
    // collect_data.py maps mirrored-frame MediaPipe Left to the right-hand slot.
    if (label === "Left") right.splice(0, 63, ...values);
    else left.splice(0, 63, ...values);
    confidence[label] = score;
  }

  handCount = Number(confidence.Left >= 0) + Number(confidence.Right >= 0);
  const features = left.concat(right);
  if (features.length !== FEATURE_COUNT) throw new Error("Feature frame must contain 126 values");
  return { features, handCount };
}

export function validFrameCount(sequence) {
  return sequence.reduce((count, frame) => count + Number(frame.some((value) => value !== 0)), 0);
}
