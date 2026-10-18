export const FRAME_COUNT = 60;
export const FEATURE_COUNT = 126;
export const EPS = 1e-8;

const ZERO_HAND = () => new Array(63).fill(0);

export function taskHandednessToTrainingLabel(label) {
  if (label === "Left") return "Right";
  if (label === "Right") return "Left";
  return label;
}

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
  return sequence.reduce((count, frame) => {
    const left = frame.slice(0, 63).some((value) => Math.abs(value) > EPS);
    const right = frame.slice(63, 126).some((value) => Math.abs(value) > EPS);
    return count + Number(left || right);
  }, 0);
}

export function sequenceFeatures(sequence, minSelectedFrames = 30) {
  if (!Array.isArray(sequence) || sequence.length !== FRAME_COUNT ||
      sequence.some((frame) => !Array.isArray(frame) || frame.length !== FEATURE_COUNT)) {
    throw new Error("Sequence must have shape (60, 126)");
  }
  const selected = sequence.filter((frame) => frame.some((value) => Math.abs(value) > EPS));
  if (selected.length < minSelectedFrames) return null;
  const rows = selected.map((row) => Float32Array.from(row));
  const mean = new Float32Array(FEATURE_COUNT);
  const std = new Float32Array(FEATURE_COUNT);
  for (let column = 0; column < FEATURE_COUNT; column += 1) {
    let sum = 0;
    for (const row of rows) sum += row[column];
    mean[column] = sum / rows.length;
    let squared = 0;
    for (const row of rows) {
      const delta = row[column] - mean[column];
      squared += delta * delta;
    }
    std[column] = Math.sqrt(squared / rows.length);
  }
  return Array.from(mean).concat(Array.from(std));
}
