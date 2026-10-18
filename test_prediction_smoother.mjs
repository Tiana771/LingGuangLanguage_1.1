import assert from "node:assert/strict";
import { PredictionSmoother } from "./prediction_smoother.mjs";

const smoother = new PredictionSmoother({ windowSize: 3, majority: 2, lockMs: 1500 });
assert.equal(smoother.push({ accepted: true, label: "A" }, 0).label, null);
assert.equal(smoother.push({ accepted: false, label: null }, 500).label, null);
const acceptedA = smoother.push({ accepted: true, label: "A" }, 1000);
assert.equal(acceptedA.label, "A");
assert.equal(acceptedA.locked, true);
assert.equal(smoother.push({ accepted: true, label: "B" }, 2000).label, "A");
assert.equal(smoother.push({ accepted: true, label: "B" }, 2600).label, null);

smoother.reset();
assert.equal(smoother.push({ accepted: true, label: "A" }, 0).label, null);
assert.equal(smoother.push({ accepted: true, label: "B" }, 500).label, null);
assert.equal(smoother.push({ accepted: false, label: null }, 1000).label, null);
console.log("prediction smoother tests passed");
