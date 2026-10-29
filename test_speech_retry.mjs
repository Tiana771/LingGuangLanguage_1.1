import assert from "node:assert/strict";
import { MAX_SPEECH_NETWORK_RETRIES, speechRetryDelay } from "./speech_retry.mjs";

assert.equal(MAX_SPEECH_NETWORK_RETRIES, 3);
assert.equal(speechRetryDelay("network", 1), 1000);
assert.equal(speechRetryDelay("network", 2), 2000);
assert.equal(speechRetryDelay("network", 3), 4000);
assert.equal(speechRetryDelay("network", 4), null);
assert.equal(speechRetryDelay("not-allowed", 1), null);
assert.equal(speechRetryDelay("network", 0), null);
console.log("Speech recognition retry policy tests passed");
