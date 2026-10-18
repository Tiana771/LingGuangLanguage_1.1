import assert from "node:assert/strict";
import { RecognitionSession } from "./recognition_session.mjs";

const session = new RecognitionSession({ noHandResetMs: 500 });
assert.equal(session.record("你好", 1000), true);
assert.equal(session.record("你好", 1100), false);
assert.equal(session.recordAccepted({ accepted: false, label: "谢谢" }, 1150), false);
assert.equal(session.history.length, 1);
assert.deepEqual(session.history.map(({ label }) => label), ["你好"]);

session.observeNoHands(1200);
assert.equal(session.observeNoHands(1699), false);
assert.equal(session.history.length, 1);
assert.equal(session.observeNoHands(1700), true);
assert.equal(session.record("你好", 1800), true);

session.record("谢谢", 1900);
assert.deepEqual(session.history.map(({ label }) => label), ["你好", "你好", "谢谢"]);
assert.equal(session.undo().label, "谢谢");
assert.equal(session.record("谢谢", 1950), true);
session.clear();
assert.equal(session.history.length, 0);

session.record("请问", 2000);
session.resetRound();
assert.equal(session.history.length, 0);
assert.equal(session.record("请问", 2100), true);
session.observeNoHands(2200);
session.observeHands();
session.observeNoHands(2300);
assert.equal(session.observeNoHands(2799), false);
assert.equal(session.observeNoHands(2800), true);
console.log("recognition session tests passed");
