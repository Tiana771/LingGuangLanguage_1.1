import assert from "node:assert/strict";
import { getTranscriptSelection, joinTranscriptSegments } from "./transcript_segments.mjs";
import { appendTranscriptText, migrateLegacySegments } from "./transcript_blocks.mjs";

const segments = [
  { id: "a", text: "下午三点", createdAt: "2026-10-20T07:00:00Z" },
  { id: "b", text: "图书馆见", createdAt: "2026-10-20T07:00:03Z" },
  { id: "c", text: "Meet", createdAt: "2026-10-20T07:00:06Z" },
  { id: "d", text: "Alex", createdAt: "2026-10-20T07:00:08Z" },
];
const transcript = joinTranscriptSegments(segments);
assert.equal(transcript.text, "下午三点图书馆见Meet Alex");
assert.deepEqual(transcript.ranges.map(({ id, start, end }) => ({ id, start, end })), [
  { id: "a", start: 0, end: 4 },
  { id: "b", start: 4, end: 8 },
  { id: "c", start: 8, end: 12 },
  { id: "d", start: 13, end: 17 },
]);

const start = {};
const end = {};
const root = { contains: (node) => node === start || node === end };
assert.equal(getTranscriptSelection({
  isCollapsed: false,
  rangeCount: 1,
  getRangeAt: () => ({ startContainer: start, endContainer: end }),
  toString: () => " 要发给 AI 的选中文字 ",
}, root), "要发给 AI 的选中文字");
assert.equal(getTranscriptSelection({
  isCollapsed: false,
  rangeCount: 1,
  getRangeAt: () => ({ startContainer: {}, endContainer: end }),
  toString: () => "外部文字",
}, root), "");

const legacyRecord = {
  id: "legacy",
  createdAt: "2026-10-20T07:00:00Z",
  segments: [
    { text: "下午三点", createdAt: "2026-10-20T07:00:00Z" },
    { text: "图书馆见", createdAt: "2026-10-20T07:00:03Z" },
  ],
};
const migrated = migrateLegacySegments(legacyRecord, () => "legacy-block");
assert.equal(migrated.length, 1);
assert.equal(migrated[0].text, "下午三点图书馆见");
assert.equal(migrated[0].id, "legacy-block");
assert.equal("segments" in legacyRecord, false);
assert.equal(migrateLegacySegments({ blocks: [{ id: "one" }] }, () => "unused")[0].id, "one");
assert.equal(appendTranscriptText("下午三点", "图书馆见"), "下午三点图书馆见");
assert.equal(appendTranscriptText("Meet", "Alex"), "Meet Alex");

console.log("Transcript block and selection tests passed");
