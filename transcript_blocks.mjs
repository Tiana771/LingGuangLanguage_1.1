export function migrateLegacySegments(record, makeId) {
  if (Array.isArray(record.blocks)) return record.blocks;
  const segments = Array.isArray(record.segments) ? record.segments : [];
  const text = segments.reduce((currentText, segment) => appendTranscriptText(currentText, segment.text), "");
  record.blocks = text ? [{
    id: makeId(),
    title: "语音记录 1",
    createdAt: segments[0]?.createdAt || record.createdAt,
    text,
  }] : [];
  delete record.segments;
  return record.blocks;
}

export function appendTranscriptText(currentText, recognizedText) {
  const current = String(currentText || "").trimEnd();
  const next = String(recognizedText || "").trim();
  if (!next) return current;
  const separator = /[A-Za-z0-9]$/.test(current) && /^[A-Za-z0-9]/.test(next) ? " " : "";
  return `${current}${separator}${next}`;
}
