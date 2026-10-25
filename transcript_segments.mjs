export function joinTranscriptSegments(segments = []) {
  let text = "";
  const ranges = [];
  for (const segment of segments) {
    const content = String(segment.text || "");
    if (!content.trim()) continue;
    const previous = text.at(-1) || "";
    const first = content[0] || "";
    const separator = /[A-Za-z0-9]$/.test(previous) && /^[A-Za-z0-9]/.test(first) ? " " : "";
    text += separator;
    const start = text.length;
    text += content;
    ranges.push({ id: segment.id, text: content, start, end: text.length, createdAt: segment.createdAt });
  }
  return { text, ranges };
}

export function getTranscriptSelection(selection, root) {
  if (!selection || selection.isCollapsed || !selection.rangeCount) return "";
  const range = selection.getRangeAt(0);
  if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) return "";
  return selection.toString().trim();
}
