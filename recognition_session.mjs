export class RecognitionSession {
  constructor({ noHandResetMs = 500, maxHistory = 30 } = {}) {
    this.noHandResetMs = noHandResetMs;
    this.maxHistory = maxHistory;
    this.noHandSince = null;
    this.armed = true;
    this.lastRecordedLabel = null;
    this.history = [];
  }

  observeHands() {
    this.noHandSince = null;
  }

  observeNoHands(now = Date.now()) {
    if (this.noHandSince === null) this.noHandSince = now;
    if (!this.armed && now - this.noHandSince >= this.noHandResetMs) {
      this.armed = true;
      this.lastRecordedLabel = null;
      this.noHandSince = null;
      return true;
    }
    return false;
  }

  record(label, now = Date.now()) {
    if (typeof label !== "string" || !label) return false;
    if (!this.armed && label === this.lastRecordedLabel) return false;
    this.armed = false;
    this.lastRecordedLabel = label;
    this.history.push({ label, timestamp: now });
    this.history = this.history.slice(-this.maxHistory);
    return true;
  }

  recordAccepted(payload, now = Date.now()) {
    if (!payload?.accepted || typeof payload.label !== "string") return false;
    return this.record(payload.label, now);
  }

  undo() {
    const removed = this.history.pop() || null;
    if (removed && removed.label === this.lastRecordedLabel) {
      this.armed = true;
      this.lastRecordedLabel = null;
    }
    return removed;
  }

  clear() {
    this.history = [];
  }

  resetRound() {
    this.noHandSince = null;
    this.armed = true;
    this.lastRecordedLabel = null;
    this.clear();
  }
}
