export class PredictionSmoother {
  constructor({ windowSize = 3, lockMs = 1500, majority = 2 } = {}) {
    if (windowSize < 3 || majority < 2 || majority > windowSize || lockMs < 0) {
      throw new Error("Invalid prediction smoother configuration");
    }
    this.windowSize = windowSize;
    this.lockMs = lockMs;
    this.majority = majority;
    this.window = [];
    this.label = null;
    this.lockedUntil = 0;
  }

  push(result, now = Date.now()) {
    if (now < this.lockedUntil) {
      return { label: this.label, locked: true, changed: false, votes: this.window.slice() };
    }
    if (this.label && now >= this.lockedUntil) this.label = null;
    this.window.push(result?.accepted && typeof result.label === "string" ? result.label : null);
    this.window = this.window.slice(-this.windowSize);
    const counts = new Map();
    for (const label of this.window) {
      if (label) counts.set(label, (counts.get(label) || 0) + 1);
    }
    const winner = [...counts].find(([, count]) => count >= this.majority)?.[0] || null;
    if (winner) {
      this.label = winner;
      this.lockedUntil = now + this.lockMs;
      return { label: winner, locked: this.lockMs > 0, changed: true, votes: this.window.slice() };
    }
    return { label: null, locked: false, changed: false, votes: this.window.slice() };
  }

  reset() {
    this.window = [];
    this.label = null;
    this.lockedUntil = 0;
  }

  isLocked(now = Date.now()) {
    return now < this.lockedUntil;
  }
}
