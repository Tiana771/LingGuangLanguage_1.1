from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import ceil


RATINGS = {"remembered", "fuzzy", "forgot"}
MAX_INTERVAL_DAYS = 180


def next_review(previous_interval_days: int, rating: str, now: datetime, *, is_new: bool) -> dict:
    """Return a deterministic, feedback-based schedule for one vocabulary item."""
    if rating not in RATINGS:
        raise ValueError("Unsupported learning rating")
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    current = max(0, int(previous_interval_days or 0))
    if is_new:
        if rating == "forgot":
            interval = 0
            due_at = now.astimezone(timezone.utc) + timedelta(minutes=10)
        elif rating == "fuzzy":
            interval = 1
            due_at = now.astimezone(timezone.utc) + timedelta(days=1)
        else:
            interval = 3
            due_at = now.astimezone(timezone.utc) + timedelta(days=3)
    elif rating == "forgot":
        interval = 1
        due_at = now.astimezone(timezone.utc) + timedelta(days=1)
    elif rating == "fuzzy":
        interval = min(MAX_INTERVAL_DAYS, max(1, ceil(max(1, current) * 1.2)))
        due_at = now.astimezone(timezone.utc) + timedelta(days=interval)
    else:
        interval = min(MAX_INTERVAL_DAYS, max(current + 1, ceil(max(1, current) * 2.2)))
        due_at = now.astimezone(timezone.utc) + timedelta(days=interval)

    return {
        "interval_days": interval,
        "next_review_at": due_at.isoformat(),
        "repetitions_delta": 0 if rating == "forgot" else 1,
        "repetitions_reset": rating == "forgot",
    }


def update_day_mastery(*, existing_session: bool, previous_required: int, previous_successes: int,
                       previous_initial_rating: str, previous_is_new: bool, is_new: bool,
                       rating: str) -> dict:
    """Track same-day consecutive correct recalls without changing long-term intervals."""
    if rating not in RATINGS:
        raise ValueError("Unsupported learning rating")
    if existing_session:
        required = max(0, int(previous_required or 0))
        successes = max(0, int(previous_successes or 0))
        initial_rating = previous_initial_rating or rating
        session_is_new = bool(previous_is_new)
        successes = successes + 1 if rating == "remembered" else 0
    else:
        required = 3 if rating == "forgot" else 2 if rating == "fuzzy" else 0
        successes = 0
        initial_rating = rating
        session_is_new = is_new
    complete = required == 0 or successes >= required
    return {"required": required, "successes": successes, "initial_rating": initial_rating,
            "session_is_new": session_is_new, "complete": complete, "repeat_today": not complete}
