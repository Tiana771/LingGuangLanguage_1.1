"""Record model calls without storing credentials or conversation content."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4


def _numeric_usage(value):
    if isinstance(value, dict):
        return {key: _numeric_usage(item) for key, item in value.items()
                if isinstance(item, (dict, int, float)) and not isinstance(item, bool)}
    return value


class ModelCallAudit:
    def __init__(self, *, logger, action, owner_id, client_ip, endpoint, model,
                 messages, max_tokens, log_path=None, key_label=None):
        self.logger = logger
        self.log_path = log_path
        self.request_id = uuid4().hex
        self.started = time.monotonic()
        self.finished = False
        self.details = {
            "request_id": self.request_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "user_id": owner_id,
            "client_ip": client_ip,
            "endpoint": endpoint,
            "key_label": key_label or "unlabelled",
            "model_requested": model,
            "message_count": len(messages),
            "message_chars": sum(len(message.get("content", "")) for message in messages),
            "max_tokens": max_tokens,
            "usage": None,
        }
        self._emit("started")

    def _emit(self, status):
        entry = {**self.details, "status": status,
                 "recorded_at": datetime.now(timezone.utc).isoformat()}
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        self.logger.info("model_usage %s", line)
        if not self.log_path:
            return
        try:
            descriptor = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(descriptor, (line + "\n").encode("utf-8"))
            finally:
                os.close(descriptor)
        except OSError as exc:
            # Logging failure must not discard an otherwise usable AI response.
            self.logger.error("model audit write failed: %s", type(exc).__name__)

    def upstream_opened(self, upstream):
        headers = getattr(upstream, "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id") or headers.get("request-id")
            if isinstance(request_id, str):
                self.details["provider_request_id"] = request_id[:200]

    def observe(self, payload):
        if isinstance(payload.get("usage"), dict):
            self.details["usage"] = _numeric_usage(payload["usage"])
        for source, target in (("model", "model_returned"), ("id", "provider_response_id")):
            if isinstance(payload.get(source), str):
                self.details[target] = payload[source][:200]
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                self.details["finish_reason"] = choice["finish_reason"][:80]

    def finish(self, status, *, http_status=None, error_type=None):
        if self.finished:
            return
        self.finished = True
        self.details["elapsed_ms"] = round((time.monotonic() - self.started) * 1000)
        self.details["usage_status"] = "reported" if self.details["usage"] is not None else "unknown"
        if http_status is not None:
            self.details["http_status"] = http_status
        if error_type:
            self.details["error_type"] = error_type
        self._emit(status)
