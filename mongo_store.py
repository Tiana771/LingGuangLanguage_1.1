from __future__ import annotations

import os
import re
from urllib.parse import unquote, urlparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
ENV_FILES = ("atlasconnection.env", ".env", "altasconnection.env", "connection.env")
LEGACY_KNOWLEDGE_COLLECTION = "SignKnowledgeChunks"
STAGING_KNOWLEDGE_COLLECTION = "SignKnowledgeChunks_v2"
CHINA_TIMEZONE = timezone(timedelta(hours=8))


def knowledge_collection_name() -> str:
    configured = os.environ.get("BOOK_KNOWLEDGE_COLLECTION", LEGACY_KNOWLEDGE_COLLECTION).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", configured):
        raise ValueError("BOOK_KNOWLEDGE_COLLECTION contains invalid characters")
    return configured


def knowledge_index_name() -> str:
    default = "sign_knowledge_vector_v2" if knowledge_collection_name() == STAGING_KNOWLEDGE_COLLECTION else "sign_knowledge_vector"
    return (os.environ.get("BOOK_EMBEDDING_INDEX_NAME") or default).strip()


def _read_env(root: Path = ROOT) -> dict[str, str]:
    values: dict[str, str] = {}
    for filename in ENV_FILES:
        path = root / filename
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            if candidate.startswith(("mongodb://", "mongodb+srv://")):
                values.setdefault("MONGO_URI", candidate)
            elif "=" in candidate:
                name, value = candidate.split("=", 1)
                values.setdefault(name.strip(), value.strip().strip('"\''))
    return values


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def learning_local_day(value: Any) -> str:
    """Return a stored UTC timestamp as a China-local calendar date without OS tzdata."""
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    elif isinstance(value, str) and value.strip():
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        return ""
    return timestamp.astimezone(CHINA_TIMEZONE).date().isoformat()


def learning_streak(reviewed_days: set[str], today: str) -> int:
    from datetime import date, timedelta

    current = date.fromisoformat(today)
    if current.isoformat() not in reviewed_days:
        current -= timedelta(days=1)
    streak = 0
    while current.isoformat() in reviewed_days:
        streak += 1
        current -= timedelta(days=1)
    return streak


def load_mongodb_uri(root: Path = ROOT) -> str:
    configured = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    if configured:
        return configured.strip().strip('"\'')

    values = _read_env(root)
    candidate = values.get("MONGODB_URI") or values.get("MONGO_URI", "")
    return candidate.strip()


def load_mongodb_database(root: Path = ROOT, uri: str | None = None) -> str:
    configured = os.environ.get("MONGODB_DATABASE", "").strip()
    if configured:
        return configured
    selected_uri = uri if uri is not None else load_mongodb_uri(root)
    if selected_uri:
        name = unquote(urlparse(selected_uri).path.lstrip("/").split("/", 1)[0]).strip()
        if name:
            return name
    return _read_env(root).get("MONGODB_DATABASE", "").strip()


def mongodb_tls_options(root: Path = ROOT) -> dict[str, bool]:
    configured = os.environ.get("MONGODB_TLS_ALLOW_INVALID_CERTIFICATES")
    if configured is None:
        configured = _read_env(root).get("MONGODB_TLS_ALLOW_INVALID_CERTIFICATES", "")
    if configured.strip().lower() in {"1", "true", "yes", "on"}:
        return {"tlsAllowInvalidCertificates": True}
    return {}


def mongodb_tls_ca_file(root: Path = ROOT) -> str:
    return (os.environ.get("MONGODB_TLS_CA_FILE") or _read_env(root).get("MONGODB_TLS_CA_FILE", "")).strip()


def create_mongodb_client(uri: str | None = None, *, root: Path = ROOT):
    """Use the same certificate bundle and bounded timeouts in app and importer."""
    import certifi
    from pymongo import MongoClient

    selected_uri = uri if uri is not None else load_mongodb_uri(root)
    if not selected_uri:
        raise RuntimeError("MongoDB Atlas 尚未配置连接串")
    return MongoClient(
        selected_uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=10000,
        socketTimeoutMS=30000,
        timeoutMS=45000,
        compressors="zlib",
        tlsCAFile=mongodb_tls_ca_file(root) or certifi.where(),
        **mongodb_tls_options(root),
    )


class MongoStore:
    def __init__(self, uri: str | None = None, client: Any | None = None):
        self.uri = uri if uri is not None else load_mongodb_uri()
        self._client = client
        self._db = None

    def _database(self):
        if self._db is not None:
            return self._db
        if self._client is None:
            if not self.uri:
                raise RuntimeError("MongoDB Atlas 尚未配置连接串")
            self._client = create_mongodb_client(self.uri)
        database_name = load_mongodb_database(uri=self.uri)
        if not database_name:
            raise RuntimeError("Atlas 连接串未指定数据库名，请在 atlasconnection.env 或 .env 中设置 MONGODB_DATABASE")
        self._db = self._client[database_name]
        return self._db

    def initialize(self) -> None:
        database = self._database()
        database.command("ping")
        database["users"].create_index("username_key", unique=True, name="users_username_unique")
        database["LingGuangChat"].create_index(
            [("owner_user_id", 1), ("updatedAt", -1)], name="conversations_owner_updated"
        )
        database["LingGuangAssistantChats"].create_index(
            [("owner_user_id", 1), ("updatedAt", -1)], name="assistant_chats_owner_updated"
        )
        database["sessions"].create_index("expires_at", expireAfterSeconds=0, name="sessions_expiry")
        database["user_learning_progress"].create_index(
            [("owner_user_id", 1), ("entry_id", 1)], unique=True, name="learning_owner_entry_unique"
        )
        database["user_learning_progress"].create_index(
            [("owner_user_id", 1), ("learning_active", 1), ("next_review_at", 1)], name="learning_due_by_owner"
        )
        database["user_study_lists"].create_index(
            [("owner_user_id", 1), ("updated_at", -1)], name="study_lists_owner_updated"
        )
        database["learning_review_events"].create_index(
            [("owner_user_id", 1), ("reviewed_at", -1)], name="learning_events_owner_reviewed"
        )
        database["learning_review_events"].create_index(
            [("owner_user_id", 1), ("event_id", 1)], unique=True, name="learning_events_owner_event_unique"
        )
        self._ensure_knowledge_lookup_index()

    def _ensure_knowledge_lookup_index(self) -> str:
        """Reuse an equivalent index, including names created by older launchers."""
        from pymongo.errors import OperationFailure

        collection = self.knowledge_chunks
        keys = [("book", 1), ("review_status", 1)]
        name = "knowledge_book_review_v2" if knowledge_collection_name() == STAGING_KNOWLEDGE_COLLECTION else "knowledge_book_review"

        def existing_equivalent():
            try:
                indexes = collection.list_indexes()
                for index in indexes:
                    if (list(index.get("key", {}).items()) == keys
                            and not any(index.get(option) for option in (
                                "unique", "sparse", "partialFilterExpression", "hidden", "collation",
                            ))):
                        return index["name"]
            except OperationFailure as exc:
                if exc.code != 26:  # A new collection may not exist yet.
                    raise
            return None

        existing = existing_equivalent()
        if existing:
            return existing
        try:
            return collection.create_index(keys, name=name)
        except OperationFailure as exc:
            # Another app/importer may have created the same index after our read.
            if exc.code in {85, 86}:
                existing = existing_equivalent()
                if existing:
                    return existing
            raise

    def initialize_staging_knowledge(self) -> None:
        if knowledge_collection_name() != STAGING_KNOWLEDGE_COLLECTION:
            raise RuntimeError("Staging importer may only initialize SignKnowledgeChunks_v2")
        database = self._database()
        database.command("ping")
        self._ensure_knowledge_lookup_index()

    @property
    def users(self):
        return self._database()["users"]

    @property
    def conversations(self):
        return self._database()["LingGuangChat"]

    @property
    def assistant_chats(self):
        return self._database()["LingGuangAssistantChats"]

    @property
    def learning_progress(self):
        return self._database()["user_learning_progress"]

    @property
    def study_lists(self):
        return self._database()["user_study_lists"]

    @property
    def learning_events(self):
        return self._database()["learning_review_events"]

    @property
    def knowledge_chunks(self):
        return self._database()[knowledge_collection_name()]

    @property
    def learning_knowledge(self):
        # Learning always reads the reviewed v2 dictionary, never the legacy dataset.
        return self._database()[STAGING_KNOWLEDGE_COLLECTION]

    @staticmethod
    def _public_learning_entry(document: dict[str, Any] | None) -> dict[str, Any] | None:
        if not document:
            return None
        return {
            "id": str(document.get("_id", "")),
            "title": str(document.get("entry_title") or ""),
            "text": str(document.get("text") or ""),
            "book": document.get("book"),
            "page": document.get("page"),
            "anchor": document.get("anchor"),
            "sourceFile": str(document.get("source_file") or ""),
            "images": [path for path in document.get("image_paths", []) if isinstance(path, str)],
        }

    def find_learning_entry(self, entry_id: str) -> dict[str, Any] | None:
        document = self.learning_knowledge.find_one(
            {"_id": entry_id, "review_status": "approved"}, {"embedding": 0}
        )
        return self._public_learning_entry(document)

    def search_learning_entries(self, query: str = "", book: int | None = None, *, limit: int = 30, skip: int = 0):
        import re

        criteria: dict[str, Any] = {"review_status": "approved", "entry_title": {"$exists": True, "$ne": ""},
                                    "text": {"$exists": True, "$ne": ""}}
        if book is not None:
            criteria["book"] = book
        if query:
            pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
            criteria["$or"] = [{"entry_title": pattern}, {"text": pattern}]
        cursor = (self.learning_knowledge.find(criteria, {"embedding": 0})
                  .sort([("book", 1), ("page", 1), ("entry_title", 1)])
                  .skip(max(0, skip)).limit(min(max(1, limit), 100)))
        return [self._public_learning_entry(item) for item in cursor]

    def count_learning_entries(self, query: str = "", book: int | None = None) -> int:
        import re

        criteria: dict[str, Any] = {"review_status": "approved", "entry_title": {"$exists": True, "$ne": ""},
                                    "text": {"$exists": True, "$ne": ""}}
        if book is not None:
            criteria["book"] = book
        if query:
            pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
            criteria["$or"] = [{"entry_title": pattern}, {"text": pattern}]
        return self.learning_knowledge.count_documents(criteria)

    def learning_entry_flags(self, owner_id: str, entry_ids: list[str]) -> dict[str, dict[str, bool]]:
        if not entry_ids:
            return {}
        return {item["entry_id"]: {"favorite": bool(item.get("favorite")),
                                    "learningActive": bool(item.get("learning_active"))}
                for item in self.learning_progress.find(
                    {"owner_user_id": owner_id, "entry_id": {"$in": entry_ids}},
                    {"entry_id": 1, "favorite": 1, "learning_active": 1},
                )}

    def update_learning_flag(self, owner_id: str, entry_id: str, field: str, value: bool) -> bool:
        if field not in {"favorite", "learning_active"} or not self.find_learning_entry(entry_id):
            return False
        now = utc_now()
        progress = self.learning_progress.find_one({"owner_user_id": owner_id, "entry_id": entry_id}) or {}
        values = {field: bool(value), "updated_at": now}
        if field == "favorite" and value:
            values["favorite_at"] = now
        if field == "learning_active" and value and not progress.get("first_studied_at"):
            values["next_review_at"] = now
        self.learning_progress.update_one(
            {"owner_user_id": owner_id, "entry_id": entry_id},
            {"$set": values,
             "$setOnInsert": {"owner_user_id": owner_id, "entry_id": entry_id, "interval_days": 0,
                               "repetitions": 0, "attempt_count": 0}},
            upsert=True,
        )
        return True

    def get_learning_settings(self, owner_id: str) -> dict[str, int]:
        user = self.users.find_one({"_id": owner_id}, {"learning_settings": 1})
        stored = user.get("learning_settings", {}) if user else {}
        target = stored.get("daily_new_target", 5) if isinstance(stored, dict) else 5
        return {"dailyNewTarget": target if target in {3, 5, 10} else 5}

    def save_learning_settings(self, owner_id: str, target: int) -> dict[str, int] | None:
        result = self.users.update_one({"_id": owner_id}, {"$set": {"learning_settings.daily_new_target": target}})
        return self.get_learning_settings(owner_id) if result.matched_count else None

    def find_user_study_list(self, owner_id: str, list_id: str):
        return self.study_lists.find_one({"_id": list_id, "owner_user_id": owner_id})

    def get_learning_today(self, owner_id: str, new_target: int) -> dict[str, Any]:
        from datetime import datetime, time, timedelta, timezone

        now = datetime.now(timezone.utc)
        local_zone = CHINA_TIMEZONE
        today = now.astimezone(local_zone).date()
        today_key = today.isoformat()
        now_iso = now.isoformat()
        day_start = datetime.combine(today, time.min, local_zone).astimezone(timezone.utc).isoformat()
        events_today = list(self.learning_events.find(
            {"owner_user_id": owner_id, "$or": [
                {"session_date": today_key},
                {"session_date": {"$exists": False}, "reviewed_at": {"$gte": day_start}},
            ]},
            {"rating": 1, "is_new": 1},
        ))
        new_done = sum(1 for event in events_today if event.get("is_new"))
        today_repeats = list(self.learning_progress.find({
            "owner_user_id": owner_id, "learning_active": True,
            "day_session_date": today_key, "day_session_complete": False,
        }).sort("updated_at", 1))
        repeat_ids = {item["entry_id"] for item in today_repeats}
        due_progress = list(self.learning_progress.find({
            "owner_user_id": owner_id, "learning_active": True,
            "next_review_at": {"$lte": now_iso},
            "entry_id": {"$nin": list(repeat_ids)},
        }).sort("next_review_at", 1).limit(100))
        due_new = [item for item in due_progress if not item.get("first_studied_at")]
        remaining_new = max(0, min(10, new_target) - new_done - len(due_new))
        studied_ids = [item["entry_id"] for item in self.learning_progress.find(
            {"owner_user_id": owner_id, "first_studied_at": {"$exists": True}}, {"entry_id": 1}
        )]
        excluded_ids = list({*studied_ids, *(item["entry_id"] for item in due_progress)})
        fresh = []
        if remaining_new:
            fresh_cursor = (self.learning_knowledge.find({
                "review_status": "approved", "entry_title": {"$exists": True, "$ne": ""},
                "text": {"$exists": True, "$ne": ""}, "_id": {"$nin": excluded_ids},
            }, {"embedding": 0}).sort([("book", 1), ("page", 1), ("entry_title", 1)])
                .limit(remaining_new))
            fresh = [self._public_learning_entry(item) for item in fresh_cursor]

        due_entries = self._entries_by_ids([item["entry_id"] for item in due_progress])
        due = []
        for progress in due_progress:
            entry = due_entries.get(progress["entry_id"])
            if entry:
                due.append({**entry, "isNew": not bool(progress.get("first_studied_at")),
                            "nextReviewAt": progress.get("next_review_at")})

        reviewed_days = sorted({event.get("session_date") or learning_local_day(event.get("reviewed_at"))
                                for event in self.learning_events.find(
            {"owner_user_id": owner_id}, {"reviewed_at": 1}
        ) if event.get("reviewed_at") and (event.get("session_date") or learning_local_day(event.get("reviewed_at")))})
        reviewed_set = set(reviewed_days)
        streak = learning_streak(reviewed_set, today_key)

        completed_today = self.learning_progress.count_documents({
            "owner_user_id": owner_id, "day_session_date": today_key, "day_session_complete": True,
        })
        repeat_entries = self._entries_by_ids([item["entry_id"] for item in today_repeats])
        repeat_cards = [{**repeat_entries[item["entry_id"]], "isNew": bool(item.get("day_session_is_new")),
                         "todayRepeat": True, "masterySuccesses": item.get("day_session_successes", 0),
                         "masteryRequired": item.get("day_session_required", 0)}
                        for item in today_repeats if item["entry_id"] in repeat_entries]

        return {
            "due": due,
            "new": fresh,
            "todayRepeat": repeat_cards,
            "counts": {"due": len(due) + len(repeat_cards), "new": len(fresh), "completed": completed_today,
                       "newCompleted": new_done, "today": today_key, "streak": streak},
        }

    def _entries_by_ids(self, entry_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not entry_ids:
            return {}
        documents = self.learning_knowledge.find(
            {"_id": {"$in": entry_ids}, "review_status": "approved"}, {"embedding": 0}
        )
        return {str(item["_id"]): self._public_learning_entry(item) for item in documents}

    def _learning_titles_by_ids(self, entry_ids: list[str]) -> dict[str, str]:
        if not entry_ids:
            return {}
        documents = self.learning_knowledge.find(
            {"_id": {"$in": entry_ids}, "review_status": "approved"}, {"entry_title": 1}
        )
        return {str(item["_id"]): str(item.get("entry_title") or "") for item in documents}

    def record_learning_review(self, owner_id: str, entry_id: str, rating: str, event_id: str):
        from datetime import datetime, timezone

        from pymongo.errors import DuplicateKeyError

        from learning_schedule import RATINGS, next_review, update_day_mastery

        if rating not in RATINGS:
            raise ValueError("Unsupported learning rating")
        entry = self.find_learning_entry(entry_id)
        if not entry:
            return None
        existing = self.learning_events.find_one({"owner_user_id": owner_id, "event_id": event_id})
        if existing:
            progress = self.learning_progress.find_one({"owner_user_id": owner_id, "entry_id": entry_id}) or {}
            return {"entry": entry, "progress": progress, "duplicate": True,
                    "dayComplete": progress.get("day_session_complete", True),
                    "repeatToday": not progress.get("day_session_complete", True),
                    "masterySuccesses": progress.get("day_session_successes", 0),
                    "masteryRequired": progress.get("day_session_required", 0),
                    "initialRating": progress.get("day_session_initial_rating")}

        now = datetime.now(timezone.utc)
        today_key = now.astimezone(CHINA_TIMEZONE).date().isoformat()
        previous = self.learning_progress.find_one({"owner_user_id": owner_id, "entry_id": entry_id}) or {}
        is_new = not bool(previous.get("first_studied_at"))
        same_day_session = previous.get("day_session_date") == today_key and not previous.get("day_session_complete")
        mastery = update_day_mastery(
            existing_session=same_day_session,
            previous_required=previous.get("day_session_required", 0),
            previous_successes=previous.get("day_session_successes", 0),
            previous_initial_rating=previous.get("day_session_initial_rating", rating),
            previous_is_new=previous.get("day_session_is_new", is_new),
            is_new=is_new,
            rating=rating,
        )
        required = mastery["required"]
        successes = mastery["successes"]
        initial_rating = mastery["initial_rating"]
        session_is_new = mastery["session_is_new"]
        day_complete = mastery["complete"]
        if day_complete:
            schedule = next_review(previous.get("interval_days", 0), "remembered", now,
                                   is_new=session_is_new)
        else:
            schedule = {"interval_days": previous.get("interval_days", 0),
                        "next_review_at": now.isoformat(), "repetitions_reset": False}
        event = {"owner_user_id": owner_id, "event_id": event_id, "entry_id": entry_id,
                 "entry_title": entry["title"], "rating": rating, "is_new": is_new,
                 "reviewed_at": now.isoformat(), "session_date": today_key,
                 "day_complete": day_complete, "mastery_successes": successes,
                 "mastery_required": required}
        try:
            self.learning_events.insert_one(event)
        except DuplicateKeyError:
            updated = self.learning_progress.find_one({"owner_user_id": owner_id, "entry_id": entry_id}) or {}
            return {"entry": entry, "progress": updated, "duplicate": True,
                    "dayComplete": updated.get("day_session_complete", True),
                    "repeatToday": not updated.get("day_session_complete", True),
                    "masterySuccesses": updated.get("day_session_successes", 0),
                    "masteryRequired": updated.get("day_session_required", 0),
                    "initialRating": updated.get("day_session_initial_rating")}
        progress_values = {"learning_active": True, "entry_title": entry["title"],
                           "interval_days": schedule["interval_days"],
                           "next_review_at": schedule["next_review_at"], "last_rating": rating,
                           "last_studied_at": now.isoformat(), "updated_at": now.isoformat(),
                           "day_session_date": today_key, "day_session_required": required,
                           "day_session_successes": successes, "day_session_initial_rating": initial_rating,
                           "day_session_is_new": session_is_new, "day_session_complete": day_complete}
        if is_new:
            progress_values["first_studied_at"] = now.isoformat()
        self.learning_progress.update_one(
            {"owner_user_id": owner_id, "entry_id": entry_id},
            {"$set": progress_values,
             "$setOnInsert": {"owner_user_id": owner_id, "entry_id": entry_id,
                               "favorite": False, "repetitions": 0},
             "$inc": {"attempt_count": 1}},
            upsert=True,
        )
        if not schedule["repetitions_reset"] and rating == "remembered":
            self.learning_progress.update_one(
                {"owner_user_id": owner_id, "entry_id": entry_id}, {"$inc": {"repetitions": 1}}
            )
        elif schedule["repetitions_reset"]:
            self.learning_progress.update_one(
                {"owner_user_id": owner_id, "entry_id": entry_id}, {"$set": {"repetitions": 0}}
            )
        progress = self.learning_progress.find_one({"owner_user_id": owner_id, "entry_id": entry_id}) or {}
        self.learning_events.update_one({"owner_user_id": owner_id, "event_id": event_id},
                                        {"$set": {"processed": True}})
        return {"entry": entry, "progress": progress, "duplicate": False,
                "dayComplete": day_complete, "repeatToday": mastery["repeat_today"],
                "masterySuccesses": successes, "masteryRequired": required,
                "initialRating": initial_rating}

    def learning_stats(self, owner_id: str, days: int = 7) -> dict[str, Any]:
        from collections import Counter
        from datetime import datetime, timedelta, timezone

        days = 30 if days == 30 else 7
        now = datetime.now(timezone.utc)
        local_zone = CHINA_TIMEZONE
        local_today = now.astimezone(local_zone).date()
        since_date = local_today - timedelta(days=days - 1)
        since = datetime.combine(since_date, datetime.min.time(), local_zone).astimezone(timezone.utc).isoformat()
        events = list(self.learning_events.find(
            {"owner_user_id": owner_id, "reviewed_at": {"$gte": since}, "processed": {"$ne": False}},
            {"entry_id": 1, "reviewed_at": 1, "session_date": 1, "rating": 1,
             "is_new": 1, "day_complete": 1},
        ))
        counts = Counter(event.get("rating", "") for event in events)
        today_key = local_today.isoformat()
        today_events = [event for event in events if (event.get("session_date") or learning_local_day(event.get("reviewed_at"))) == today_key]
        today_first_event_by_entry: dict[str, dict[str, Any]] = {}
        for event in sorted(today_events, key=lambda item: item.get("reviewed_at", "")):
            if event.get("entry_id"):
                today_first_event_by_entry.setdefault(event["entry_id"], event)
        today_new_ids = {entry_id for entry_id, event in today_first_event_by_entry.items() if event.get("is_new")}
        today_review_ids = set(today_first_event_by_entry) - today_new_ids
        per_day: dict[str, Counter] = {}
        for event in events:
            day = event.get("session_date") or learning_local_day(event.get("reviewed_at"))
            per_day.setdefault(day, Counter())[event.get("rating", "reviewed")] += 1
        chart = []
        for offset in range(days - 1, -1, -1):
            day = (local_today - timedelta(days=offset)).isoformat()
            daily = per_day.get(day, Counter())
            chart.append({"date": day, "total": sum(daily.values()), "remembered": daily["remembered"],
                          "fuzzy": daily["fuzzy"], "forgot": daily["forgot"]})
        active_progress = list(self.learning_progress.find({"owner_user_id": owner_id}, {
            "first_studied_at": 1, "learning_active": 1, "next_review_at": 1, "last_rating": 1,
            "entry_id": 1, "entry_title": 1, "day_session_date": 1,
            "day_session_required": 1, "day_session_successes": 1,
        }))
        learned = [item for item in active_progress if item.get("first_studied_at")]
        now_iso = now.isoformat()
        return {
            "days": days,
            "totals": {"learnedWords": len(learned), "todayWords": len(today_new_ids | today_review_ids),
                       "todayNewWords": len(today_new_ids), "todayReviewWords": len(today_review_ids),
                       "practiceCount": len(events),
                       "remembered": counts["remembered"], "fuzzy": counts["fuzzy"], "forgot": counts["forgot"],
                       "due": sum(1 for item in active_progress if item.get("learning_active") and item.get("next_review_at", "") <= now_iso)},
            "chart": chart,
            "hardWords": self._learning_hard_words(owner_id, local_today.isoformat()),
        }

    def _learning_hard_words(self, owner_id: str, today_key: str) -> list[dict[str, Any]]:
        from collections import defaultdict
        from datetime import datetime, timedelta

        zone = CHINA_TIMEZONE
        events = list(self.learning_events.find(
            {"owner_user_id": owner_id, "rating": {"$in": ["forgot", "fuzzy"]},
             "processed": {"$ne": False}},
            {"entry_id": 1, "entry_title": 1, "rating": 1, "session_date": 1, "reviewed_at": 1},
        ).sort("reviewed_at", -1).limit(2000))
        days_by_entry: dict[str, set[str]] = defaultdict(set)
        titles: dict[str, str] = {}
        forgotten_today: dict[str, dict[str, Any]] = {}
        for event in events:
            entry_id = str(event.get("entry_id", ""))
            if not entry_id:
                continue
            reviewed_at = event.get("reviewed_at", "")
            day = event.get("session_date") or learning_local_day(reviewed_at)
            title = event.get("entry_title", "")
            titles[entry_id] = title
            days_by_entry[entry_id].add(day)
            if day == today_key and event.get("rating") == "forgot":
                forgotten_today[entry_id] = {"id": entry_id, "title": title, "reason": "今天忘记了", "date": day}

        result = list(forgotten_today.values())
        seen = set(forgotten_today)
        repeated = []
        for entry_id, days in days_by_entry.items():
            if entry_id in seen or len(days) < 2:
                continue
            ordered = sorted(days, reverse=True)
            latest_day = datetime.fromisoformat(ordered[0]).date()
            if latest_day not in {datetime.fromisoformat(today_key).date(),
                                  datetime.fromisoformat(today_key).date() - timedelta(days=1)}:
                continue
            streak = 1
            for previous_day, next_day in zip(ordered, ordered[1:]):
                if datetime.fromisoformat(previous_day).date().toordinal() - datetime.fromisoformat(next_day).date().toordinal() != 1:
                    break
                streak += 1
            if streak >= 2:
                repeated.append(entry_id)
        repeated.sort(key=lambda entry_id: max(days_by_entry[entry_id]), reverse=True)
        result.extend({"id": entry_id, "title": titles[entry_id], "reason": "多天反馈模糊或忘记",
                       "date": max(days_by_entry[entry_id])} for entry_id in repeated[:8])
        return result[:16]

    def learning_stat_words(self, owner_id: str, category: str, days: int = 7) -> list[dict[str, Any]]:
        from collections import defaultdict
        from datetime import datetime, timedelta, timezone

        zone = CHINA_TIMEZONE
        today = datetime.now(timezone.utc).astimezone(zone).date()
        since_day = today - timedelta(days=max(1, min(days, 30)) - 1)
        since = datetime.combine(since_day, datetime.min.time(), zone).astimezone(timezone.utc).isoformat()
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        if category == "today":
            today_key = today.isoformat()
            day_start = datetime.combine(today, datetime.min.time(), zone).astimezone(timezone.utc).isoformat()
            events = self.learning_events.find(
                {"owner_user_id": owner_id, "processed": {"$ne": False}, "$or": [
                    {"session_date": today_key},
                    {"session_date": {"$exists": False}, "reviewed_at": {"$gte": day_start}},
                ]},
                {"entry_id": 1, "entry_title": 1, "reviewed_at": 1, "is_new": 1},
            ).sort("reviewed_at", 1)
            first_event_by_entry: dict[str, dict[str, Any]] = {}
            for event in events:
                entry_id = str(event.get("entry_id", ""))
                if entry_id:
                    first_event_by_entry.setdefault(entry_id, event)
            for entry_id, event in first_event_by_entry.items():
                key = "new" if event.get("is_new") else "review"
                grouped[key][entry_id] = {"id": entry_id, "title": event.get("entry_title", "")}
            return [{"date": today_key, "type": key,
                     "entries": sorted(entries.values(), key=lambda item: item["title"])}
                    for key, entries in (("new", grouped["new"]), ("review", grouped["review"]))
                    if entries]
        if category in {"all", "remembered", "fuzzy", "forgot"}:
            event_filter: dict[str, Any] = {"owner_user_id": owner_id, "reviewed_at": {"$gte": since},
                                            "processed": {"$ne": False}}
            if category != "all":
                event_filter["rating"] = category
            for event in self.learning_events.find(
                event_filter,
                {"entry_id": 1, "entry_title": 1, "reviewed_at": 1, "session_date": 1},
            ).sort("reviewed_at", -1):
                day = event.get("session_date") or learning_local_day(event.get("reviewed_at"))
                entry_id = str(event.get("entry_id", ""))
                if entry_id:
                    grouped[day][entry_id] = {"id": entry_id, "title": event.get("entry_title", "")}
        elif category == "learned":
            progress_filter = {"owner_user_id": owner_id, "first_studied_at": {"$exists": True}}
            learned_progress = list(self.learning_progress.find(
                progress_filter,
                {"entry_id": 1, "entry_title": 1, "first_studied_at": 1},
            ).sort("first_studied_at", -1))
            missing_titles = [item["entry_id"] for item in learned_progress if not item.get("entry_title")]
            title_fallbacks = self._entries_by_ids(missing_titles)
            for progress in learned_progress:
                timestamp = progress.get("first_studied_at", "")
                try:
                    day = learning_local_day(timestamp)
                except ValueError:
                    continue
                if progress.get("entry_id"):
                    title = progress.get("entry_title") or title_fallbacks.get(progress["entry_id"], {}).get("title", "")
                    grouped[day][progress["entry_id"]] = {"id": progress["entry_id"], "title": title}
        elif category == "due":
            now_iso = datetime.now(timezone.utc).isoformat()
            progress_items = list(self.learning_progress.find(
                {"owner_user_id": owner_id, "learning_active": True, "next_review_at": {"$lte": now_iso}},
                {"entry_id": 1, "next_review_at": 1},
            ).sort("next_review_at", 1).limit(500))
            entries = self._entries_by_ids([item["entry_id"] for item in progress_items])
            for item in progress_items:
                entry = entries.get(item["entry_id"])
                if not entry:
                    continue
                try:
                    due_date = datetime.fromisoformat(item.get("next_review_at", "").replace("Z", "+00:00")).astimezone(zone).date()
                    day = "逾期" if due_date < today else due_date.isoformat()
                except ValueError:
                    day = "逾期"
                grouped[day][item["entry_id"]] = {"id": item["entry_id"], "title": entry["title"]}
        return [{"date": day, "entries": sorted(items.values(), key=lambda value: value["title"])}
                for day, items in sorted(grouped.items(), reverse=True)]

    def list_learning_lists(self, owner_id: str):
        favorites = list(self.learning_progress.find(
            {"owner_user_id": owner_id, "favorite": True}, {"entry_id": 1, "favorite_at": 1}
        ).sort("favorite_at", -1))
        favorite_entries = self._learning_titles_by_ids([item["entry_id"] for item in favorites])
        favorite_entries = [{"id": item["entry_id"], "title": favorite_entries[item["entry_id"]],
                             "favoritedAt": item.get("favorite_at", "")}
                            for item in favorites if item["entry_id"] in favorite_entries]
        custom = []
        for item in self.study_lists.find({"owner_user_id": owner_id}).sort("updated_at", -1):
            entries = self._learning_titles_by_ids(item.get("entry_ids", []))
            custom.append({"id": item["_id"], "title": item.get("title", "未命名词单"),
                           "entries": [{"id": key, "title": entries[key]}
                                       for key in item.get("entry_ids", []) if key in entries],
                           "updatedAt": item.get("updated_at", "")})
        return {"favorites": favorite_entries, "lists": custom}

    def create_learning_list(self, owner_id: str, title: str):
        from uuid import uuid4

        document = {"_id": str(uuid4()), "owner_user_id": owner_id, "title": title,
                    "entry_ids": [], "created_at": utc_now(), "updated_at": utc_now()}
        self.study_lists.insert_one(document)
        return {"id": document["_id"], "title": title, "entries": [], "updatedAt": document["updated_at"]}

    def rename_learning_list(self, owner_id: str, list_id: str, title: str) -> bool:
        result = self.study_lists.update_one({"_id": list_id, "owner_user_id": owner_id},
                                             {"$set": {"title": title, "updated_at": utc_now()}})
        return result.matched_count == 1

    def delete_learning_list(self, owner_id: str, list_id: str) -> bool:
        return self.study_lists.delete_one({"_id": list_id, "owner_user_id": owner_id}).deleted_count == 1

    def update_learning_list_entry(self, owner_id: str, list_id: str, entry_id: str, add: bool) -> bool:
        if not self.find_learning_entry(entry_id):
            return False
        operation = {"$addToSet": {"entry_ids": entry_id}} if add else {"$pull": {"entry_ids": entry_id}}
        operation["$set"] = {"updated_at": utc_now()}
        result = self.study_lists.update_one({"_id": list_id, "owner_user_id": owner_id}, operation)
        return result.matched_count == 1

    def batch_save_learning_entries(self, owner_id: str, entry_ids: list[str], *, action: str,
                                    list_id: str = "", list_title: str = "") -> dict[str, Any]:
        from datetime import datetime, timezone
        from uuid import uuid4

        unique_ids = list(dict.fromkeys(entry_ids))[:50]
        valid_ids = [str(item["_id"]) for item in self.learning_knowledge.find(
            {"_id": {"$in": unique_ids}, "review_status": "approved"}, {"_id": 1}
        )]
        valid_set = set(valid_ids)
        skipped = len(unique_ids) - len(valid_set)
        now = utc_now()
        if action == "favorite":
            added = 0
            for entry_id in valid_ids:
                current = self.learning_progress.find_one(
                    {"owner_user_id": owner_id, "entry_id": entry_id}, {"favorite": 1}
                ) or {}
                if current.get("favorite"):
                    continue
                self.learning_progress.update_one(
                    {"owner_user_id": owner_id, "entry_id": entry_id},
                    {"$set": {"favorite": True, "favorite_at": now, "updated_at": now},
                     "$setOnInsert": {"owner_user_id": owner_id, "entry_id": entry_id,
                                       "interval_days": 0, "repetitions": 0, "attempt_count": 0}},
                    upsert=True,
                )
                added += 1
            return {"action": action, "added": added, "alreadyPresent": len(valid_ids) - added,
                    "skipped": skipped}

        if action == "plan":
            already_due = {
                item["entry_id"] for item in self.learning_progress.find(
                    {"owner_user_id": owner_id, "entry_id": {"$in": valid_ids}, "learning_active": True},
                    {"entry_id": 1, "next_review_at": 1},
                ) if item.get("next_review_at", "") <= now
            }
            for entry_id in valid_ids:
                self.learning_progress.update_one(
                    {"owner_user_id": owner_id, "entry_id": entry_id},
                    {"$set": {"learning_active": True, "next_review_at": now, "updated_at": now},
                     "$setOnInsert": {"owner_user_id": owner_id, "entry_id": entry_id,
                                       "interval_days": 0, "repetitions": 0, "attempt_count": 0}},
                    upsert=True,
                )
            return {"action": action, "added": len(valid_ids) - len(already_due),
                    "alreadyPresent": len(already_due),
                    "skipped": skipped}

        if action != "list" or (not list_id and not list_title):
            raise ValueError("A list target is required")
        target = None
        if list_id:
            target = self.study_lists.find_one({"_id": list_id, "owner_user_id": owner_id})
            if not target:
                return {"missingList": True}
        else:
            target = self.study_lists.find_one({"owner_user_id": owner_id, "title": list_title})
            if not target:
                target = {"_id": str(uuid4()), "owner_user_id": owner_id, "title": list_title,
                          "entry_ids": [], "created_at": now, "updated_at": now}
                self.study_lists.insert_one(target)
        existing = set(target.get("entry_ids", []))
        new_ids = [entry_id for entry_id in valid_ids if entry_id not in existing]
        if new_ids:
            self.study_lists.update_one(
                {"_id": target["_id"], "owner_user_id": owner_id},
                {"$addToSet": {"entry_ids": {"$each": new_ids}}, "$set": {"updated_at": now}},
            )
        return {"action": action, "listId": target["_id"], "listTitle": target["title"],
                "added": len(new_ids), "alreadyPresent": len(valid_ids) - len(new_ids), "skipped": skipped}

    def upsert_knowledge_chunks(self, documents: list[dict[str, Any]]) -> int:
        from pymongo import UpdateOne

        if not documents:
            return 0
        operations = [UpdateOne({"_id": item["_id"]}, {"$setOnInsert": item}, upsert=True) for item in documents]
        result = self.knowledge_chunks.bulk_write(operations, ordered=False)
        return result.upserted_count

    def upsert_staging_knowledge_chunks(self, documents: list[dict[str, Any]]) -> int:
        from pymongo import UpdateOne

        if knowledge_collection_name() != STAGING_KNOWLEDGE_COLLECTION:
            raise RuntimeError("Refusing staging upsert unless the v2 knowledge collection is selected")
        if not documents:
            return 0
        operations = []
        for item in documents:
            document = {
                key: value for key, value in item.items()
                if key not in {"_id", "embedding", "embedding_content_hash", "review_status"}
            }
            unchanged = {"$eq": ["$content_hash", {"$literal": item.get("content_hash", "")}]}
            document["review_status"] = {"$cond": [
                unchanged,
                {"$ifNull": ["$review_status", "pending"]},
                {"$literal": item.get("review_status", "pending")},
            ]}
            document["embedding"] = {"$cond": [unchanged, "$embedding", None]}
            document["embedding_content_hash"] = {"$cond": [unchanged, "$embedding_content_hash", None]}
            pipeline = [{"$set": {
                key: value if key in {"review_status", "embedding", "embedding_content_hash"}
                else {"$literal": value}
                for key, value in document.items()
            }}]
            operations.append(UpdateOne(
                {"_id": item["_id"]}, pipeline, upsert=True,
            ))
        result = self.knowledge_chunks.bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count

    def existing_knowledge_ids(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()
        return {
            item["_id"] for item in self.knowledge_chunks.find(
                {"_id": {"$in": chunk_ids}}, {"_id": 1},
            )
        }

    def knowledge_ids_without_embeddings(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()
        return {
            item["_id"] for item in self.knowledge_chunks.find(
                {"_id": {"$in": chunk_ids}, "$or": [
                    {"embedding": {"$exists": False}}, {"embedding": None},
                    {"$expr": {"$ne": ["$embedding_content_hash", "$content_hash"]}},
                ]},
                {"_id": 1},
            )
        }

    def save_knowledge_embeddings(self, documents: list[dict[str, Any]]) -> int:
        from pymongo import UpdateOne

        if not documents:
            return 0
        operations = [
            UpdateOne({"_id": item["_id"]}, {"$set": {
                "embedding": item["embedding"],
                "embedding_content_hash": item.get("content_hash", ""),
            }})
            for item in documents
        ]
        return self.knowledge_chunks.bulk_write(operations, ordered=False).modified_count

    def create_knowledge_vector_index(self, name: str, dimensions: int) -> str:
        return self.knowledge_chunks.create_search_index({
            "name": name,
            "type": "vectorSearch",
            "definition": {"fields": [
                {"type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine"},
                {"type": "filter", "path": "review_status"},
            ]},
        })

    def search_knowledge_keywords(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        import re
        from book_knowledge import keyword_search, keyword_terms

        exact_results = self.search_knowledge_exact(query, limit=max(limit, 20))
        if exact_results:
            return exact_results[:limit]

        clauses = []
        for term in keyword_terms(query):
            condition = {"$regex": re.escape(term), "$options": "i"}
            clauses.extend(({"entry_title": condition}, {"text": condition}))
        if not clauses:
            return []
        candidates = list(self.knowledge_chunks.find(
            {"review_status": "approved", "$or": clauses}, {"embedding": 0}
        ).limit(5000))
        return keyword_search(candidates, query, limit)

    def search_knowledge_exact(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        import re
        from book_knowledge import _canonical_entry_title, _entry_query_term, keyword_search

        entry_term = _entry_query_term(query)
        if not entry_term:
            return []
        key = _canonical_entry_title(entry_term)
        has_number = bool(re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]$", key))
        if has_number:
            exact_pattern = re.escape(key) + r"(?:$|[\s（(])"
        else:
            exact_pattern = re.escape(key) + r"(?:[①②③④⑤⑥⑦⑧⑨⑩]|[（(][一二三四五六七八九十0-9]+[）)])?(?:$|[\s（(])"
        exact_title = re.compile(r"^" + exact_pattern, re.IGNORECASE)
        candidates = list(self.knowledge_chunks.find(
            {"review_status": "approved", "entry_title": exact_title}, {"embedding": 0}
        ).limit(max(limit * 4, 20)))
        return keyword_search(candidates, query, limit=limit)

    def search_knowledge_vectors(self, vector: list[float], index_name: str, limit: int = 5) -> list[dict[str, Any]]:
        pipeline = [{"$vectorSearch": {
            "index": index_name,
            "path": "embedding",
            "queryVector": vector,
            "numCandidates": max(limit * 20, 100),
            "limit": limit,
            "filter": {"review_status": "approved"},
        }}, {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}}]
        return list(self.knowledge_chunks.aggregate(pipeline))

    def search_knowledge_hybrid(
        self, vector: list[float], index_name: str, query: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        from book_knowledge import merge_knowledge_results

        vector_results = self.search_knowledge_vectors(vector, index_name, limit=limit)
        keyword_results = self.search_knowledge_keywords(query, limit=limit)
        return merge_knowledge_results(vector_results, keyword_results, query, limit=limit)

    def has_knowledge_vector_index(self, index_name: str) -> bool:
        return any(item.get("name") == index_name for item in self.knowledge_chunks.list_search_indexes())

    def list_knowledge_pending(self, limit: int = 200) -> list[dict[str, Any]]:
        return list(self.knowledge_chunks.find({"review_status": "pending"}, {"embedding": 0}).limit(limit))

    def review_knowledge_chunk(self, chunk_id: str, approved: bool, *, expected_content_hash: str | None = None) -> bool:
        query = {"_id": chunk_id}
        if expected_content_hash is not None:
            query["content_hash"] = expected_content_hash
        result = self.knowledge_chunks.update_one(
            query,
            {"$set": {"review_status": "approved" if approved else "rejected", "reviewedAt": utc_now()}},
        )
        return result.matched_count == 1

    def find_user_by_name(self, username_key: str):
        return self.users.find_one({"username_key": username_key})

    def insert_user(self, username: str, username_key: str, password_hash: str):
        document = {
            "_id": str(uuid4()),
            "username": username,
            "username_key": username_key,
            "password_hash": password_hash,
            "createdAt": utc_now(),
        }
        self.users.insert_one(document)
        return document

    def find_user_by_id(self, user_id: str):
        return self.users.find_one({"_id": user_id})

    def update_user_profile(self, user_id: str, display_name: str, bio: str, avatar_data_url: str):
        result = self.users.find_one_and_update(
            {"_id": user_id},
            {"$set": {
                "display_name": display_name,
                "bio": bio,
                "avatar_data_url": avatar_data_url,
                "updatedAt": utc_now(),
            }},
            return_document=True,
        )
        return _public_user(result) if result else None

    def change_password(self, user_id: str, password_hash: str) -> bool:
        result = self.users.update_one({"_id": user_id}, {"$set": {"password_hash": password_hash}})
        return result.modified_count == 1

    def list_conversations(self, owner_id: str) -> list[dict[str, Any]]:
        return [_public_conversation(item) for item in self.conversations.find({"owner_user_id": owner_id})]

    def find_conversation(self, owner_id: str, record_id: str):
        item = self.conversations.find_one({"_id": record_id, "owner_user_id": owner_id})
        return _public_conversation(item) if item else None

    def insert_conversation(self, owner_id: str, record: dict[str, Any]) -> dict[str, Any]:
        document = _clean_conversation(record)
        document["_id"] = str(record.get("id") or uuid4())
        document.pop("id", None)
        document["owner_user_id"] = owner_id
        document.setdefault("createdAt", utc_now())
        document.setdefault("updatedAt", document["createdAt"])
        self.conversations.insert_one(document)
        return _public_conversation(document)

    def update_conversation(self, owner_id: str, record_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.conversations.find_one({"_id": record_id, "owner_user_id": owner_id})
        if not existing:
            return None
        document = _clean_conversation(record)
        document.pop("id", None)
        document["owner_user_id"] = owner_id
        document["createdAt"] = existing.get("createdAt", document["createdAt"])
        document["updatedAt"] = utc_now()
        result = self.conversations.replace_one({"_id": record_id, "owner_user_id": owner_id}, document)
        if result.matched_count != 1:
            return None
        return self.find_conversation(owner_id, record_id)

    def delete_conversation(self, owner_id: str, record_id: str) -> bool:
        result = self.conversations.delete_one({"_id": record_id, "owner_user_id": owner_id})
        return result.deleted_count == 1

    def delete_all_conversations(self, owner_id: str) -> int:
        return self.conversations.delete_many({"owner_user_id": owner_id}).deleted_count

    def search_conversations(self, owner_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        documents = self.conversations.find({"owner_user_id": owner_id}).sort("updatedAt", -1).limit(500)
        query_terms = _search_terms(query)
        matches = []
        for record in documents:
            summary_record = record.get("summary") if isinstance(record.get("summary"), dict) else {}
            reply_record = record.get("reply") if isinstance(record.get("reply"), dict) else {}
            for block in record.get("blocks", []):
                transcript = str(block.get("text") or "")
                summary = str(block.get("summary") or summary_record.get("text") or "")
                reply = str(block.get("reply") or reply_record.get("text") or "")
                haystack = " ".join((transcript, summary, reply, str(block.get("title") or ""), str(record.get("title") or "")))
                normalized = haystack.casefold()
                score = sum(normalized.count(term.casefold()) for term in query_terms if term)
                normalized_query = query.strip().casefold()
                if normalized_query and normalized_query in normalized:
                    score += len(query_terms) + 4
                if score <= 0:
                    continue
                source_text = transcript or summary or reply
                matches.append({
                    "conversationId": str(record.get("_id", "")),
                    "conversationTitle": str(record.get("title") or "未命名沟通"),
                    "blockId": str(block.get("id") or ""),
                    "blockTitle": str(block.get("title") or "语音框"),
                    "createdAt": str(block.get("createdAt") or record.get("createdAt") or ""),
                    "originalText": source_text[:1200],
                    "score": score,
                })
        matches.sort(key=lambda item: (item["score"], item["createdAt"]), reverse=True)
        return matches[:limit]

    def save_assistant_turn(self, owner_id: str, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        self.assistant_chats.insert_one({
            "_id": str(uuid4()),
            "owner_user_id": owner_id,
            "query": query,
            "answer": answer,
            "sources": [{key: value for key, value in source.items() if key != "score"} for source in sources],
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
        })

    def list_assistant_chats(self, owner_id: str, limit: int = 50) -> list[dict[str, Any]]:
        chats = []
        for item in self.assistant_chats.find({"owner_user_id": owner_id}).sort("updatedAt", -1).limit(limit):
            chats.append({key: value for key, value in item.items() if key not in {"_id", "owner_user_id"}} | {"id": str(item.get("_id", ""))})
        return chats

    @property
    def sessions(self):
        return self._database()["sessions"]

    def get_session(self, session_id: str):
        document = self.sessions.find_one({"_id": session_id, "expires_at": {"$gt": datetime.now(timezone.utc)}})
        return document.get("data") if document else None

    def save_session(self, session_id: str, data: dict[str, Any], expires_at: datetime) -> None:
        self.sessions.replace_one(
            {"_id": session_id},
            {"_id": session_id, "data": data, "expires_at": expires_at},
            upsert=True,
        )

    def delete_session(self, session_id: str) -> None:
        self.sessions.delete_one({"_id": session_id})


def _public_conversation(document: dict[str, Any] | None):
    if not document:
        return None
    result = dict(document)
    result["id"] = str(result.pop("_id", ""))
    result.pop("owner_user_id", None)
    return result


def _public_user(document: dict[str, Any] | None):
    if not document:
        return None
    return {
        "id": str(document.get("_id", "")),
        "username": str(document.get("username", "")),
        "displayName": str(document.get("display_name") or document.get("username", "")),
        "bio": str(document.get("bio", "")),
        "avatar": str(document.get("avatar_data_url", "")),
    }


def _clean_conversation(record: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    for block in record.get("blocks", []) if isinstance(record.get("blocks"), list) else []:
        if not isinstance(block, dict):
            continue
        blocks.append({
            "id": str(block.get("id") or uuid4()),
            "title": str(block.get("title") or "未命名主题")[:120],
            "createdAt": str(block.get("createdAt") or utc_now()),
            "text": str(block.get("text") or "")[:50000],
            "summary": str(block.get("summary") or "")[:12000],
            "reply": str(block.get("reply") or "")[:12000],
        })
    now = utc_now()
    return {
        "title": str(record.get("title") or "未命名沟通")[:160],
        "createdAt": str(record.get("createdAt") or now),
        "updatedAt": str(record.get("updatedAt") or now),
        "blocks": blocks,
        "summary": record.get("summary") if isinstance(record.get("summary"), dict) else None,
        "reply": record.get("reply") if isinstance(record.get("reply"), dict) else None,
    }


def _search_terms(query: str) -> list[str]:
    normalized = re.sub(r"\s+", "", query).casefold()
    terms = set(re.findall(r"[a-z0-9]{2,}", normalized))
    chinese = re.findall(r"[\u3400-\u9fff]+", normalized)
    for part in chinese:
        if len(part) == 1:
            terms.add(part)
        else:
            terms.update(part[index:index + 2] for index in range(len(part) - 1))
    return sorted(terms, key=len, reverse=True)
