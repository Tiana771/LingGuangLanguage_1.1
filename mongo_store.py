from __future__ import annotations

import os
import re
from urllib.parse import unquote, urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
ENV_FILES = ("atlasconnection.env", ".env", "altasconnection.env", "connection.env")
LEGACY_KNOWLEDGE_COLLECTION = "SignKnowledgeChunks"
STAGING_KNOWLEDGE_COLLECTION = "SignKnowledgeChunks_v2"


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
    def knowledge_chunks(self):
        return self._database()[knowledge_collection_name()]

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
