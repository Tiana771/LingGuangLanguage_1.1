import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4
from werkzeug.security import check_password_hash

from serve_sign_api import CSRF_SESSION_KEY, SESSION_USER_KEY, _deepseek_messages, _iter_deepseek_sse, _sse_event, create_app
from mongo_store import load_mongodb_database, load_mongodb_uri
from mongo_store import MongoStore


def login_test_session(client):
    with client.session_transaction() as session:
        session[SESSION_USER_KEY] = "user-a"
        session[CSRF_SESSION_KEY] = "test-csrf"


def owner_header(owner_id="user-a"):
    return {"X-Linguang-User-ID": owner_id}


class FakeMongoStore:
    def __init__(self):
        self.records = {}
        self.users = {}
        self.sessions = {}

    def initialize(self):
        pass

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def save_session(self, session_id, data, expires_at):
        self.sessions[session_id] = dict(data)

    def delete_session(self, session_id):
        self.sessions.pop(session_id, None)

    def find_user_by_name(self, username_key):
        return next((user for user in self.users.values() if user["username_key"] == username_key), None)

    def insert_user(self, username, username_key, password_hash):
        if self.find_user_by_name(username_key):
            raise type("DuplicateKeyError", (Exception,), {})()
        user = {"_id": str(uuid4()), "username": username, "username_key": username_key,
                "password_hash": password_hash}
        self.users[user["_id"]] = user
        return user

    def find_user_by_id(self, user_id):
        return self.users.get(user_id)

    def change_password(self, user_id, password_hash):
        if user_id not in self.users:
            return False
        self.users[user_id]["password_hash"] = password_hash
        return True

    def update_user_profile(self, user_id, display_name, bio, avatar_data_url):
        user = self.users.get(user_id)
        if not user:
            return None
        user.update({"display_name": display_name, "bio": bio, "avatar_data_url": avatar_data_url})
        return {"id": user_id, "username": user["username"], "displayName": display_name,
                "bio": bio, "avatar": avatar_data_url}

    def insert_conversation(self, owner_id, payload):
        record_id = payload.get("id") or str(uuid4())
        record = {**payload, "id": record_id}
        self.records[(owner_id, record_id)] = record
        return record

    def list_conversations(self, owner_id):
        return [record for (owner, _), record in self.records.items() if owner == owner_id]

    def update_conversation(self, owner_id, record_id, payload):
        key = (owner_id, record_id)
        if key not in self.records:
            return None
        self.records[key] = {**payload, "id": record_id}
        return self.records[key]

    def delete_conversation(self, owner_id, record_id):
        return self.records.pop((owner_id, record_id), None) is not None

    def search_conversations(self, owner_id, query, limit=5):
        matches = []
        for (owner, record_id), record in self.records.items():
            if owner != owner_id:
                continue
            for block in record.get("blocks", []):
                text = block.get("text", "")
                if query in text:
                    matches.append({"conversationId": record_id, "conversationTitle": record.get("title", ""),
                                    "blockId": block.get("id", ""), "blockTitle": block.get("title", ""),
                                    "createdAt": block.get("createdAt", ""), "originalText": text, "score": 1})
        return matches[:limit]

    def save_assistant_turn(self, owner_id, query, answer, sources):
        self.sessions.setdefault(f"assistant:{owner_id}", []).append({"query": query, "answer": answer, "sources": sources})

    def list_assistant_chats(self, owner_id, limit=50):
        return self.sessions.get(f"assistant:{owner_id}", [])[-limit:]


class DeepSeekAssistantTests(unittest.TestCase):
    def test_connection_loader_keeps_raw_srv_uri_with_query_parameters(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MONGODB_URI": "", "MONGO_URI": ""}):
            root = Path(directory)
            expected = "mongodb+srv://example.mongodb.net/?retryWrites=true&w=majority"
            (root / "altasconnection.env").write_text(expected + "\n", encoding="utf-8")
            self.assertEqual(load_mongodb_uri(root), expected)

    def test_connection_loader_reads_connection_env(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MONGODB_URI": "", "MONGO_URI": ""}):
            root = Path(directory)
            expected = "mongodb+srv://example.mongodb.net/LingGuang?compressors=zlib"
            (root / "connection.env").write_text(expected + "\n", encoding="utf-8")
            self.assertEqual(load_mongodb_uri(root), expected)
            self.assertEqual(load_mongodb_database(root, expected), "LingGuang")

    def test_connection_loader_prefers_named_atlas_file_without_exposing_values(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MONGODB_URI": "", "MONGO_URI": ""}):
            root = Path(directory)
            named = "mongodb+srv://atlas.example.net/LingGuang?retryWrites=true"
            legacy = "mongodb+srv://legacy.example.net/old"
            (root / "atlasconnection.env").write_text(f"MONGO_URI={named}\n", encoding="utf-8")
            (root / ".env").write_text(f"MONGO_URI={legacy}\n", encoding="utf-8")
            self.assertEqual(load_mongodb_uri(root), named)

    def test_tls_certificate_bypass_is_opt_in(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            self.assertEqual(MongoStore(uri="mongodb://localhost", client=object()).uri, "mongodb://localhost")
            from mongo_store import mongodb_tls_options
            self.assertEqual(mongodb_tls_options(root), {})
            (root / ".env").write_text("MONGODB_TLS_ALLOW_INVALID_CERTIFICATES=true\n", encoding="utf-8")
            self.assertEqual(mongodb_tls_options(root), {"tlsAllowInvalidCertificates": True})

    def test_mongo_store_persists_owner_and_scopes_mutations(self):
        client = MagicMock()
        database = MagicMock()
        conversations = MagicMock()
        client.__getitem__.return_value = database
        database.__getitem__.return_value = conversations
        conversations.find_one.side_effect = [
            {"_id": "record-1", "owner_user_id": "user-a", "createdAt": "original-date"},
            {"_id": "record-1", "owner_user_id": "user-a", "createdAt": "original-date", "title": "changed"},
        ]
        conversations.replace_one.return_value.matched_count = 1
        conversations.delete_one.return_value.deleted_count = 1
        store = MongoStore(uri="mongodb://example.test/LingGuang", client=client)

        inserted = store.insert_conversation("user-a", {"title": "first", "blocks": []})
        inserted_document = conversations.insert_one.call_args.args[0]
        self.assertEqual(inserted_document["owner_user_id"], "user-a")
        self.assertNotIn("owner_user_id", inserted)

        updated = store.update_conversation("user-a", "record-1", {"title": "changed", "blocks": []})
        update_filter = conversations.replace_one.call_args.args[0]
        self.assertEqual(update_filter, {"_id": "record-1", "owner_user_id": "user-a"})
        self.assertEqual(conversations.replace_one.call_args.args[1]["createdAt"], "original-date")
        self.assertEqual(updated["id"], "record-1")

        store.delete_conversation("user-a", "record-1")
        self.assertEqual(conversations.delete_one.call_args.args[0], {"_id": "record-1", "owner_user_id": "user-a"})

    def test_history_search_queries_only_the_authenticated_owner(self):
        client = MagicMock()
        database = MagicMock()
        conversations = MagicMock()
        client.__getitem__.return_value = database
        database.__getitem__.return_value = conversations
        conversations.find.return_value.sort.return_value.limit.return_value = []
        store = MongoStore(uri="mongodb://example.test/LingGuang", client=client)

        self.assertEqual(store.search_conversations("user-a", "图书馆"), [])
        self.assertEqual(conversations.find.call_args.args[0], {"owner_user_id": "user-a"})

    def test_database_name_must_be_explicit_when_uri_has_no_database_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MONGODB_DATABASE": ""}):
            root = Path(directory)
            self.assertEqual(load_mongodb_database(root, "mongodb+srv://example.mongodb.net/"), "")
            self.assertEqual(load_mongodb_database(root, "mongodb+srv://example.mongodb.net/linguang"), "linguang")
            (root / "altasconnection.env").write_text("MONGODB_DATABASE=existing_db\n", encoding="utf-8")
            self.assertEqual(load_mongodb_database(root, "mongodb+srv://example.mongodb.net/"), "existing_db")

    def test_prompts_keep_generation_grounded_in_transcript(self):
        summary, user = _deepseek_messages("summarize", "下午三点图书馆见")
        self.assertIn("不得补充原文没有的事实", summary["content"])
        self.assertIn("下午三点图书馆见", user["content"])
        reply, _ = _deepseek_messages("reply", "下午三点图书馆见")
        self.assertIn("不得虚构", reply["content"])

    def test_upstream_requires_done_marker(self):
        upstream = [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        ]
        events = list(_iter_deepseek_sse(type("Upstream", (), {"__iter__": lambda self: iter(upstream), "close": lambda self: None})()))
        self.assertTrue(any("event: token" in event for event in events))
        self.assertTrue(any("event: error" in event for event in events))
        self.assertFalse(any("event: done" in event for event in events))

    def test_endpoint_validates_and_reports_missing_key_without_transcript(self):
        app = create_app(mongo_store=FakeMongoStore())
        client = app.test_client()
        login_test_session(client)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEP_SEEK_API_KEY", None)
            invalid = client.post("/api/assistant/stream", json={"action": "other", "text": "sensitive"}, headers={"X-CSRF-Token": "test-csrf", **owner_header()})
            missing = client.post("/api/assistant/stream", json={"action": "summarize", "text": "sensitive text"}, headers={"X-CSRF-Token": "test-csrf", **owner_header()})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 503)
        self.assertNotIn("sensitive", missing.get_data(as_text=True))

    def test_stream_route_sends_streaming_request_and_returns_events(self):
        app = create_app(mongo_store=FakeMongoStore())
        client = app.test_client()
        login_test_session(client)
        response = type("Response", (), {
            "__iter__": lambda self: iter([
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                b"data: [DONE]\n",
            ]),
            "close": lambda self: None,
        })()
        with patch.dict(os.environ, {"DEEP_SEEK_API_KEY": "test-key"}), patch("serve_sign_api.urlopen", return_value=response) as mocked:
            result = client.post("/api/assistant/stream", json={"action": "reply", "text": "原始对话"}, headers={"X-CSRF-Token": "test-csrf", **owner_header()}, buffered=False)
            first_event = next(result.response).decode("utf-8")
            remaining = "".join(chunk.decode("utf-8") for chunk in result.response)
        self.assertEqual(result.status_code, 200)
        self.assertIn("event: meta", first_event)
        self.assertIn("等待模型首段内容", first_event)
        self.assertIn("event: token", remaining)
        self.assertIn("hello", remaining)
        self.assertIn("event: done", remaining)
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["max_tokens"], 220)
        self.assertEqual(payload["messages"][1]["content"].count("原始对话"), 1)

    def test_assistant_routes_require_login_and_csrf(self):
        client = create_app(mongo_store=FakeMongoStore()).test_client()
        self.assertEqual(client.get("/api/auth/csrf").status_code, 200)
        self.assertFalse(client.get("/api/auth/me").json["authenticated"])
        module_response = client.get("/auth_client.mjs")
        self.assertEqual(module_response.status_code, 200)
        module_response.close()
        self.assertEqual(client.get("/api/communications").status_code, 401)
        self.assertEqual(client.post("/api/auth/password", json={}).status_code, 401)
        login_test_session(client)
        self.assertEqual(client.get("/api/communications").status_code, 200)
        self.assertEqual(client.post("/api/communications", json={"title": "x"}, headers=owner_header()).status_code, 403)
        self.assertEqual(client.get("/api/communications", headers=owner_header("user-b")).status_code, 200)

    def test_password_change_verifies_current_password_and_saves_new_hash(self):
        repository = FakeMongoStore()
        client = create_app(mongo_store=repository).test_client()
        token = client.get("/api/auth/csrf").json["csrfToken"]
        created = client.post("/api/auth/register", json={"username": "password_user", "password": "old-password"},
                              headers={"X-CSRF-Token": token})
        self.assertEqual(created.status_code, 201)
        failed = client.post("/api/auth/password", json={"currentPassword": "bad-password", "newPassword": "new-password"},
                             headers={"X-CSRF-Token": created.json["csrfToken"]})
        self.assertEqual(failed.status_code, 400)
        changed = client.post("/api/auth/password", json={"currentPassword": "old-password", "newPassword": "new-password"},
                              headers={"X-CSRF-Token": created.json["csrfToken"]})
        self.assertEqual(changed.status_code, 200)
        user = next(iter(repository.users.values()))
        self.assertTrue(check_password_hash(user["password_hash"], "new-password"))
        self.assertFalse(check_password_hash(user["password_hash"], "old-password"))

    def test_profile_is_login_protected_and_saved_to_current_account(self):
        repository = FakeMongoStore()
        client = create_app(mongo_store=repository).test_client()
        self.assertEqual(client.get("/api/auth/profile").status_code, 401)
        token = client.get("/api/auth/csrf").json["csrfToken"]
        registered = client.post("/api/auth/register", json={"username": "profile_user", "password": "long-password"},
                                 headers={"X-CSRF-Token": token})
        profile = client.put("/api/auth/profile", json={"displayName": "小溪", "bio": "喜欢交流", "avatar": ""},
                             headers={"X-CSRF-Token": registered.json["csrfToken"], **owner_header("another-user")})
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json["profile"]["displayName"], "小溪")
        stored = next(iter(repository.users.values()))
        self.assertEqual(stored["display_name"], "小溪")
        self.assertEqual(stored["bio"], "喜欢交流")
        self.assertEqual(client.get("/api/auth/profile").json["profile"]["username"], "profile_user")

    def test_profile_rejects_oversized_or_non_image_avatar_data(self):
        repository = FakeMongoStore()
        client = create_app(mongo_store=repository).test_client()
        token = client.get("/api/auth/csrf").json["csrfToken"]
        registered = client.post("/api/auth/register", json={"username": "avatar_user", "password": "long-password"},
                                 headers={"X-CSRF-Token": token})
        headers = {"X-CSRF-Token": registered.json["csrfToken"]}
        invalid = client.put("/api/auth/profile", json={"displayName": "User", "avatar": "data:image/svg+xml;base64,PHN2Zz4="}, headers=headers)
        self.assertEqual(invalid.status_code, 400)
        oversized = client.put("/api/auth/profile", json={"displayName": "User", "avatar": "x" * 360001}, headers=headers)
        self.assertEqual(oversized.status_code, 400)

    def test_registration_hashes_password_and_normalizes_username_uniqueness(self):
        repository = FakeMongoStore()
        client = create_app(mongo_store=repository).test_client()
        token = client.get("/api/auth/csrf").json["csrfToken"]
        first = client.post("/api/auth/register", json={"username": "  TeSt_User  ", "password": "long-password"},
                            headers={"X-CSRF-Token": token})
        self.assertEqual(first.status_code, 201)
        user = next(iter(repository.users.values()))
        self.assertEqual(user["username"], "TeSt_User")
        self.assertNotEqual(user["password_hash"], "long-password")
        self.assertTrue(check_password_hash(user["password_hash"], "long-password"))
        duplicate = client.post("/api/auth/register", json={"username": "test_user", "password": "different-password"},
                                headers={"X-CSRF-Token": first.json["csrfToken"]})
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(len(repository.users), 1)

    def test_server_session_cookie_contains_only_opaque_id_and_logout_removes_session(self):
        repository = FakeMongoStore()
        client = create_app(mongo_store=repository).test_client()
        token = client.get("/api/auth/csrf").json["csrfToken"]
        response = client.post("/api/auth/register", json={"username": "session_user", "password": "long-password"},
                               headers={"X-CSRF-Token": token})
        cookie = response.headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie)
        cookie_value = cookie.split("=", 1)[1].split(";", 1)[0]
        self.assertNotIn(response.json["user"]["id"], cookie_value)
        self.assertTrue(any(SESSION_USER_KEY in data for data in repository.sessions.values()))
        csrf = response.json["csrfToken"]
        client.post("/api/auth/logout", json={}, headers={"X-CSRF-Token": csrf})
        self.assertFalse(any(SESSION_USER_KEY in data for data in repository.sessions.values()))

    def test_communication_records_are_scoped_to_session_owner(self):
        repository = FakeMongoStore()
        client = create_app(mongo_store=repository).test_client()
        login_test_session(client)
        created = client.post("/api/communications", json={"title": "private", "blocks": []},
                              headers={"X-CSRF-Token": "test-csrf", **owner_header("attacker")})
        self.assertEqual(created.status_code, 201)
        record_id = created.json["record"]["id"]
        self.assertIn(("user-a", record_id), repository.records)
        with client.session_transaction() as session:
            session[SESSION_USER_KEY] = "user-b"
        self.assertEqual(client.get("/api/communications", headers=owner_header("user-b")).json["records"], [])
        denied = client.put(f"/api/communications/{record_id}", json={"title": "stolen"},
                            headers={"X-CSRF-Token": "test-csrf", **owner_header("user-b")})
        self.assertEqual(denied.status_code, 404)
        self.assertEqual(client.delete(f"/api/communications/{record_id}", headers={"X-CSRF-Token": "test-csrf"}).status_code, 404)
        with client.session_transaction() as session:
            session[SESSION_USER_KEY] = "user-a"
        updated = client.put(f"/api/communications/{record_id}", json={"title": "updated", "blocks": []},
                             headers={"X-CSRF-Token": "test-csrf", **owner_header("user-b")})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json["record"]["title"], "updated")
        self.assertEqual(client.delete(f"/api/communications/{record_id}", headers={"X-CSRF-Token": "test-csrf"}).status_code, 200)


class AtlasLiveCheck(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LINGUANG_RUN_ATLAS_CHECK") == "1", "manual Atlas connectivity check")
    def test_atlas_connection_and_required_indexes(self):
        try:
            MongoStore().initialize()
        except Exception as exc:
            self.fail(f"Atlas initialization failed ({type(exc).__name__}); verify Atlas network access and credentials")


if __name__ == "__main__":
    unittest.main()
