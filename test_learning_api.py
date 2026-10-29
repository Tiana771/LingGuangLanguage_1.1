import unittest

from serve_sign_api import CSRF_SESSION_KEY, SESSION_USER_KEY, create_app


class LearningApiStore:
    def __init__(self):
        self.sessions = {}
        self.owners = []

    def initialize(self):
        pass

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def save_session(self, session_id, data, expires_at):
        self.sessions[session_id] = dict(data)

    def delete_session(self, session_id):
        self.sessions.pop(session_id, None)

    def get_learning_settings(self, owner_id):
        self.owners.append(owner_id)
        return {"dailyNewTarget": 5}

    def get_learning_today(self, owner_id, target):
        self.owners.append(owner_id)
        return {"due": [], "new": [], "counts": {"due": 0, "new": 0, "completed": 0, "streak": 0}}

    def search_learning_entries(self, query, book, *, limit, skip):
        self.owners.append("search-called")
        return [{"id": "entry-1", "title": "谢谢", "text": "双手动作示意"}]

    def count_learning_entries(self, query, book):
        return 1

    def learning_entry_flags(self, owner_id, entry_ids):
        self.owners.append(owner_id)
        return {"entry-1": {"favorite": False, "learningActive": False}}

    def update_learning_flag(self, owner_id, entry_id, field, value):
        self.owners.append(owner_id)
        return entry_id == "entry-1"

    def record_learning_review(self, owner_id, entry_id, rating, event_id):
        self.owners.append(owner_id)
        return {"duplicate": False, "progress": {"interval_days": 3, "next_review_at": "2026-01-05T00:00:00+00:00"}}

    def learning_stat_words(self, owner_id, category, days):
        self.owners.append(owner_id)
        return [{"date": "2026-10-25", "type": "new" if category == "today" else None,
                 "entries": [{"id": "entry-1", "title": category}]}]

    def batch_save_learning_entries(self, owner_id, entry_ids, **kwargs):
        self.owners.append(owner_id)
        self.batch = {"ids": entry_ids, **kwargs}
        return {"action": kwargs["action"], "added": len(entry_ids), "alreadyPresent": 0, "skipped": 0,
                "listId": "list-a", "listTitle": kwargs.get("list_title") or "主题词单"}


class LearningApiTests(unittest.TestCase):
    def setUp(self):
        self.store = LearningApiStore()
        self.client = create_app(mongo_store=self.store).test_client()

    def login(self, owner="account-a", csrf="learning-csrf"):
        with self.client.session_transaction() as session:
            session[SESSION_USER_KEY] = owner
            session[CSRF_SESSION_KEY] = csrf
        return {"X-CSRF-Token": csrf}

    def test_learning_api_rejects_signed_out_users(self):
        response = self.client.get("/api/learning/today")
        self.assertEqual(response.status_code, 401)

    def test_learning_reads_use_session_owner_not_client_owner(self):
        self.login("account-a")
        response = self.client.get("/api/learning/today?owner_user_id=account-b")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.owners, ["account-a", "account-a"])

    def test_learning_mutations_require_csrf(self):
        self.login()
        response = self.client.post("/api/learning/review", json={
            "entryId": "entry-1", "rating": "remembered", "eventId": "event-12345678",
            "owner_user_id": "account-b",
        })
        self.assertEqual(response.status_code, 403)

    def test_entry_reads_are_session_scoped(self):
        headers = self.login("account-a")
        response = self.client.get("/api/learning/entries?owner_user_id=account-b")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["entries"][0]["favorite"], False)
        self.assertEqual(self.store.owners[-1], "account-a")

    def test_favorite_update_cannot_choose_another_owner(self):
        headers = self.login("account-a")
        response = self.client.post("/api/learning/entries/entry-1/favorite", headers=headers,
                                    json={"favorite": True, "owner_user_id": "account-b"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.owners[-1], "account-a")

    def test_review_records_use_authenticated_owner(self):
        headers = self.login("account-a")
        response = self.client.post("/api/learning/review", headers=headers, json={
            "entryId": "entry-1", "rating": "remembered", "eventId": "event-12345678",
            "owner_user_id": "account-b",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.owners[-1], "account-a")

    def test_stats_words_require_login_and_scope_to_session_owner(self):
        self.assertEqual(self.client.get("/api/learning/stats/words?category=forgot").status_code, 401)
        self.login("account-a")
        response = self.client.get("/api/learning/stats/words?category=forgot&days=7&owner=account-b")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["groups"][0]["entries"][0]["title"], "forgot")
        self.assertEqual(self.store.owners[-1], "account-a")

    def test_stats_words_reject_unknown_category(self):
        self.login()
        self.assertEqual(self.client.get("/api/learning/stats/words?category=unknown").status_code, 400)

    def test_today_words_stats_uses_authenticated_owner(self):
        self.login("account-a")
        response = self.client.get("/api/learning/stats/words?category=today&days=7&owner=account-b")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["category"], "today")
        self.assertEqual(self.store.owners[-1], "account-a")

    def test_batch_save_rejects_signed_out_and_requires_csrf(self):
        payload = {"action": "favorite", "entryIds": ["entry-1"]}
        self.assertEqual(self.client.post("/api/learning/batch-save", json=payload).status_code, 401)
        self.login("account-a")
        self.assertEqual(self.client.post("/api/learning/batch-save", json=payload).status_code, 403)

    def test_batch_save_uses_session_owner_and_passes_selected_ids(self):
        headers = self.login("account-a")
        response = self.client.post("/api/learning/batch-save", headers=headers, json={
            "action": "list", "entryIds": ["entry-1", "entry-2"], "listTitle": "医院场景",
            "owner_user_id": "account-b",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["added"], 2)
        self.assertEqual(self.store.owners[-1], "account-a")
        self.assertEqual(self.store.batch["ids"], ["entry-1", "entry-2"])

    def test_batch_plan_uses_session_owner_and_selected_ids(self):
        headers = self.login("account-a")
        response = self.client.post("/api/learning/batch-save", headers=headers, json={
            "action": "plan", "entryIds": ["entry-1", "entry-2"], "owner_user_id": "account-b",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["action"], "plan")
        self.assertEqual(response.json["added"], 2)
        self.assertEqual(self.store.owners[-1], "account-a")
        self.assertEqual(self.store.batch["action"], "plan")
        self.assertEqual(self.store.batch["ids"], ["entry-1", "entry-2"])

    def test_batch_save_validates_id_count_and_list_target(self):
        headers = self.login()
        too_many = self.client.post("/api/learning/batch-save", headers=headers, json={
            "action": "favorite", "entryIds": [f"entry-{index}" for index in range(51)],
        })
        no_target = self.client.post("/api/learning/batch-save", headers=headers, json={
            "action": "list", "entryIds": ["entry-1"],
        })
        malformed_target = self.client.post("/api/learning/batch-save", headers=headers, json={
            "action": "favorite", "entryIds": ["entry-1"], "listTitle": [],
        })
        self.assertEqual(too_many.status_code, 400)
        self.assertEqual(no_target.status_code, 400)
        self.assertEqual(malformed_target.status_code, 400)


if __name__ == "__main__":
    unittest.main()
