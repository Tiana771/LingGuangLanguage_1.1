import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from mongo_store import CHINA_TIMEZONE, MongoStore


class FakeCursor(list):
    def sort(self, key, direction=1):
        return FakeCursor(sorted(self, key=lambda item: item.get(key, ""), reverse=direction < 0))


class FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def find(self, *_args, **_kwargs):
        return FakeCursor(self.documents)


class FakeDatabase(dict):
    pass


class LearningStatsTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc).astimezone(CHINA_TIMEZONE)
        day = now.date().isoformat()
        start = (now - timedelta(minutes=10)).astimezone(timezone.utc).isoformat()
        later = (now - timedelta(minutes=5)).astimezone(timezone.utc).isoformat()
        events = [
            {"entry_id": "new-entry", "entry_title": "父亲", "rating": "forgot", "is_new": True,
             "session_date": day, "reviewed_at": start, "processed": True},
            {"entry_id": "new-entry", "entry_title": "父亲", "rating": "remembered", "is_new": False,
             "session_date": day, "reviewed_at": later, "processed": True},
            {"entry_id": "old-entry", "entry_title": "母亲", "rating": "remembered", "is_new": False,
             "session_date": day, "reviewed_at": later, "processed": True},
        ]
        self.database = FakeDatabase({
            "learning_review_events": FakeCollection(events),
            "user_learning_progress": FakeCollection([
                {"entry_id": "new-entry", "first_studied_at": start},
                {"entry_id": "old-entry", "first_studied_at": "2026-01-01T00:00:00+00:00"},
            ]),
        })
        self.store = MongoStore(client=object())
        self.store._db = self.database
        self.store._learning_hard_words = lambda _owner, _today: []

    def test_today_stats_count_unique_words_and_separate_new_from_reviews(self):
        result = self.store.learning_stats("account-a", 7)
        self.assertEqual(result["totals"]["todayWords"], 2)
        self.assertEqual(result["totals"]["todayNewWords"], 1)
        self.assertEqual(result["totals"]["todayReviewWords"], 1)

    def test_today_word_groups_use_first_review_event_and_deduplicate(self):
        groups = self.store.learning_stat_words("account-a", "today", 7)
        self.assertEqual([(item["type"], [entry["id"] for entry in item["entries"]]) for item in groups], [
            ("new", ["new-entry"]), ("review", ["old-entry"]),
        ])

    def test_favorite_and_list_indexes_return_only_word_labels(self):
        favorite_cursor = FakeCursor([{"entry_id": "favorite", "favorite_at": "2026-10-26T00:00:00+00:00"}])
        list_cursor = FakeCursor([{"_id": "list-a", "title": "日常", "entry_ids": ["list-word"], "updated_at": "2026-10-26"}])
        self.database["user_learning_progress"] = MagicMock()
        self.database["user_learning_progress"].find.return_value.sort.return_value = favorite_cursor
        self.database["user_study_lists"] = MagicMock()
        self.database["user_study_lists"].find.return_value.sort.return_value = list_cursor
        with patch.object(self.store, "_learning_titles_by_ids", side_effect=[
            {"favorite": "父亲"}, {"list-word": "母亲"},
        ]) as titles:
            result = self.store.list_learning_lists("account-a")
        self.assertEqual(result["favorites"], [{"id": "favorite", "title": "父亲", "favoritedAt": "2026-10-26T00:00:00+00:00"}])
        self.assertEqual(result["lists"][0]["entries"], [{"id": "list-word", "title": "母亲"}])
        self.assertEqual(titles.call_count, 2)


if __name__ == "__main__":
    unittest.main()
