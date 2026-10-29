import unittest
from datetime import datetime, timezone

from learning_schedule import next_review, update_day_mastery
from mongo_store import learning_local_day, learning_streak


class LearningScheduleTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)

    def test_new_word_feedback_sets_expected_first_interval(self):
        self.assertEqual(next_review(0, "forgot", self.now, is_new=True)["interval_days"], 0)
        self.assertEqual(next_review(0, "fuzzy", self.now, is_new=True)["interval_days"], 1)
        self.assertEqual(next_review(0, "remembered", self.now, is_new=True)["interval_days"], 3)

    def test_review_feedback_changes_interval_and_caps_growth(self):
        forgotten = next_review(8, "forgot", self.now, is_new=False)
        fuzzy = next_review(8, "fuzzy", self.now, is_new=False)
        remembered = next_review(8, "remembered", self.now, is_new=False)
        self.assertEqual(forgotten["interval_days"], 1)
        self.assertTrue(forgotten["repetitions_reset"])
        self.assertEqual(fuzzy["interval_days"], 10)
        self.assertEqual(remembered["interval_days"], 18)
        self.assertEqual(next_review(180, "remembered", self.now, is_new=False)["interval_days"], 180)

    def test_invalid_rating_is_rejected(self):
        with self.assertRaises(ValueError):
            next_review(0, "maybe", self.now, is_new=True)

    def test_forgot_word_requires_three_consecutive_correct_recalls(self):
        state = update_day_mastery(existing_session=False, previous_required=0, previous_successes=0,
                                   previous_initial_rating="", previous_is_new=False, is_new=True,
                                   rating="forgot")
        self.assertEqual(state["required"], 3)
        self.assertEqual(state["successes"], 0)
        for expected in (1, 2):
            state = update_day_mastery(existing_session=True, previous_required=state["required"],
                                       previous_successes=state["successes"],
                                       previous_initial_rating=state["initial_rating"],
                                       previous_is_new=state["session_is_new"], is_new=False,
                                       rating="remembered")
            self.assertEqual(state["successes"], expected)
            self.assertFalse(state["complete"])
        state = update_day_mastery(existing_session=True, previous_required=state["required"],
                                   previous_successes=state["successes"],
                                   previous_initial_rating=state["initial_rating"],
                                   previous_is_new=state["session_is_new"], is_new=False,
                                   rating="remembered")
        self.assertTrue(state["complete"])

    def test_fuzzy_word_requires_two_consecutive_correct_recalls_and_resets_on_miss(self):
        state = update_day_mastery(existing_session=False, previous_required=0, previous_successes=0,
                                   previous_initial_rating="", previous_is_new=False, is_new=False,
                                   rating="fuzzy")
        state = update_day_mastery(existing_session=True, previous_required=state["required"],
                                   previous_successes=state["successes"],
                                   previous_initial_rating=state["initial_rating"],
                                   previous_is_new=state["session_is_new"], is_new=False,
                                   rating="remembered")
        self.assertEqual(state["successes"], 1)
        state = update_day_mastery(existing_session=True, previous_required=state["required"],
                                   previous_successes=state["successes"],
                                   previous_initial_rating=state["initial_rating"],
                                   previous_is_new=state["session_is_new"], is_new=False,
                                   rating="fuzzy")
        self.assertEqual(state["successes"], 0)
        state = update_day_mastery(existing_session=True, previous_required=state["required"],
                                   previous_successes=state["successes"],
                                   previous_initial_rating=state["initial_rating"],
                                   previous_is_new=state["session_is_new"], is_new=False,
                                   rating="remembered")
        state = update_day_mastery(existing_session=True, previous_required=state["required"],
                                   previous_successes=state["successes"],
                                   previous_initial_rating=state["initial_rating"],
                                   previous_is_new=state["session_is_new"], is_new=False,
                                   rating="remembered")
        self.assertTrue(state["complete"])

    def test_first_try_remembered_finishes_today_without_repeats(self):
        state = update_day_mastery(existing_session=False, previous_required=0, previous_successes=0,
                                   previous_initial_rating="", previous_is_new=False, is_new=True,
                                   rating="remembered")
        self.assertTrue(state["complete"])
        self.assertFalse(state["repeat_today"])

    def test_china_local_day_handles_utc_midnight_and_legacy_datetime(self):
        self.assertEqual(learning_local_day("2026-10-25T16:30:00+00:00"), "2026-10-26")
        self.assertEqual(learning_local_day(datetime(2026, 10, 25, 16, 30, tzinfo=timezone.utc)), "2026-10-26")

    def test_streak_counts_yesterday_when_today_has_no_review_and_keeps_gap(self):
        self.assertEqual(learning_streak({"2026-10-25"}, "2026-10-26"), 1)
        self.assertEqual(learning_streak({"2026-10-24", "2026-10-25"}, "2026-10-26"), 2)
        self.assertEqual(learning_streak({"2026-10-23", "2026-10-25"}, "2026-10-26"), 1)


if __name__ == "__main__":
    unittest.main()
