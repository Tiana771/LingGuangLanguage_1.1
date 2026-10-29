import unittest
from unittest.mock import patch

from serve_sign_api import _fallback_knowledge_plan, _keyword_scenario_candidates, _qwen_select_knowledge


class KnowledgeIntelligenceTests(unittest.TestCase):
    def test_qwen_scene_selection_discards_ids_outside_retrieved_candidates(self):
        candidates = [
            {"id": "entry-hospital", "title": "医院", "text": "医院 看病"},
            {"id": "entry-market", "title": "市场", "text": "买菜"},
        ]
        with patch("serve_sign_api._qwen_book_json", return_value={
            "summary": "已找到相关词条", "items": [
                {"id": "entry-hospital", "reason": "场景相关"},
                {"id": "fabricated", "reason": "模型编造"},
            ],
        }):
            result = _qwen_select_knowledge("医院里有哪些词", candidates, "医院")
        self.assertEqual(result["items"], [{"id": "entry-hospital", "reason": "场景相关"}])

    def test_keyword_scene_fallback_requires_visible_text_match(self):
        candidates = [
            {"id": "doctor", "title": "医生", "text": "医院里看病的人。"},
            {"id": "market", "title": "市场", "text": "买菜的地方。"},
            {"id": "nurse", "title": "护士", "text": "在医院协助医生。"},
        ]
        selected = _keyword_scenario_candidates(candidates, ["医生"], "医院")
        self.assertEqual([item["id"] for item in selected], ["doctor", "nurse"])

    def test_keyword_scene_fallback_returns_empty_without_evidence(self):
        candidates = [{"id": "unrelated", "title": "早上", "text": "太阳升起。"}]
        self.assertEqual(_keyword_scenario_candidates(candidates, [], "医院"), [])

    def test_keyword_plan_extracts_scene_without_location_suffix(self):
        plan = _fallback_knowledge_plan("医院里会用到哪些词？")
        self.assertEqual(plan["intent"], "scenario")
        self.assertEqual(plan["topic"], "医院")
        self.assertEqual(plan["search_terms"], ["医院"])


if __name__ == "__main__":
    unittest.main()
