import json
import os
import unittest
from unittest.mock import patch

from agent.llm_planner import plan_with_llm


class LlmPlannerTests(unittest.TestCase):
    def setUp(self):
        self.dataset_profile = {
            "shape": {"rows": 4, "columns": 3},
            "columns": ["segment", "age", "income"],
            "column_types": {
                "numeric": ["age", "income"],
                "categorical": ["segment"],
            },
        }
        self.candidate_actions = [
            {
                "tool": "missing_analysis",
                "args": {},
                "priority": 1,
                "reason": "Check missing values.",
            },
            {
                "tool": "numeric_summary",
                "args": {"columns": ["age", "income"]},
                "priority": 2,
                "reason": "Summarize numeric columns.",
            },
            {
                "tool": "group_summary",
                "args": {"group_col": "segment", "value_col": "income"},
                "priority": 3,
                "reason": "Compare groups.",
            },
        ]

    def test_missing_api_key_uses_fallback_without_api_call(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("agent.llm_planner._request_llm_plan") as request:
                result = plan_with_llm(
                    "Explore this dataset",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=2,
                )

        self.assertEqual(result["mode"], "fallback")
        self.assertEqual(result["selected_actions"], self.candidate_actions[:2])
        self.assertIn("OPENAI_API_KEY is not set", result["warnings"])
        request.assert_not_called()

    def test_invalid_json_uses_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value="not json"):
                result = plan_with_llm(
                    "Explore this dataset",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=2,
                )

        self.assertEqual(result["mode"], "fallback")
        self.assertEqual(result["selected_actions"], self.candidate_actions[:2])
        self.assertTrue(any("invalid JSON" in error for error in result["errors"]))

    def test_llm_selects_valid_candidate_indexes(self):
        payload = json.dumps({"selected_indexes": [2, 0], "reason": "Group comparison then quality check."})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "Compare income by segment",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=3,
                )

        self.assertEqual(result["mode"], "llm")
        self.assertEqual(result["selected_actions"], [self.candidate_actions[2], self.candidate_actions[0]])
        self.assertEqual(result["reason"], "Group comparison then quality check.")

    def test_out_of_range_indexes_are_discarded(self):
        payload = json.dumps({"selected_indexes": [99, 1], "reason": "Use numeric summary."})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "Summarize numeric fields",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=3,
                )

        self.assertEqual(result["mode"], "llm")
        self.assertEqual(result["selected_actions"], [self.candidate_actions[1]])
        self.assertIn("Discarded out-of-range selected index: 99", result["warnings"])

    def test_duplicate_indexes_are_deduplicated(self):
        payload = json.dumps({"selected_indexes": [1, 1, 2], "reason": "Avoid duplicate work."})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "Compare income",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=3,
                )

        self.assertEqual(result["mode"], "llm")
        self.assertEqual(result["selected_actions"], [self.candidate_actions[1], self.candidate_actions[2]])

    def test_max_actions_is_respected(self):
        payload = json.dumps({"selected_indexes": [0, 1, 2], "reason": "Ranked choices."})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "General EDA",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=2,
                )

        self.assertEqual(result["selected_actions"], self.candidate_actions[:2])

    def test_stable_response_shape(self):
        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm("", self.dataset_profile, self.candidate_actions)

        self.assertEqual(
            set(result.keys()),
            {"mode", "selected_actions", "reason", "errors", "warnings"},
        )
        json.dumps(result)

    def test_no_candidate_actions_handled_gracefully(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan") as request:
                result = plan_with_llm("Anything", self.dataset_profile, [], max_actions=3)

        self.assertEqual(result["mode"], "fallback")
        self.assertEqual(result["selected_actions"], [])
        self.assertIn("No candidate actions were provided", result["warnings"])
        request.assert_not_called()

    def test_candidate_action_objects_are_preserved_exactly(self):
        payload = json.dumps(
            {
                "selected_indexes": [1],
                "selected_actions": [{"tool": "invented_tool", "args": {"columns": ["fake"]}}],
                "reason": "Model tried to rewrite actions.",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "Summarize",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=1,
                )

        self.assertIs(result["selected_actions"][0], self.candidate_actions[1])
        self.assertEqual(result["selected_actions"][0]["tool"], "numeric_summary")

    def test_all_invalid_llm_indexes_fallback(self):
        payload = json.dumps({"selected_indexes": [9, 10], "reason": "Bad indexes."})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "Explore this dataset",
                    self.dataset_profile,
                    self.candidate_actions,
                    max_actions=2,
                )

        self.assertEqual(result["mode"], "fallback")
        self.assertEqual(result["selected_actions"], self.candidate_actions[:2])
        self.assertIn("No valid LLM-selected action indexes remained", result["warnings"])

    def test_correlation_question_fallback_prioritizes_correlation_analysis(self):
        actions = self._rich_candidate_actions()

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "Can you see any correlation here?",
                self.dataset_profile,
                actions,
                max_actions=2,
            )

        self.assertEqual(result["mode"], "fallback")
        self.assertEqual(result["selected_actions"][0]["tool"], "correlation_analysis")

    def test_affects_salary_question_prioritizes_relationship_actions(self):
        actions = self._rich_candidate_actions()

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "What affects salary?",
                self.dataset_profile,
                actions,
                max_actions=3,
            )

        selected_tools = [action["tool"] for action in result["selected_actions"]]
        self.assertIn("simple_linear_regression", selected_tools)
        self.assertIn("correlation_analysis", selected_tools)

    def test_compare_salary_by_department_prioritizes_group_summary(self):
        actions = self._rich_candidate_actions()

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "Compare salary by department",
                self.dataset_profile,
                actions,
                max_actions=2,
            )

        self.assertEqual(result["selected_actions"][0]["tool"], "group_summary")

    def test_binary_group_effect_question_prioritizes_group_comparison(self):
        actions = self._rich_candidate_actions()

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "Does remote work affect salary?",
                self.dataset_profile,
                actions,
                max_actions=2,
            )

        selected_tools = [action["tool"] for action in result["selected_actions"]]
        self.assertIn(selected_tools[0], {"t_test_by_group", "group_summary"})

    def test_vague_explore_question_still_selects_broad_eda(self):
        actions = self._rich_candidate_actions()

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "Explore this dataset",
                self.dataset_profile,
                actions,
                max_actions=3,
            )

        self.assertEqual(
            [action["tool"] for action in result["selected_actions"]],
            ["missing_analysis", "numeric_summary", "categorical_summary"],
        )

    def test_relationship_question_fallback_prioritizes_correlation_graph_helper(self):
        actions = self._rich_candidate_actions() + [
            {
                "tool": "top_correlation_plots",
                "args": {"cols": ["age", "income"], "top_n": 3},
                "priority": 9,
                "reason": "Visualize strongest numeric relationships.",
            }
        ]

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "Show important relationships",
                self.dataset_profile,
                actions,
                max_actions=3,
            )

        selected_tools = [action["tool"] for action in result["selected_actions"]]
        self.assertIn("top_correlation_plots", selected_tools)

    def test_fallback_prefers_cohesive_relationship_action_for_specific_pair(self):
        actions = [
            {
                "tool": "distribution_analysis",
                "args": {"column": "salary"},
                "priority": 1,
                "reason": "Summarize salary.",
            },
            {
                "tool": "relationship_analysis",
                "args": {"x_col": "years_experience", "y_col": "salary"},
                "priority": 2,
                "reason": "Analyze the requested pair.",
            },
        ]

        with patch.dict(os.environ, {}, clear=True):
            result = plan_with_llm(
                "Is salary related to years_experience?",
                self.dataset_profile,
                actions,
                max_actions=2,
            )

        self.assertEqual(result["selected_actions"][0]["tool"], "relationship_analysis")

    def test_llm_broad_selection_is_reordered_for_obvious_correlation_intent(self):
        actions = self._rich_candidate_actions()
        payload = json.dumps({"selected_indexes": [0, 1, 2], "reason": "Default broad checks."})

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("agent.llm_planner._request_llm_plan", return_value=payload):
                result = plan_with_llm(
                    "Can you see any correlation here?",
                    self.dataset_profile,
                    actions,
                    max_actions=3,
                )

        self.assertEqual(result["mode"], "llm")
        self.assertEqual(result["selected_actions"][0]["tool"], "correlation_analysis")
        self.assertEqual(len(result["selected_actions"]), 3)

    def _rich_candidate_actions(self):
        return [
            {
                "tool": "missing_analysis",
                "args": {},
                "priority": 1,
                "reason": "Check missing values.",
            },
            {
                "tool": "numeric_summary",
                "args": {"columns": ["age", "income"]},
                "priority": 2,
                "reason": "Summarize numeric columns.",
            },
            {
                "tool": "categorical_summary",
                "args": {"columns": ["segment", "remote_work"]},
                "priority": 3,
                "reason": "Summarize categorical columns.",
            },
            {
                "tool": "correlation_analysis",
                "args": {"columns": ["age", "income"]},
                "priority": 4,
                "reason": "Check numeric relationships.",
            },
            {
                "tool": "group_summary",
                "args": {"group_col": "segment", "value_col": "income"},
                "priority": 5,
                "reason": "Compare income across segments.",
            },
            {
                "tool": "t_test_by_group",
                "args": {"group_col": "remote_work", "value_col": "income"},
                "priority": 6,
                "reason": "Compare two group means.",
            },
            {
                "tool": "chi_square_test",
                "args": {"col_a": "segment", "col_b": "remote_work"},
                "priority": 7,
                "reason": "Test relationship between categories.",
            },
            {
                "tool": "simple_linear_regression",
                "args": {"feature_col": "age", "target_col": "income"},
                "priority": 8,
                "reason": "Fit a simple numeric relationship.",
            },
        ]


if __name__ == "__main__":
    unittest.main()
