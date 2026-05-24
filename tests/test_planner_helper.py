import json
import unittest

import pandas as pd

from agent.planner_helper import recommend_actions
from agent.tool_registry import TOOL_REGISTRY
from agent.verifier import verify_action


class PlannerHelperTests(unittest.TestCase):
    def test_numeric_only_dataframe_recommends_numeric_actions(self):
        df = pd.DataFrame({"age": [21, 35, 42], "income": [40, 62, 80]})

        actions = recommend_actions(df)
        names = [action["tool"] for action in actions]

        self.assertEqual(names[0], "missing_analysis")
        self.assertIn("numeric_summary", names)
        self.assertIn("correlation_analysis", names)
        self.assertIn("simple_linear_regression", names)
        self.assertNotIn("categorical_summary", names)
        self.assertNotIn("group_summary", names)

    def test_categorical_only_dataframe_recommends_categorical_actions(self):
        df = pd.DataFrame(
            {
                "segment": ["A", "B", "A"],
                "region": ["north", "south", "north"],
            }
        )

        actions = recommend_actions(df)
        names = [action["tool"] for action in actions]

        self.assertEqual(names[0], "missing_analysis")
        self.assertIn("categorical_summary", names)
        self.assertIn("chi_square_test", names)
        self.assertNotIn("numeric_summary", names)
        self.assertNotIn("t_test_by_group", names)

    def test_mixed_dataframe_recommends_grouped_and_multivariate_actions(self):
        df = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "age": [21, 35, 42, 58],
                "income": [40, 62, 80, 91],
            }
        )

        actions = recommend_actions(df)
        names = [action["tool"] for action in actions]

        self.assertIn("numeric_summary", names)
        self.assertIn("categorical_summary", names)
        self.assertIn("correlation_analysis", names)
        self.assertIn("group_summary", names)
        self.assertIn("chi_square_test", names)
        self.assertIn("simple_linear_regression", names)

        group_action = next(action for action in actions if action["tool"] == "group_summary")
        self.assertEqual(group_action["args"], {"group_col": "segment", "value_col": "age"})

    def test_binary_group_and_numeric_dataframe_recommends_t_test(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [1.0, 2.0, 3.0, 4.0]})

        actions = recommend_actions(df)
        t_test_action = next(action for action in actions if action["tool"] == "t_test_by_group")

        self.assertEqual(t_test_action["args"], {"group_col": "group", "value_col": "score"})

    def test_empty_dataframe_only_recommends_missing_analysis(self):
        actions = recommend_actions(pd.DataFrame())

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "missing_analysis")
        self.assertEqual(actions[0]["args"], {})

    def test_max_actions_limits_result_length(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "age": [21, 35, 42, 58],
                "income": [40, 62, 80, 91],
            }
        )

        actions = recommend_actions(df, max_actions=3)

        self.assertEqual(len(actions), 3)
        self.assertEqual([action["priority"] for action in actions], [1, 2, 3])

    def test_recommendations_use_existing_tool_names_and_stable_shape(self):
        df = pd.DataFrame({"group": ["A", "B"], "score": [1, 2]})

        actions = recommend_actions(df)

        for action in actions:
            self.assertIn(action["tool"], TOOL_REGISTRY)
            self.assertEqual(set(action.keys()), {"tool", "args", "priority", "reason"})
            self.assertIsInstance(action["args"], dict)
            self.assertIsInstance(action["priority"], int)
            self.assertIsInstance(action["reason"], str)
        json.dumps(actions)

    def test_recommended_actions_can_be_passed_to_verifier(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "age": [21, 35, 42, 58],
                "income": [40, 62, 80, 91],
            }
        )

        actions = recommend_actions(df)
        verification_results = [verify_action(df, action) for action in actions]

        self.assertTrue(all(result["tool"] for result in verification_results))
        self.assertTrue(all(isinstance(result["args"], dict) for result in verification_results))
        self.assertTrue(all(result["valid"] for result in verification_results))


if __name__ == "__main__":
    unittest.main()
