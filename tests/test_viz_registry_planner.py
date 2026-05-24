import unittest

import pandas as pd

from agent.planner_helper import recommend_actions
from agent.tool_registry import get_tool_spec, list_available_tools, run_tool
from agent.verifier import verify_action


class VizRegistryPlannerTests(unittest.TestCase):
    def test_graph_helpers_are_registered(self):
        names = {tool["name"] for tool in list_available_tools()}

        for name in {
            "numeric_distribution_plot",
            "scatter_plot",
            "top_correlation_plots",
            "group_mean_bar_chart",
            "missing_value_bar_chart",
            "regression_plot",
        }:
            self.assertIn(name, names)
            self.assertIsNotNone(get_tool_spec(name))

    def test_verifier_accepts_valid_graph_helper_action(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

        result = verify_action(df, {"tool": "scatter_plot", "args": {"x_col": "x", "y_col": "y"}})

        self.assertTrue(result["valid"], result["errors"])

    def test_verifier_rejects_invalid_graph_helper_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "category": ["a", "b", "c"]})

        result = verify_action(
            df,
            {"tool": "scatter_plot", "args": {"x_col": "x", "y_col": "category"}},
        )

        self.assertFalse(result["valid"])
        self.assertTrue(any("must be numeric" in error for error in result["errors"]))

    def test_registry_runs_graph_helper(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

        result = run_tool("top_correlation_plots", df, top_n=1)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"]["tool_name"], "top_correlation_plots")

    def test_planner_recommends_graph_helpers_for_numeric_data(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4],
                "sales": [10, 20, 30, 40],
                "engagement": [1, 3, 5, 7],
                "region": ["A", "A", "B", "B"],
            }
        )

        actions = recommend_actions(df)
        by_tool = {action["tool"]: action for action in actions}

        self.assertIn("top_correlation_plots", by_tool)
        self.assertIn("missing_value_bar_chart", by_tool)
        self.assertIn("group_mean_bar_chart", by_tool)
        self.assertNotIn("customer_id", by_tool["top_correlation_plots"]["args"].get("cols", []))


if __name__ == "__main__":
    unittest.main()
