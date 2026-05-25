import json
import unittest

import pandas as pd

from tools.viz_tools import (
    group_mean_bar_chart,
    missing_value_bar_chart,
    numeric_distribution_plot,
    regression_plot,
    scatter_plot,
    top_correlation_plots,
)


class VizToolsTests(unittest.TestCase):
    def test_numeric_distribution_plot_on_numeric_data(self):
        df = pd.DataFrame({"sales": [10, 20, 20, 30, 40, 50]})

        result = numeric_distribution_plot(df, "sales")

        self.assertEqual(result["tool_name"], "numeric_distribution_plot")
        self.assertIn("sales", result["title"])
        self.assertIn(result["chart_type"], {"histogram", "box"})
        self.assertTrue(result["data"])
        self.assertEqual(result["warnings"], [])
        for key in ["count", "missing_count", "min", "max", "mean", "median", "std", "q1", "q3", "iqr"]:
            self.assertIn(key, result["metadata"])
        json.dumps(result)

    def test_numeric_distribution_plot_uses_bar_for_few_unique_values(self):
        df = pd.DataFrame({"rating": [1, 1, 2, 2, 2, 3]})

        result = numeric_distribution_plot(df, "rating")

        self.assertEqual(result["chart_type"], "bar")
        self.assertEqual(result["x_col"], "rating")
        self.assertEqual(result["y_col"], "count")

    def test_scatter_plot_on_two_numeric_columns(self):
        df = pd.DataFrame({"ad_spend": [1, 2, 3], "revenue": [2, 4, 6]})

        result = scatter_plot(df, "ad_spend", "revenue")

        self.assertEqual(result["chart_type"], "scatter")
        self.assertEqual(result["x"], "ad_spend")
        self.assertEqual(result["y"], "revenue")
        self.assertEqual(len(result["data"]), 3)
        self.assertIn("ad_spend vs revenue", result["title"])
        json.dumps(result)

    def test_top_correlation_plots_returns_top_three_and_excludes_ids(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "row_index": [10, 20, 30, 40, 50],
                "ad_spend": [1, 2, 3, 4, 5],
                "revenue": [2, 4, 6, 8, 10],
                "profit": [1, 1.5, 3, 3.5, 5],
                "discount": [5, 4, 3, 2, 1],
            }
        )

        result = top_correlation_plots(df, top_n=3)

        self.assertEqual(result["tool_name"], "top_correlation_plots")
        self.assertEqual(result["title"], "Top 3 strongest numeric relationships")
        self.assertEqual(len(result["data"]), 3)
        selected_columns = {
            column
            for graph in result["data"]
            for point in graph["data"]
            for column in (graph["x"], graph["y"])
        }
        self.assertNotIn("customer_id", selected_columns)
        self.assertNotIn("row_index", selected_columns)
        self.assertIn("Correlation:", result["data"][0]["title"])
        json.dumps(result)

    def test_top_correlation_plots_keeps_explicit_id_columns(self):
        df = pd.DataFrame({"customer_id": [1, 2, 3], "score": [10, 20, 30]})

        result = top_correlation_plots(df, cols=["customer_id", "score"], top_n=1)

        self.assertEqual(len(result["data"]), 1)
        self.assertEqual({result["data"][0]["x"], result["data"][0]["y"]}, {"customer_id", "score"})

    def test_top_correlation_plots_can_focus_on_target_column(self):
        df = pd.DataFrame(
            {
                "age": [20, 30, 40, 50, 60],
                "years_experience": [1, 4, 7, 10, 13],
                "salary": [45, 65, 85, 105, 125],
                "bonus": [2, 3, 5, 8, 13],
            }
        )

        result = top_correlation_plots(df, target_col="age", top_n=2)

        self.assertEqual(result["title"], "Top 2 correlations with age")
        self.assertEqual(len(result["data"]), 2)
        self.assertTrue(all(graph["y_col"] == "age" for graph in result["data"]))
        self.assertFalse(
            any({graph["x_col"], graph["y_col"]} == {"years_experience", "salary"} for graph in result["data"])
        )

    def test_group_mean_bar_chart_ranks_groups(self):
        df = pd.DataFrame(
            {
                "region": ["North", "South", "North", "West"],
                "sales": [100, 50, 200, 75],
            }
        )

        result = group_mean_bar_chart(df, "region", "sales")

        self.assertEqual(result["chart_type"], "bar")
        self.assertEqual(result["x_col"], "region")
        self.assertEqual(result["y_col"], "mean_sales")
        self.assertEqual(result["data"][0]["region"], "North")
        self.assertEqual(result["data"][0]["mean_sales"], 150.0)
        self.assertEqual(result["table"], result["data"])
        json.dumps(result)

    def test_missing_value_bar_chart_excludes_zero_missing_columns(self):
        df = pd.DataFrame({"complete": [1, 2, 3], "missing": [1, None, None]})

        result = missing_value_bar_chart(df)

        self.assertEqual(result["chart_type"], "bar")
        self.assertEqual([row["column"] for row in result["data"]], ["missing"])
        json.dumps(result)

    def test_safe_fallback_when_columns_invalid_or_unsuitable(self):
        df = pd.DataFrame({"category": ["a", "b"], "value": [1, 2]})

        missing = scatter_plot(df, "value", "missing")
        non_numeric = numeric_distribution_plot(df, "category")
        constant = numeric_distribution_plot(pd.DataFrame({"x": [1, 1, 1]}), "x")

        self.assertIn("Column not found", missing["warnings"][0])
        self.assertIn("must be numeric", non_numeric["warnings"][0])
        self.assertEqual(constant["chart_type"], "box")

    def test_scatter_sampling_is_deterministic_for_large_data(self):
        df = pd.DataFrame({"x": range(1200), "y": range(1200)})

        first = scatter_plot(df, "x", "y", max_points=500)
        second = scatter_plot(df, "x", "y", max_points=500)

        self.assertEqual(len(first["data"]), 500)
        self.assertEqual(first["data"], second["data"])
        self.assertTrue(first["metadata"]["sampled"])

    def test_regression_plot_returns_scatter_and_line_data(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

        result = regression_plot(df, "x", "y")

        self.assertEqual(result["chart_type"], "scatter_with_line")
        self.assertIn("points", result["metadata"])
        self.assertIn("line", result["metadata"])


if __name__ == "__main__":
    unittest.main()
