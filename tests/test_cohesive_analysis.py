import json
import unittest

import pandas as pd

from tools.cohesive_analysis import (
    distribution_analysis,
    group_comparison_analysis,
    missingness_analysis,
    outlier_analysis,
    relationship_analysis,
    target_relationship_analysis,
)


class CohesiveAnalysisTests(unittest.TestCase):
    def test_distribution_analysis_returns_statistics_and_chart(self):
        df = pd.DataFrame({"salary": [50, 60, 70, 80, 90, None]})

        result = distribution_analysis(df, "salary")

        self.assertEqual(result["analysis_type"], "distribution_analysis")
        self.assertEqual(result["column"], "salary")
        self.assertEqual(result["statistics"]["count"], 5)
        self.assertEqual(result["statistics"]["missing_count"], 1)
        self.assertEqual(result["statistics"]["min"], 50)
        self.assertEqual(result["statistics"]["max"], 90)
        self.assertIn("mean", result["statistics"])
        self.assertEqual(result["statistics"]["median"], 70)
        self.assertIn("std", result["statistics"])
        self.assertIn("q1", result["statistics"])
        self.assertIn("q3", result["statistics"])
        self.assertIn("iqr", result["statistics"])
        self.assertIn(result["chart"]["chart_type"], {"histogram", "bar", "box"})
        json.dumps(result)

    def test_relationship_analysis_uses_only_explicit_pair(self):
        df = pd.DataFrame(
            {
                "age": [20, 30, 40, 50],
                "salary": [50, 70, 90, 110],
                "years_experience": [1, 4, 7, 10],
            }
        )

        result = relationship_analysis(df, "years_experience", "salary")

        self.assertEqual(result["analysis_type"], "relationship_analysis")
        self.assertEqual(result["x_col"], "years_experience")
        self.assertEqual(result["y_col"], "salary")
        self.assertNotIn("age", result["columns"])
        self.assertEqual(result["chart"]["chart_type"], "scatter")

    def test_target_relationship_analysis_excludes_identifier_predictors(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E0001", "E0002", "E0003", "E0004", "E0005"],
                "salary": [50, 60, 70, 80, 90],
                "years_experience": [1, 2, 3, 4, 5],
                "age": [22, 25, 29, 32, 35],
            }
        )

        result = target_relationship_analysis(df, "salary", top_n=2)

        self.assertEqual(result["analysis_type"], "target_relationship_analysis")
        self.assertEqual(result["target_col"], "salary")
        predictors = [item["predictor_col"] for item in result["relationships"]]
        self.assertNotIn("employee_id", predictors)
        self.assertLessEqual(len(predictors), 2)
        self.assertTrue(all(item.get("chart") for item in result["relationships"]))

    def test_group_comparison_analysis_returns_stats_and_bar_chart(self):
        df = pd.DataFrame(
            {
                "department": ["Sales", "Finance", "Sales", "Finance"],
                "salary": [90, 70, 100, 80],
            }
        )

        result = group_comparison_analysis(df, "department", "salary")

        self.assertEqual(result["analysis_type"], "group_comparison_analysis")
        self.assertEqual(result["group_col"], "department")
        self.assertEqual(result["value_col"], "salary")
        self.assertEqual(result["chart"]["chart_type"], "bar")
        self.assertEqual(result["ranked_groups"][0]["group"], "Sales")
        self.assertIn("Sales", result["summary"])
        self.assertIn("inferential_test", result)
        self.assertEqual(result["inferential_test"]["test"], "t_test_by_group")

    def test_group_comparison_uses_anova_for_three_or_more_groups(self):
        df = pd.DataFrame(
            {
                "department": ["Sales", "Sales", "Finance", "Finance", "HR", "HR"],
                "salary": [90, 100, 70, 80, 60, 65],
            }
        )

        result = group_comparison_analysis(df, "department", "salary")

        self.assertEqual(result["inferential_test"]["test"], "anova_by_group")
        self.assertIn("f_statistic", result["inferential_test"])
        self.assertIn("p_value", result["inferential_test"])
        self.assertEqual(result["inferential_test"]["number_of_groups"], 3)
        self.assertIn("Sales", result["summary"])
        self.assertIn("HR", result["summary"])
        self.assertIn("F =", result["summary"])
        self.assertIn("p =", result["summary"])
        self.assertIn("statistically", result["summary"])

    def test_group_comparison_warns_when_identifier_group_is_forced(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E0001", "E0002", "E0003"],
                "salary": [90, 70, 100],
            }
        )

        result = group_comparison_analysis(df, "employee_id", "salary")

        self.assertIn("identifier-like", " ".join(result["warnings"]))

    def test_outlier_analysis_includes_visualization_data_and_caution(self):
        df = pd.DataFrame({"salary": [10, 11, 12, 13, 100]})

        result = outlier_analysis(df, "salary")

        self.assertEqual(result["analysis_type"], "outlier_analysis")
        self.assertEqual(result["count"], 1)
        self.assertIn("q1", result)
        self.assertIn("q3", result)
        self.assertIn("iqr", result)
        self.assertIn("min", result)
        self.assertIn("max", result)
        self.assertTrue(result["chart"])
        self.assertIn("statistical flags only", " ".join(result["warnings"]))

    def test_outlier_analysis_shows_thresholds_and_chart_when_no_outliers(self):
        df = pd.DataFrame({"salary": [10, 11, 12, 13, 14, 15]})

        result = outlier_analysis(df, "salary")

        self.assertEqual(result["count"], 0)
        self.assertIsNotNone(result["lower_bound"])
        self.assertIsNotNone(result["upper_bound"])
        self.assertTrue(result["chart"])
        self.assertIn("No values fall outside", result["summary"])

    def test_missingness_analysis_has_no_default_chart(self):
        df = pd.DataFrame(
            {
                "salary": [1, None, None, 4],
                "age": [20, 30, None, 40],
                "bonus": [None, None, None, 5],
                "complete": [1, 2, 3, 4],
            }
        )

        result = missingness_analysis(df)

        self.assertEqual(result["analysis_type"], "missingness_analysis")
        self.assertIsNone(result["chart"])
        self.assertEqual(result["total_missing_cells"], 6)
        self.assertEqual(
            [row["column"] for row in result["ranked_missing_columns"]],
            ["bonus", "salary", "age"],
        )
        self.assertEqual(result["ranked_missing_columns"][0]["missing_count"], 3)
        self.assertEqual(result["ranked_missing_columns"][0]["missing_percent"], 75.0)

    def test_cohesive_summaries_start_with_capital_letter(self):
        relationship = relationship_analysis(
            pd.DataFrame({"salary": [1, 2, 3], "years_experience": [1, 2, 3]}),
            "years_experience",
            "salary",
        )
        distribution = distribution_analysis(pd.DataFrame({"salary": [1, 2, 3]}), "salary")

        self.assertTrue(relationship["summary"][0].isupper())
        self.assertTrue(distribution["summary"][0].isupper())


if __name__ == "__main__":
    unittest.main()
