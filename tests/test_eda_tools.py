import json
import unittest

import pandas as pd

from tools.eda_tools import (
    anova_by_group,
    categorical_summary,
    chi_square_test,
    correlation_analysis,
    group_summary,
    missing_analysis,
    numeric_summary,
    simple_linear_regression,
    outlier_detection,
    t_test_by_group,
    target_group_summary,
)


class EdaToolsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "B"],
                "target": ["yes", "no", "yes", "no", "yes"],
                "x": [1.0, 2.0, 3.0, 4.0, None],
                "y": [2.0, 4.0, 6.0, 8.0, 10.0],
                "category": ["red", "blue", "red", "red", None],
                "mostly_missing": [None, None, None, "seen", None],
            }
        )

    def test_missing_analysis_is_compact_and_json_serializable(self):
        result = missing_analysis(self.df)

        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["columns"]["mostly_missing"]["missing_count"], 4)
        self.assertEqual(result["columns"]["mostly_missing"]["missing_percent"], 80.0)
        self.assertEqual(result["high_missingness_columns"], ["mostly_missing"])
        json.dumps(result)

    def test_numeric_summary_describes_numeric_columns(self):
        result = numeric_summary(self.df, columns=["x", "y", "missing_column"])

        self.assertIn("missing_column not found", result["warnings"])
        self.assertEqual(result["columns"]["x"]["count"], 4)
        self.assertEqual(result["columns"]["x"]["mean"], 2.5)
        self.assertEqual(result["columns"]["x"]["min"], 1.0)
        self.assertIn("skewness", result["columns"]["y"])
        json.dumps(result)

    def test_numeric_summary_handles_no_numeric_columns(self):
        result = numeric_summary(pd.DataFrame({"name": ["Ada", "Grace"]}))

        self.assertEqual(result["columns"], {})
        self.assertIn("No numeric columns available", result["warnings"])

    def test_categorical_summary_counts_values_and_handles_no_categorical_columns(self):
        result = categorical_summary(self.df, columns=["group"], top_n=2)

        self.assertEqual(result["columns"]["group"]["unique_values"], 2)
        self.assertEqual(result["columns"]["group"]["top_values"][0]["value"], "B")
        self.assertEqual(result["columns"]["group"]["top_values"][0]["count"], 3)
        json.dumps(result)

        no_categorical = categorical_summary(pd.DataFrame({"score": [1, 2, 3]}))
        self.assertEqual(no_categorical["columns"], {})
        self.assertIn("No categorical columns available", no_categorical["warnings"])

    def test_correlation_analysis_returns_matrix_and_strong_pairs(self):
        result = correlation_analysis(self.df, columns=["x", "y"])

        self.assertEqual(result["method"], "pearson")
        self.assertIn("x", result["correlation_matrix"])
        self.assertIn("strongest_positive", result)
        json.dumps(result)

    def test_correlation_analysis_handles_fewer_than_two_numeric_columns(self):
        result = correlation_analysis(pd.DataFrame({"x": [1, 2], "name": ["a", "b"]}))

        self.assertEqual(result["correlation_matrix"], {})
        self.assertIn("At least two numeric columns are required", result["warnings"])

    def test_group_summary_compares_numeric_values_by_category(self):
        result = group_summary(self.df, "group", "y")

        self.assertEqual(result["group_col"], "group")
        self.assertEqual(result["value_col"], "y")
        self.assertEqual(result["groups"]["A"]["count"], 2)
        self.assertEqual(result["groups"]["A"]["mean"], 3.0)
        json.dumps(result)

    def test_group_summary_handles_invalid_columns(self):
        result = group_summary(self.df, "missing", "y")
        self.assertIn("group_col missing not found", result["warnings"])

        result = group_summary(self.df, "group", "category")
        self.assertIn("value_col category must be numeric", result["warnings"])

    def test_target_group_summary_for_binary_categorical_target(self):
        result = target_group_summary(self.df, "target")

        self.assertEqual(result["target_col"], "target")
        self.assertEqual(result["class_distribution"]["yes"]["count"], 3)
        self.assertIn("x", result["numeric_by_target"])
        self.assertEqual(result["numeric_by_target"]["y"]["yes"]["mean"], 6.0)
        json.dumps(result)

    def test_target_group_summary_handles_invalid_target(self):
        result = target_group_summary(self.df, "missing")
        self.assertIn("target_col missing not found", result["warnings"])

    def test_simple_linear_regression_returns_coefficients_and_interpretation(self):
        result = simple_linear_regression(
            pd.DataFrame({"feature": [1, 2, 3, 4], "target": [2, 4, 6, 8]}),
            "target",
            "feature",
        )

        self.assertEqual(result["slope"], 2.0)
        self.assertEqual(result["intercept"], 0.0)
        self.assertEqual(result["r_squared"], 1.0)
        self.assertIn("For each 1-unit increase", result["interpretation"])
        json.dumps(result)

    def test_chi_square_and_t_test_return_p_values_with_scipy(self):
        chi = chi_square_test(self.df, "group", "target")
        self.assertEqual(chi["degrees_of_freedom"], 1)
        self.assertIsNotNone(chi["p_value"])
        self.assertNotIn("p_value unavailable", chi["warnings"])
        json.dumps(chi)

        t_test = t_test_by_group(self.df, "group", "y")
        self.assertEqual(set(t_test["groups"].keys()), {"A", "B"})
        self.assertIn("t_statistic", t_test)
        self.assertIsNotNone(t_test["p_value"])
        self.assertNotIn("p_value unavailable", t_test["warnings"])
        json.dumps(t_test)

    def test_anova_by_group_returns_p_value_for_three_groups(self):
        df = pd.DataFrame(
            {
                "department": ["A", "A", "B", "B", "C", "C"],
                "salary": [10, 12, 20, 22, 30, 32],
            }
        )

        result = anova_by_group(df, "department", "salary")

        self.assertEqual(result["test"], "anova_by_group")
        self.assertEqual(result["number_of_groups"], 3)
        self.assertEqual(result["rows_used"], 6)
        self.assertIsNotNone(result["f_statistic"])
        self.assertIsNotNone(result["p_value"])

    def test_t_test_handles_non_binary_group(self):
        result = t_test_by_group(
            pd.DataFrame({"group": ["A", "B", "C"], "value": [1, 2, 3]}),
            "group",
            "value",
        )

        self.assertIn("group_col group must contain exactly two non-missing groups", result["warnings"])

    def test_outlier_detection_uses_iqr_rule(self):
        result = outlier_detection(pd.DataFrame({"salary": [10, 11, 12, 13, 100]}), "salary")

        self.assertEqual(result["column"], "salary")
        self.assertEqual(result["method"], "iqr")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["outliers"][0]["value"], 100)
        self.assertIn("potential anomalies", result["warnings"][0])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
