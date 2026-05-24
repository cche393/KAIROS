import json
import unittest

from agent.result_interpreter import interpret_result


class ResultInterpreterTests(unittest.TestCase):
    def assert_stable_shape(self, interpretation):
        self.assertEqual(
            set(interpretation.keys()),
            {"summary", "key_findings", "cautions", "method_note"},
        )
        self.assertIsInstance(interpretation["summary"], str)
        self.assertIsInstance(interpretation["key_findings"], list)
        self.assertIsInstance(interpretation["cautions"], list)
        self.assertIsInstance(interpretation["method_note"], str)
        json.dumps(interpretation)

    def test_correlation_interpreter_identifies_strongest_positive(self):
        result = {
            "method": "pearson",
            "strongest_positive": [
                {"columns": ["customer_id", "salary"], "correlation": 0.99},
                {"columns": ["experience_years", "salary"], "correlation": 0.9},
            ],
            "strongest_negative": [],
        }

        interpretation = interpret_result("correlation_analysis", result)

        self.assert_stable_shape(interpretation)
        self.assertIn("experience_years and salary", interpretation["summary"])
        self.assertIn("strong positive", interpretation["summary"])
        self.assertTrue(any("Correlation does not prove causation" in item for item in interpretation["cautions"]))

    def test_correlation_interpreter_identifies_strongest_negative(self):
        result = {
            "method": "pearson",
            "strongest_positive": [],
            "strongest_negative": [
                {"columns": ["age", "risk_score"], "correlation": -0.82},
                {"columns": ["salary", "debt"], "correlation": -0.31},
            ],
        }

        interpretation = interpret_result("correlation_analysis", result)

        self.assertIn("age and risk_score", " ".join(interpretation["key_findings"]))
        self.assertIn("strong negative", " ".join(interpretation["key_findings"]))

    def test_correlation_interpreter_ignores_self_correlation(self):
        result = {
            "method": "pearson",
            "correlation_matrix": {
                "age": {"age": 1.0, "salary": 0.72},
                "salary": {"age": 0.72, "salary": 1.0},
            },
        }

        interpretation = interpret_result("correlation_analysis", result)

        text = " ".join([interpretation["summary"]] + interpretation["key_findings"])
        self.assertNotIn("age and age", text)
        self.assertIn("age and salary", text)

    def test_missing_analysis_no_missing_summary(self):
        result = {
            "total_missing_cells": 0,
            "columns": {
                "age": {"missing_count": 0, "missing_percent": 0.0},
                "salary": {"missing_count": 0, "missing_percent": 0.0},
            },
        }

        interpretation = interpret_result("missing_analysis", result)

        self.assertEqual(interpretation["summary"], "No missing values were detected.")
        self.assertEqual(interpretation["key_findings"], [])

    def test_missing_analysis_with_missing_columns(self):
        result = {
            "total_missing_cells": 7,
            "columns": {
                "salary": {"missing_count": 5, "missing_percent": 50.0},
                "age": {"missing_count": 2, "missing_percent": 20.0},
            },
        }

        interpretation = interpret_result("missing_analysis", result)

        self.assertIn("7 missing cells", interpretation["summary"])
        self.assertIn("salary", interpretation["key_findings"][0])

    def test_numeric_summary_stable_explanation(self):
        result = {
            "columns": {
                "age": {"count": 3, "mean": 30, "std": 5, "min": 20, "max": 40},
                "salary": {"count": 3, "mean": 80000, "std": 25000, "min": 50000, "max": 120000},
            }
        }

        interpretation = interpret_result("numeric_summary", result)

        self.assertIn("2 numeric columns", interpretation["summary"])
        self.assertTrue(any("salary" in item for item in interpretation["key_findings"]))

    def test_categorical_summary_stable_explanation(self):
        result = {
            "columns": {
                "department": {
                    "unique_values": 2,
                    "top_values": [{"value": "Engineering", "count": 8, "proportion": 0.8}],
                }
            }
        }

        interpretation = interpret_result("categorical_summary", result)

        self.assertIn("1 categorical column", interpretation["summary"])
        self.assertIn("Engineering", " ".join(interpretation["key_findings"]))
        self.assertTrue(any("imbalanced" in item for item in interpretation["key_findings"]))

    def test_group_summary_highest_and_lowest_group(self):
        result = {
            "group_col": "department",
            "value_col": "salary",
            "groups": {
                "Engineering": {"count": 3, "mean": 110000, "median": 108000},
                "HR": {"count": 2, "mean": 65000, "median": 64000},
            },
        }

        interpretation = interpret_result("group_summary", result)

        self.assertIn("Engineering", interpretation["summary"])
        self.assertIn("HR", interpretation["summary"])

    def test_t_test_p_value_interpretation(self):
        result = {
            "group_col": "remote",
            "value_col": "salary",
            "groups": {
                "Yes": {"mean": 90000, "count": 10},
                "No": {"mean": 75000, "count": 10},
            },
            "p_value": 0.03,
        }

        interpretation = interpret_result("t_test_by_group", result)

        self.assertIn("statistically significant", interpretation["summary"])
        self.assertTrue(any("sample size" in item for item in interpretation["cautions"]))

    def test_chi_square_p_value_interpretation(self):
        result = {
            "col_a": "region",
            "col_b": "churn",
            "chi_square_statistic": 9.4,
            "degrees_of_freedom": 2,
            "p_value": 0.02,
        }

        interpretation = interpret_result("chi_square_test", result)

        self.assertIn("association", interpretation["summary"])
        self.assertTrue(any("causation" in item for item in interpretation["cautions"]))

    def test_simple_linear_regression_stable_explanation(self):
        result = {
            "feature_col": "experience_years",
            "target_col": "salary",
            "slope": 4200,
            "intercept": 50000,
            "r_squared": 0.64,
            "n": 30,
        }

        interpretation = interpret_result("simple_linear_regression", result)

        self.assertIn("positive", interpretation["summary"])
        self.assertIn("R-squared", " ".join(interpretation["key_findings"]))

    def test_unknown_tool_returns_stable_shape(self):
        interpretation = interpret_result("unknown_tool", {"anything": "goes"})

        self.assert_stable_shape(interpretation)
        self.assertIn("No specialised interpretation", interpretation["summary"])

    def test_interpreter_does_not_crash_on_unexpected_structure(self):
        for tool_name, result in [
            ("correlation_analysis", None),
            ("numeric_summary", []),
            ("categorical_summary", {"columns": []}),
            ("group_summary", {"groups": None}),
            ("t_test_by_group", {"groups": "bad"}),
        ]:
            with self.subTest(tool_name=tool_name):
                self.assert_stable_shape(interpret_result(tool_name, result))


if __name__ == "__main__":
    unittest.main()
