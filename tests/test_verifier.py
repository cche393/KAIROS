import json
import unittest

import pandas as pd

from agent.verifier import verify_action


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "age": [21, 35, 42, 58],
                "income": [40000, 62000, 80000, 91000],
                "segment": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "outcome": ["yes", "no", "yes", "no"],
                "three_group": ["low", "medium", "high", "low"],
            }
        )

    def test_valid_numeric_summary_is_accepted(self):
        result = verify_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": ["age", "income"]}},
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["tool"], "numeric_summary")
        self.assertEqual(result["args"], {"columns": ["age", "income"]})
        self.assertEqual(result["errors"], [])
        json.dumps(result)

    def test_unknown_tool_is_rejected(self):
        result = verify_action(self.df, {"tool": "made_up_tool", "args": {}})

        self.assertFalse(result["valid"])
        self.assertIn("Unknown tool: made_up_tool", result["errors"])

    def test_missing_required_args_are_rejected(self):
        result = verify_action(self.df, {"tool": "group_summary", "args": {"group_col": "segment"}})

        self.assertFalse(result["valid"])
        self.assertIn("Missing required arg: value_col", result["errors"])

    def test_missing_column_is_rejected(self):
        result = verify_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": ["age", "missing"]}},
        )

        self.assertFalse(result["valid"])
        self.assertIn("Column not found: missing", result["errors"])

    def test_numeric_tool_with_categorical_column_is_rejected(self):
        result = verify_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": ["segment"]}},
        )

        self.assertFalse(result["valid"])
        self.assertIn("Column must be numeric for numeric_summary: segment", result["errors"])

    def test_group_summary_with_nonnumeric_value_column_is_rejected(self):
        result = verify_action(
            self.df,
            {"tool": "group_summary", "args": {"group_col": "segment", "value_col": "region"}},
        )

        self.assertFalse(result["valid"])
        self.assertIn("value_col must be numeric: region", result["errors"])

    def test_chi_square_test_with_numeric_column_is_rejected(self):
        result = verify_action(
            self.df,
            {"tool": "chi_square_test", "args": {"col_a": "segment", "col_b": "age"}},
        )

        self.assertFalse(result["valid"])
        self.assertIn("Column must be categorical for chi_square_test: age", result["errors"])

    def test_t_test_by_group_with_non_binary_group_is_rejected(self):
        result = verify_action(
            self.df,
            {"tool": "t_test_by_group", "args": {"group_col": "three_group", "value_col": "income"}},
        )

        self.assertFalse(result["valid"])
        self.assertIn("group_col must contain exactly two non-missing groups: three_group", result["errors"])

    def test_valid_t_test_by_group_is_accepted(self):
        result = verify_action(
            self.df,
            {"tool": "t_test_by_group", "args": {"group_col": "segment", "value_col": "income"}},
        )

        self.assertTrue(result["valid"])

    def test_valid_anova_by_group_is_accepted(self):
        df = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B", "C", "C"],
                "age": [20, 21, 30, 31, 40, 41],
            }
        )

        result = verify_action(
            df,
            {"tool": "anova_by_group", "args": {"group_col": "segment", "value_col": "age"}},
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_invalid_parameter_types_are_rejected(self):
        result = verify_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": "age"}},
        )

        self.assertFalse(result["valid"])
        self.assertIn("Arg columns must be a list of strings", result["errors"])

    def test_result_structure_is_stable_for_bad_action_shape(self):
        result = verify_action(self.df, ["not", "a", "dict"])

        self.assertEqual(set(result.keys()), {"valid", "tool", "args", "errors", "warnings"})
        self.assertFalse(result["valid"])
        self.assertIsNone(result["tool"])
        self.assertEqual(result["args"], {})
        self.assertIn("Action proposal must be a dictionary", result["errors"])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
