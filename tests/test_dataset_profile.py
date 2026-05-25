import json
import unittest

import pandas as pd

from tools.dataset_profile import build_dataset_profile, dataset_overview


class DatasetProfileTests(unittest.TestCase):
    def test_build_dataset_profile_detects_types_and_basic_metadata(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E001", "E002", "E003"],
                "age": [25, 35, 45],
                "department": ["Sales", "Finance", "Sales"],
                "active": [True, False, True],
                "order_date": ["2026-01-01", "2026-01-02", "bad-date"],
            }
        )

        profile = build_dataset_profile(df)

        self.assertEqual(profile["shape"], {"rows": 3, "columns": 5})
        self.assertEqual(profile["row_count"], 3)
        self.assertEqual(profile["column_count"], 5)
        self.assertIn("age", profile["column_types"]["numeric"])
        self.assertIn("department", profile["column_types"]["categorical"])
        self.assertIn("active", profile["column_types"]["boolean"])
        self.assertIn("order_date", profile["column_types"]["datetime"])
        self.assertIn("order_date", profile["column_types"]["datetime_like"])
        json.dumps(profile)

    def test_build_dataset_profile_reports_missing_statistics_and_top_values(self):
        df = pd.DataFrame(
            {
                "salary": [100.0, None, 130.0, 150.0],
                "department": ["Sales", "Sales", None, "HR"],
            }
        )

        profile = build_dataset_profile(df)

        self.assertEqual(profile["missing_values"]["columns"]["salary"]["missing_count"], 1)
        self.assertEqual(profile["missing_values"]["columns"]["salary"]["missing_percent"], 25.0)
        self.assertEqual(profile["numeric_statistics"]["salary"]["mean"], 126.666667)
        self.assertEqual(profile["numeric_statistics"]["salary"]["min"], 100.0)
        self.assertEqual(profile["numeric_statistics"]["salary"]["max"], 150.0)
        self.assertEqual(profile["categorical_statistics"]["department"]["unique_count"], 2)
        self.assertEqual(profile["categorical_statistics"]["department"]["top_values"][0]["value"], "Sales")

    def test_build_dataset_profile_flags_quality_and_structural_hints(self):
        df = pd.DataFrame(
            {
                "employee_id": [1001, 1002, 1003, 1004, 1005],
                "constant_flag": ["yes", "yes", "yes", "yes", "yes"],
                "city": ["A", "B", "C", "D", "E"],
                "salary": [10, 20, 30, 40, 50],
            }
        )

        profile = build_dataset_profile(df)
        issues = profile["potential_issues"]

        self.assertIn("constant_flag", issues["constant_value_columns"])
        self.assertIn("employee_id", issues["likely_id_columns"])
        self.assertIn("city", issues["high_cardinality_categorical_columns"])
        self.assertTrue(any("employee_id appears to be an identifier" in note for note in profile["quality_notes"]))
        self.assertTrue(any("city has high cardinality" in note for note in profile["quality_notes"]))

    def test_dataset_overview_returns_schema_quality_and_reuses_profile(self):
        df = pd.DataFrame({"age": [25], "department": ["Sales"]})
        profile = build_dataset_profile(df)
        profile["row_count"] = 999
        profile["shape"]["rows"] = 999

        result = dataset_overview(df, dataset_profile=profile)

        self.assertEqual(result["analysis_type"], "dataset_overview")
        self.assertEqual(result["row_count"], 999)
        self.assertEqual(result["column_count"], 2)
        self.assertEqual(result["columns"], ["age", "department"])
        self.assertIn("age", result["column_types"]["numeric"])
        self.assertIn("department", result["column_types"]["categorical"])
        self.assertIn("missing_values", result)
        self.assertIn("likely_id_columns", result["potential_issues"])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
