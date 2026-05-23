import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agent.observer import (
    detect_column_types,
    inspect_dataset,
    load_csv,
    suggest_target_columns,
    summarize_missing_values,
)


class ObserverTests(unittest.TestCase):
    def test_load_csv_reads_file_and_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text("name,score\nAda,10\nGrace,12\n", encoding="utf-8")

            df = load_csv(csv_path)

        self.assertEqual(list(df.columns), ["name", "score"])
        self.assertEqual(df.shape, (2, 2))

    def test_load_csv_returns_empty_dataframe_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text("", encoding="utf-8")

            df = load_csv(csv_path)

        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), [])

    def test_detect_column_types_handles_common_and_mixed_columns(self):
        df = pd.DataFrame(
            {
                "age": [29, 41, 35],
                "joined": ["2025-01-01", "2025-01-02", "bad-date"],
                "region": ["north", "south", "north"],
                "active": [True, False, True],
                "notes": [
                    "long free text about a customer",
                    "another longer note for the account",
                    "short note with words",
                ],
                "mixed": [1, "unknown", None],
            }
        )

        column_types = detect_column_types(df)

        self.assertEqual(column_types["numeric"], ["age"])
        self.assertEqual(column_types["boolean"], ["active"])
        self.assertEqual(column_types["datetime_like"], ["joined"])
        self.assertIn("region", column_types["categorical"])
        self.assertIn("notes", column_types["text_like"])
        self.assertIn("mixed", column_types["text_like"])

    def test_summarize_missing_values_reports_counts_percentages_and_high_missingness(self):
        df = pd.DataFrame(
            {
                "mostly_missing": [None, None, None, "seen"],
                "some_missing": [1, None, 3, 4],
                "complete": ["a", "b", "c", "d"],
            }
        )

        summary = summarize_missing_values(df)

        self.assertEqual(summary["total_missing_cells"], 4)
        self.assertEqual(summary["columns"]["mostly_missing"]["missing_count"], 3)
        self.assertEqual(summary["columns"]["mostly_missing"]["missing_percent"], 75.0)
        self.assertEqual(summary["columns"]["some_missing"]["missing_percent"], 25.0)
        self.assertEqual(summary["high_missingness_columns"], ["mostly_missing"])

    def test_suggest_target_columns_prefers_common_target_names_and_binary_columns(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4],
                "churn": ["yes", "no", "no", "yes"],
                "segment": ["a", "b", "c", "d"],
                "is_fraud": [0, 1, 0, 0],
                "notes": ["x", "y", "z", "w"],
            }
        )

        suggestions = suggest_target_columns(df)

        self.assertEqual(suggestions[0]["column"], "churn")
        self.assertEqual(suggestions[0]["reason"], "known target-like name")
        self.assertIn("is_fraud", [item["column"] for item in suggestions])

    def test_inspect_dataset_returns_compact_json_serializable_summary(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02", "2025-01-03", "bad-date"],
                "region": ["north", "north", "south", "south"],
                "sales": [100.5, 125.0, None, 125.0],
                "churn": ["no", "no", "yes", "yes"],
            }
        )
        df = pd.concat([df, df.iloc[[1]]], ignore_index=True)

        summary = inspect_dataset(df)

        self.assertEqual(summary["shape"], {"rows": 5, "columns": 4})
        self.assertEqual(summary["columns"], ["date", "region", "sales", "churn"])
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertLessEqual(len(summary["sample_rows"]), 5)
        self.assertIn("sales", summary["column_types"]["numeric"])
        self.assertIn("churn", [item["column"] for item in summary["suggested_target_columns"]])
        json.dumps(summary)

    def test_inspect_dataset_handles_empty_and_one_column_dataframes(self):
        empty_summary = inspect_dataset(pd.DataFrame())
        self.assertEqual(empty_summary["shape"], {"rows": 0, "columns": 0})
        self.assertEqual(empty_summary["columns"], [])
        self.assertEqual(empty_summary["sample_rows"], [])

        one_column_summary = inspect_dataset(pd.DataFrame({"status": ["ok", "fail", "ok"]}))
        self.assertEqual(one_column_summary["shape"], {"rows": 3, "columns": 1})
        self.assertEqual(one_column_summary["column_types"]["numeric"], [])
        self.assertIn("status", one_column_summary["column_types"]["categorical"])


if __name__ == "__main__":
    unittest.main()
