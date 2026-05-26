import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agent.memory_log import (
    append_log_entry,
    create_log_entry,
    export_log_jsonl,
    summarize_log_entries,
)


class MemoryLogTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "row_count": 500,
            "column_count": 8,
            "column_types": {
                "numeric": ["age", "salary"],
                "categorical": ["department"],
                "datetime": ["joined_date"],
            },
        }
        self.plan = {
            "mode": "llm",
            "reason": "The question asks for relationships centered on age.",
            "selected_actions": [
                {"tool": "target_relationship_analysis", "args": {"target_col": "age"}},
            ],
        }
        self.trace = {
            "analysis_type": "targeted_relationship_analysis",
            "analysis_focus": "relationships centered on age",
            "target_column": "age",
        }
        self.results = [
            {
                "executed": True,
                "tool": "target_relationship_analysis",
                "verification": {"valid": True, "warnings": []},
                "warnings": [],
                "errors": [],
                "result": {
                    "summary": "Selected variables most associated with age.",
                    "relationships": [{"predictor_col": "years_experience"}],
                },
            }
        ]

    def test_create_log_entry_records_planning_verification_and_execution(self):
        final_report = {
            "question_answered": "What correlates with age?",
            "key_findings": ["Selected variables most associated with age."],
            "limitations": ["Correlation does not imply causation."],
            "suggested_next_analyses": ["Compare the target variable across a categorical group."],
        }
        entry = create_log_entry(
            step=1,
            user_question="What correlates with age?",
            dataset_profile=self.profile,
            plan=self.plan,
            planning_trace=self.trace,
            selected_actions=self.plan["selected_actions"],
            execution_results=self.results,
            final_report=final_report,
            timestamp="2026-05-25T12:00:00+00:00",
        )

        self.assertEqual(entry["step"], 1)
        self.assertEqual(entry["timestamp"], "2026-05-25T12:00:00+00:00")
        self.assertEqual(entry["dataset_summary"]["rows"], 500)
        self.assertEqual(entry["dataset_summary"]["numeric_columns"], ["age", "salary"])
        self.assertEqual(entry["planner"]["mode"], "llm")
        self.assertEqual(entry["planner"]["analysis_type"], "targeted_relationship_analysis")
        self.assertEqual(entry["planner"]["selected_tools"], ["target_relationship_analysis"])
        self.assertEqual(entry["verification"]["status"], "passed")
        self.assertEqual(entry["execution"]["status"], "success")
        self.assertIn("Selected variables", entry["execution"]["result_summary"])
        self.assertEqual(
            entry["final_report_summary"],
            {
                "question_answered": "What correlates with age?",
                "top_findings": ["Selected variables most associated with age."],
                "limitations": ["Correlation does not imply causation."],
                "suggested_next_analyses": ["Compare the target variable across a categorical group."],
            },
        )
        json.dumps(entry)

    def test_append_log_entry_assigns_increasing_step_numbers(self):
        entries = []

        first = append_log_entry(entries, create_log_entry(user_question="First"))
        second = append_log_entry(entries, create_log_entry(user_question="Second"))

        self.assertEqual(first["step"], 1)
        self.assertEqual(second["step"], 2)
        self.assertEqual([entry["step"] for entry in entries], [1, 2])

    def test_summarize_log_entries_returns_compact_rows(self):
        entries = []
        append_log_entry(
            entries,
            create_log_entry(
                user_question="What correlates with age?",
                planning_trace=self.trace,
                selected_actions=self.plan["selected_actions"],
                execution_results=self.results,
            ),
        )

        rows = summarize_log_entries(entries)

        self.assertEqual(
            rows,
            [
                {
                    "step": 1,
                    "user_question": "What correlates with age?",
                    "planner_mode": "deterministic",
                    "analysis_type": "targeted_relationship_analysis",
                    "selected_tools": ["target_relationship_analysis"],
                    "verification_status": "passed",
                    "execution_status": "success",
                    "result_summary": "Selected variables most associated with age.",
                }
            ],
        )

    def test_create_log_entry_does_not_store_raw_dataframe(self):
        entry = create_log_entry(
            user_question="Analyze this dataset",
            dataset_profile={"sample_rows": [{"secret": "raw row"}], **self.profile},
            execution_results=self.results,
            raw_dataframe=pd.DataFrame({"secret": [1, 2, 3]}),
        )

        encoded = json.dumps(entry)
        self.assertNotIn("raw_dataframe", entry)
        self.assertNotIn("sample_rows", encoded)
        self.assertNotIn("raw row", encoded)
        self.assertNotIn("secret", encoded)

    def test_missing_optional_fields_are_handled_gracefully(self):
        entry = create_log_entry(user_question=None)

        self.assertEqual(entry["user_question"], "")
        self.assertEqual(entry["dataset_summary"]["rows"], 0)
        self.assertEqual(entry["planner"]["mode"], "deterministic")
        self.assertEqual(entry["planner"]["selected_tools"], [])
        self.assertEqual(entry["verification"]["status"], "failed")
        self.assertEqual(entry["execution"]["status"], "failed")
        json.dumps(entry)

    def test_export_log_jsonl_writes_compact_json_lines(self):
        entry = create_log_entry(user_question="Export this step", timestamp="2026-05-25T12:00:00+00:00")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "logs" / "analysis_log.jsonl"

            returned_path = export_log_jsonl([entry], path)

            self.assertEqual(returned_path, path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["user_question"], "Export this step")
            self.assertNotIn("raw_dataframe", parsed)


if __name__ == "__main__":
    unittest.main()
