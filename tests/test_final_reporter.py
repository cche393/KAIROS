import json
import unittest

from agent.final_reporter import (
    build_final_report,
    describe_analysis_run,
    extract_key_findings,
    generate_limitations,
    suggest_next_analyses,
    summarize_tools_run,
)


class FinalReporterTests(unittest.TestCase):
    def test_dataset_overview_report_uses_profile_and_interpretation(self):
        profile = {
            "row_count": 500,
            "column_count": 3,
            "column_types": {"numeric": ["age"], "categorical": ["department"]},
        }
        actions = [{"tool": "dataset_overview", "args": {}}]
        results = [
            {
                "executed": True,
                "tool": "dataset_overview",
                "args": {},
                "verification": {"valid": True, "warnings": []},
                "warnings": [],
                "errors": [],
                "result": {
                    "analysis_type": "dataset_overview",
                    "row_count": 500,
                    "column_count": 3,
                    "column_types": {"numeric": ["age"], "categorical": ["department"]},
                    "quality_notes": ["department has no missing values."],
                },
            }
        ]

        report = build_final_report(
            user_question="What is in this dataset?",
            dataset_profile=profile,
            planning_trace={"analysis_type": "dataset_overview"},
            selected_actions=actions,
            execution_results=results,
        )

        self.assertEqual(report["question_answered"], "What is in this dataset?")
        self.assertEqual(report["analyses_run"][0]["analysis_type"], "dataset_overview")
        self.assertEqual(report["analyses_run"][0]["tools"], ["dataset_overview"])
        self.assertEqual(report["analyses_run"][0]["status"], "success")
        self.assertTrue(any("500 rows and 3 columns" in item for item in report["key_findings"]))
        self.assertIn(
            "This describes dataset structure only; it does not test relationships.",
            report["limitations"],
        )
        self.assertIn("Ask about missing values.", report["suggested_next_analyses"])
        json.dumps(report)

    def test_targeted_relationship_report_adds_correlation_cautions(self):
        results = [
            {
                "executed": True,
                "tool": "target_relationship_analysis",
                "args": {"target_col": "salary"},
                "verification": {"valid": True, "warnings": []},
                "warnings": [],
                "errors": [],
                "result": {
                    "analysis_type": "targeted_relationship_analysis",
                    "target_col": "salary",
                    "summary": "Years_experience is most associated with salary.",
                    "relationships": [
                        {
                            "predictor_col": "years_experience",
                            "summary": "years_experience has a strong positive association with salary.",
                        }
                    ],
                },
            }
        ]

        report = build_final_report(
            user_question="What relates to salary?",
            dataset_profile={"column_types": {"numeric": ["salary", "years_experience"]}},
            planning_trace={"analysis_type": "targeted_relationship_analysis", "target_column": "salary"},
            selected_actions=[
                {"tool": "target_relationship_analysis", "args": {"target_col": "salary"}}
            ],
            execution_results=results,
        )

        self.assertEqual(report["analyses_run"][0]["target_column"], "salary")
        self.assertTrue(any("years_experience" in item for item in report["key_findings"]))
        self.assertIn("Correlation does not imply causation.", report["limitations"])
        self.assertIn(
            "Compare the target variable across a categorical group.",
            report["suggested_next_analyses"],
        )

    def test_group_and_missing_templates_are_deterministic(self):
        group_limitations = generate_limitations(
            "group_comparison_analysis",
            [{"verification": {"valid": True, "warnings": []}, "warnings": []}],
        )
        missing_suggestions = suggest_next_analyses(
            "missing_analysis",
            dataset_profile={"column_types": {"categorical": ["region"]}},
            target_column="",
            verification_warnings=[],
        )

        self.assertIn(
            "Group summaries describe differences but do not prove causation.",
            group_limitations,
        )
        self.assertIn("Inspect rows with missing values.", missing_suggestions)

    def test_failed_verification_report_includes_warning_and_safe_suggestion(self):
        actions = [{"tool": "group_summary", "args": {"group_col": "missing", "value_col": "salary"}}]
        results = [
            {
                "executed": False,
                "tool": "group_summary",
                "args": {"group_col": "missing", "value_col": "salary"},
                "verification": {
                    "valid": False,
                    "errors": ["Column not found: missing"],
                    "warnings": ["salary has missing values."],
                },
                "warnings": ["salary has missing values."],
                "errors": ["Column not found: missing"],
                "result": None,
            }
        ]

        report = build_final_report(
            user_question="Compare salary by missing.",
            dataset_profile={
                "column_types": {"numeric": ["salary"], "categorical": ["department"]}
            },
            planning_trace={"analysis_type": "group_comparison_analysis"},
            selected_actions=actions,
            execution_results=results,
        )

        self.assertEqual(report["analyses_run"][0]["status"], "blocked")
        self.assertIn("Column not found: missing", " ".join(report["limitations"]))
        self.assertIn("salary has missing values.", " ".join(report["limitations"]))
        self.assertTrue(any("department" in item and "salary" in item for item in report["suggested_next_analyses"]))

    def test_helper_functions_keep_report_compact(self):
        actions = [
            {"tool": "numeric_summary", "args": {"columns": ["salary"]}},
            {"tool": "numeric_distribution_plot", "args": {"column": "salary"}},
        ]
        results = [
            {
                "executed": True,
                "tool": "numeric_summary",
                "args": {"columns": ["salary"]},
                "verification": {"valid": True, "warnings": []},
                "warnings": [],
                "errors": [],
                "result": {"columns": {"salary": {"mean": 100}}},
            },
            {
                "executed": True,
                "tool": "numeric_distribution_plot",
                "args": {"column": "salary"},
                "verification": {"valid": True, "warnings": []},
                "warnings": [],
                "errors": [],
                "result": {"tool_name": "numeric_distribution_plot", "chart_type": "histogram", "data": []},
            },
        ]

        analyses = summarize_tools_run(actions, results, {"analysis_type": "distribution_analysis"})
        findings = extract_key_findings(results, user_question="Summarise salary")

        self.assertEqual(analyses[0]["tools"], ["numeric_summary", "numeric_distribution_plot"])
        self.assertLessEqual(len(findings), 6)
        self.assertTrue(all(isinstance(item, str) for item in findings))

    def test_analysis_run_description_is_natural_and_hides_function_names(self):
        description = describe_analysis_run(
            {
                "analysis_type": "missingness_analysis",
                "tools": ["missingness_analysis"],
                "target_column": "",
                "status": "success",
            }
        )

        self.assertEqual(
            description,
            "We ran a missing-value analysis to identify which columns contain incomplete data. The analysis completed successfully.",
        )
        self.assertNotIn("missingness_analysis", description)

    def test_group_comparison_description_uses_selected_columns(self):
        description = describe_analysis_run(
            {
                "analysis_type": "group_comparison_analysis",
                "tools": ["group_summary"],
                "target_column": "salary",
                "group_column": "department",
                "value_column": "salary",
                "status": "success",
            }
        )

        self.assertEqual(
            description,
            "We compared salary across department groups. The analysis completed successfully.",
        )
        self.assertNotIn("group_summary", description)

    def test_missingness_key_findings_do_not_duplicate_top_column(self):
        findings = extract_key_findings(
            [
                {
                    "executed": True,
                    "tool": "missingness_analysis",
                    "args": {},
                    "verification": {"valid": True, "warnings": []},
                    "warnings": [],
                    "errors": [],
                    "result": {
                        "analysis_type": "missingness_analysis",
                        "summary": "Satisfaction_score has the most missing values.",
                        "ranked_missing_columns": [
                            {
                                "column": "satisfaction_score",
                                "missing_count": 83,
                                "missing_percent": 6.92,
                            },
                            {"column": "gender", "missing_count": 66, "missing_percent": 5.5},
                        ],
                    },
                }
            ]
        )

        self.assertEqual(
            findings,
            [
                "satisfaction_score has the highest missingness at 6.92% (83 rows).",
                "gender has 5.5% missing values (66 rows).",
            ],
        )


if __name__ == "__main__":
    unittest.main()
