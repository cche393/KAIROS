import unittest

from app import _action_label, _chart_dataframe, _cohesive_analysis_sections


class AppCohesiveRenderingTests(unittest.TestCase):
    def test_cohesive_analysis_is_one_section_without_separate_graph(self):
        actions = [
            {
                "tool": "group_comparison_analysis",
                "args": {"group_col": "department", "value_col": "salary"},
                "priority": 1,
                "reason": "Compare salary by department.",
            }
        ]
        results = [
            {
                "executed": True,
                "tool": "group_comparison_analysis",
                "args": actions[0]["args"],
                "verification": {"valid": True},
                "result": {
                    "analysis_type": "group_comparison_analysis",
                    "chart": {"tool_name": "group_mean_bar_chart", "chart_type": "bar", "data": []},
                },
                "errors": [],
                "warnings": [],
            }
        ]

        sections = _cohesive_analysis_sections(actions, results)

        self.assertEqual(len(sections), 1)
        self.assertIsNone(sections[0]["chart_action"])
        self.assertIsNone(sections[0]["chart_result"])

    def test_new_cohesive_actions_have_human_labels(self):
        self.assertEqual(_action_label({"tool": "distribution_analysis"}), "Distribution analysis")
        self.assertEqual(_action_label({"tool": "group_comparison_analysis"}), "Group comparison")

    def test_histogram_chart_spec_becomes_renderable_bar_dataframe(self):
        chart = {
            "tool_name": "numeric_distribution_plot",
            "chart_type": "histogram",
            "x_col": "salary",
            "y_col": "count",
            "data": [
                {"bin_start": 10, "bin_end": 20, "count": 2},
                {"bin_start": 20, "bin_end": 30, "count": 3},
            ],
        }

        frame = _chart_dataframe(chart)

        self.assertEqual(list(frame.columns), ["count"])
        self.assertEqual(frame.index.tolist(), ["10-20", "20-30"])
        self.assertEqual(frame["count"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
