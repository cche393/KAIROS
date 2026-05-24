import unittest

from app import (
    _analysis_result_pairs,
    _chart_dataframe,
    _cohesive_analysis_sections,
    _top_correlation_chart_frames,
)


class AppLayoutTests(unittest.TestCase):
    def test_analysis_result_pairs_preserve_selected_action_order(self):
        actions = [
            {"tool": "numeric_summary", "reason": "Summarise numeric fields."},
            {"tool": "correlation_analysis", "reason": "Check relationships."},
        ]
        results = [
            {"tool": "numeric_summary", "result": {"columns": {}}},
            {"tool": "correlation_analysis", "result": {"strongest_positive": []}},
        ]

        pairs = _analysis_result_pairs(actions, results)

        self.assertEqual(len(pairs), 2)
        self.assertIs(pairs[0][0], actions[0])
        self.assertIs(pairs[0][1], results[0])
        self.assertIs(pairs[1][0], actions[1])
        self.assertIs(pairs[1][1], results[1])

    def test_analysis_result_pairs_handles_missing_result(self):
        actions = [{"tool": "numeric_summary"}]

        pairs = _analysis_result_pairs(actions, [])

        self.assertEqual(pairs, [(actions[0], None)])

    def test_combines_group_summary_with_matching_group_mean_chart(self):
        actions = [
            {
                "tool": "group_summary",
                "args": {"group_col": "department", "value_col": "salary"},
            },
            {
                "tool": "group_mean_bar_chart",
                "args": {"group_col": "department", "value_col": "salary"},
            },
        ]
        results = [
            {"tool": "group_summary", "result": {"groups": {}}},
            {"tool": "group_mean_bar_chart", "result": {"chart_type": "bar", "data": []}},
        ]

        sections = _cohesive_analysis_sections(actions, results)

        self.assertEqual(len(sections), 1)
        self.assertIs(sections[0]["action"], actions[0])
        self.assertIs(sections[0]["result"], results[0])
        self.assertIs(sections[0]["chart_action"], actions[1])
        self.assertIs(sections[0]["chart_result"], results[1])

    def test_combines_correlation_with_matching_scatter_chart(self):
        actions = [
            {"tool": "correlation_analysis", "args": {"columns": ["salary", "years_experience"]}},
            {"tool": "scatter_plot", "args": {"x_col": "years_experience", "y_col": "salary"}},
        ]
        results = [
            {"tool": "correlation_analysis", "result": {"strongest_positive": []}},
            {"tool": "scatter_plot", "result": {"chart_type": "scatter", "data": []}},
        ]

        sections = _cohesive_analysis_sections(actions, results)

        self.assertEqual(len(sections), 1)
        self.assertIs(sections[0]["chart_action"], actions[1])

    def test_combines_numeric_summary_with_distribution_chart(self):
        actions = [
            {"tool": "numeric_summary", "args": {"columns": ["salary"]}},
            {"tool": "numeric_distribution_plot", "args": {"column": "salary"}},
        ]
        results = [
            {"tool": "numeric_summary", "result": {"columns": {"salary": {}}}},
            {"tool": "numeric_distribution_plot", "result": {"chart_type": "histogram", "data": []}},
        ]

        sections = _cohesive_analysis_sections(actions, results)

        self.assertEqual(len(sections), 1)
        self.assertIs(sections[0]["chart_action"], actions[1])

    def test_chart_dataframe_recognizes_renderable_bar_chart(self):
        chart_result = {
            "chart_type": "bar",
            "x_col": "department",
            "y_col": "mean_salary",
            "data": [
                {"department": "Sales", "mean_salary": 92500},
                {"department": "Finance", "mean_salary": 77500},
            ],
        }

        chart_df = _chart_dataframe(chart_result)

        self.assertEqual(list(chart_df.index), ["Sales", "Finance"])
        self.assertEqual(list(chart_df.columns), ["mean_salary"])

    def test_top_correlation_specs_are_split_into_renderable_frames(self):
        chart_result = {
            "tool_name": "top_correlation_plots",
            "data": [
                {
                    "title": "Correlation: salary vs years_experience",
                    "x_col": "years_experience",
                    "y_col": "salary",
                    "data": [
                        {"years_experience": 1, "salary": 70000},
                        {"years_experience": 5, "salary": 90000},
                    ],
                }
            ],
        }

        frames = _top_correlation_chart_frames(chart_result)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["title"], "Correlation: salary vs years_experience")
        self.assertEqual(frames[0]["x_col"], "years_experience")
        self.assertEqual(frames[0]["y_col"], "salary")


if __name__ == "__main__":
    unittest.main()
