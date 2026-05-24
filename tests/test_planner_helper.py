import json
import unittest

import pandas as pd

from agent.planner_helper import recommend_actions
from agent.tool_registry import TOOL_REGISTRY
from agent.verifier import verify_action


class PlannerHelperTests(unittest.TestCase):
    def test_numeric_only_dataframe_recommends_numeric_actions(self):
        df = pd.DataFrame({"age": [21, 35, 42], "income": [40, 62, 80]})

        actions = recommend_actions(df)
        names = [action["tool"] for action in actions]

        self.assertEqual(names[0], "missing_analysis")
        self.assertIn("numeric_summary", names)
        self.assertIn("correlation_analysis", names)
        self.assertIn("simple_linear_regression", names)
        self.assertNotIn("categorical_summary", names)
        self.assertNotIn("group_summary", names)

    def test_categorical_only_dataframe_recommends_categorical_actions(self):
        df = pd.DataFrame(
            {
                "segment": ["A", "B", "A"],
                "region": ["north", "south", "north"],
            }
        )

        actions = recommend_actions(df)
        names = [action["tool"] for action in actions]

        self.assertEqual(names[0], "missing_analysis")
        self.assertIn("categorical_summary", names)
        self.assertIn("chi_square_test", names)
        self.assertNotIn("numeric_summary", names)
        self.assertNotIn("t_test_by_group", names)

    def test_mixed_dataframe_recommends_grouped_and_multivariate_actions(self):
        df = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "age": [21, 35, 42, 58],
                "income": [40, 62, 80, 91],
            }
        )

        actions = recommend_actions(df)
        names = [action["tool"] for action in actions]

        self.assertIn("numeric_summary", names)
        self.assertIn("categorical_summary", names)
        self.assertIn("correlation_analysis", names)
        self.assertIn("group_summary", names)
        self.assertIn("chi_square_test", names)
        self.assertIn("simple_linear_regression", names)

        group_action = next(action for action in actions if action["tool"] == "group_summary")
        self.assertEqual(group_action["args"], {"group_col": "segment", "value_col": "age"})

    def test_binary_group_and_numeric_dataframe_recommends_t_test(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [1.0, 2.0, 3.0, 4.0]})

        actions = recommend_actions(df)
        t_test_action = next(action for action in actions if action["tool"] == "t_test_by_group")

        self.assertEqual(t_test_action["args"], {"group_col": "group", "value_col": "score"})

    def test_empty_dataframe_only_recommends_missing_analysis(self):
        actions = recommend_actions(pd.DataFrame())

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "missing_analysis")
        self.assertEqual(actions[0]["args"], {})

    def test_max_actions_limits_result_length(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "age": [21, 35, 42, 58],
                "income": [40, 62, 80, 91],
            }
        )

        actions = recommend_actions(df, max_actions=3)

        self.assertEqual(len(actions), 3)
        self.assertEqual([action["priority"] for action in actions], [1, 2, 3])

    def test_recommendations_use_existing_tool_names_and_stable_shape(self):
        df = pd.DataFrame({"group": ["A", "B"], "score": [1, 2]})

        actions = recommend_actions(df)

        for action in actions:
            self.assertIn(action["tool"], TOOL_REGISTRY)
            self.assertEqual(set(action.keys()), {"tool", "args", "priority", "reason"})
            self.assertIsInstance(action["args"], dict)
            self.assertIsInstance(action["priority"], int)
            self.assertIsInstance(action["reason"], str)
        json.dumps(actions)

    def test_recommended_actions_can_be_passed_to_verifier(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
                "age": [21, 35, 42, 58],
                "income": [40, 62, 80, 91],
            }
        )

        actions = recommend_actions(df)
        verification_results = [verify_action(df, action) for action in actions]

        self.assertTrue(all(result["tool"] for result in verification_results))
        self.assertTrue(all(isinstance(result["args"], dict) for result in verification_results))
        self.assertTrue(all(result["valid"] for result in verification_results))

    def test_goal_mentions_salary_so_group_comparison_uses_salary_value(self):
        df = pd.DataFrame(
            {
                "department": ["Sales", "Finance", "Sales", "Finance"],
                "age": [25, 41, 29, 38],
                "salary": [90000, 75000, 95000, 80000],
            }
        )

        actions = recommend_actions(df, goal="Does the mean salary vary among different groups?")
        group_action = next(action for action in actions if action["tool"] == "group_comparison_analysis")

        self.assertEqual(group_action["args"], {"group_col": "department", "value_col": "salary"})
        self.assertIn("department", group_action["reason"])

    def test_goal_mentions_income_as_salary_synonym(self):
        df = pd.DataFrame(
            {
                "department": ["Sales", "Finance"],
                "age": [25, 41],
                "annual_salary": [90000, 75000],
            }
        )

        actions = recommend_actions(df, goal="Does pay vary across departments?")
        group_action = next(action for action in actions if action["tool"] == "group_comparison_analysis")

        self.assertEqual(group_action["args"]["value_col"], "annual_salary")

    def test_goal_avoids_identifier_numeric_columns_by_default(self):
        df = pd.DataFrame(
            {
                "employee_id": [1, 2, 3],
                "department": ["Sales", "Finance", "Sales"],
                "salary": [90000, 75000, 95000],
            }
        )

        actions = recommend_actions(df, goal="Does salary vary among groups?")
        group_action = next(action for action in actions if action["tool"] == "group_comparison_analysis")

        self.assertEqual(group_action["args"]["value_col"], "salary")
        self.assertNotEqual(group_action["args"]["value_col"], "employee_id")

    def test_explicit_pair_uses_only_named_pair_not_age(self):
        df = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "salary": [70000, 90000, 120000],
                "years_experience": [1, 5, 10],
            }
        )

        actions = recommend_actions(df, goal="Is salary related to years_experience?")
        names = [action["tool"] for action in actions]
        relationship = actions[0]

        self.assertEqual(names, ["relationship_analysis"])
        self.assertEqual(relationship["args"], {"x_col": "years_experience", "y_col": "salary"})
        self.assertNotIn("top_correlation_plots", names)
        self.assertNotIn("age", relationship["args"].values())

    def test_explicit_pair_scope_ignores_max_actions(self):
        df = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "salary": [70000, 90000, 120000],
                "years_experience": [1, 5, 10],
            }
        )

        actions = recommend_actions(df, max_actions=10, goal="Is salary related to years_experience?")

        self.assertEqual([action["tool"] for action in actions], ["relationship_analysis"])

    def test_strongest_relationships_returns_top_correlations_non_id(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E00001", "E00002", "E00003", "E00004"],
                "age": [20, 30, 40, 50],
                "salary": [50, 70, 90, 110],
                "years_experience": [1, 4, 7, 10],
            }
        )

        actions = recommend_actions(df, goal="What are the strongest relationships?")
        first = actions[0]

        self.assertEqual(first["tool"], "global_relationship_analysis")
        self.assertEqual(first["args"]["top_n"], 3)
        self.assertNotIn("employee_id", first["args"]["cols"])

    def test_salary_distribution_uses_summary_and_distribution_only(self):
        df = pd.DataFrame({"age": [25, 35, 45], "salary": [70000, 90000, 120000]})

        actions = recommend_actions(df, goal="Show salary distribution")
        names = [action["tool"] for action in actions]

        self.assertEqual(names, ["distribution_analysis"])
        self.assertEqual(actions[0]["args"], {"column": "salary"})

    def test_missingness_question_runs_missing_analysis_without_graph(self):
        df = pd.DataFrame({"salary": [1, None], "age": [20, 30]})

        actions = recommend_actions(df, goal="Which columns have missing values?")

        self.assertEqual([action["tool"] for action in actions], ["missingness_analysis"])

    def test_outlier_question_runs_outlier_detection(self):
        df = pd.DataFrame({"salary": [10, 11, 12, 1000], "age": [20, 21, 22, 23]})

        actions = recommend_actions(df, goal="Are there outliers in salary?")

        self.assertEqual(actions[0]["tool"], "outlier_analysis")
        self.assertEqual(actions[0]["args"], {"column": "salary"})

    def test_promotion_prediction_treats_promotion_as_target_and_excludes_id(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E00001", "E00002", "E00003", "E00004"],
                "department": ["Sales", "Finance", "Sales", "Finance"],
                "salary": [70, 80, 90, 100],
                "promotion": ["yes", "no", "yes", "no"],
            }
        )

        actions = recommend_actions(df, goal="What predicts promotion?")
        target_action = actions[0]

        self.assertEqual(target_action["tool"], "target_relationship_analysis")
        self.assertEqual(target_action["args"], {"target_col": "promotion"})
        self.assertFalse(any("employee_id" in action["args"].values() for action in actions))

    def test_variables_most_strongly_related_to_salary_uses_target_relationship(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E00001", "E00002", "E00003", "E00004"],
                "department": ["Sales", "Finance", "Sales", "Finance"],
                "salary": [70, 80, 90, 100],
                "years_experience": [1, 3, 5, 7],
                "age": [24, 29, 35, 41],
            }
        )

        actions = recommend_actions(df, goal="What variables are most strongly related to salary?")

        self.assertEqual(actions[0]["tool"], "target_relationship_analysis")
        self.assertEqual(actions[0]["args"], {"target_col": "salary"})
        self.assertNotEqual(actions[0]["tool"], "group_comparison_analysis")

    def test_relationship_target_wording_prefers_target_relationship(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E00001", "E00002", "E00003", "E00004"],
                "department": ["Sales", "Finance", "Sales", "Finance"],
                "salary": [70, 80, 90, 100],
                "years_experience": [1, 3, 5, 7],
                "age": [24, 29, 35, 41],
            }
        )

        prompts = [
            "What factors relate to salary?",
            "Which variables are associated with salary?",
            "What drives salary?",
            "What affects salary?",
            "What predicts salary?",
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                actions = recommend_actions(df, goal=prompt)
                self.assertEqual(actions[0]["tool"], "target_relationship_analysis")
                self.assertEqual(actions[0]["args"], {"target_col": "salary"})

    def test_global_correlation_wording_prefers_global_relationships(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E00001", "E00002", "E00003", "E00004"],
                "age": [20, 30, 40, 50],
                "salary": [50, 70, 90, 110],
                "years_experience": [1, 4, 7, 10],
            }
        )

        prompts = [
            "What are the strongest correlations?",
            "Show the strongest relationships in the dataset.",
            "Which variables are most correlated?",
            "Show the top 3 strongest correlations visually.",
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                actions = recommend_actions(df, goal=prompt)
                self.assertEqual(actions[0]["tool"], "global_relationship_analysis")
                self.assertEqual(actions[0]["args"]["top_n"], 3)
                self.assertNotIn("employee_id", actions[0]["args"]["cols"])

    def test_salary_vary_by_department_remains_group_comparison(self):
        df = pd.DataFrame(
            {
                "department": ["Sales", "Finance", "Sales"],
                "salary": [90, 75, 95],
                "age": [30, 40, 35],
            }
        )

        actions = recommend_actions(df, goal="Does salary vary by department?")

        self.assertEqual(actions[0]["tool"], "group_comparison_analysis")
        self.assertEqual(actions[0]["args"], {"group_col": "department", "value_col": "salary"})

    def test_paid_groups_prioritizes_group_comparison_before_summary(self):
        df = pd.DataFrame(
            {
                "department": ["Sales", "Finance", "Sales"],
                "salary": [90, 75, 95],
                "age": [30, 40, 35],
            }
        )

        actions = recommend_actions(df, goal="Which groups are paid the most?")

        self.assertEqual(actions[0]["tool"], "group_comparison_analysis")
        self.assertEqual(actions[0]["args"], {"group_col": "department", "value_col": "salary"})

    def test_compare_age_across_categories_prioritizes_group_comparison(self):
        df = pd.DataFrame({"category": ["A", "B", "A"], "age": [20, 30, 25], "salary": [1, 2, 3]})

        actions = recommend_actions(df, goal="Compare age across categories")

        self.assertEqual(actions[0]["tool"], "group_comparison_analysis")
        self.assertEqual(actions[0]["args"], {"group_col": "category", "value_col": "age"})

    def test_explicit_identifier_group_is_allowed_with_warning_reason(self):
        df = pd.DataFrame(
            {
                "employee_id": ["E0001", "E0002", "E0003"],
                "department": ["Sales", "Finance", "Sales"],
                "salary": [90, 75, 95],
            }
        )

        actions = recommend_actions(df, goal="Compare salary by employee_id")

        self.assertEqual(actions[0]["tool"], "group_comparison_analysis")
        self.assertEqual(actions[0]["args"], {"group_col": "employee_id", "value_col": "salary"})
        self.assertIn("identifier-like", actions[0]["reason"])

    def test_fallback_overview_prioritizes_distribution_before_missingness(self):
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55],
                "salary": [90, 75, 95, 110],
                "bonus": [5, 4, 8, 10],
                "department": ["Sales", "Finance", "Sales", "HR"],
            }
        )

        actions = recommend_actions(df, max_actions=6, goal="Analyze this dataset")
        names = [action["tool"] for action in actions]

        self.assertEqual(names[:3], ["distribution_analysis", "distribution_analysis", "distribution_analysis"])
        self.assertEqual([action["args"]["column"] for action in actions[:3]], ["salary", "bonus", "age"])
        self.assertIn("global_relationship_analysis", names)
        self.assertIn("missingness_analysis", names)
        if "group_comparison_analysis" in names:
            self.assertLess(names.index("missingness_analysis"), names.index("group_comparison_analysis"))


if __name__ == "__main__":
    unittest.main()
