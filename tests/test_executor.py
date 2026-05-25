import json
import unittest

import pandas as pd

import agent.executor as executor
from agent.executor import execute_action
from agent.tool_registry import TOOL_REGISTRY
from tools.dataset_profile import build_dataset_profile


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "age": [21, 35, 42, 58],
                "income": [40000, 62000, 80000, 91000],
                "segment": ["A", "A", "B", "B"],
                "region": ["north", "south", "north", "south"],
            }
        )

    def test_valid_action_executes_successfully(self):
        response = execute_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": ["age", "income"]}},
        )

        self.assertTrue(response["executed"])
        self.assertTrue(response["verification"]["valid"])
        self.assertEqual(response["tool"], "numeric_summary")
        self.assertEqual(response["errors"], [])
        self.assertEqual(response["result"]["columns"]["age"]["mean"], 39.0)
        json.dumps(response)

    def test_invalid_action_is_blocked_before_execution(self):
        response = execute_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": ["segment"]}},
        )

        self.assertFalse(response["executed"])
        self.assertFalse(response["verification"]["valid"])
        self.assertIsNone(response["result"])
        self.assertIn("Column must be numeric for numeric_summary: segment", response["errors"])

    def test_unknown_tool_is_blocked(self):
        response = execute_action(self.df, {"tool": "unknown_tool", "args": {}})

        self.assertFalse(response["executed"])
        self.assertIsNone(response["result"])
        self.assertIn("Unknown tool: unknown_tool", response["errors"])

    def test_missing_column_is_blocked(self):
        response = execute_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": ["missing"]}},
        )

        self.assertFalse(response["executed"])
        self.assertIn("Column not found: missing", response["errors"])

    def test_invalid_arg_type_is_blocked(self):
        response = execute_action(
            self.df,
            {"tool": "numeric_summary", "args": {"columns": "age"}},
        )

        self.assertFalse(response["executed"])
        self.assertIn("Arg columns must be a list of strings", response["errors"])

    def test_runtime_tool_exception_is_safely_handled(self):
        def exploding_tool(_df, **_kwargs):
            raise RuntimeError("boom")

        original = TOOL_REGISTRY.get("explode")
        TOOL_REGISTRY["explode"] = {
            "description": "Intentional test tool",
            "args": {},
            "function": exploding_tool,
        }
        try:
            response = execute_action(self.df, {"tool": "explode", "args": {}})
        finally:
            if original is None:
                del TOOL_REGISTRY["explode"]
            else:
                TOOL_REGISTRY["explode"] = original

        self.assertFalse(response["executed"])
        self.assertTrue(response["verification"]["valid"])
        self.assertIsNone(response["result"])
        self.assertIn("Runtime error while executing explode: boom", response["errors"])

    def test_structured_response_format_is_stable(self):
        response = execute_action(self.df, ["bad", "shape"])

        self.assertEqual(
            set(response.keys()),
            {"executed", "verification", "tool", "args", "result", "errors", "warnings"},
        )
        self.assertFalse(response["executed"])
        self.assertIsNone(response["tool"])
        self.assertEqual(response["args"], {})
        json.dumps(response)

    def test_executor_reuses_verifier(self):
        calls = []

        def fake_verify(_df, action):
            calls.append(action)
            return {
                "valid": False,
                "tool": "numeric_summary",
                "args": {},
                "errors": ["blocked by fake verifier"],
                "warnings": [],
            }

        original = executor.verify_action
        executor.verify_action = fake_verify
        try:
            response = execute_action(self.df, {"tool": "numeric_summary", "args": {}})
        finally:
            executor.verify_action = original

        self.assertEqual(len(calls), 1)
        self.assertFalse(response["executed"])
        self.assertIn("blocked by fake verifier", response["errors"])

    def test_executor_dispatches_through_tool_registry(self):
        calls = []

        def fake_tool(_df, columns=None):
            calls.append(columns)
            return {"ok": True, "warnings": ["from fake tool"]}

        original = TOOL_REGISTRY["numeric_summary"]["function"]
        TOOL_REGISTRY["numeric_summary"]["function"] = fake_tool
        try:
            response = execute_action(
                self.df,
                {"tool": "numeric_summary", "args": {"columns": ["age"]}},
            )
        finally:
            TOOL_REGISTRY["numeric_summary"]["function"] = original

        self.assertTrue(response["executed"])
        self.assertEqual(calls, [["age"]])
        self.assertEqual(response["result"], {"ok": True, "warnings": ["from fake tool"]})
        self.assertEqual(response["warnings"], ["from fake tool"])

    def test_executor_passes_profile_to_verifier_when_available(self):
        df = pd.DataFrame(
            {
                "employee_id": [1, 2, 3, 4],
                "salary": [70, 80, 90, 100],
            }
        )
        profile = build_dataset_profile(df)

        response = execute_action(
            df,
            {"tool": "global_relationship_analysis", "args": {"cols": ["employee_id", "salary"]}},
            dataset_profile=profile,
        )

        self.assertFalse(response["executed"])
        self.assertIn("employee_id appears to be an identifier", " ".join(response["errors"]))

    def test_executor_passes_existing_profile_to_dataset_overview_tool(self):
        df = pd.DataFrame({"age": [25], "department": ["Sales"]})
        profile = build_dataset_profile(df)
        profile["row_count"] = 999
        profile["shape"]["rows"] = 999

        response = execute_action(df, {"tool": "dataset_overview", "args": {}}, dataset_profile=profile)

        self.assertTrue(response["executed"])
        self.assertEqual(response["result"]["analysis_type"], "dataset_overview")
        self.assertEqual(response["result"]["row_count"], 999)


if __name__ == "__main__":
    unittest.main()
