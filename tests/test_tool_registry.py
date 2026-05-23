import json
import unittest

import pandas as pd

from agent.tool_registry import get_tool_spec, list_available_tools, run_tool


class ToolRegistryTests(unittest.TestCase):
    def test_registry_lists_allowed_tools_without_callables(self):
        tools = list_available_tools()
        names = {tool["name"] for tool in tools}

        self.assertIn("missing_analysis", names)
        self.assertIn("numeric_summary", names)
        self.assertIn("t_test_by_group", names)
        self.assertNotIn("simple_logistic_regression", names)
        self.assertTrue(all("function" not in tool for tool in tools))
        json.dumps(tools)

    def test_get_tool_spec_returns_safe_spec_or_none(self):
        spec = get_tool_spec("group_summary")

        self.assertEqual(spec["name"], "group_summary")
        self.assertIn("group_col", spec["args"])
        self.assertNotIn("function", spec)
        self.assertIsNone(get_tool_spec("not_allowed"))
        json.dumps(spec)

    def test_run_tool_dispatches_allowed_tool_and_rejects_unknown_tool(self):
        df = pd.DataFrame({"group": ["A", "B"], "value": [1, 2]})

        result = run_tool("numeric_summary", df, columns=["value"])
        self.assertEqual(result["tool"], "numeric_summary")
        self.assertEqual(result["result"]["columns"]["value"]["mean"], 1.5)
        json.dumps(result)

        rejected = run_tool("not_allowed", df)
        self.assertEqual(rejected["status"], "error")
        self.assertIn("not allowed", rejected["warnings"][0])


if __name__ == "__main__":
    unittest.main()
