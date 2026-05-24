"""Guarded deterministic tool execution for KAIROS."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.tool_registry import TOOL_REGISTRY
from agent.verifier import verify_action


def execute_action(df: pd.DataFrame, action: dict[str, Any] | Any) -> dict[str, Any]:
    """Verify and execute one proposed tool action."""
    verification = verify_action(df, action)
    response = {
        "executed": False,
        "verification": verification,
        "tool": verification.get("tool"),
        "args": verification.get("args", {}),
        "result": None,
        "errors": list(verification.get("errors", [])),
        "warnings": list(verification.get("warnings", [])),
    }

    if not verification.get("valid", False):
        return response

    tool_name = verification["tool"]
    args = verification["args"]
    tool_spec = TOOL_REGISTRY.get(tool_name)
    if tool_spec is None:
        response["errors"].append(f"Tool disappeared from registry before execution: {tool_name}")
        return response

    try:
        result = tool_spec["function"](df, **args)
    except Exception as exc:
        response["errors"].append(f"Runtime error while executing {tool_name}: {exc}")
        return response

    response["executed"] = True
    response["result"] = result
    if isinstance(result, dict):
        response["warnings"].extend(result.get("warnings", []))
    return response
