"""Fixed deterministic tool registry for the future KAIROS planner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tools.eda_tools import (
    categorical_summary,
    chi_square_test,
    correlation_analysis,
    group_summary,
    missing_analysis,
    numeric_summary,
    simple_linear_regression,
    t_test_by_group,
    target_group_summary,
)


TOOL_REGISTRY = {
    "missing_analysis": {
        "description": "Summarize missing counts, missing percentages, and high-missingness columns.",
        "args": {},
        "function": missing_analysis,
    },
    "numeric_summary": {
        "description": "Summarize numeric columns with count, mean, standard deviation, quartiles, range, and skewness.",
        "args": {"columns": "optional list of numeric column names"},
        "function": numeric_summary,
    },
    "categorical_summary": {
        "description": "Summarize categorical columns with value counts, proportions, and unique counts.",
        "args": {
            "columns": "optional list of categorical column names",
            "top_n": "maximum number of values to return per column",
        },
        "function": categorical_summary,
    },
    "correlation_analysis": {
        "description": "Compute numeric Pearson correlations and strongest positive/negative column pairs.",
        "args": {"columns": "optional list of numeric column names"},
        "function": correlation_analysis,
    },
    "group_summary": {
        "description": "Compare a numeric value column across categories of a grouping column.",
        "args": {"group_col": "categorical grouping column", "value_col": "numeric value column"},
        "function": group_summary,
    },
    "target_group_summary": {
        "description": "Summarize class balance and numeric columns by a categorical target column.",
        "args": {"target_col": "categorical or binary target column"},
        "function": target_group_summary,
    },
    "simple_linear_regression": {
        "description": "Fit a one-feature linear regression using deterministic pandas arithmetic.",
        "args": {"target_col": "numeric target column", "feature_col": "numeric feature column"},
        "function": simple_linear_regression,
    },
    "chi_square_test": {
        "description": "Compute a chi-square statistic for two categorical columns. P-value is unavailable without scipy.",
        "args": {"col_a": "first categorical column", "col_b": "second categorical column"},
        "function": chi_square_test,
    },
    "t_test_by_group": {
        "description": "Compute a two-group t-statistic for a numeric value column. P-value is unavailable without scipy.",
        "args": {"group_col": "binary categorical group column", "value_col": "numeric value column"},
        "function": t_test_by_group,
    },
}


def list_available_tools() -> list[dict[str, Any]]:
    """Return planner-safe tool specs without callable objects."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "args": spec["args"],
        }
        for name, spec in TOOL_REGISTRY.items()
    ]


def get_tool_spec(tool_name: str) -> dict[str, Any] | None:
    """Return one planner-safe tool spec by name."""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return None
    return {"name": tool_name, "description": spec["description"], "args": spec["args"]}


def run_tool(tool_name: str, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Run an allowed deterministic tool and return a structured result."""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return {
            "tool": tool_name,
            "status": "error",
            "result": None,
            "warnings": [f"Tool {tool_name} is not allowed"],
        }

    try:
        result = spec["function"](df, **kwargs)
    except TypeError as exc:
        return {
            "tool": tool_name,
            "status": "error",
            "result": None,
            "warnings": [f"Invalid arguments for {tool_name}: {exc}"],
        }

    return {
        "tool": tool_name,
        "status": "ok",
        "result": result,
        "warnings": result.get("warnings", []) if isinstance(result, dict) else [],
    }
