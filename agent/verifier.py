"""Validate proposed KAIROS tool actions before execution."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.tool_registry import TOOL_REGISTRY


OPTIONAL_ARGS = {
    "numeric_summary": {"columns"},
    "categorical_summary": {"columns", "top_n"},
    "correlation_analysis": {"columns"},
}

REQUIRED_ARGS = {
    name: set(spec["args"].keys()) - OPTIONAL_ARGS.get(name, set())
    for name, spec in TOOL_REGISTRY.items()
}


def verify_action(df: pd.DataFrame, action: dict[str, Any] | Any) -> dict[str, Any]:
    """Return a structured validation result for a proposed tool action."""
    result = {"valid": False, "tool": None, "args": {}, "errors": [], "warnings": []}

    if not isinstance(action, dict):
        result["errors"].append("Action proposal must be a dictionary")
        return result

    tool = action.get("tool")
    args = action.get("args", {})
    result["tool"] = tool if isinstance(tool, str) else None
    result["args"] = args if isinstance(args, dict) else {}

    if not isinstance(tool, str) or not tool:
        result["errors"].append("Action tool must be a non-empty string")
        return result
    if tool not in TOOL_REGISTRY:
        result["errors"].append(f"Unknown tool: {tool}")
        return result
    if not isinstance(args, dict):
        result["errors"].append("Action args must be a dictionary")
        return result

    _validate_required_args(tool, args, result["errors"])
    _validate_known_args(tool, args, result["warnings"])
    _validate_arg_types(tool, args, result["errors"])

    if not result["errors"]:
        _validate_tool_semantics(df, tool, args, result["errors"], result["warnings"])

    result["valid"] = not result["errors"]
    return result


def _validate_required_args(tool: str, args: dict[str, Any], errors: list[str]) -> None:
    for arg_name in sorted(REQUIRED_ARGS.get(tool, set())):
        if arg_name not in args:
            errors.append(f"Missing required arg: {arg_name}")


def _validate_known_args(tool: str, args: dict[str, Any], warnings: list[str]) -> None:
    allowed = set(TOOL_REGISTRY[tool]["args"].keys())
    for arg_name in sorted(args.keys()):
        if arg_name not in allowed:
            warnings.append(f"Unknown arg for {tool}: {arg_name}")


def _validate_arg_types(tool: str, args: dict[str, Any], errors: list[str]) -> None:
    for arg_name in _column_arg_names(tool):
        if arg_name in args and not isinstance(args[arg_name], str):
            errors.append(f"Arg {arg_name} must be a string")

    if "columns" in args and not _is_list_of_strings(args["columns"]):
        errors.append("Arg columns must be a list of strings")

    if "top_n" in args and not isinstance(args["top_n"], int):
        errors.append("Arg top_n must be an integer")


def _validate_tool_semantics(
    df: pd.DataFrame,
    tool: str,
    args: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if tool == "missing_analysis":
        return

    if tool in {"numeric_summary", "correlation_analysis"}:
        columns = args.get("columns")
        target_columns = columns if columns is not None else _numeric_columns(df)
        _validate_column_list_exists(df, target_columns, errors)
        for column in target_columns:
            if column in df.columns and not _is_numeric(df, column):
                errors.append(f"Column must be numeric for {tool}: {column}")
        if tool == "correlation_analysis" and len([c for c in target_columns if c in df.columns]) < 2:
            errors.append("correlation_analysis requires at least two numeric columns")
        return

    if tool == "categorical_summary":
        columns = args.get("columns")
        target_columns = columns if columns is not None else _categorical_columns(df)
        _validate_column_list_exists(df, target_columns, errors)
        for column in target_columns:
            if column in df.columns and _is_numeric(df, column):
                errors.append(f"Column must be categorical for categorical_summary: {column}")
        return

    if tool == "group_summary":
        group_col = args.get("group_col")
        value_col = args.get("value_col")
        _validate_column_exists(df, group_col, "group_col", errors)
        _validate_column_exists(df, value_col, "value_col", errors)
        if value_col in df.columns and not _is_numeric(df, value_col):
            errors.append(f"value_col must be numeric: {value_col}")
        return

    if tool == "target_group_summary":
        target_col = args.get("target_col")
        _validate_column_exists(df, target_col, "target_col", errors)
        if target_col in df.columns and _is_numeric(df, target_col):
            warnings.append(f"target_col is numeric and may not be categorical: {target_col}")
        return

    if tool == "simple_linear_regression":
        target_col = args.get("target_col")
        feature_col = args.get("feature_col")
        _validate_column_exists(df, target_col, "target_col", errors)
        _validate_column_exists(df, feature_col, "feature_col", errors)
        if target_col in df.columns and not _is_numeric(df, target_col):
            errors.append(f"target_col must be numeric: {target_col}")
        if feature_col in df.columns and not _is_numeric(df, feature_col):
            errors.append(f"feature_col must be numeric: {feature_col}")
        return

    if tool == "chi_square_test":
        for arg_name in ("col_a", "col_b"):
            column = args.get(arg_name)
            _validate_column_exists(df, column, arg_name, errors)
            if column in df.columns and _is_numeric(df, column):
                errors.append(f"Column must be categorical for chi_square_test: {column}")
        return

    if tool == "t_test_by_group":
        group_col = args.get("group_col")
        value_col = args.get("value_col")
        _validate_column_exists(df, group_col, "group_col", errors)
        _validate_column_exists(df, value_col, "value_col", errors)
        if value_col in df.columns and not _is_numeric(df, value_col):
            errors.append(f"value_col must be numeric: {value_col}")
        if group_col in df.columns and _non_missing_unique_count(df[group_col]) != 2:
            errors.append(f"group_col must contain exactly two non-missing groups: {group_col}")


def _column_arg_names(tool: str) -> set[str]:
    return {
        name
        for name in TOOL_REGISTRY[tool]["args"].keys()
        if name.endswith("_col") or name in {"col_a", "col_b", "feature_col", "target_col"}
    }


def _validate_column_list_exists(
    df: pd.DataFrame, columns: list[str], errors: list[str]
) -> None:
    for column in columns:
        if column not in df.columns:
            errors.append(f"Column not found: {column}")


def _validate_column_exists(
    df: pd.DataFrame, column: Any, arg_name: str, errors: list[str]
) -> None:
    if isinstance(column, str) and column not in df.columns:
        errors.append(f"Column not found for {arg_name}: {column}")


def _is_list_of_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_numeric(df: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[column])


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if _is_numeric(df, column)]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if not _is_numeric(df, column)]


def _non_missing_unique_count(series: pd.Series) -> int:
    return int(series.dropna().nunique(dropna=True))
