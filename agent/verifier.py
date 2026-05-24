"""Validate proposed KAIROS tool actions before execution."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.tool_registry import TOOL_REGISTRY


OPTIONAL_ARGS = {
    "target_relationship_analysis": {"top_n"},
    "global_relationship_analysis": {"cols", "top_n"},
    "outlier_analysis": {"method"},
    "numeric_summary": {"columns"},
    "categorical_summary": {"columns", "top_n"},
    "correlation_analysis": {"columns"},
    "numeric_distribution_plot": {"bins"},
    "scatter_plot": {"max_points"},
    "top_correlation_plots": {"cols", "top_n", "max_points"},
    "group_mean_bar_chart": {"top_n"},
    "missing_value_bar_chart": {"include_zero"},
    "regression_plot": {"max_points"},
    "outlier_detection": {"method"},
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

    if "cols" in args and not _is_list_of_strings(args["cols"]):
        errors.append("Arg cols must be a list of strings")

    for int_arg in ("top_n", "bins", "max_points"):
        if int_arg in args and not isinstance(args[int_arg], int):
            errors.append(f"Arg {int_arg} must be an integer")

    if "include_zero" in args and not isinstance(args["include_zero"], bool):
        errors.append("Arg include_zero must be a boolean")

    if "method" in args and not isinstance(args["method"], str):
        errors.append("Arg method must be a string")


def _validate_tool_semantics(
    df: pd.DataFrame,
    tool: str,
    args: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if tool == "missing_analysis":
        return

    if tool == "missingness_analysis":
        return

    if tool == "distribution_analysis":
        column = args.get("column")
        _validate_column_exists(df, column, "column", errors)
        if column in df.columns and not _is_numeric_or_coercible(df, column):
            errors.append(f"column must be numeric: {column}")
        return

    if tool == "relationship_analysis":
        x_col = args.get("x_col")
        y_col = args.get("y_col")
        _validate_column_exists(df, x_col, "x_col", errors)
        _validate_column_exists(df, y_col, "y_col", errors)
        if x_col in df.columns and not _is_numeric_or_coercible(df, x_col):
            errors.append(f"x_col must be numeric: {x_col}")
        if y_col in df.columns and not _is_numeric_or_coercible(df, y_col):
            errors.append(f"y_col must be numeric: {y_col}")
        return

    if tool == "target_relationship_analysis":
        target_col = args.get("target_col")
        _validate_column_exists(df, target_col, "target_col", errors)
        return

    if tool == "global_relationship_analysis":
        columns = args.get("cols")
        target_columns = columns if columns is not None else _numeric_columns(df)
        _validate_column_list_exists(df, target_columns, errors)
        for column in target_columns:
            if column in df.columns and not _is_numeric_or_coercible(df, column):
                errors.append(f"Column must be numeric for global_relationship_analysis: {column}")
        if len([column for column in target_columns if column in df.columns]) < 2:
            errors.append("global_relationship_analysis requires at least two numeric columns")
        return

    if tool == "group_comparison_analysis":
        group_col = args.get("group_col")
        value_col = args.get("value_col")
        _validate_column_exists(df, group_col, "group_col", errors)
        _validate_column_exists(df, value_col, "value_col", errors)
        if value_col in df.columns and not _is_numeric_or_coercible(df, value_col):
            errors.append(f"value_col must be numeric: {value_col}")
        return

    if tool == "outlier_analysis":
        column = args.get("column")
        _validate_column_exists(df, column, "column", errors)
        if column in df.columns and not _is_numeric_or_coercible(df, column):
            errors.append(f"column must be numeric: {column}")
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
        return

    if tool == "anova_by_group":
        group_col = args.get("group_col")
        value_col = args.get("value_col")
        _validate_column_exists(df, group_col, "group_col", errors)
        _validate_column_exists(df, value_col, "value_col", errors)
        if value_col in df.columns and not _is_numeric(df, value_col):
            errors.append(f"value_col must be numeric: {value_col}")
        if group_col in df.columns and _non_missing_unique_count(df[group_col]) < 3:
            errors.append(f"group_col must contain at least three non-missing groups: {group_col}")
        return

    if tool == "numeric_distribution_plot":
        column = args.get("column")
        _validate_column_exists(df, column, "column", errors)
        if column in df.columns and not _is_numeric_or_coercible(df, column):
            errors.append(f"column must be numeric: {column}")
        return

    if tool in {"scatter_plot", "regression_plot"}:
        x_col = args.get("x_col")
        y_col = args.get("y_col")
        _validate_column_exists(df, x_col, "x_col", errors)
        _validate_column_exists(df, y_col, "y_col", errors)
        if x_col in df.columns and not _is_numeric_or_coercible(df, x_col):
            errors.append(f"x_col must be numeric: {x_col}")
        if y_col in df.columns and not _is_numeric_or_coercible(df, y_col):
            errors.append(f"y_col must be numeric: {y_col}")
        return

    if tool == "top_correlation_plots":
        columns = args.get("cols")
        target_columns = columns if columns is not None else _numeric_columns(df)
        _validate_column_list_exists(df, target_columns, errors)
        for column in target_columns:
            if column in df.columns and not _is_numeric_or_coercible(df, column):
                errors.append(f"Column must be numeric for top_correlation_plots: {column}")
        if len([column for column in target_columns if column in df.columns]) < 2:
            errors.append("top_correlation_plots requires at least two numeric columns")
        return

    if tool == "group_mean_bar_chart":
        group_col = args.get("group_col")
        value_col = args.get("value_col")
        _validate_column_exists(df, group_col, "group_col", errors)
        _validate_column_exists(df, value_col, "value_col", errors)
        if value_col in df.columns and not _is_numeric_or_coercible(df, value_col):
            errors.append(f"value_col must be numeric: {value_col}")
        return

    if tool == "missing_value_bar_chart":
        return

    if tool == "outlier_detection":
        column = args.get("column")
        _validate_column_exists(df, column, "column", errors)
        if column in df.columns and not _is_numeric_or_coercible(df, column):
            errors.append(f"column must be numeric: {column}")
        return


def _column_arg_names(tool: str) -> set[str]:
    return {
        name
        for name in TOOL_REGISTRY[tool]["args"].keys()
        if name.endswith("_col") or name in {"column", "col_a", "col_b", "feature_col", "target_col"}
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


def _is_numeric_or_coercible(df: pd.DataFrame, column: str) -> bool:
    if _is_numeric(df, column):
        return True
    series = df[column]
    coerced = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    return non_missing > 0 and int(coerced.notna().sum()) / non_missing >= 0.9


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if _is_numeric(df, column)]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if not _is_numeric(df, column)]


def _non_missing_unique_count(series: pd.Series) -> int:
    return int(series.dropna().nunique(dropna=True))
