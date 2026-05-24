"""Deterministic action recommendations for KAIROS."""

from __future__ import annotations

from typing import Any

import pandas as pd


def recommend_actions(df: pd.DataFrame, max_actions: int | None = None) -> list[dict[str, Any]]:
    """Recommend candidate analysis actions from simple DataFrame properties."""
    numeric_columns = _numeric_columns(df)
    categorical_columns = _categorical_columns(df)
    binary_categorical_columns = _binary_categorical_columns(df, categorical_columns)

    recommendations = [
        _action(
            "missing_analysis",
            {},
            "Check missing values before deeper analysis.",
        )
    ]

    if numeric_columns:
        recommendations.append(
            _action(
                "numeric_summary",
                {"columns": numeric_columns},
                "Summarize numeric columns to understand scale and spread.",
            )
        )

    if categorical_columns:
        recommendations.append(
            _action(
                "categorical_summary",
                {"columns": categorical_columns},
                "Summarize categorical distributions and common values.",
            )
        )

    if len(numeric_columns) >= 2:
        recommendations.append(
            _action(
                "correlation_analysis",
                {"columns": numeric_columns},
                "Check pairwise numeric relationships.",
            )
        )

    if categorical_columns and numeric_columns:
        recommendations.append(
            _action(
                "group_summary",
                {"group_col": categorical_columns[0], "value_col": numeric_columns[0]},
                "Compare a numeric column across category groups.",
            )
        )

    if binary_categorical_columns and numeric_columns:
        recommendations.append(
            _action(
                "t_test_by_group",
                {"group_col": binary_categorical_columns[0], "value_col": numeric_columns[0]},
                "Compare a numeric column between two groups.",
            )
        )

    if len(categorical_columns) >= 2:
        recommendations.append(
            _action(
                "chi_square_test",
                {"col_a": categorical_columns[0], "col_b": categorical_columns[1]},
                "Check whether two categorical columns are associated.",
            )
        )

    if len(numeric_columns) >= 2:
        recommendations.append(
            _action(
                "simple_linear_regression",
                {"feature_col": numeric_columns[0], "target_col": numeric_columns[1]},
                "Fit a simple one-feature linear relationship between numeric columns.",
            )
        )

    for priority, recommendation in enumerate(recommendations, start=1):
        recommendation["priority"] = priority

    if max_actions is None:
        return recommendations
    return recommendations[: max(int(max_actions), 0)]


def _action(tool: str, args: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"tool": tool, "args": args, "priority": 0, "reason": reason}


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]


def _binary_categorical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if int(df[column].dropna().nunique(dropna=True)) == 2
    ]
