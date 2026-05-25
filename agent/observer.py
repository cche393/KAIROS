"""Deterministic dataset inspection for KAIROS."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from tools.dataset_profile import build_dataset_profile


KNOWN_TARGET_NAMES = {
    "target",
    "label",
    "class",
    "churn",
    "survived",
    "status",
    "diagnosis",
    "outcome",
    "result",
    "default",
    "fraud",
    "is_fraud",
    "response",
}

BOOLEAN_STRINGS = {"true", "false", "yes", "no", "y", "n", "0", "1"}
DATETIME_NAME_HINTS = ("date", "time", "timestamp", "joined", "created", "updated")


def load_csv(file: str | Path | Any) -> pd.DataFrame:
    """Load a CSV into a DataFrame, returning an empty DataFrame for empty files."""
    try:
        return pd.read_csv(file)
    except EmptyDataError:
        return pd.DataFrame()
    except (ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not load CSV safely: {exc}") from exc


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into compact type groups useful for planning."""
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime_like": [],
        "boolean": [],
        "text_like": [],
    }

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        if _is_boolean_series(series):
            column_types["boolean"].append(str(column))
        elif pd.api.types.is_numeric_dtype(series):
            column_types["numeric"].append(str(column))
        elif _is_datetime_like(column, non_null):
            column_types["datetime_like"].append(str(column))
        elif _is_categorical_like(non_null, len(df)):
            column_types["categorical"].append(str(column))
        else:
            column_types["text_like"].append(str(column))

    return column_types


def summarize_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Return per-column missing counts and percentages."""
    row_count = int(len(df))
    total_cells = int(row_count * len(df.columns))
    columns: dict[str, dict[str, float | int]] = {}
    high_missingness_columns: list[str] = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percent = _round_percent(missing_count, row_count)
        column_name = str(column)
        columns[column_name] = {
            "missing_count": missing_count,
            "missing_percent": missing_percent,
        }
        if missing_percent >= 50.0 and row_count > 0:
            high_missingness_columns.append(column_name)

    return {
        "total_missing_cells": int(df.isna().sum().sum()) if total_cells else 0,
        "total_cells": total_cells,
        "columns": columns,
        "high_missingness_columns": high_missingness_columns,
    }


def suggest_target_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Suggest likely target columns using names and simple cardinality signals."""
    suggestions: list[dict[str, Any]] = []

    for position, column in enumerate(df.columns):
        column_name = str(column)
        normalized = _normalize_column_name(column_name)
        non_null = df[column].dropna()
        unique_count = int(non_null.nunique(dropna=True))
        score = 0
        reason = ""

        if normalized in KNOWN_TARGET_NAMES:
            score = 100
            reason = "known target-like name"
        elif normalized.startswith(("is_", "has_")) and 1 < unique_count <= 2:
            score = 80
            reason = "binary indicator name"
        elif 1 < unique_count <= 2 and not _looks_like_identifier(normalized):
            score = 50
            reason = "binary column"

        if score:
            suggestions.append(
                {
                    "column": column_name,
                    "reason": reason,
                    "unique_values": unique_count,
                    "missing_percent": _round_percent(int(df[column].isna().sum()), len(df)),
                    "_score": score,
                    "_position": position,
                }
            )

    suggestions.sort(key=lambda item: (-item["_score"], item["_position"]))
    for item in suggestions:
        del item["_score"]
        del item["_position"]
    return suggestions[:5]


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact JSON-serializable summary of a DataFrame."""
    profile = build_dataset_profile(df)
    profile["suggested_target_columns"] = suggest_target_columns(df)
    return profile


def _is_boolean_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    values = {str(value).strip().lower() for value in non_null.unique()}
    return 0 < len(values) <= 2 and values.issubset(BOOLEAN_STRINGS)


def _is_datetime_like(column: Any, non_null: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return True
    if non_null.empty:
        return False

    column_name = str(column).lower()
    values = non_null.astype(str)
    has_date_signal = (
        any(hint in column_name for hint in DATETIME_NAME_HINTS)
        or values.str.contains(r"[-/:]", regex=True).mean() >= 0.6
    )
    if not has_date_signal:
        return False

    parsed = pd.to_datetime(values, errors="coerce")
    return float(parsed.notna().mean()) >= 0.6


def _is_categorical_like(non_null: pd.Series, row_count: int) -> bool:
    if non_null.empty:
        return True

    unique_count = int(non_null.nunique(dropna=True))
    if unique_count <= 1:
        return True

    if _has_mixed_python_types(non_null):
        return False

    text_values = non_null.astype(str)
    average_length = float(text_values.str.len().mean())
    average_words = float(text_values.str.split().str.len().mean())
    unique_ratio = unique_count / max(row_count, 1)

    if average_length > 30 or average_words > 4:
        return False
    return unique_count <= 20 or unique_ratio <= 0.5


def _has_mixed_python_types(series: pd.Series) -> bool:
    observed_types = {type(value).__name__ for value in series}
    return len(observed_types) > 1


def _sample_rows(df: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return [
        {str(column): _json_safe_value(value) for column, value in row.items()}
        for row in df.head(limit).to_dict(orient="records")
    ]


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _round_percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100, 2)


def _normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_").replace("-", "_")


def _looks_like_identifier(normalized_column_name: str) -> bool:
    return normalized_column_name == "id" or normalized_column_name.endswith("_id")
