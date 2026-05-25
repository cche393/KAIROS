"""Lightweight deterministic dataset profiling for KAIROS."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


BOOLEAN_STRINGS = {"true", "false", "yes", "no", "y", "n", "0", "1"}
DATETIME_NAME_HINTS = ("date", "time", "timestamp", "joined", "created", "updated")
ID_NAME_TOKENS = ("id", "uuid", "key", "code", "serial", "number", "no")


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact reusable profile for planning and UI explanation."""
    row_count = int(len(df))
    column_names = [str(column) for column in df.columns]
    column_profiles = {}
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "datetime_like": [],
        "boolean": [],
        "text_like": [],
    }
    numeric_statistics = {}
    categorical_statistics = {}
    constant_columns = []
    likely_id_columns = []
    high_cardinality_columns = []

    for column in df.columns:
        name = str(column)
        series = df[column]
        non_missing = series.dropna()
        missing_count = int(series.isna().sum())
        unique_count = int(non_missing.nunique(dropna=True))
        inferred_type = _infer_column_type(name, series)

        if inferred_type == "datetime":
            column_types["datetime"].append(name)
            column_types["datetime_like"].append(name)
        else:
            column_types[inferred_type].append(name)

        missing_percent = _percent(missing_count, row_count)
        is_constant = bool(row_count > 0 and unique_count <= 1)
        is_likely_id = _is_likely_id(name, series)
        is_high_cardinality = _is_high_cardinality(name, series, inferred_type)

        if is_constant:
            constant_columns.append(name)
        if is_likely_id:
            likely_id_columns.append(name)
        if is_high_cardinality:
            high_cardinality_columns.append(name)

        profile = {
            "name": name,
            "type": inferred_type,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "is_constant": is_constant,
            "is_likely_id": is_likely_id,
            "is_high_cardinality": is_high_cardinality,
        }
        column_profiles[name] = profile

        if inferred_type == "numeric":
            numeric_statistics[name] = _numeric_stats(series)
        elif inferred_type in {"categorical", "boolean", "text_like"}:
            categorical_statistics[name] = _categorical_stats(series)

    missing_values = _missing_values(df)
    profile = {
        "row_count": row_count,
        "column_count": int(len(df.columns)),
        "shape": {"rows": row_count, "columns": int(len(df.columns))},
        "columns": column_names,
        "column_types": column_types,
        "missing_values": missing_values,
        "potential_issues": {
            "constant_value_columns": constant_columns,
            "likely_id_columns": likely_id_columns,
            "high_cardinality_categorical_columns": high_cardinality_columns,
        },
        "quality_notes": _quality_notes(missing_values, constant_columns, likely_id_columns, high_cardinality_columns),
        "numeric_statistics": numeric_statistics,
        "categorical_statistics": categorical_statistics,
        "column_profiles": column_profiles,
        "duplicate_rows": int(df.duplicated().sum()) if len(df.columns) else 0,
        "sample_rows": _sample_rows(df),
    }
    return profile


def dataset_overview(
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a concise structure/schema overview using an existing profile when available."""
    profile = dataset_profile if isinstance(dataset_profile, dict) else build_dataset_profile(df)
    issues = profile.get("potential_issues", {}) if isinstance(profile.get("potential_issues"), dict) else {}
    return {
        "analysis_type": "dataset_overview",
        "title": "Dataset overview",
        "summary": (
            f"This dataset has {profile.get('row_count', profile.get('shape', {}).get('rows', 0))} rows "
            f"and {profile.get('column_count', profile.get('shape', {}).get('columns', 0))} columns."
        ),
        "row_count": int(profile.get("row_count", profile.get("shape", {}).get("rows", 0)) or 0),
        "column_count": int(profile.get("column_count", profile.get("shape", {}).get("columns", 0)) or 0),
        "columns": list(profile.get("columns", [])),
        "column_types": profile.get("column_types", {}),
        "missing_values": profile.get("missing_values", {}),
        "potential_issues": {
            "likely_id_columns": list(issues.get("likely_id_columns", [])),
            "constant_value_columns": list(issues.get("constant_value_columns", [])),
            "high_cardinality_categorical_columns": list(issues.get("high_cardinality_categorical_columns", [])),
        },
        "quality_notes": list(profile.get("quality_notes", [])),
        "table": _column_type_rows(profile.get("column_types", {})),
        "warnings": [],
        "method_note": "Dataset overview reports schema, inferred types, missingness, and structural quality hints from the reusable dataset profile.",
    }


def _column_type_rows(column_types: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    seen = set()
    type_order = ["numeric", "categorical", "datetime", "boolean", "text_like"]
    for type_name in type_order:
        values = column_types.get(type_name, [])
        if not isinstance(values, list):
            continue
        for column in values:
            column_name = str(column)
            if column_name in seen:
                continue
            seen.add(column_name)
            rows.append({"column": column_name, "type": type_name})
    return rows


def _infer_column_type(name: str, series: pd.Series) -> str:
    non_missing = series.dropna()
    if _is_boolean_series(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _is_datetime_like(name, non_missing):
        return "datetime"
    if _is_categorical_like(non_missing, len(series)):
        return "categorical"
    return "text_like"


def _missing_values(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    total_cells = int(row_count * len(df.columns))
    columns = {}
    high_missingness_columns = []
    for column in df.columns:
        name = str(column)
        missing_count = int(df[column].isna().sum())
        missing_percent = _percent(missing_count, row_count)
        columns[name] = {"missing_count": missing_count, "missing_percent": missing_percent}
        if row_count > 0 and missing_percent >= 50.0:
            high_missingness_columns.append(name)
    return {
        "total_missing_cells": int(df.isna().sum().sum()) if total_cells else 0,
        "total_cells": total_cells,
        "columns": columns,
        "high_missingness_columns": high_missingness_columns,
    }


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "count": int(numeric.count()),
        "mean": _safe_number(numeric.mean()),
        "std": _safe_number(numeric.std()),
        "min": _safe_number(numeric.min()),
        "max": _safe_number(numeric.max()),
    }


def _categorical_stats(series: pd.Series, top_n: int = 5) -> dict[str, Any]:
    non_missing = series.dropna()
    counts = series.fillna("<missing>").astype(str).value_counts(dropna=False).head(max(int(top_n), 1))
    return {
        "unique_count": int(non_missing.nunique(dropna=True)),
        "top_values": [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ],
    }


def _quality_notes(
    missing_values: dict[str, Any],
    constant_columns: list[str],
    likely_id_columns: list[str],
    high_cardinality_columns: list[str],
) -> list[str]:
    notes = []
    for column, values in missing_values.get("columns", {}).items():
        percent = values.get("missing_percent", 0)
        if percent:
            notes.append(f"Missing values in {column} ({percent}%).")
    for column in constant_columns:
        notes.append(f"{column} is constant and is unlikely to help relationship analysis.")
    for column in likely_id_columns:
        notes.append(f"{column} appears to be an identifier and is excluded from relationship planning by default.")
    for column in high_cardinality_columns:
        notes.append(f"{column} has high cardinality and may be poor for grouped charts.")
    return notes


def _is_boolean_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    values = {str(value).strip().lower() for value in non_missing.unique()}
    return 0 < len(values) <= 2 and values.issubset(BOOLEAN_STRINGS)


def _is_datetime_like(name: str, non_missing: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(non_missing):
        return True
    if non_missing.empty:
        return False
    values = non_missing.astype(str)
    has_signal = any(hint in name.lower() for hint in DATETIME_NAME_HINTS) or values.str.contains(r"[-/:]", regex=True).mean() >= 0.6
    if not has_signal:
        return False
    parsed = pd.to_datetime(values, errors="coerce")
    return float(parsed.notna().mean()) >= 0.6


def _is_categorical_like(non_missing: pd.Series, row_count: int) -> bool:
    if non_missing.empty:
        return True
    unique_count = int(non_missing.nunique(dropna=True))
    if unique_count <= 1:
        return True
    if _has_mixed_python_types(non_missing):
        return False
    text_values = non_missing.astype(str)
    average_length = float(text_values.str.len().mean())
    average_words = float(text_values.str.split().str.len().mean())
    unique_ratio = unique_count / max(row_count, 1)
    if average_length > 30 or average_words > 4:
        return False
    return unique_count <= 20 or unique_ratio <= 0.5


def _is_likely_id(name: str, series: pd.Series) -> bool:
    normalised = _normalise_name(name)
    name_says_id = normalised == "id" or normalised.endswith("_id") or any(token in normalised.split("_") for token in ID_NAME_TOKENS)
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    unique_ratio = int(non_missing.nunique(dropna=True)) / max(len(non_missing), 1)
    code_like = non_missing.astype(str).head(20).str.match(r"^[A-Za-z]{1,6}[-_]?\d{2,}$").mean()
    return bool((name_says_id and unique_ratio >= 0.8) or code_like > 0.8)


def _is_high_cardinality(name: str, series: pd.Series, inferred_type: str) -> bool:
    if inferred_type not in {"categorical", "text_like"}:
        return False
    if _is_likely_id(name, series):
        return False
    non_missing = series.dropna()
    if len(non_missing) < 4:
        return False
    unique_count = int(non_missing.nunique(dropna=True))
    unique_ratio = unique_count / max(len(non_missing), 1)
    return bool(unique_count > 20 or (unique_count >= 4 and unique_ratio >= 0.8))


def _has_mixed_python_types(series: pd.Series) -> bool:
    return len({type(value).__name__ for value in series}) > 1


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


def _safe_number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, int):
        return int(value)
    return value


def _percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100, 2)


def _normalise_name(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", value.lower()))
