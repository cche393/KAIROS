"""Safe deterministic EDA and introductory statistics tools."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def missing_analysis(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    total_cells = int(row_count * len(df.columns))
    columns = {}
    high_missingness_columns = []

    for column in df.columns:
        name = str(column)
        count = int(df[column].isna().sum())
        percent = _percent(count, row_count)
        columns[name] = {"missing_count": count, "missing_percent": percent}
        if row_count > 0 and percent >= 50.0:
            high_missingness_columns.append(name)

    return {
        "row_count": row_count,
        "column_count": int(len(df.columns)),
        "total_cells": total_cells,
        "total_missing_cells": int(df.isna().sum().sum()) if total_cells else 0,
        "columns": columns,
        "high_missingness_columns": high_missingness_columns,
        "warnings": [],
    }


def numeric_summary(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    selected, warnings = _select_columns(df, columns)
    numeric_columns = [column for column in selected if pd.api.types.is_numeric_dtype(df[column])]
    for column in selected:
        if column not in numeric_columns:
            warnings.append(f"{column} must be numeric")

    result = {"columns": {}, "warnings": warnings}
    if not numeric_columns:
        result["warnings"].append("No numeric columns available")
        return result

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        result["columns"][str(column)] = {
            "count": int(series.count()),
            "mean": _safe_number(series.mean()),
            "std": _safe_number(series.std()),
            "min": _safe_number(series.min()),
            "q1": _safe_number(series.quantile(0.25)),
            "median": _safe_number(series.median()),
            "q3": _safe_number(series.quantile(0.75)),
            "max": _safe_number(series.max()),
            "skewness": _safe_number(series.skew()),
        }
    return result


def categorical_summary(
    df: pd.DataFrame, columns: list[str] | None = None, top_n: int = 10
) -> dict[str, Any]:
    selected, warnings = _select_columns(df, columns)
    categorical_columns = [
        column for column in selected if not pd.api.types.is_numeric_dtype(df[column])
    ]
    for column in selected:
        if column not in categorical_columns:
            warnings.append(f"{column} must be categorical")

    result = {"columns": {}, "warnings": warnings}
    if not categorical_columns:
        result["warnings"].append("No categorical columns available")
        return result

    limit = max(int(top_n), 1)
    row_count = max(int(len(df)), 1)
    for column in categorical_columns:
        counts = df[column].fillna("<missing>").astype(str).value_counts(dropna=False).head(limit)
        result["columns"][str(column)] = {
            "unique_values": int(df[column].nunique(dropna=True)),
            "top_values": [
                {
                    "value": str(value),
                    "count": int(count),
                    "proportion": _safe_number(count / row_count),
                }
                for value, count in counts.items()
            ],
        }
    return result


def correlation_analysis(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    selected, warnings = _select_columns(df, columns)
    numeric_columns = [column for column in selected if pd.api.types.is_numeric_dtype(df[column])]
    for column in selected:
        if column not in numeric_columns:
            warnings.append(f"{column} must be numeric")

    if len(numeric_columns) < 2:
        warnings.append("At least two numeric columns are required")
        return {
            "method": "pearson",
            "columns": [str(column) for column in numeric_columns],
            "correlation_matrix": {},
            "strongest_positive": [],
            "strongest_negative": [],
            "warnings": warnings,
        }

    corr = df[numeric_columns].corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(numeric_columns):
        for right in numeric_columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"columns": [str(left), str(right)], "correlation": _safe_number(value)})

    positives = sorted(
        [pair for pair in pairs if pair["correlation"] is not None and pair["correlation"] >= 0],
        key=lambda pair: pair["correlation"],
        reverse=True,
    )[:5]
    negatives = sorted(
        [pair for pair in pairs if pair["correlation"] is not None and pair["correlation"] < 0],
        key=lambda pair: pair["correlation"],
    )[:5]

    return {
        "method": "pearson",
        "columns": [str(column) for column in numeric_columns],
        "correlation_matrix": _dataframe_to_nested_dict(corr),
        "strongest_positive": positives,
        "strongest_negative": negatives,
        "warnings": warnings,
    }


def group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    warnings = []
    if group_col not in df.columns:
        warnings.append(f"group_col {group_col} not found")
    if value_col not in df.columns:
        warnings.append(f"value_col {value_col} not found")
    if warnings:
        return _group_result(group_col, value_col, {}, warnings)
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _group_result(group_col, value_col, {}, [f"value_col {value_col} must be numeric"])

    grouped = df.groupby(group_col, dropna=False)[value_col]
    groups = {}
    for group, series in grouped:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        groups[_label(group)] = {
            "count": int(clean.count()),
            "mean": _safe_number(clean.mean()),
            "median": _safe_number(clean.median()),
            "std": _safe_number(clean.std()),
        }
    return _group_result(group_col, value_col, groups, [])


def target_group_summary(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    warnings = []
    if target_col not in df.columns:
        warnings.append(f"target_col {target_col} not found")
    if warnings:
        return {
            "target_col": target_col,
            "class_distribution": {},
            "numeric_by_target": {},
            "warnings": warnings,
        }

    target = df[target_col].fillna("<missing>").astype(str)
    counts = target.value_counts(dropna=False)
    row_count = max(int(len(df)), 1)
    class_distribution = {
        str(value): {"count": int(count), "proportion": _safe_number(count / row_count)}
        for value, count in counts.items()
    }
    if len(class_distribution) > 20:
        warnings.append("target_col has more than 20 classes; summary may be less useful")

    numeric_by_target = {}
    numeric_columns = [
        column
        for column in df.columns
        if column != target_col and pd.api.types.is_numeric_dtype(df[column])
    ]
    grouped_df = df.assign(_kairos_target=target)
    for column in numeric_columns:
        numeric_by_target[str(column)] = group_summary(grouped_df, "_kairos_target", column)["groups"]

    return {
        "target_col": target_col,
        "class_distribution": class_distribution,
        "numeric_by_target": numeric_by_target,
        "warnings": warnings,
    }


def simple_linear_regression(
    df: pd.DataFrame, target_col: str, feature_col: str
) -> dict[str, Any]:
    warnings = _validate_columns(df, [target_col, feature_col])
    base = {
        "target_col": target_col,
        "feature_col": feature_col,
        "slope": None,
        "intercept": None,
        "r_squared": None,
        "n": 0,
        "interpretation": "",
        "warnings": warnings,
    }
    if warnings:
        return base
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        base["warnings"].append(f"target_col {target_col} must be numeric")
    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        base["warnings"].append(f"feature_col {feature_col} must be numeric")
    if base["warnings"]:
        return base

    data = df[[feature_col, target_col]].dropna()
    x = pd.to_numeric(data[feature_col], errors="coerce")
    y = pd.to_numeric(data[target_col], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    base["n"] = int(len(x))
    if len(x) < 2 or _safe_number(x.var()) == 0.0:
        base["warnings"].append("At least two rows with non-constant numeric feature values are required")
        return base

    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    intercept = y_mean - slope * x_mean
    predictions = intercept + slope * x
    ss_res = ((y - predictions) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    base["slope"] = _safe_number(slope)
    base["intercept"] = _safe_number(intercept)
    base["r_squared"] = _safe_number(r_squared)
    base["interpretation"] = (
        f"For each 1-unit increase in {feature_col}, {target_col} changes by "
        f"about {base['slope']} units on average. R-squared is {base['r_squared']}."
    )
    return base


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    warnings = _validate_columns(df, [col_a, col_b])
    base = {
        "col_a": col_a,
        "col_b": col_b,
        "contingency_table": {},
        "chi_square_statistic": None,
        "degrees_of_freedom": None,
        "p_value": None,
        "warnings": warnings,
    }
    if warnings:
        return base

    table = pd.crosstab(df[col_a].fillna("<missing>"), df[col_b].fillna("<missing>"))
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        base["warnings"].append("Both columns must have at least two observed categories")
        base["contingency_table"] = _dataframe_to_nested_dict(table)
        return base

    total = table.to_numpy().sum()
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    statistic = 0.0
    for row in table.index:
        for col in table.columns:
            expected = row_totals[row] * col_totals[col] / total
            if expected > 0:
                statistic += ((table.loc[row, col] - expected) ** 2) / expected

    base["contingency_table"] = _dataframe_to_nested_dict(table)
    base["chi_square_statistic"] = _safe_number(statistic)
    base["degrees_of_freedom"] = int((table.shape[0] - 1) * (table.shape[1] - 1))
    base["warnings"].append("p_value unavailable")
    base["warnings"].append("scipy is not a project dependency")
    return base


def t_test_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    warnings = _validate_columns(df, [group_col, value_col])
    base = {
        "group_col": group_col,
        "value_col": value_col,
        "groups": {},
        "mean_difference": None,
        "t_statistic": None,
        "p_value": None,
        "warnings": warnings,
    }
    if warnings:
        return base
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        base["warnings"].append(f"value_col {value_col} must be numeric")
        return base

    non_missing = df[[group_col, value_col]].dropna()
    group_values = list(non_missing[group_col].drop_duplicates())
    if len(group_values) != 2:
        base["warnings"].append(
            f"group_col {group_col} must contain exactly two non-missing groups"
        )
        return base

    series_a = pd.to_numeric(
        non_missing.loc[non_missing[group_col] == group_values[0], value_col], errors="coerce"
    ).dropna()
    series_b = pd.to_numeric(
        non_missing.loc[non_missing[group_col] == group_values[1], value_col], errors="coerce"
    ).dropna()
    base["groups"] = {
        _label(group_values[0]): _series_stats(series_a),
        _label(group_values[1]): _series_stats(series_b),
    }
    base["mean_difference"] = _safe_number(series_a.mean() - series_b.mean())

    if len(series_a) < 2 or len(series_b) < 2:
        base["warnings"].append("Each group needs at least two numeric observations")
        return base

    variance_a = series_a.var()
    variance_b = series_b.var()
    denominator = math.sqrt((variance_a / len(series_a)) + (variance_b / len(series_b)))
    base["t_statistic"] = _safe_number((series_a.mean() - series_b.mean()) / denominator) if denominator else None
    base["warnings"].append("p_value unavailable")
    base["warnings"].append("scipy is not a project dependency")
    return base


def _select_columns(df: pd.DataFrame, columns: list[str] | None) -> tuple[list[str], list[str]]:
    if columns is None:
        return list(df.columns), []
    selected = []
    warnings = []
    for column in columns:
        if column in df.columns:
            selected.append(column)
        else:
            warnings.append(f"{column} not found")
    return selected, warnings


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [f"{column} not found" for column in columns if column not in df.columns]


def _group_result(
    group_col: str, value_col: str, groups: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    return {"group_col": group_col, "value_col": value_col, "groups": groups, "warnings": warnings}


def _series_stats(series: pd.Series) -> dict[str, Any]:
    return {
        "count": int(series.count()),
        "mean": _safe_number(series.mean()),
        "median": _safe_number(series.median()),
        "std": _safe_number(series.std()),
    }


def _dataframe_to_nested_dict(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(index): {str(column): _safe_number(value) for column, value in row.items()}
        for index, row in df.iterrows()
    }


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


def _label(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)
