"""Scope-aware cohesive analysis objects for KAIROS."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tools.eda_tools import (
    anova_by_group,
    correlation_analysis,
    group_summary,
    missing_analysis,
    numeric_summary,
    outlier_detection,
    simple_linear_regression,
    t_test_by_group,
    target_group_summary,
)
from tools.viz_tools import (
    group_mean_bar_chart,
    numeric_distribution_plot,
    scatter_plot,
    top_correlation_plots,
)


ID_TOKENS = ("id", "index", "row", "serial", "number", "no", "code", "key", "uuid")


def distribution_analysis(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return statistics, explanation, and chart data for one numeric column."""
    summary = numeric_summary(df, [column])
    chart = numeric_distribution_plot(df, column)
    stats = dict(summary.get("columns", {}).get(column, {}))
    stats.setdefault("missing_count", int(df[column].isna().sum()) if column in df.columns else 0)
    if "q1" in stats and "q3" in stats:
        stats["iqr"] = _safe_number(stats["q3"] - stats["q1"]) if stats["q1"] is not None and stats["q3"] is not None else None
    chart_stats = chart.get("metadata", {}) if isinstance(chart, dict) else {}
    for key in ("count", "missing_count", "min", "max", "mean", "median", "std", "q1", "q3", "iqr"):
        if key not in stats and key in chart_stats:
            stats[key] = chart_stats[key]

    warnings = _combine_warnings(summary, chart)
    return {
        "analysis_type": "distribution_analysis",
        "title": f"Distribution of {column}",
        "column": column,
        "summary": _distribution_summary(column, stats),
        "statistics": stats,
        "chart": chart,
        "table": [stats] if stats else [],
        "warnings": warnings,
        "method_note": "Numeric distribution analysis uses deterministic pandas summary statistics and chart-ready bins or counts.",
    }


def relationship_analysis(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    """Return statistics, explanation, and scatter data for one explicit pair."""
    corr = correlation_analysis(df, [x_col, y_col])
    regression = simple_linear_regression(df, target_col=y_col, feature_col=x_col)
    chart = scatter_plot(df, x_col, y_col)
    r_value = _pair_correlation(corr, x_col, y_col)
    warnings = _combine_warnings(corr, regression, chart)
    if _is_identifier_like_column(df, x_col) or _is_identifier_like_column(df, y_col):
        warnings.append("One selected variable appears identifier-like, so this relationship may not be meaningful.")
    return {
        "analysis_type": "relationship_analysis",
        "title": f"{y_col} vs {x_col}",
        "x_col": x_col,
        "y_col": y_col,
        "columns": [x_col, y_col],
        "summary": _relationship_summary(x_col, y_col, r_value),
        "statistics": {
            "method": "pearson",
            "correlation": r_value,
            "regression": {
                "slope": regression.get("slope"),
                "intercept": regression.get("intercept"),
                "r_squared": regression.get("r_squared"),
                "n": regression.get("n"),
            },
        },
        "chart": chart,
        "table": [{"x_col": x_col, "y_col": y_col, "correlation": r_value}],
        "warnings": warnings,
        "method_note": "Relationship analysis uses Pearson correlation and a simple one-feature linear fit for numeric columns.",
    }


def global_relationship_analysis(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    top_n: int = 3,
) -> dict[str, Any]:
    """Return top non-identifier numeric relationships as cohesive relationship items."""
    selected = _selected_numeric_columns(df, cols, exclude_ids=cols is None)
    chart = top_correlation_plots(df, selected, top_n=top_n)
    relationships = []
    for graph in chart.get("data", []):
        metadata = graph.get("metadata", {})
        relationships.append(
            {
                "x_col": graph.get("x_col"),
                "y_col": graph.get("y_col"),
                "correlation": metadata.get("correlation"),
                "chart": graph,
                "summary": graph.get("finding", ""),
            }
        )
    return {
        "analysis_type": "global_relationship_analysis",
        "title": f"Top {top_n} numeric relationships",
        "summary": f"Selected {len(relationships)} strongest non-identifier numeric relationships.",
        "relationships": relationships,
        "chart": chart,
        "table": [
            {
                "x_col": item["x_col"],
                "y_col": item["y_col"],
                "correlation": item["correlation"],
            }
            for item in relationships
        ],
        "warnings": chart.get("warnings", []),
        "method_note": "Global relationship analysis ranks absolute Pearson correlations after excluding identifier-like columns by default.",
    }


def target_relationship_analysis(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 3,
) -> dict[str, Any]:
    """Rank variables that are most associated with one target column."""
    warnings = []
    if target_col not in df.columns:
        return _target_result(target_col, [], None, [f"Column not found: {target_col}"])

    if pd.api.types.is_numeric_dtype(df[target_col]):
        relationships = _numeric_target_relationships(df, target_col, top_n, warnings)
        return _target_result(
            target_col,
            relationships,
            None,
            warnings,
            "Numeric target relationships are ranked by absolute Pearson correlation.",
        )

    summary = target_group_summary(df, target_col)
    relationships = _categorical_target_relationships(df, target_col, top_n, warnings)
    warnings.extend(summary.get("warnings", []))
    return _target_result(
        target_col,
        relationships,
        summary,
        warnings,
        "Categorical target relationships use deterministic group summaries for numeric predictors.",
    )


def group_comparison_analysis(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    """Compare one numeric value across one grouping column with a ranked chart."""
    summary = group_summary(df, group_col, value_col)
    chart = group_mean_bar_chart(df, group_col, value_col)
    warnings = _combine_warnings(summary, chart)
    if _is_identifier_like_column(df, group_col):
        warnings.append(f"{group_col} appears identifier-like; grouped averages by unique IDs are usually not meaningful.")

    ranked = _ranked_groups(summary.get("groups", {}))
    inferential_test = _group_inferential_test(df, group_col, value_col, ranked)
    warnings.extend(inferential_test.get("warnings", []))
    high = ranked[0]["group"] if ranked else None
    low = ranked[-1]["group"] if ranked else None
    text = _group_summary_text(value_col, group_col, ranked, inferential_test)
    return {
        "analysis_type": "group_comparison_analysis",
        "title": f"Average {value_col} by {group_col}",
        "group_col": group_col,
        "value_col": value_col,
        "summary": text,
        "ranked_groups": ranked,
        "statistics": summary,
        "inferential_test": inferential_test,
        "chart": chart,
        "table": ranked,
        "warnings": warnings,
        "method_note": "Group comparison ranks groups by mean numeric value and adds a basic inferential test when the group count supports it.",
    }


def outlier_analysis(df: pd.DataFrame, column: str, method: str = "iqr") -> dict[str, Any]:
    """Detect potential outliers and include distribution chart data."""
    detection = outlier_detection(df, column, method=method)
    chart = numeric_distribution_plot(df, column)
    warnings = _combine_warnings(detection, chart)
    warnings = [warning for warning in warnings if "Statistical outliers are potential anomalies only" not in warning]
    warnings.append("Outliers are statistical flags only; interpretation depends on domain context.")
    stats = chart.get("metadata", {}) if isinstance(chart, dict) else {}
    summary = detection.get("summary", "")
    if detection.get("count", 0) == 0 and detection.get("lower_bound") is not None and detection.get("upper_bound") is not None:
        summary = f"No values fall outside the IQR rule bounds for {column}."
    return {
        "analysis_type": "outlier_analysis",
        "title": f"Potential outliers in {column}",
        "column": column,
        "method": method,
        "summary": summary,
        "q1": stats.get("q1"),
        "q3": stats.get("q3"),
        "iqr": stats.get("iqr"),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "count": detection.get("count", 0),
        "lower_bound": detection.get("lower_bound"),
        "upper_bound": detection.get("upper_bound"),
        "outliers": detection.get("outliers", []),
        "chart": chart,
        "table": detection.get("outliers", []),
        "warnings": warnings,
        "method_note": "Outlier analysis uses the IQR rule: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged.",
    }


def missingness_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Return dataset missingness diagnostics without a default graph."""
    result = missing_analysis(df)
    ranked = list(result.get("ranked_missing_columns", []))
    if not ranked:
        ranked = [
            {"column": column, **values}
            for column, values in result.get("columns", {}).items()
            if values.get("missing_count", 0) > 0
        ]
        ranked.sort(key=lambda row: (row["missing_percent"], row["missing_count"]), reverse=True)
    if ranked:
        summary = f"{ranked[0]['column']} has the most missing values ({ranked[0]['missing_percent']}%)."
    else:
        summary = "No missing values were detected."
    return {
        "analysis_type": "missingness_analysis",
        "title": "Missing values",
        "summary": _sentence_case(summary),
        "row_count": result.get("row_count", 0),
        "column_count": result.get("column_count", 0),
        "total_missing_cells": result.get("total_missing_cells", 0),
        "columns": result.get("columns", {}),
        "ranked_missing_columns": ranked,
        "chart": None,
        "table": ranked,
        "warnings": result.get("warnings", []),
        "method_note": "Missingness analysis counts null values per column without imputing or modifying the dataset.",
    }


def _target_result(
    target_col: str,
    relationships: list[dict[str, Any]],
    target_summary: dict[str, Any] | None,
    warnings: list[str],
    method_note: str = "Target relationship analysis ranks deterministic associations with the target.",
) -> dict[str, Any]:
    return {
        "analysis_type": "targeted_relationship_analysis",
        "title": f"Variables related to {target_col}",
        "target_col": target_col,
        "target_column": target_col,
        "analysis_focus": f"relationships centered on {target_col}",
        "summary": (
            f"Selected {len(relationships)} variables with the strongest available association to {target_col}."
            if relationships
            else f"No suitable non-identifier predictors were found for {target_col}."
        ),
        "relationships": relationships,
        "target_summary": target_summary,
        "chart": None,
        "table": [
            {
                "predictor_col": item.get("predictor_col"),
                "association": item.get("association"),
                "association_type": item.get("association_type"),
            }
            for item in relationships
        ],
        "warnings": warnings,
        "method_note": method_note,
    }


def _numeric_target_relationships(
    df: pd.DataFrame,
    target_col: str,
    top_n: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    predictors = [
        column
        for column in _selected_numeric_columns(df, None, exclude_ids=True)
        if column != target_col
    ]
    rows = []
    target = pd.to_numeric(df[target_col], errors="coerce")
    for predictor in predictors:
        series = pd.to_numeric(df[predictor], errors="coerce")
        paired = pd.DataFrame({predictor: series, target_col: target}).dropna()
        if len(paired) < 2 or paired[predictor].nunique(dropna=True) <= 1:
            continue
        correlation = _safe_number(paired[predictor].corr(paired[target_col]))
        if correlation is None:
            continue
        rows.append((predictor, correlation))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    relationships = []
    for predictor, correlation in rows[: max(int(top_n), 0)]:
        chart = scatter_plot(df, predictor, target_col)
        relationships.append(
            {
                "predictor_col": predictor,
                "target_col": target_col,
                "association": correlation,
                "association_type": "correlation",
                "summary": _relationship_summary(predictor, target_col, correlation),
                "chart": chart,
            }
        )
    return relationships


def _categorical_target_relationships(
    df: pd.DataFrame,
    target_col: str,
    top_n: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    numeric_predictors = [
        column
        for column in _selected_numeric_columns(df, None, exclude_ids=True)
        if column != target_col
    ]
    relationships = []
    target = df[target_col].fillna("<missing>").astype(str)
    for predictor in numeric_predictors:
        groups = group_summary(df.assign(_kairos_target=target), "_kairos_target", predictor)
        ranked = _ranked_groups(groups.get("groups", {}))
        if not ranked:
            continue
        means = [row["mean"] for row in ranked if row.get("mean") is not None]
        spread = _safe_number(max(means) - min(means)) if len(means) >= 2 else None
        chart = group_mean_bar_chart(df.assign(_kairos_target=target), "_kairos_target", predictor)
        chart["title"] = f"Mean {predictor} by {target_col}"
        relationships.append(
            {
                "predictor_col": predictor,
                "target_col": target_col,
                "association": spread,
                "association_type": "group_mean_spread",
                "summary": f"{predictor} varies across {target_col} groups by about {spread}.",
                "chart": chart,
            }
        )
    relationships.sort(key=lambda item: abs(item["association"] or 0), reverse=True)
    return relationships[: max(int(top_n), 0)]


def _group_inferential_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    ranked_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    group_count = len(ranked_groups)
    if group_count == 2:
        result = t_test_by_group(df, group_col, value_col)
        return {"test": "t_test_by_group", **result}
    if group_count > 2:
        return anova_by_group(df, group_col, value_col)
    return {
        "test": "descriptive_only",
        "p_value": None,
        "warnings": ["Inferential testing requires at least two observed groups; showing descriptive group comparison only."],
    }


def _group_summary_text(
    value_col: str,
    group_col: str,
    ranked: list[dict[str, Any]],
    inferential_test: dict[str, Any],
) -> str:
    if ranked:
        high = ranked[0]
        low = ranked[-1]
        text = (
            f"{_label_name(value_col)} differs across {group_col} descriptively: "
            f"{high.get('group')} has the highest mean {value_col} ({high.get('mean')}), "
            f"while {low.get('group')} has the lowest ({low.get('mean')})."
        )
    else:
        text = f"Grouped statistics were computed for {value_col} by {group_col}."

    if inferential_test.get("test") == "anova_by_group" and inferential_test.get("p_value") is not None:
        p_value = inferential_test.get("p_value")
        notable = "statistically notable" if p_value < 0.05 else "not statistically notable"
        text += (
            f" ANOVA found F = {inferential_test.get('f_statistic')}, "
            f"p = {p_value}, which is {notable} under the common 0.05 threshold."
        )
    elif inferential_test.get("test") == "t_test_by_group" and inferential_test.get("p_value") is not None:
        p_value = inferential_test.get("p_value")
        notable = "statistically notable" if p_value < 0.05 else "not statistically notable"
        text += (
            f" The two-group t-test found t = {inferential_test.get('t_statistic')}, "
            f"p = {p_value}, which is {notable} under the common 0.05 threshold."
        )
    return _sentence_case(text)


def _selected_numeric_columns(df: pd.DataFrame, cols: list[str] | None, exclude_ids: bool) -> list[str]:
    selected = list(df.columns) if cols is None else cols
    columns = []
    for column in selected:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if exclude_ids and _is_identifier_like_column(df, str(column)):
                continue
            columns.append(str(column))
    return columns


def _ranked_groups(groups: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group, values in groups.items():
        if not isinstance(values, dict):
            continue
        rows.append({"group": group, **values})
    rows.sort(key=lambda row: (row.get("mean") is not None, row.get("mean") or 0), reverse=True)
    return rows


def _pair_correlation(correlation_result: dict[str, Any], x_col: str, y_col: str) -> float | None:
    matrix = correlation_result.get("correlation_matrix", {})
    for left, right in ((x_col, y_col), (y_col, x_col)):
        value = matrix.get(left, {}).get(right) if isinstance(matrix, dict) else None
        if value is not None:
            return value
    pairs = correlation_result.get("strongest_positive", []) + correlation_result.get("strongest_negative", [])
    for pair in pairs:
        if set(pair.get("columns", [])) == {x_col, y_col}:
            return pair.get("correlation")
    return None


def _distribution_summary(column: str, stats: dict[str, Any]) -> str:
    if not stats:
        return _sentence_case(f"Distribution analysis could not summarize {column}.")
    return _sentence_case(
        f"{column} has mean {stats.get('mean')}, median {stats.get('median')}, "
        f"and ranges from {stats.get('min')} to {stats.get('max')}."
    )


def _relationship_summary(x_col: str, y_col: str, correlation: float | None) -> str:
    if correlation is None:
        return _sentence_case(f"The relationship between {x_col} and {y_col} could not be estimated.")
    direction = "positive" if correlation >= 0 else "negative"
    return _sentence_case(f"{x_col} and {y_col} show a {_strength(correlation)} {direction} association (r = {correlation}).")


def _label_name(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _strength(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "very weak"


def _is_identifier_like_column(df: pd.DataFrame, column: str) -> bool:
    normalised = str(column).lower()
    if any(token in normalised for token in ID_TOKENS):
        return True
    if column not in df.columns:
        return False
    series = df[column].dropna()
    if series.empty:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return False
    if int(series.nunique(dropna=True)) / len(series) > 0.9:
        return True
    code_like = series.astype(str).head(20).str.match(r"^[A-Za-z]{1,4}[-_]?\d{3,}$").mean()
    return bool(code_like > 0.8)


def _combine_warnings(*results: dict[str, Any]) -> list[str]:
    warnings = []
    for result in results:
        if isinstance(result, dict):
            warnings.extend([str(warning) for warning in result.get("warnings", [])])
    return list(dict.fromkeys(warnings))


def _safe_number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            return None
        return round(value, 6)
    if isinstance(value, int):
        return int(value)
    return value
