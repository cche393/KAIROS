"""Deterministic chart-spec helpers for KAIROS analysis results."""

from __future__ import annotations

from typing import Any

import pandas as pd


ID_TOKENS = ("id", "index", "row", "serial", "number", "no", "code", "key")
DEFAULT_MAX_POINTS = 500


def numeric_distribution_plot(
    df: pd.DataFrame,
    column: str,
    bins: int = 10,
) -> dict[str, Any]:
    warnings = []
    series, series_warnings = _numeric_series(df, column)
    raw_missing = int(df[column].isna().sum()) if column in df.columns else 0
    warnings.extend(series_warnings)
    base = _chart(
        "numeric_distribution_plot",
        f"Distribution of {column}",
        f"Shows the distribution of {column}.",
        "",
        "histogram",
        column,
        "count",
        [],
        warnings,
        {"column": column},
    )
    if warnings:
        base["chart_type"] = "box"
        return base

    stats = _distribution_stats(column, series, raw_missing)
    if len(series) < 2 or int(series.nunique(dropna=True)) <= 1:
        base["chart_type"] = "box"
        base["finding"] = f"{column} has too little variation for a useful histogram."
        base["data"] = [stats]
        base["table"] = base["data"]
        base["metadata"].update(stats)
        return base

    if int(series.nunique(dropna=True)) <= 4:
        counts = series.value_counts().sort_index()
        base["chart_type"] = "bar"
        base["x_col"] = column
        base["y_col"] = "count"
        base["x"] = column
        base["y"] = "count"
        base["data"] = [{column: _safe_number(value), "count": int(count)} for value, count in counts.items()]
        base["table"] = base["data"]
        base["finding"] = f"{column} has {int(series.nunique(dropna=True))} distinct observed values."
        base["metadata"].update(stats)
        return base

    bin_count = max(min(int(bins), 30), 1)
    try:
        cut = pd.cut(series, bins=bin_count, duplicates="drop")
        counts = cut.value_counts(sort=False)
        data = [
            {
                "bin_start": _safe_number(interval.left),
                "bin_end": _safe_number(interval.right),
                "count": int(count),
            }
            for interval, count in counts.items()
        ]
    except ValueError:
        base["chart_type"] = "box"
        base["finding"] = f"{column} could not be binned, so a box-style summary is returned."
        base["data"] = [_box_summary(column, series)]
        return base

    base["data"] = data
    base["table"] = data
    base["finding"] = f"{column} ranges from {_safe_number(series.min())} to {_safe_number(series.max())} across {int(series.count())} usable rows."
    base["metadata"].update(stats)
    return base


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    x, y, warnings = _paired_numeric(df, x_col, y_col)
    title = f"{x_col} vs {y_col}"
    base = _chart(
        "scatter_plot",
        title,
        f"Shows the relationship between {x_col} and {y_col}.",
        "",
        "scatter",
        x_col,
        y_col,
        [],
        warnings,
        {"sampled": False, "row_count": 0},
    )
    if warnings:
        return base

    points = pd.DataFrame({x_col: x, y_col: y}).dropna()
    if points.empty:
        base["warnings"].append("No complete numeric rows are available for the selected columns")
        return base

    limit = max(int(max_points), 1)
    sampled = len(points) > limit
    if sampled:
        points = points.sample(n=limit, random_state=42).sort_index()

    base["data"] = [
        {x_col: _safe_number(row[x_col]), y_col: _safe_number(row[y_col])}
        for _, row in points.iterrows()
    ]
    correlation = _safe_number(points[x_col].corr(points[y_col])) if len(points) >= 2 else None
    base["finding"] = (
        f"The chart contains {len(points)} points"
        + (f" with sample correlation r = {correlation}." if correlation is not None else ".")
    )
    base["metadata"] = {
        "sampled": sampled,
        "row_count": int(len(x.dropna())),
        "points_returned": int(len(points)),
        "correlation": correlation,
    }
    return base


def top_correlation_plots(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    top_n: int = 3,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    warnings = []
    explicit_cols = cols is not None
    numeric_cols = _selected_numeric_columns(df, cols, warnings)
    if not explicit_cols:
        numeric_cols = [column for column in numeric_cols if not _is_identifier_like(column, df[column])]
    if len(numeric_cols) < 2:
        warnings.append("At least two suitable numeric columns are required for top correlations")
        return _chart(
            "top_correlation_plots",
            f"Top {top_n} strongest numeric relationships",
            "Automatically selects strong numeric relationships for charting.",
            "",
            "scatter",
            "",
            "",
            [],
            warnings,
            {"columns_considered": numeric_cols},
        )

    corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append((left, right, float(value)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    selected = pairs[: max(int(top_n), 0)]
    graphs = []
    for left, right, value in selected:
        graph = scatter_plot(df, left, right, max_points=max_points)
        graph["tool_name"] = "scatter_plot"
        graph["title"] = f"Correlation: {left} vs {right}"
        graph["finding"] = f"{left} and {right} have correlation r = {_safe_number(value)}."
        graph["metadata"]["correlation"] = _safe_number(value)
        graphs.append(graph)

    return _chart(
        "top_correlation_plots",
        f"Top {max(int(top_n), 0)} strongest numeric relationships",
        "Automatically selects the strongest absolute numeric correlations.",
        f"Selected {len(graphs)} numeric relationship charts.",
        "scatter",
        "",
        "",
        graphs,
        warnings,
        {"columns_considered": numeric_cols},
    )


def group_mean_bar_chart(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    top_n: int = 15,
) -> dict[str, Any]:
    warnings = []
    if group_col not in df.columns:
        warnings.append(f"Column not found: {group_col}")
    series, value_warnings = _numeric_series(df, value_col)
    warnings.extend(value_warnings)
    base = _chart(
        "group_mean_bar_chart",
        f"Mean {value_col} by {group_col}",
        f"Shows average {value_col} for each {group_col} category.",
        "",
        "bar",
        group_col,
        f"mean_{value_col}",
        [],
        warnings,
        {"limited": False},
    )
    if warnings:
        return base

    grouped_df = pd.DataFrame({group_col: df[group_col], value_col: series}).dropna()
    if grouped_df.empty:
        base["warnings"].append("No complete rows are available for the selected columns")
        return base

    grouped = (
        grouped_df.groupby(group_col, dropna=False)[value_col]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    limit = max(int(top_n), 1)
    limited = len(grouped) > limit
    if limited:
        grouped = grouped.head(limit)
        base["warnings"].append(f"Showing top {limit} groups by mean because there are many categories")
    value_key = f"mean_{value_col}"
    base["data"] = [
        {
            group_col: str(row[group_col]),
            value_key: _safe_number(row["mean"]),
            "count": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]
    base["table"] = base["data"]
    base["finding"] = (
        f"{base['data'][0][group_col]} has the highest average {value_col}."
        if base["data"]
        else ""
    )
    base["metadata"]["limited"] = limited
    return base


def missing_value_bar_chart(
    df: pd.DataFrame,
    include_zero: bool = False,
) -> dict[str, Any]:
    row_count = int(len(df))
    rows = []
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        if missing_count == 0 and not include_zero:
            continue
        percent = 0.0 if row_count == 0 else round((missing_count / row_count) * 100, 2)
        rows.append(
            {
                "column": str(column),
                "missing_count": missing_count,
                "missing_percent": percent,
            }
        )
    rows.sort(key=lambda row: (row["missing_percent"], row["missing_count"]), reverse=True)
    return _chart(
        "missing_value_bar_chart",
        "Missing values by column",
        "Shows columns with missing values.",
        f"{len(rows)} columns have missing values." if rows else "No missing values were found.",
        "bar",
        "column",
        "missing_percent",
        rows,
        [],
        {"include_zero": include_zero, "row_count": row_count},
    )


def regression_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    scatter = scatter_plot(df, x_col, y_col, max_points=max_points)
    scatter["tool_name"] = "regression_plot"
    scatter["title"] = f"Regression: {x_col} vs {y_col}"
    scatter["chart_type"] = "scatter_with_line"
    if scatter["warnings"] or len(scatter["data"]) < 2:
        return scatter

    points = pd.DataFrame(scatter["data"])
    x = points[x_col]
    y = points[y_col]
    if _safe_number(x.var()) == 0.0:
        scatter["warnings"].append(f"{x_col} is constant; regression line cannot be estimated")
        return scatter
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    intercept = y.mean() - slope * x.mean()
    x_min = x.min()
    x_max = x.max()
    line = [
        {x_col: _safe_number(x_min), y_col: _safe_number(intercept + slope * x_min)},
        {x_col: _safe_number(x_max), y_col: _safe_number(intercept + slope * x_max)},
    ]
    scatter["finding"] = f"The fitted line has slope {_safe_number(slope)}."
    scatter["metadata"].update(
        {
            "points": scatter["data"],
            "line": line,
            "slope": _safe_number(slope),
            "intercept": _safe_number(intercept),
        }
    )
    return scatter


def _chart(
    tool_name: str,
    title: str,
    topic: str,
    finding: str,
    chart_type: str,
    x: str,
    y: str,
    data: list[dict[str, Any]],
    warnings: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "title": title,
        "topic": topic,
        "finding": finding,
        "chart_type": chart_type,
        "x_col": x,
        "y_col": y,
        "x": x,
        "y": y,
        "data": data,
        "table": data,
        "warnings": warnings,
        "metadata": metadata,
    }


def _numeric_series(df: pd.DataFrame, column: str) -> tuple[pd.Series, list[str]]:
    if column not in df.columns:
        return pd.Series(dtype="float64"), [f"Column not found: {column}"]
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return numeric, [] if not numeric.empty else [f"{column} has no non-missing numeric values"]

    coerced = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    valid = int(coerced.notna().sum())
    if non_missing > 0 and valid / non_missing >= 0.9:
        return coerced.dropna(), [f"{column} was safely coerced to numeric values"]
    return pd.Series(dtype="float64"), [f"{column} must be numeric"]


def _paired_numeric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[pd.Series, pd.Series, list[str]]:
    x, x_warnings = _numeric_series(df, x_col)
    y, y_warnings = _numeric_series(df, y_col)
    warnings = x_warnings + y_warnings
    if warnings:
        return x, y, warnings
    paired = pd.DataFrame({x_col: x, y_col: y}).dropna()
    return paired[x_col], paired[y_col], []


def _selected_numeric_columns(
    df: pd.DataFrame,
    cols: list[str] | None,
    warnings: list[str],
) -> list[str]:
    selected = list(df.columns) if cols is None else cols
    numeric_cols = []
    for column in selected:
        if column not in df.columns:
            warnings.append(f"Column not found: {column}")
            continue
        series, series_warnings = _numeric_series(df, str(column))
        if series_warnings and not any("safely coerced" in warning for warning in series_warnings):
            warnings.extend(series_warnings)
            continue
        numeric_cols.append(str(column))
    return numeric_cols


def _box_summary(column: str, series: pd.Series) -> dict[str, Any]:
    return _distribution_stats(column, series, 0)


def _distribution_stats(column: str, series: pd.Series, missing_count: int) -> dict[str, Any]:
    q1 = series.quantile(0.25) if not series.empty else None
    q3 = series.quantile(0.75) if not series.empty else None
    return {
        "column": column,
        "count": int(series.count()),
        "missing_count": int(missing_count),
        "min": _safe_number(series.min()),
        "mean": _safe_number(series.mean()),
        "median": _safe_number(series.median()),
        "std": _safe_number(series.std()),
        "q1": _safe_number(q1),
        "q3": _safe_number(q3),
        "iqr": _safe_number(q3 - q1) if q1 is not None and q3 is not None else None,
        "max": _safe_number(series.max()),
    }


def _is_identifier_like(column: str, series: pd.Series | None = None) -> bool:
    normalised = str(column).lower()
    if any(token in normalised for token in ID_TOKENS):
        return True
    if series is None:
        return False
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    if not pd.api.types.is_numeric_dtype(non_missing) and int(non_missing.nunique(dropna=True)) / len(non_missing) > 0.9:
        return True
    code_like = non_missing.astype(str).head(20).str.match(r"^[A-Za-z]{1,4}[-_]?\d{3,}$").mean()
    return bool(code_like > 0.8)


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
