"""Deterministic human-readable interpretations for KAIROS tool results."""

from __future__ import annotations

from typing import Any


def interpret_result(
    tool_name: str | None,
    result: Any,
    user_goal: str | None = None,
) -> dict[str, Any]:
    """Interpret one executed tool result without inventing findings."""
    interpreters = {
        "missing_analysis": _interpret_missing_analysis,
        "numeric_summary": _interpret_numeric_summary,
        "categorical_summary": _interpret_categorical_summary,
        "correlation_analysis": _interpret_correlation_analysis,
        "group_summary": _interpret_group_summary,
        "target_group_summary": _interpret_target_group_summary,
        "t_test_by_group": _interpret_t_test_by_group,
        "chi_square_test": _interpret_chi_square_test,
        "simple_linear_regression": _interpret_simple_linear_regression,
        "outlier_detection": _interpret_outlier_detection,
    }
    interpreter = interpreters.get(str(tool_name or ""))
    if interpreter is None:
        if isinstance(result, dict) and result.get("analysis_type"):
            return _interpret_cohesive_analysis(result)
        if _is_chart_spec(result):
            return _interpret_chart_spec(result)
        return _response(
            "No specialised interpretation is available for this analysis.",
            method_note="The raw structured result is available in technical details.",
        )

    try:
        return interpreter(result, user_goal=user_goal)
    except Exception as exc:
        return _response(
            "This result could not be interpreted in detail.",
            cautions=[f"Interpreter handled an unexpected result structure: {exc}"],
            method_note="The raw structured result is available in technical details.",
        )


def _interpret_correlation_analysis(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    pairs = _correlation_pairs(data)
    if not pairs:
        return _response(
            "No usable pairwise correlations were available.",
            cautions=["Correlation does not prove causation."],
            method_note="Pearson correlation requires at least two numeric columns.",
        )

    positive = [pair for pair in pairs if pair["correlation"] >= 0]
    negative = [pair for pair in pairs if pair["correlation"] < 0]
    strongest_positive = max(positive, key=lambda pair: pair["correlation"], default=None)
    strongest_negative = min(negative, key=lambda pair: pair["correlation"], default=None)
    strongest_overall = max(pairs, key=lambda pair: abs(pair["correlation"]))

    summary = _correlation_sentence(strongest_overall, prefix="The strongest relationship is")
    findings = []
    if strongest_positive:
        findings.append(_correlation_sentence(strongest_positive, prefix="Strongest positive relationship:"))
    if strongest_negative:
        findings.append(_correlation_sentence(strongest_negative, prefix="Strongest negative relationship:"))

    return _response(
        summary,
        findings,
        ["Correlation does not prove causation."],
        "Pearson correlation measures linear association between numeric columns.",
    )


def _interpret_numeric_summary(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    columns = _as_dict(_as_dict(result).get("columns"))
    count = len(columns)
    if count == 0:
        return _response(
            "No numeric columns were summarised.",
            method_note="Numeric summary reports count, centre, spread, quartiles, range, and skewness where available.",
        )

    spread_rows = []
    for column, stats in columns.items():
        values = _as_dict(stats)
        spread = _number(values.get("std"))
        if spread is None:
            min_value = _number(values.get("min"))
            max_value = _number(values.get("max"))
            spread = None if min_value is None or max_value is None else max_value - min_value
        if spread is not None:
            spread_rows.append((str(column), spread))
    spread_rows.sort(key=lambda item: item[1], reverse=True)

    findings = []
    if spread_rows:
        findings.append(
            f"{spread_rows[0][0]} has the largest spread among the summarised numeric columns."
        )
    return _response(
        f"{count} numeric {_plural(count, 'column')} were summarised.",
        findings,
        method_note="Numeric summary reports descriptive statistics only; it does not test relationships.",
    )


def _interpret_categorical_summary(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    columns = _as_dict(_as_dict(result).get("columns"))
    count = len(columns)
    if count == 0:
        return _response(
            "No categorical columns were summarised.",
            method_note="Categorical summary reports observed category counts and proportions.",
        )

    findings = []
    for column, details in columns.items():
        top_values = _as_list(_as_dict(details).get("top_values"))
        if not top_values:
            continue
        top = _as_dict(top_values[0])
        value = top.get("value")
        item_count = top.get("count")
        proportion = _number(top.get("proportion"))
        finding = f"In {column}, the most common category is {value}"
        if item_count is not None:
            finding += f" ({item_count} rows"
            if proportion is not None:
                finding += f", {_format_percent(proportion)}"
            finding += ")"
        finding += "."
        findings.append(finding)
        if proportion is not None and proportion >= 0.75:
            findings.append(f"{column} appears imbalanced because {value} accounts for {_format_percent(proportion)}.")

    return _response(
        f"{count} categorical {_plural(count, 'column')} were summarised.",
        findings,
        method_note="Categorical summary uses observed counts and proportions for each selected column.",
    )


def _interpret_missing_analysis(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    total = int(_number(data.get("total_missing_cells")) or 0)
    columns = _as_dict(data.get("columns"))
    if total == 0:
        return _response(
            "No missing values were detected.",
            method_note="Missing analysis counts blank/null values by column.",
        )

    ranked = []
    source_rows = _as_list(data.get("ranked_missing_columns")) or _as_list(data.get("table"))
    if source_rows:
        ranked = [_as_dict(row) for row in source_rows if _as_dict(row).get("missing_count", 0)]
    else:
        for column, details in columns.items():
            values = _as_dict(details)
            count = int(_number(values.get("missing_count")) or 0)
            percent = _number(values.get("missing_percent"))
            if count > 0:
                ranked.append({"column": str(column), "missing_count": count, "missing_percent": percent})
        ranked.sort(key=lambda item: (item.get("missing_percent") or 0, item.get("missing_count") or 0), reverse=True)
    findings = _missingness_findings(ranked)
    top = ranked[0] if ranked else {}
    summary = (
        f"{top.get('column')} has the most missing values ({top.get('missing_percent')}%)."
        if top
        else f"{total} missing {_plural(total, 'cell')} were detected."
    )
    return _response(
        summary,
        findings,
        ["High missingness may affect downstream summaries and comparisons."] if ranked else [],
        "Missing analysis counts blank/null values by column.",
    )


def _interpret_group_summary(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    return _group_mean_response(
        result,
        group_label_key="group_col",
        value_label_key="value_col",
        method_note="Group summary compares a numeric column across observed groups.",
    )


def _interpret_target_group_summary(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    target_col = data.get("target_col", "target")
    class_distribution = _as_dict(data.get("class_distribution"))
    numeric_by_target = _as_dict(data.get("numeric_by_target"))
    findings = []

    if class_distribution:
        largest_class = max(
            class_distribution.items(),
            key=lambda item: _number(_as_dict(item[1]).get("count")) or 0,
        )
        details = _as_dict(largest_class[1])
        findings.append(
            f"The largest {target_col} group is {largest_class[0]} with {details.get('count')} rows."
        )

    for column, groups in numeric_by_target.items():
        group_result = {"group_col": target_col, "value_col": column, "groups": groups}
        group_interpretation = _interpret_group_summary(group_result)
        if group_interpretation["key_findings"]:
            findings.append(group_interpretation["summary"])
            break

    return _response(
        f"Rows were summarised by target column {target_col}.",
        findings,
        ["Group-level differences are descriptive and should be checked with appropriate tests."],
        "Target group summary compares class distribution and numeric columns by target group.",
    )


def _interpret_t_test_by_group(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    groups = _as_dict(data.get("groups"))
    value_col = data.get("value_col", "value")
    p_value = _number(data.get("p_value"))
    findings = _group_mean_findings(groups, value_col)

    if p_value is None:
        summary = f"The two-group comparison for {value_col} was computed, but no p-value is available."
    elif p_value < 0.05:
        summary = f"The difference in {value_col} is statistically notable under the common 0.05 threshold."
    else:
        summary = f"The difference in {value_col} is not statistically notable under the common 0.05 threshold."

    return _response(
        summary,
        findings,
        [
            "A small p-value suggests the observed group difference would be unlikely under the no-difference assumption; it does not prove causation.",
            "Practical size should be considered separately from statistical significance.",
            "Interpret t-tests cautiously when sample sizes are small or group variances differ.",
        ],
        "Two-sample t-tests compare the means of exactly two groups.",
    )


def _interpret_chi_square_test(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    col_a = data.get("col_a", "first categorical column")
    col_b = data.get("col_b", "second categorical column")
    p_value = _number(data.get("p_value"))
    statistic = _number(data.get("chi_square_statistic"))

    if p_value is None:
        summary = f"The chi-square statistic for {col_a} and {col_b} was computed, but no p-value is available."
    elif p_value < 0.05:
        summary = f"The p-value suggests an association between {col_a} and {col_b} that is statistically notable under the common 0.05 threshold."
    else:
        summary = f"The p-value is not statistically notable under the common 0.05 threshold for association between {col_a} and {col_b}."

    findings = []
    if statistic is not None:
        findings.append(f"Chi-square statistic: {_format_number(statistic)}.")
    return _response(
        summary,
        findings,
        [
            "A small p-value suggests the observed pattern would be unlikely under the no-association assumption; it does not prove causation.",
            "Practical size should be considered separately from statistical significance.",
        ],
        "Chi-square tests compare observed categorical counts with expected counts under independence.",
    )


def _interpret_simple_linear_regression(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    feature = data.get("feature_col", "feature")
    target = data.get("target_col", "target")
    slope = _number(data.get("slope"))
    r_squared = _number(data.get("r_squared"))
    if slope is None:
        summary = f"A simple linear relationship between {feature} and {target} could not be estimated."
    else:
        direction = "positive" if slope > 0 else "negative" if slope < 0 else "flat"
        summary = (
            f"The fitted relationship between {feature} and {target} is {direction}: "
            f"each 1-unit increase in {feature} changes {target} by about {_format_number(slope)} units on average."
        )

    findings = []
    if r_squared is not None:
        findings.append(f"R-squared is {_format_number(r_squared)}, indicating the share of variation explained by this one-feature linear model.")
    n = data.get("n")
    if n is not None:
        findings.append(f"The fit used {n} rows.")
    return _response(
        summary,
        findings,
        ["A simple linear model assumes an approximately linear relationship and does not prove causation."],
        "Simple linear regression estimates one numeric target from one numeric feature.",
    )


def _interpret_outlier_detection(result: Any, user_goal: str | None = None) -> dict[str, Any]:
    data = _as_dict(result)
    column = data.get("column", "selected column")
    count = int(_number(data.get("count")) or 0)
    lower = data.get("lower_bound")
    upper = data.get("upper_bound")
    findings = []
    if lower is not None and upper is not None:
        findings.append(f"IQR bounds were {lower} to {upper}.")
    return _response(
        f"{count} potential {_plural(count, 'outlier')} were detected in {column}.",
        findings,
        data.get("warnings", []) if isinstance(data.get("warnings"), list) else ["Statistical outliers are potential anomalies only."],
        "Outlier detection uses the IQR rule by default.",
    )


def _interpret_chart_spec(result: Any) -> dict[str, Any]:
    data = _as_dict(result)
    title = data.get("title", "chart")
    finding = data.get("finding") or data.get("topic") or f"Chart-ready data was prepared for {title}."
    chart_type = data.get("chart_type", "chart")
    return _response(
        str(finding),
        [f"{title} is ready to view as a {chart_type} chart."],
        data.get("warnings", []) if isinstance(data.get("warnings"), list) else [],
        "Graph helpers return deterministic chart specifications; the UI decides how to render them.",
    )


def _interpret_cohesive_analysis(result: Any) -> dict[str, Any]:
    data = _as_dict(result)
    analysis_type = str(data.get("analysis_type", "analysis"))
    summary = str(data.get("summary") or "The analysis completed.")
    findings = []

    if analysis_type in {"relationship_analysis", "global_relationship_analysis"}:
        relationships = _as_list(data.get("relationships"))
        for item in relationships[:5]:
            details = _as_dict(item)
            if details.get("summary"):
                findings.append(str(details["summary"]))
            elif details.get("x_col") and details.get("y_col"):
                findings.append(
                    f"{details.get('x_col')} and {details.get('y_col')} have association {details.get('correlation')}."
                )
    elif analysis_type in {"target_relationship_analysis", "targeted_relationship_analysis"}:
        for item in _as_list(data.get("relationships"))[:5]:
            details = _as_dict(item)
            if details.get("summary"):
                findings.append(str(details["summary"]))
            elif details.get("predictor_col"):
                findings.append(
                    f"{details.get('predictor_col')} has association {details.get('association')} with {data.get('target_col')}."
                )
    elif analysis_type == "dataset_overview":
        summary = f"This dataset has {data.get('row_count', 0)} rows and {data.get('column_count', 0)} columns."
        findings.extend(_dataset_overview_findings(data))
    elif analysis_type == "group_comparison_analysis":
        ranked = _as_list(data.get("ranked_groups"))
        if ranked:
            top = _as_dict(ranked[0])
            bottom = _as_dict(ranked[-1])
            findings.append(
                f"{top.get('group')} has the highest mean; {bottom.get('group')} has the lowest."
            )
        test = _as_dict(data.get("inferential_test"))
        p_value = _number(test.get("p_value"))
        if p_value is not None:
            if p_value < 0.05:
                findings.append("The p-value is statistically notable under the common 0.05 threshold.")
            else:
                findings.append("The p-value is not statistically notable under the common 0.05 threshold.")
    elif analysis_type == "distribution_analysis":
        stats = _as_dict(data.get("statistics"))
        if stats:
            findings.append(
                f"Mean {stats.get('mean')}; median {stats.get('median')}; range {stats.get('min')} to {stats.get('max')}."
            )
    elif analysis_type == "outlier_analysis":
        findings.append(f"{data.get('count', 0)} potential outliers were flagged.")
    elif analysis_type == "missingness_analysis":
        ranked = _as_list(data.get("ranked_missing_columns"))
        if ranked:
            findings.extend(_missingness_findings(ranked))

    cautions = data.get("warnings", []) if isinstance(data.get("warnings"), list) else []
    if _as_dict(data.get("inferential_test")).get("p_value") is not None:
        cautions = cautions + [
            "A small p-value suggests the observed group difference would be unlikely under the no-difference assumption; it does not prove causation.",
            "Practical size should be considered separately from statistical significance.",
        ]
    return _response(
        summary,
        findings,
        cautions,
        str(data.get("method_note") or "This cohesive analysis combines deterministic statistics with chart-ready data."),
    )


def _dataset_overview_findings(data: dict[str, Any]) -> list[str]:
    findings = []
    column_types = _as_dict(data.get("column_types"))
    for type_name in ("numeric", "categorical", "datetime", "boolean", "text_like"):
        for column in _as_list(column_types.get(type_name)):
            findings.append(f"{column}: {type_name}.")
    quality_notes = _as_list(data.get("quality_notes"))
    findings.extend(str(note) for note in quality_notes[:5])
    if not quality_notes:
        issues = _as_dict(data.get("potential_issues"))
        for column in _as_list(issues.get("likely_id_columns")):
            findings.append(f"{column} appears to be an identifier.")
        for column in _as_list(issues.get("constant_value_columns")):
            findings.append(f"{column} is constant.")
        for column in _as_list(issues.get("high_cardinality_categorical_columns")):
            findings.append(f"{column} has high cardinality.")
    return findings


def _missingness_findings(ranked_missing_columns: list[Any]) -> list[str]:
    findings = []
    for item in ranked_missing_columns:
        details = _as_dict(item)
        column = details.get("column")
        count = details.get("missing_count")
        percent = details.get("missing_percent")
        if not column:
            continue
        if percent is not None and count is not None:
            findings.append(f"{column}: {percent}% missing ({count} rows).")
        elif percent is not None:
            findings.append(f"{column}: {percent}% missing.")
        elif count is not None:
            findings.append(f"{column}: {count} missing rows.")
    return findings


def _group_mean_response(
    result: Any,
    group_label_key: str,
    value_label_key: str,
    method_note: str,
) -> dict[str, Any]:
    data = _as_dict(result)
    groups = _as_dict(data.get("groups"))
    group_col = data.get(group_label_key, "group")
    value_col = data.get(value_label_key, "value")
    findings = _group_mean_findings(groups, value_col)
    if not findings:
        return _response(
            f"No group means were available for {value_col} by {group_col}.",
            method_note=method_note,
        )

    return _response(
        findings[0],
        findings[1:],
        method_note=method_note,
    )


def _group_mean_findings(groups: dict[str, Any], value_col: Any) -> list[str]:
    means = []
    for group, details in groups.items():
        mean = _number(_as_dict(details).get("mean"))
        if mean is not None:
            means.append((str(group), mean))
    if not means:
        return []
    highest = max(means, key=lambda item: item[1])
    lowest = min(means, key=lambda item: item[1])
    findings = [
        f"{highest[0]} has the highest average {value_col}, while {lowest[0]} has the lowest."
    ]
    for group, mean in means[:4]:
        findings.append(f"{group}: mean {value_col} = {_format_number(mean)}.")
    return findings


def _correlation_pairs(data: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = []
    for key in ("strongest_positive", "strongest_negative"):
        for pair in _as_list(data.get(key)):
            parsed = _parse_correlation_pair(pair)
            if parsed:
                pairs.append(parsed)
    if pairs:
        return _unique_correlation_pairs(pairs)

    matrix = _as_dict(data.get("correlation_matrix"))
    for left, row in matrix.items():
        for right, value in _as_dict(row).items():
            parsed = _parse_correlation_pair({"columns": [left, right], "correlation": value})
            if parsed:
                pairs.append(parsed)
    return _unique_correlation_pairs(pairs)


def _parse_correlation_pair(pair: Any) -> dict[str, Any] | None:
    values = _as_dict(pair)
    columns = _as_list(values.get("columns"))
    if len(columns) != 2:
        return None
    left, right = str(columns[0]), str(columns[1])
    if left == right or _is_identifier_like(left) or _is_identifier_like(right):
        return None
    correlation = _number(values.get("correlation"))
    if correlation is None:
        return None
    return {"columns": [left, right], "correlation": correlation}


def _unique_correlation_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for pair in pairs:
        key = tuple(sorted(pair["columns"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def _correlation_sentence(pair: dict[str, Any], prefix: str) -> str:
    left, right = pair["columns"]
    value = pair["correlation"]
    direction = "positive" if value >= 0 else "negative"
    strength = _correlation_strength(abs(value))
    return (
        f"{prefix} between {left} and {right} (r = {_format_number(value)}), "
        f"suggesting a {strength} {direction} association."
    )


def _correlation_strength(value: float) -> str:
    if value >= 0.7:
        return "strong"
    if value >= 0.4:
        return "moderate"
    if value >= 0.2:
        return "weak"
    return "very weak"


def _is_identifier_like(column: str) -> bool:
    normalised = column.lower().strip()
    tokens = [token for token in normalised.replace("-", "_").split("_") if token]
    return normalised == "id" or normalised.endswith("_id") or tokens == ["id"]


def _response(
    summary: str,
    key_findings: list[str] | None = None,
    cautions: list[str] | None = None,
    method_note: str = "",
) -> dict[str, Any]:
    return {
        "summary": _sentence_case(str(summary or "")),
        "key_findings": [str(item) for item in (key_findings or []) if item],
        "cautions": [str(item) for item in (cautions or []) if item],
        "method_note": str(method_note or ""),
    }


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_chart_spec(value: Any) -> bool:
    data = _as_dict(value)
    return {"tool_name", "chart_type", "data"}.issubset(data.keys())


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_percent(proportion: float) -> str:
    return f"{proportion * 100:.1f}%"


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else f"{noun}s"
