"""Deterministic action recommendations for KAIROS."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


def recommend_actions(
    df: pd.DataFrame,
    max_actions: int | None = None,
    goal: str | None = None,
    dataset_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Recommend candidate analysis actions from simple DataFrame properties."""
    numeric_columns = _numeric_columns(df, dataset_profile)
    relationship_numeric_columns = _relationship_numeric_columns(df, dataset_profile)
    categorical_columns = _categorical_columns(df, dataset_profile)
    analysis_categorical_columns = _analysis_categorical_columns(df, dataset_profile)
    binary_categorical_columns = _binary_categorical_columns(df, analysis_categorical_columns)
    preferred_numeric = _choose_numeric_column(goal, relationship_numeric_columns)
    preferred_group = _choose_group_column(goal, analysis_categorical_columns)
    if _is_dataset_overview_question(_goal_tokens(goal)):
        overview = [_action("dataset_overview", {}, "Task type: dataset inspection. Analysis focus: dataset structure overview.")]
        overview[0]["priority"] = 1
        return overview[: max(int(max_actions), 0)] if max_actions is not None else overview
    scoped = _scoped_recommendations(
        df,
        goal,
        dataset_profile,
        relationship_numeric_columns,
        analysis_categorical_columns,
        binary_categorical_columns,
        preferred_numeric,
        preferred_group,
    )
    if scoped is not None:
        for priority, recommendation in enumerate(scoped, start=1):
            recommendation["priority"] = priority
        if max_actions is None:
            return scoped
        return scoped[: max(int(max_actions), 0)]

    recommendations = [
        _action(
            "missing_analysis",
            {},
            "Check missing values before deeper analysis.",
        )
    ]

    if len(df.columns) > 0:
        recommendations.append(
            _action(
                "missing_value_bar_chart",
                {},
                "Visualize missing values by column.",
            )
        )

    if numeric_columns:
        recommendations.append(
            _action(
                "numeric_summary",
                {"columns": numeric_columns},
                "Summarize numeric columns to understand scale and spread.",
            )
        )
        first_numeric = preferred_numeric or (relationship_numeric_columns[0] if relationship_numeric_columns else numeric_columns[0])
        recommendations.append(
            _action(
                "numeric_distribution_plot",
                {"column": first_numeric},
                "Show the distribution of a key numeric column.",
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

    if len(relationship_numeric_columns) >= 2:
        relationship_reason = "Check pairwise numeric relationships."
        exclusions = _profile_relationship_exclusions(dataset_profile)
        if exclusions:
            relationship_reason += " " + " ".join(exclusions)
        recommendations.append(
            _action(
                "correlation_analysis",
                {"columns": relationship_numeric_columns},
                relationship_reason,
            )
        )
        recommendations.append(
            _action(
                "top_correlation_plots",
                {"cols": relationship_numeric_columns, "top_n": 3},
                "Visualize the strongest numeric relationships while avoiding identifier-like columns.",
            )
        )
        recommendations.append(
            _action(
                "scatter_plot",
                {"x_col": relationship_numeric_columns[0], "y_col": relationship_numeric_columns[1]},
                "Show a point-level view of a numeric relationship.",
            )
        )

    if categorical_columns and relationship_numeric_columns:
        group_col = preferred_group or categorical_columns[0]
        value_col = preferred_numeric or relationship_numeric_columns[0]
        assumption = ""
        if goal and preferred_numeric and not _goal_mentions_any_column(goal, categorical_columns):
            assumption = f" No group column was specified, so {group_col} was used as the grouping variable."
        recommendations.append(
            _action(
                "group_summary",
                {"group_col": group_col, "value_col": value_col},
                f"Compare {value_col} across {group_col} groups.{assumption}",
            )
        )
        recommendations.append(
            _action(
                "group_mean_bar_chart",
                {"group_col": group_col, "value_col": value_col},
                f"Visualize mean {value_col} by {group_col}.{assumption}",
            )
        )

    if binary_categorical_columns and relationship_numeric_columns:
        recommendations.append(
            _action(
                "t_test_by_group",
                {"group_col": binary_categorical_columns[0], "value_col": relationship_numeric_columns[0]},
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

    if len(relationship_numeric_columns) >= 2:
        recommendations.append(
            _action(
                "simple_linear_regression",
                {"feature_col": relationship_numeric_columns[0], "target_col": relationship_numeric_columns[1]},
                "Fit a simple one-feature linear relationship between numeric columns.",
            )
        )
        recommendations.append(
            _action(
                "regression_plot",
                {"x_col": relationship_numeric_columns[0], "y_col": relationship_numeric_columns[1]},
                "Visualize a simple fitted linear relationship.",
            )
        )

    for priority, recommendation in enumerate(recommendations, start=1):
        recommendation["priority"] = priority

    if max_actions is None:
        return recommendations
    return recommendations[: max(int(max_actions), 0)]


def describe_planning_scope(
    goal: str | None,
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a lightweight deterministic trace of the inferred planning scope."""
    return _describe_planning_scope(goal, df, dataset_profile)


def _describe_planning_scope(
    goal: str | None,
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = _goal_tokens(goal)
    explicit_columns = _explicit_columns(goal, df)
    target = _choose_target_column(goal, df)
    time_column = _choose_datetime_column(goal, dataset_profile)
    base = {
        "analysis_type": None,
        "target_column": target,
        "analysis_focus": None,
    }
    if _is_time_question(tokens) and time_column:
        value_column = target or _choose_numeric_column(goal, _relationship_numeric_columns(df, dataset_profile))
        return {
            **base,
            "analysis_type": "time_context_analysis",
            "target_column": value_column,
            "time_column": time_column,
            "analysis_focus": f"{time_column} was detected as datetime-like; time-series plotting can be added as a future tool.",
            "scope": "time_context",
            "trigger": "time",
            "target": value_column,
        }
    if _is_dataset_overview_question(tokens):
        return {
            **base,
            "analysis_type": "dataset_overview",
            "target_column": None,
            "analysis_focus": "dataset structure overview",
            "scope": "dataset_overview",
            "trigger": "schema overview",
            "target": None,
        }
    if tokens & {"missing", "missingness", "null", "nulls", "blank", "blanks", "incomplete"}:
        return {**base, "analysis_type": "missingness_analysis", "target_column": None, "scope": "missingness", "trigger": _first_token(tokens, {"missing", "missingness", "null", "nulls", "blank", "blanks", "incomplete"}), "target": None}
    if tokens & {"outlier", "outliers", "anomaly", "anomalies"}:
        return {**base, "analysis_type": "outlier_analysis", "analysis_focus": f"one-variable analysis for {target}" if target else None, "scope": "one_variable", "trigger": _first_token(tokens, {"outlier", "outliers", "anomaly", "anomalies"}), "target": target}
    if len([column for column in explicit_columns if pd.api.types.is_numeric_dtype(df[column])]) >= 2:
        return {**base, "analysis_type": "relationship_analysis", "target_column": None, "scope": "explicit_pair", "trigger": "two explicit numeric variables", "target": None}
    if _is_distribution_question(tokens):
        return {**base, "analysis_type": "distribution_analysis", "analysis_focus": f"one-variable analysis for {target}" if target else None, "scope": "one_variable", "trigger": _first_token(tokens, {"distribution", "histogram", "spread", "look"}), "target": target}
    if _is_relationship_question(tokens):
        analysis_type = "targeted_relationship_analysis" if target is not None else "relationship_analysis"
        return {
            **base,
            "analysis_type": analysis_type,
            "target_column": target,
            "analysis_focus": f"relationships centered on {target}" if target else "global numeric relationships",
            "scope": "target_driven" if target is not None else "global_relationships",
            "trigger": _relationship_trigger(tokens),
            "target": target,
        }
    if _is_target_question(tokens):
        return {**base, "analysis_type": "targeted_relationship_analysis", "analysis_focus": f"relationships centered on {target}" if target else None, "scope": "target_driven", "trigger": _target_trigger(tokens), "target": target}
    if _is_group_question(tokens):
        return {**base, "analysis_type": "group_comparison_analysis", "target_column": None, "scope": "group_comparison", "trigger": _group_trigger(tokens), "target": None}
    if _is_fallback_overview_question(tokens):
        return {**base, "analysis_type": "exploratory_analysis", "target_column": None, "analysis_focus": "broad exploratory overview", "scope": "fallback_overview", "trigger": "open-ended overview", "target": None}
    return {**base, "analysis_type": "exploratory_analysis", "target_column": None, "scope": "legacy_schema_recommendation", "trigger": "", "target": None}


def detect_referenced_columns(goal: str | None, df: pd.DataFrame) -> list[str]:
    """Detect dataset columns explicitly referenced in a user goal."""
    if not goal:
        return []
    goal_text = _normalise_text(goal)
    goal_compact = _compact(goal)
    matches = []
    for column in df.columns:
        name = str(column)
        column_text = _normalise_text(name)
        column_compact = _compact(name)
        if not column_text:
            continue
        if _contains_normalised_phrase(goal_text, column_text) or column_compact in goal_compact.split():
            matches.append(name)
            continue
        if _high_confidence_fuzzy_match(goal_text, column_text):
            matches.append(name)
    return matches


def _action(tool: str, args: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"tool": tool, "args": args, "priority": 0, "reason": reason}


def _scoped_recommendations(
    df: pd.DataFrame,
    goal: str | None,
    dataset_profile: dict[str, Any] | None,
    numeric_columns: list[str],
    categorical_columns: list[str],
    binary_categorical_columns: list[str],
    preferred_numeric: str | None,
    preferred_group: str | None,
) -> list[dict[str, Any]] | None:
    if not goal:
        return None
    tokens = _goal_tokens(goal)
    explicit_columns = _explicit_columns(goal, df)
    explicit_numeric = [column for column in explicit_columns if column in numeric_columns]

    if tokens & {"missing", "missingness", "null", "nulls", "blank", "blanks", "incomplete"}:
        return [_action("missingness_analysis", {}, "Check which columns have missing values.")]

    if tokens & {"outlier", "outliers", "anomaly", "anomalies"}:
        column = preferred_numeric or (explicit_numeric[0] if explicit_numeric else (numeric_columns[0] if numeric_columns else None))
        if column is None:
            return [_action("missingness_analysis", {}, "No suitable numeric column was found for outlier analysis.")]
        return [_action("outlier_analysis", {"column": column}, f"Detect potential outliers in {column}.")]

    if len(explicit_numeric) >= 2:
        x_col, y_col = explicit_numeric[1], explicit_numeric[0]
        return [
            _action(
                "relationship_analysis",
                {"x_col": x_col, "y_col": y_col},
                f"Analyze only the explicitly requested pair: {explicit_numeric[0]} and {explicit_numeric[1]}.",
            ),
        ]

    if _is_distribution_question(tokens):
        column = preferred_numeric or (explicit_numeric[0] if explicit_numeric else (numeric_columns[0] if numeric_columns else None))
        if column is None:
            return [_action("missingness_analysis", {}, "No suitable numeric column was found for distribution analysis.")]
        return [
            _action("distribution_analysis", {"column": column}, f"Summarize and visualize the distribution of {column}."),
        ]

    if _is_relationship_question(tokens):
        target = _choose_target_column(goal, df)
        if target is not None:
            return [_action(
                "target_relationship_analysis",
                {"target_col": target},
                f"Detected target variable: {target}. Analysis mode: targeted relationship analysis.",
            )]
        if len(numeric_columns) < 2:
            return [_action("missingness_analysis", {}, "No suitable numeric relationships are available.")]
        return [
            _action(
                "global_relationship_analysis",
                {"cols": numeric_columns, "top_n": 3},
                _with_profile_exclusions(
                    "Show the strongest non-identifier numeric relationships as cohesive relationship results.",
                    dataset_profile,
                ),
            )
        ]

    if _is_target_question(tokens):
        target = _choose_target_column(goal, df)
        if target is not None and not _is_explicit_group_comparison(goal, categorical_columns):
            return [_action(
                "target_relationship_analysis",
                {"target_col": target},
                f"Detected target variable: {target}. Analysis mode: targeted relationship analysis.",
            )]

    if _is_group_question(tokens):
        value_col = preferred_numeric or (explicit_numeric[0] if explicit_numeric else _implied_value_column(tokens, numeric_columns))
        group_col = preferred_group or _implied_group_column(tokens, categorical_columns)
        if value_col is None or group_col is None:
            return [_action("missingness_analysis", {}, "A valid group and numeric value column were not both available.")]
        assumption = ""
        if not _goal_mentions_any_column(goal, categorical_columns):
            assumption = f" No group column was specified, so {group_col} was used as the grouping variable."
        if _is_identifier_like_column(df, group_col):
            assumption += f" {group_col} appears identifier-like, so this grouping may not be meaningful."
        elif group_col in _profile_columns(dataset_profile, "categorical"):
            assumption += f" {group_col} appears categorical, so group comparison analysis was selected."
        return [
            _action("group_comparison_analysis", {"group_col": group_col, "value_col": value_col}, f"Compare {value_col} across {group_col} groups.{assumption}"),
        ]

    if _is_global_relationship_question(tokens):
        if len(numeric_columns) < 2:
            return [_action("distribution_analysis", {"column": numeric_columns[0]}, "No pairwise numeric relationships are available.") if numeric_columns else _action("missingness_analysis", {}, "No suitable numeric columns are available.")]
        return [
            _action(
                "global_relationship_analysis",
                {"cols": numeric_columns, "top_n": 3},
                _with_profile_exclusions(
                    "Show the strongest non-identifier numeric relationships as cohesive relationship results.",
                    dataset_profile,
                ),
            )
        ]

    if _is_target_question(tokens):
        target = _choose_target_column(goal, df)
        if target is None:
            return None
        return [_action(
            "target_relationship_analysis",
            {"target_col": target},
            f"Detected target variable: {target}. Analysis mode: targeted relationship analysis.",
        )]

    if _is_fallback_overview_question(tokens):
        return _fallback_overview_actions(numeric_columns, categorical_columns, preferred_numeric, preferred_group)
    return None


def _numeric_columns(
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> list[str]:
    columns = _profile_columns(dataset_profile, "numeric")
    if columns:
        return [column for column in columns if column in df.columns]
    return [str(column) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]


def _relationship_numeric_columns(
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> list[str]:
    numeric = _numeric_columns(df, dataset_profile)
    issue_columns = set(_profile_issue_columns(dataset_profile, "likely_id_columns"))
    issue_columns.update(_profile_issue_columns(dataset_profile, "constant_value_columns"))
    meaningful = [
        column
        for column in numeric
        if column not in issue_columns and not _is_identifier_like_column(df, column)
    ]
    return meaningful or numeric


def _categorical_columns(
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> list[str]:
    columns = _profile_columns(dataset_profile, "categorical") + _profile_columns(dataset_profile, "boolean")
    if columns:
        return [column for column in columns if column in df.columns]
    return [str(column) for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]


def _analysis_categorical_columns(
    df: pd.DataFrame,
    dataset_profile: dict[str, Any] | None = None,
) -> list[str]:
    high_cardinality = set(_profile_issue_columns(dataset_profile, "high_cardinality_categorical_columns"))
    likely_ids = set(_profile_issue_columns(dataset_profile, "likely_id_columns"))
    return [
        column
        for column in _categorical_columns(df, dataset_profile)
        if column not in high_cardinality and column not in likely_ids
    ]


def _binary_categorical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if int(df[column].dropna().nunique(dropna=True)) == 2
    ]


def _is_identifier_like(column: str) -> bool:
    normalised = column.lower()
    return any(token in normalised for token in ("id", "index", "row", "serial", "number", "no", "code", "key", "uuid"))


def _is_identifier_like_column(df: pd.DataFrame, column: str) -> bool:
    if _is_identifier_like(column):
        return True
    series = df[column].dropna()
    if series.empty:
        return False
    if not pd.api.types.is_numeric_dtype(series) and int(series.nunique(dropna=True)) / len(series) > 0.9:
        return True
    sample = series.astype(str).head(20)
    code_like = sample.str.match(r"^[A-Za-z]{1,4}[-_]?\d{3,}$").mean()
    return bool(code_like > 0.8)


def _explicit_columns(goal: str | None, df: pd.DataFrame) -> list[str]:
    return detect_referenced_columns(goal, df)


def _is_distribution_question(tokens: set[str]) -> bool:
    return bool(tokens & {"distribution", "histogram", "spread"} or {"look", "like"}.issubset(tokens))


def _is_time_question(tokens: set[str]) -> bool:
    return bool(tokens & {"time", "trend", "trends", "over", "timeline"} or {"over", "time"}.issubset(tokens))


def _is_dataset_overview_question(tokens: set[str]) -> bool:
    if not tokens:
        return False
    analytical_terms = {
        "missing", "missingness", "null", "distribution", "correlation", "correlations",
        "correlated", "correlate", "correlates", "relationship", "relationships",
        "related", "associated", "association", "compare", "comparison", "by",
        "predict", "predicts", "affect", "affects", "trend", "plot", "chart",
    }
    if tokens & analytical_terms:
        return False
    schema_terms = {"column", "columns", "field", "fields", "variable", "variables", "schema"}
    dataset_terms = {"dataset", "data", "csv", "file"}
    if tokens & {"variable", "variables"} and not (tokens & {"available", "have"} or tokens & dataset_terms):
        return False
    if tokens & schema_terms and (tokens & {"what", "which", "show", "list", "available", "have"} or tokens & dataset_terms):
        return True
    if {"dataset", "overview"}.issubset(tokens) or {"data", "overview"}.issubset(tokens):
        return True
    if tokens & {"describe", "summarize", "summarise"} and tokens & dataset_terms and not tokens & {"distribution", "correlation", "relationship", "compare", "by", "plot", "chart", "trend"}:
        return True
    if {"what", "is", "in"}.issubset(tokens) and tokens & dataset_terms:
        return True
    return False


def _is_group_question(tokens: set[str]) -> bool:
    return bool(
        tokens & {"group", "groups", "department", "departments", "team", "teams", "category", "categories", "across", "by", "paid", "pay", "highest", "vary", "differ", "differs", "compare"}
    )


def _is_global_relationship_question(tokens: set[str]) -> bool:
    return bool(
        ("strongest" in tokens and "relationships" in tokens)
        or ("strongest" in tokens and "correlations" in tokens)
        or ("top" in tokens and "correlations" in tokens)
        or {"variables", "correlated"}.issubset(tokens)
        or {"important", "relationships"}.issubset(tokens)
        or {"variables", "matter"}.issubset(tokens)
    )


def _is_relationship_question(tokens: set[str]) -> bool:
    return bool(
        tokens & {"correlate", "correlates", "correlated", "correlation", "correlations", "relationship", "relationships"}
        or {"related", "to"}.issubset(tokens)
        or {"associated", "with"}.issubset(tokens)
        or (tokens & {"variables", "factors"} and tokens & {"related", "associated", "correlated", "relationship", "relationships"})
        or {"strongly", "related"}.issubset(tokens)
    )


def _is_target_question(tokens: set[str]) -> bool:
    return bool(
        tokens & {"predict", "predicts", "affect", "affects", "drive", "drives", "influence", "influences", "factors", "factor"}
        or (tokens & {"variables", "variable"} and tokens & {"related", "relate", "associated", "association", "strongly"})
        or (tokens & {"related", "associated"} and tokens & {"to", "with"})
    )


def _is_explicit_group_comparison(goal: str | None, categorical_columns: list[str]) -> bool:
    tokens = _goal_tokens(goal)
    return bool(
        tokens & {"by", "across", "department", "departments", "team", "teams", "group", "groups", "category", "categories"}
        and _goal_mentions_any_column(goal, categorical_columns)
    )


def _is_fallback_overview_question(tokens: set[str]) -> bool:
    return bool(
        not tokens
        or tokens & {"analyze", "analyse", "explore", "overview", "summarize", "summarise"}
        or {"this", "dataset"}.issubset(tokens)
    )


def _fallback_overview_actions(
    numeric_columns: list[str],
    categorical_columns: list[str],
    preferred_numeric: str | None,
    preferred_group: str | None,
) -> list[dict[str, Any]]:
    actions = []
    important_numeric = _important_numeric_columns(numeric_columns, preferred_numeric)
    for column in important_numeric[:3]:
        actions.append(_action("distribution_analysis", {"column": column}, f"Start with the distribution of {column}."))
    if len(numeric_columns) >= 2:
        actions.append(
            _action(
                "global_relationship_analysis",
                {"cols": numeric_columns, "top_n": 3},
                "Then check the strongest non-identifier numeric relationships.",
            )
        )
    actions.append(_action("missingness_analysis", {}, "Check missing values after the main overview analyses."))
    if numeric_columns and categorical_columns:
        value_col = preferred_numeric or (important_numeric[0] if important_numeric else numeric_columns[0])
        group_col = preferred_group or _choose_group_column("groups", categorical_columns) or categorical_columns[0]
        actions.append(
            _action(
                "group_comparison_analysis",
                {"group_col": group_col, "value_col": value_col},
                f"Compare {value_col} across {group_col} as a compact group view.",
            )
        )
    return actions


def _important_numeric_columns(numeric_columns: list[str], preferred_numeric: str | None) -> list[str]:
    scored = []
    for index, column in enumerate(numeric_columns):
        normalised = column.lower()
        score = 0
        if column == preferred_numeric:
            score += 200
        priority_terms = (
            "salary",
            "revenue",
            "sales",
            "profit",
            "bonus",
            "income",
            "conversion",
            "performance",
            "satisfaction",
            "engagement",
            "age",
        )
        for rank, term in enumerate(priority_terms):
            if term in normalised:
                score += 100 - rank
                break
        if _is_identifier_like(column):
            score -= 500
        scored.append((score, -index, column))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [column for _, _, column in scored]


def _implied_value_column(tokens: set[str], numeric_columns: list[str]) -> str | None:
    if tokens & {"paid", "pay", "compensation", "income"}:
        return _first_matching_alias(numeric_columns, {"salary", "income", "pay", "bonus"})
    if "happiness" in tokens:
        return _first_matching_alias(numeric_columns, {"satisfaction_score", "satisfaction"})
    if "performance" in tokens:
        return _first_matching_alias(numeric_columns, {"performance_score", "performance"})
    if "experience" in tokens:
        return _first_matching_alias(numeric_columns, {"years_experience", "experience"})
    return numeric_columns[0] if numeric_columns else None


def _implied_group_column(tokens: set[str], categorical_columns: list[str]) -> str | None:
    if not categorical_columns:
        return None
    explicit = [column for column in categorical_columns if _semantic_column_score(tokens, column) > 0]
    if explicit:
        return explicit[0]
    return _choose_group_column("groups", categorical_columns) or categorical_columns[0]


def _choose_target_column(goal: str | None, df: pd.DataFrame) -> str | None:
    explicit = _explicit_columns(goal, df)
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        return None
    tokens = _goal_tokens(goal)
    for column in df.columns:
        if _semantic_column_score(tokens, str(column)) > 0:
            return str(column)
    return None


def _choose_datetime_column(
    goal: str | None,
    dataset_profile: dict[str, Any] | None,
) -> str | None:
    datetime_columns = _profile_columns(dataset_profile, "datetime") or _profile_columns(dataset_profile, "datetime_like")
    if not datetime_columns:
        return None
    tokens = _goal_tokens(goal)
    for column in datetime_columns:
        if _column_explicitly_requested(tokens, column) or _semantic_column_score(tokens, column) > 0:
            return column
    return datetime_columns[0]


def _first_matching_alias(columns: list[str], aliases: set[str]) -> str | None:
    for column in columns:
        normalised = column.lower()
        if any(alias in normalised for alias in aliases):
            return column
    return columns[0] if columns else None


def _choose_numeric_column(goal: str | None, numeric_columns: list[str]) -> str | None:
    if not numeric_columns:
        return None
    tokens = _goal_tokens(goal)
    scored = []
    for index, column in enumerate(numeric_columns):
        score = _semantic_column_score(tokens, column)
        if _is_identifier_like(column) and not _column_explicitly_requested(tokens, column):
            score -= 100
        scored.append((score, -index, column))
    best_score, _, best_column = max(scored)
    return best_column if best_score > 0 else None


def _choose_group_column(goal: str | None, categorical_columns: list[str]) -> str | None:
    if not categorical_columns:
        return None
    if not goal:
        return None
    tokens = _goal_tokens(goal)
    preferred_names = ("department", "platform", "region", "category", "segment", "group", "team")
    scored = []
    for index, column in enumerate(categorical_columns):
        score = _semantic_column_score(tokens, column)
        normalised = column.lower()
        for rank, name in enumerate(preferred_names):
            if name in normalised:
                score += 50 - rank
        scored.append((score, -index, column))
    return max(scored)[2]


def _goal_mentions_any_column(goal: str | None, columns: list[str]) -> bool:
    tokens = _goal_tokens(goal)
    return any(_column_explicitly_requested(tokens, column) for column in columns)


def _semantic_column_score(goal_tokens: set[str], column: str) -> int:
    column_tokens = set(re.findall(r"[a-z0-9]+", column.lower().replace("_", " ")))
    score = 0
    if column_tokens & goal_tokens:
        score += 100 + len(column_tokens & goal_tokens)
    for token in goal_tokens:
        if token in _semantic_aliases_for(column):
            score += 90
    return score


def _semantic_aliases_for(column: str) -> set[str]:
    normalised = column.lower()
    aliases = set()
    if any(term in normalised for term in ("salary", "income", "pay", "wage", "compensation")):
        aliases.update({"salary", "income", "pay", "wage", "wages", "compensation", "earnings"})
    if "age" in normalised:
        aliases.add("age")
    if any(term in normalised for term in ("sales", "revenue")):
        aliases.update({"sales", "sale", "revenue"})
    if any(term in normalised for term in ("engagement", "likes", "comments", "shares")):
        aliases.update({"engagement", "like", "likes", "comment", "comments", "share", "shares"})
    if any(term in normalised for term in ("promotion", "promoted")):
        aliases.update({"promotion", "promoted", "promote", "predicts"})
    if "attrition" in normalised:
        aliases.update({"attrition", "retention", "churn"})
    if any(term in normalised for term in ("performance", "score")):
        aliases.update({"performance", "score"})
    if "experience" in normalised:
        aliases.update({"experience", "years"})
    if "satisfaction" in normalised:
        aliases.update({"happiness", "morale", "satisfaction"})
    return aliases


def _column_explicitly_requested(goal_tokens: set[str], column: str) -> bool:
    column_tokens = set(re.findall(r"[a-z0-9]+", column.lower().replace("_", " ")))
    return bool(column_tokens and column_tokens.issubset(goal_tokens))


def _goal_tokens(goal: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (goal or "").lower().replace("_", " ")))


def _profile_columns(dataset_profile: dict[str, Any] | None, type_name: str) -> list[str]:
    if not isinstance(dataset_profile, dict):
        return []
    column_types = dataset_profile.get("column_types", {})
    if not isinstance(column_types, dict):
        return []
    values = column_types.get(type_name, [])
    return [str(value) for value in values] if isinstance(values, list) else []


def _profile_issue_columns(dataset_profile: dict[str, Any] | None, issue_name: str) -> list[str]:
    if not isinstance(dataset_profile, dict):
        return []
    issues = dataset_profile.get("potential_issues", {})
    if not isinstance(issues, dict):
        return []
    values = issues.get(issue_name, [])
    return [str(value) for value in values] if isinstance(values, list) else []


def _profile_relationship_exclusions(dataset_profile: dict[str, Any] | None) -> list[str]:
    notes = []
    for column in _profile_issue_columns(dataset_profile, "likely_id_columns"):
        notes.append(f"{column} appears to be an identifier and was excluded.")
    for column in _profile_issue_columns(dataset_profile, "constant_value_columns"):
        notes.append(f"{column} is constant and was excluded.")
    return notes


def _with_profile_exclusions(reason: str, dataset_profile: dict[str, Any] | None) -> str:
    exclusions = _profile_relationship_exclusions(dataset_profile)
    if not exclusions:
        return reason
    return f"{reason} {' '.join(exclusions)}"


def _normalise_text(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower().replace("_", " ")))


def _compact(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _contains_normalised_phrase(goal_text: str, column_text: str) -> bool:
    return bool(re.search(rf"(^| ){re.escape(column_text)}($| )", goal_text))


def _high_confidence_fuzzy_match(goal_text: str, column_text: str) -> bool:
    goal_tokens = goal_text.split()
    column_tokens = column_text.split()
    if not goal_tokens or not column_tokens:
        return False
    width = len(column_tokens)
    if width > len(goal_tokens):
        return False
    for index in range(0, len(goal_tokens) - width + 1):
        candidate = " ".join(goal_tokens[index : index + width])
        ratio = SequenceMatcher(None, candidate, column_text).ratio()
        compact_ratio = SequenceMatcher(None, candidate.replace(" ", ""), column_text.replace(" ", "")).ratio()
        if ratio >= 0.94 and compact_ratio >= 0.94:
            return True
    return False


def _first_token(tokens: set[str], candidates: set[str]) -> str:
    return next((candidate for candidate in sorted(candidates) if candidate in tokens), "")


def _relationship_trigger(tokens: set[str]) -> str:
    phrases = [
        ({"most", "strongly", "related"}, "most strongly related"),
        ({"strongest", "correlations"}, "strongest correlations"),
        ({"strongest", "relationships"}, "strongest relationships"),
        ({"variables", "related"}, "variables related"),
        ({"factors", "related"}, "factors related"),
        ({"associated", "with"}, "associated with"),
        ({"related", "to"}, "related to"),
    ]
    for required, label in phrases:
        if required.issubset(tokens):
            return label
    return _first_token(tokens, {"correlate", "correlates", "correlated", "correlation", "correlations", "relationship", "relationships", "related", "associated"})


def _target_trigger(tokens: set[str]) -> str:
    return _first_token(tokens, {"predict", "predicts", "affect", "affects", "drive", "drives", "influence", "influences", "factor", "factors"})


def _group_trigger(tokens: set[str]) -> str:
    phrases = [
        ({"vary", "by"}, "vary by"),
        ({"differ", "by"}, "differ by"),
        ({"compare", "across"}, "compare across"),
        ({"paid", "most"}, "paid the most"),
    ]
    for required, label in phrases:
        if required.issubset(tokens):
            return label
    return _first_token(tokens, {"by", "across", "group", "groups", "department", "departments", "team", "teams", "category", "categories"})
