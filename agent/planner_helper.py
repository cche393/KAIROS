"""Deterministic action recommendations for KAIROS."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def recommend_actions(
    df: pd.DataFrame,
    max_actions: int | None = None,
    goal: str | None = None,
) -> list[dict[str, Any]]:
    """Recommend candidate analysis actions from simple DataFrame properties."""
    numeric_columns = _numeric_columns(df)
    relationship_numeric_columns = _relationship_numeric_columns(df)
    categorical_columns = _categorical_columns(df)
    binary_categorical_columns = _binary_categorical_columns(df, categorical_columns)
    preferred_numeric = _choose_numeric_column(goal, relationship_numeric_columns)
    preferred_group = _choose_group_column(goal, categorical_columns)
    scoped = _scoped_recommendations(
        df,
        goal,
        relationship_numeric_columns,
        categorical_columns,
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
        recommendations.append(
            _action(
                "correlation_analysis",
                {"columns": relationship_numeric_columns},
                "Check pairwise numeric relationships.",
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


def describe_planning_scope(goal: str | None, df: pd.DataFrame) -> dict[str, Any]:
    """Return a lightweight deterministic trace of the inferred planning scope."""
    tokens = _goal_tokens(goal)
    explicit_columns = _explicit_columns(goal, df)
    target = _choose_target_column(goal, df)
    if tokens & {"missing", "missingness", "null", "nulls", "blank", "blanks", "incomplete"}:
        return {"scope": "missingness", "trigger": _first_token(tokens, {"missing", "missingness", "null", "nulls", "blank", "blanks", "incomplete"}), "target": None}
    if tokens & {"outlier", "outliers", "anomaly", "anomalies"}:
        return {"scope": "one_variable", "trigger": _first_token(tokens, {"outlier", "outliers", "anomaly", "anomalies"}), "target": target}
    if len([column for column in explicit_columns if pd.api.types.is_numeric_dtype(df[column])]) >= 2:
        return {"scope": "explicit_pair", "trigger": "two explicit numeric variables", "target": None}
    if _is_distribution_question(tokens):
        return {"scope": "one_variable", "trigger": _first_token(tokens, {"distribution", "histogram", "spread", "look"}), "target": target}
    if _is_relationship_question(tokens):
        return {
            "scope": "target_driven" if target is not None else "global_relationships",
            "trigger": _relationship_trigger(tokens),
            "target": target,
        }
    if _is_target_question(tokens):
        return {"scope": "target_driven", "trigger": _target_trigger(tokens), "target": target}
    if _is_group_question(tokens):
        return {"scope": "group_comparison", "trigger": _group_trigger(tokens), "target": None}
    if _is_fallback_overview_question(tokens):
        return {"scope": "fallback_overview", "trigger": "open-ended overview", "target": None}
    return {"scope": "legacy_schema_recommendation", "trigger": "", "target": None}


def _action(tool: str, args: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"tool": tool, "args": args, "priority": 0, "reason": reason}


def _scoped_recommendations(
    df: pd.DataFrame,
    goal: str | None,
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
            return [_action("target_relationship_analysis", {"target_col": target}, f"Rank variables associated with target column {target}.")]
        if len(numeric_columns) < 2:
            return [_action("missingness_analysis", {}, "No suitable numeric relationships are available.")]
        return [
            _action(
                "global_relationship_analysis",
                {"cols": numeric_columns, "top_n": 3},
                "Show the strongest non-identifier numeric relationships as cohesive relationship results.",
            )
        ]

    if _is_target_question(tokens):
        target = _choose_target_column(goal, df)
        if target is not None and not _is_explicit_group_comparison(goal, categorical_columns):
            return [_action("target_relationship_analysis", {"target_col": target}, f"Rank variables associated with target column {target}.")]

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
                "Show the strongest non-identifier numeric relationships as cohesive relationship results.",
            )
        ]

    if _is_target_question(tokens):
        target = _choose_target_column(goal, df)
        if target is None:
            return None
        return [_action("target_relationship_analysis", {"target_col": target}, f"Rank variables associated with target column {target}.")]

    if _is_fallback_overview_question(tokens):
        return _fallback_overview_actions(numeric_columns, categorical_columns, preferred_numeric, preferred_group)
    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]


def _relationship_numeric_columns(df: pd.DataFrame) -> list[str]:
    meaningful = [column for column in _numeric_columns(df) if not _is_identifier_like_column(df, column)]
    return meaningful or _numeric_columns(df)


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]


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
    tokens = _goal_tokens(goal)
    matches = []
    for column in df.columns:
        name = str(column)
        if _column_explicitly_requested(tokens, name):
            matches.append(name)
            continue
        column_tokens = set(re.findall(r"[a-z0-9]+", name.lower().replace("_", " ")))
        if len(column_tokens) > 1 and column_tokens.issubset(tokens):
            matches.append(name)
    return matches


def _is_distribution_question(tokens: set[str]) -> bool:
    return bool(tokens & {"distribution", "histogram", "spread"} or {"look", "like"}.issubset(tokens))


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
        tokens & {"correlation", "correlations", "correlated", "relationship", "relationships"}
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
    if explicit:
        return explicit[0]
    tokens = _goal_tokens(goal)
    for column in df.columns:
        if _semantic_column_score(tokens, str(column)) > 0:
            return str(column)
    return None


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
    return _first_token(tokens, {"correlation", "correlations", "correlated", "relationship", "relationships", "related", "associated"})


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
