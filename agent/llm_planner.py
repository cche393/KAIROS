"""Optional LLM planner for selecting existing KAIROS candidate actions."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.groq_client import DEFAULT_GROQ_MODEL, get_llm_config, request_chat_completion
from agent.tool_registry import TOOL_REGISTRY


DEFAULT_MODEL = DEFAULT_GROQ_MODEL


def plan_with_llm(
    goal: str,
    dataset_profile: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    max_actions: int = 3,
) -> dict[str, Any]:
    """Select and rank candidate actions with an optional hosted LLM call."""
    limit = _safe_limit(max_actions)
    response = _empty_response()

    if not candidate_actions:
        response["warnings"].append("No candidate actions were provided")
        response["selected_actions"] = []
        response["reason"] = "No candidate actions available."
        return response

    config = get_llm_config()
    if not config["api_key_configured"]:
        return _fallback(
            goal,
            candidate_actions,
            limit,
            warnings=list(config.get("warnings", [])) + [f"{config['api_key_name']} is not set"],
            reason="Using deterministic fallback because no API key is configured.",
            fallback_cause="no_api_key",
        )

    try:
        raw_content = _request_llm_plan(goal, dataset_profile, candidate_actions)
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        return _fallback(
            goal,
            candidate_actions,
            limit,
            errors=[f"LLM returned invalid JSON: {exc}"],
            reason="Using deterministic fallback because the LLM response was not valid JSON.",
            fallback_cause="llm_invalid_json",
        )
    except Exception as exc:
        return _fallback(
            goal,
            candidate_actions,
            limit,
            errors=[f"LLM planning failed: {exc}"],
            reason="Using deterministic fallback because LLM planning failed.",
            fallback_cause="llm_error",
        )

    selected_actions, warnings = _actions_from_indexes(parsed, candidate_actions, limit)
    if not selected_actions:
        indexes = parsed.get("selected_indexes", [])
        empty_selection = isinstance(indexes, list) and len(indexes) == 0
        return _fallback(
            goal,
            candidate_actions,
            limit,
            warnings=warnings + [
                "LLM returned no selected action indexes"
                if empty_selection
                else "No valid LLM-selected action indexes remained"
            ],
            reason=(
                "Using deterministic fallback because the LLM returned no runnable action."
                if empty_selection
                else "Using deterministic fallback because no valid LLM selections remained."
            ),
            fallback_cause="llm_empty_selection" if empty_selection else "llm_invalid_selection",
        )

    selected_actions = _merge_goal_ranked_actions(goal, candidate_actions, selected_actions, limit)

    return {
        "mode": "llm",
        "selected_actions": selected_actions,
        "reason": str(parsed.get("reason") or "LLM selected candidate action indexes."),
        "errors": [],
        "warnings": list(config.get("warnings", [])) + warnings,
    }


def _request_llm_plan(
    goal: str,
    dataset_profile: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
) -> str:
    """Call the configured hosted LLM and return the raw JSON string."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a controlled planner for a data analysis agent. "
                "You must choose only from the candidate actions. Return JSON only. "
                "Do not invent tools. Do not invent columns. The user's question is "
                "the primary decision signal. Do not automatically choose missing, "
                "numeric, or categorical summaries for every request. Choose analyses "
                "that directly answer the question, and choose at most the requested "
                "maximum number of actions. Prefer broad EDA actions only when the "
                "question is vague or asks to explore the dataset."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_goal": goal or "",
                    "dataset_profile": dataset_profile,
                    "allowed_tool_names": sorted(TOOL_REGISTRY.keys()),
                    "candidate_actions": _numbered_candidates(candidate_actions),
                    "goal_relevant_candidate_indexes": _goal_relevant_indexes(goal, candidate_actions),
                    "selection_guidance": [
                        "You must return at least one selected index whenever candidate_actions is non-empty.",
                        "If the dataset_profile already answers the user directly, select the best lightweight overview action rather than returning an empty list.",
                        "If the user asks about columns, fields, variables, schema, dataset structure, what is in the CSV, or a general dataset description, choose dataset_overview when it is available.",
                        "For dataset overview questions, do not infer extra analytical goals from columns such as salary or department unless the user explicitly asks for comparison, relationship, prediction, correlation, trend, distribution, or visualization.",
                        "If the user asks about one explicit numeric pair, prioritize relationship_analysis when it is available.",
                        "If the user asks about correlation, relationship, relatedness, or association between numeric variables, prioritize relationship_analysis or global_relationship_analysis, with correlation_analysis as a lower-level fallback.",
                        "If the user asks what predicts, affects, impacts, or influences a target variable, prioritize target_relationship_analysis when available.",
                        "If the user asks about differences across groups or categories, prioritize group_comparison_analysis, or group_summary/t_test_by_group if cohesive analyses are unavailable.",
                        "If the user asks whether two categorical variables are related, prioritize chi_square_test.",
                        "If the user asks about one numeric variable's distribution, prioritize distribution_analysis.",
                        "If the user asks a broad or vague question such as 'Explore this dataset', choose broad overview actions such as distribution_analysis, global_relationship_analysis, group_comparison_analysis, and missingness_analysis when available.",
                    ],
                    "response_schema": {
                        "selected_indexes": [0, 1, 2],
                        "reason": "short reason for the selected action order",
                    },
                },
                ensure_ascii=True,
            ),
        },
    ]
    return request_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )


def _actions_from_indexes(
    parsed: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    max_actions: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    selected = []
    seen = set()
    indexes = parsed.get("selected_indexes", [])

    if not isinstance(indexes, list):
        return [], ["LLM selected_indexes was not a list"]

    for index in indexes:
        if len(selected) >= max_actions:
            break
        if not isinstance(index, int):
            warnings.append(f"Discarded non-integer selected index: {index}")
            continue
        if index in seen:
            continue
        seen.add(index)
        if index < 0 or index >= len(candidate_actions):
            warnings.append(f"Discarded out-of-range selected index: {index}")
            continue
        selected.append(candidate_actions[index])

    return selected, warnings


def _fallback(
    goal: str,
    candidate_actions: list[dict[str, Any]],
    max_actions: int,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    reason: str = "Using deterministic fallback.",
    fallback_cause: str = "deterministic_fallback",
) -> dict[str, Any]:
    return {
        "mode": "fallback",
        "fallback_cause": fallback_cause,
        "selected_actions": _rank_candidates_for_goal(goal, candidate_actions)[:max_actions],
        "reason": reason,
        "errors": errors or [],
        "warnings": warnings or [],
    }


def _empty_response() -> dict[str, Any]:
    return {
        "mode": "fallback",
        "fallback_cause": "empty",
        "selected_actions": [],
        "reason": "",
        "errors": [],
        "warnings": [],
    }


def _safe_limit(max_actions: int) -> int:
    try:
        return max(int(max_actions), 0)
    except (TypeError, ValueError):
        return 3


def _numbered_candidates(candidate_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "tool": action.get("tool"),
            "args": action.get("args", {}),
            "priority": action.get("priority"),
            "reason": action.get("reason", ""),
        }
        for index, action in enumerate(candidate_actions)
    ]


def _merge_goal_ranked_actions(
    goal: str,
    candidate_actions: list[dict[str, Any]],
    selected_actions: list[dict[str, Any]],
    max_actions: int,
) -> list[dict[str, Any]]:
    if not _has_specific_intent(goal):
        return selected_actions[:max_actions]

    ranked_matches = _top_specific_matches(goal, candidate_actions)
    if not ranked_matches:
        return selected_actions[:max_actions]
    if any(action in selected_actions for action in ranked_matches):
        return selected_actions[:max_actions]

    merged = []
    for action in ranked_matches + selected_actions:
        if len(merged) >= max_actions:
            break
        if not any(existing is action for existing in merged):
            merged.append(action)
    return merged


def _rank_candidates_for_goal(
    goal: str,
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if _is_dataset_overview_question(goal):
        overview = [action for action in candidate_actions if action.get("tool") == "dataset_overview"]
        if overview:
            return overview
    scored = [
        (_intent_score(goal, action), index, action)
        for index, action in enumerate(candidate_actions)
    ]
    if not any(score > 0 for score, _, _ in scored):
        return candidate_actions
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [action for _, _, action in scored]


def _top_specific_matches(
    goal: str,
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored = [
        (_specific_intent_score(goal, action), index, action)
        for index, action in enumerate(candidate_actions)
    ]
    best_score = max((score for score, _, _ in scored), default=0)
    if best_score <= 0:
        return []
    return [
        action
        for score, _, action in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score == best_score
    ]


def _goal_relevant_indexes(goal: str, candidate_actions: list[dict[str, Any]]) -> list[int]:
    scored = [
        (_intent_score(goal, action), index)
        for index, action in enumerate(candidate_actions)
    ]
    return [index for score, index in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0]


def _intent_score(goal: str, action: dict[str, Any]) -> int:
    tool = str(action.get("tool", ""))
    tokens = _goal_tokens(goal)
    score = 0

    score += _specific_intent_score(goal, action)

    return score


def _specific_intent_score(goal: str, action: dict[str, Any]) -> int:
    tool = str(action.get("tool", ""))
    tokens = _goal_tokens(goal)
    text = " ".join(tokens)
    score = 0

    if _is_dataset_overview_question(goal):
        score += _tool_score(
            tool,
            {
                "dataset_overview": 500,
                "distribution_analysis": -200,
                "global_relationship_analysis": -200,
                "relationship_analysis": -200,
                "group_comparison_analysis": -200,
                "correlation_analysis": -200,
                "group_summary": -200,
            },
        )

    if tokens & {"missing", "null", "incomplete", "blank", "empty"}:
        score += _tool_score(tool, {"missingness_analysis": 140, "missing_analysis": 120, "missing_value_bar_chart": 100})

    if tokens & {"correlate", "correlates", "correlated", "correlation", "correlations", "relationship", "relationships", "related", "association", "associations"}:
        score += _tool_score(
            tool,
            {
                "target_relationship_analysis": 185,
                "relationship_analysis": 170,
                "global_relationship_analysis": 150,
                "correlation_analysis": 140,
                "top_correlation_plots": 135,
                "scatter_plot": 90,
                "simple_linear_regression": 70,
                "numeric_summary": 30,
            },
        )

    if tokens & {"predict", "predicts", "affect", "affects", "effect", "effects", "impact", "impacts", "influence", "influences"}:
        score += _tool_score(
            tool,
            {
                "target_relationship_analysis": 190,
                "relationship_analysis": 145,
                "global_relationship_analysis": 130,
                "simple_linear_regression": 125,
                "top_correlation_plots": 115,
                "correlation_analysis": 105,
                "regression_plot": 100,
                "t_test_by_group": 95,
                "group_summary": 85,
                "group_mean_bar_chart": 80,
                "numeric_summary": 25,
            },
        )

    target_col = action.get("args", {}).get("target_col")
    if tool == "target_relationship_analysis" and _column_name_matches_goal(tokens, target_col):
        score += 120

    if tool in {"group_summary", "t_test_by_group"}:
        group_col = action.get("args", {}).get("group_col")
        if _column_name_matches_goal(tokens, group_col):
            score += 80

    if tokens & {"difference", "differences", "compare", "compares", "comparison", "across", "between", "groups", "group"} or " by " in f" {text} ":
        score += _tool_score(
            tool,
            {
                "group_comparison_analysis": 150,
                "group_summary": 130,
                "group_mean_bar_chart": 125,
                "t_test_by_group": 115,
                "chi_square_test": 55,
                "categorical_summary": 25,
            },
        )

    if tokens & {"category", "categories", "categorical", "proportion", "proportions", "distribution", "distributions"}:
        score += _tool_score(
            tool,
            {
                "distribution_analysis": 135,
                "categorical_summary": 110,
                "chi_square_test": 80,
            },
        )

    if tokens & {"show", "plot", "chart", "visualize", "visualise", "graph"}:
        score += _tool_score(
            tool,
            {
                "global_relationship_analysis": 85,
                "top_correlation_plots": 70,
                "relationship_analysis": 65,
                "scatter_plot": 60,
                "group_comparison_analysis": 58,
                "group_mean_bar_chart": 55,
                "missing_value_bar_chart": 50,
                "distribution_analysis": 48,
                "numeric_distribution_plot": 45,
                "regression_plot": 45,
            },
        )

    if tokens & {"explore", "overview", "summarize", "summarise", "summary", "describe", "eda"}:
        score += _tool_score(
            tool,
            {
                "distribution_analysis": 90,
                "global_relationship_analysis": 85,
                "group_comparison_analysis": 82,
                "missingness_analysis": 50,
                "missing_analysis": 80,
                "numeric_summary": 70,
                "categorical_summary": 60,
            },
        )

    return score


def _tool_score(tool: str, scores: dict[str, int]) -> int:
    return scores.get(tool, 0)


def _has_specific_intent(goal: str) -> bool:
    tokens = _goal_tokens(goal)
    specific_tokens = {
        "missing",
        "null",
        "incomplete",
        "blank",
        "empty",
        "correlation",
        "correlate",
        "correlates",
        "correlated",
        "correlations",
        "relationship",
        "relationships",
        "related",
        "association",
        "associations",
        "predict",
        "predicts",
        "affect",
        "affects",
        "effect",
        "effects",
        "impact",
        "impacts",
        "influence",
        "influences",
        "difference",
        "differences",
        "compare",
        "compares",
        "comparison",
        "across",
        "between",
        "groups",
        "group",
        "category",
        "categories",
        "categorical",
        "proportion",
        "proportions",
        "distribution",
        "distributions",
        "show",
        "plot",
        "chart",
        "visualize",
        "visualise",
        "graph",
    }
    return bool(tokens & specific_tokens or "by" in tokens)


def _is_dataset_overview_question(goal: str) -> bool:
    tokens = _goal_tokens(goal)
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


def _goal_tokens(goal: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (goal or "").lower().replace("_", " ")))


def _column_name_matches_goal(goal_tokens: set[str], value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    column_tokens = set(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))
    return bool(column_tokens and column_tokens.issubset(goal_tokens))
