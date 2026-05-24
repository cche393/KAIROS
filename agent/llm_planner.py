"""Optional LLM planner for selecting existing KAIROS candidate actions."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agent.tool_registry import TOOL_REGISTRY


DEFAULT_MODEL = "gpt-4o-mini"


def plan_with_llm(
    goal: str,
    dataset_profile: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    max_actions: int = 3,
) -> dict[str, Any]:
    """Select and rank candidate actions with an optional OpenAI call."""
    limit = _safe_limit(max_actions)
    response = _empty_response()

    if not candidate_actions:
        response["warnings"].append("No candidate actions were provided")
        response["selected_actions"] = []
        response["reason"] = "No candidate actions available."
        return response

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback(
            goal,
            candidate_actions,
            limit,
            warnings=["OPENAI_API_KEY is not set"],
            reason="Using deterministic fallback because no API key is configured.",
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
        )
    except Exception as exc:
        return _fallback(
            goal,
            candidate_actions,
            limit,
            errors=[f"LLM planning failed: {exc}"],
            reason="Using deterministic fallback because LLM planning failed.",
        )

    selected_actions, warnings = _actions_from_indexes(parsed, candidate_actions, limit)
    if not selected_actions:
        return _fallback(
            goal,
            candidate_actions,
            limit,
            warnings=warnings + ["No valid LLM-selected action indexes remained"],
            reason="Using deterministic fallback because no valid LLM selections remained.",
        )

    selected_actions = _merge_goal_ranked_actions(goal, candidate_actions, selected_actions, limit)

    return {
        "mode": "llm",
        "selected_actions": selected_actions,
        "reason": str(parsed.get("reason") or "LLM selected candidate action indexes."),
        "errors": [],
        "warnings": warnings,
    }


def _request_llm_plan(
    goal: str,
    dataset_profile: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
) -> str:
    """Call OpenAI and return the raw JSON string from the model."""
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"OpenAI SDK import failed: {exc}") from exc

    client = OpenAI()
    model = os.getenv("KAIROS_LLM_MODEL", DEFAULT_MODEL)
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
                        "If the user asks about correlation, relationship, relatedness, or association between numeric variables, prioritize correlation_analysis.",
                        "If the user asks whether one numeric variable predicts, affects, impacts, or influences another, consider simple_linear_regression and correlation_analysis.",
                        "If the user asks about differences across groups or categories, prioritize group_summary or t_test_by_group depending on binary versus multi-category groups.",
                        "If the user asks whether two categorical variables are related, prioritize chi_square_test.",
                        "If the user asks a broad or vague question such as 'Explore this dataset', choose broad EDA actions such as missing_analysis, numeric_summary, and categorical_summary.",
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
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return completion.choices[0].message.content or "{}"


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
) -> dict[str, Any]:
    return {
        "mode": "fallback",
        "selected_actions": _rank_candidates_for_goal(goal, candidate_actions)[:max_actions],
        "reason": reason,
        "errors": errors or [],
        "warnings": warnings or [],
    }


def _empty_response() -> dict[str, Any]:
    return {
        "mode": "fallback",
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

    if tokens & {"missing", "null", "incomplete", "blank", "empty"}:
        score += _tool_score(tool, {"missing_analysis": 120})

    if tokens & {"correlation", "correlations", "relationship", "relationships", "related", "association", "associations"}:
        score += _tool_score(
            tool,
            {
                "correlation_analysis": 140,
                "simple_linear_regression": 70,
                "numeric_summary": 30,
            },
        )

    if tokens & {"predict", "predicts", "affect", "affects", "effect", "effects", "impact", "impacts", "influence", "influences"}:
        score += _tool_score(
            tool,
            {
                "simple_linear_regression": 125,
                "correlation_analysis": 105,
                "t_test_by_group": 95,
                "group_summary": 85,
                "numeric_summary": 25,
            },
        )

    if tool in {"group_summary", "t_test_by_group"}:
        group_col = action.get("args", {}).get("group_col")
        if _column_name_matches_goal(tokens, group_col):
            score += 80

    if tokens & {"difference", "differences", "compare", "compares", "comparison", "across", "between", "groups", "group"} or " by " in f" {text} ":
        score += _tool_score(
            tool,
            {
                "group_summary": 130,
                "t_test_by_group": 115,
                "chi_square_test": 55,
                "categorical_summary": 25,
            },
        )

    if tokens & {"category", "categories", "categorical", "proportion", "proportions", "distribution", "distributions"}:
        score += _tool_score(
            tool,
            {
                "categorical_summary": 110,
                "chi_square_test": 80,
            },
        )

    if tokens & {"explore", "overview", "summarize", "summarise", "summary", "describe", "eda"}:
        score += _tool_score(
            tool,
            {
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
    }
    return bool(tokens & specific_tokens or "by" in tokens)


def _goal_tokens(goal: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (goal or "").lower().replace("_", " ")))


def _column_name_matches_goal(goal_tokens: set[str], value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    column_tokens = set(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))
    return bool(column_tokens and column_tokens.issubset(goal_tokens))
