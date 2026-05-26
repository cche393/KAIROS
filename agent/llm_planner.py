"""LLM planner for selecting existing KAIROS candidate actions."""

from __future__ import annotations

import json
from typing import Any

from agent.groq_client import get_llm_config, request_chat_completion
from agent.tool_registry import TOOL_REGISTRY


def plan_with_llm(
    goal: str,
    dataset_profile: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    max_actions: int = 3,
) -> dict[str, Any]:
    """Select and rank candidate actions with an optional provider-aware LLM call."""
    limit = _safe_limit(max_actions)
    response = _empty_response()

    if not candidate_actions:
        response["warnings"].append("No candidate actions were provided")
        response["selected_actions"] = []
        response["reason"] = "No candidate actions available."
        return response

    config = get_llm_config()
    if config.get("provider") == "deterministic":
        return _fallback(
            candidate_actions,
            limit,
            warnings=list(config.get("warnings", [])),
            reason="Using deterministic planner mode.",
            cause="deterministic_mode",
            goal=goal,
        )

    if not config.get("api_key_configured"):
        key_name = str(config.get("api_key_name") or "API key")
        return _fallback(
            candidate_actions,
            limit,
            warnings=list(config.get("warnings", [])) + [f"{key_name} is not set"],
            reason="Using deterministic fallback because no API key is configured.",
            cause="no_api_key",
            goal=goal,
        )

    try:
        raw_content = _request_llm_plan(goal, dataset_profile, candidate_actions)
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        return _fallback(
            candidate_actions,
            limit,
            errors=[f"LLM returned invalid JSON: {exc}"],
            reason="Using deterministic fallback because the LLM response was not valid JSON.",
            cause="llm_invalid_json",
            goal=goal,
        )
    except Exception as exc:
        return _fallback(
            candidate_actions,
            limit,
            errors=[f"LLM planning failed: {exc}"],
            reason="Using deterministic fallback because LLM planning failed.",
            cause="llm_error",
            goal=goal,
        )

    selected_actions, warnings = _actions_from_indexes(parsed, candidate_actions, limit)
    if not selected_actions:
        fallback_cause = "llm_empty_selection" if parsed.get("selected_indexes") == [] else "llm_invalid_selection"
        if parsed.get("selected_indexes") == []:
            warnings.append("LLM returned no selected action indexes")
        return _fallback(
            candidate_actions,
            limit,
            warnings=warnings + ["No valid LLM-selected action indexes remained"],
            reason="Using deterministic fallback because no valid LLM selections remained.",
            cause=fallback_cause,
            goal=goal,
        )

    if _needs_goal_correction(goal):
        selected_actions = _fallback_actions(candidate_actions, limit, goal)
    else:
        selected_actions = selected_actions[:limit]
    return {
        "mode": "llm",
        "fallback_cause": "",
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
    """Call the configured hosted provider and return the raw JSON string."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a controlled planner for a data analysis agent. "
                "You must choose only from the candidate actions. Return JSON only. "
                "Do not invent tools. Do not invent columns. Prefer safe broad checks "
                "before narrow tests. Choose actions that best match the user goal."
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
        messages,
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
    candidate_actions: list[dict[str, Any]],
    max_actions: int,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    reason: str = "Using deterministic fallback.",
    cause: str = "no_api_key",
    goal: str | None = None,
) -> dict[str, Any]:
    return {
        "mode": "fallback",
        "fallback_cause": cause,
        "selected_actions": _fallback_actions(candidate_actions, max_actions, goal),
        "reason": reason,
        "errors": errors or [],
        "warnings": warnings or [],
    }


def _empty_response() -> dict[str, Any]:
    return {
        "mode": "fallback",
        "fallback_cause": "",
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


def _fallback_actions(
    candidate_actions: list[dict[str, Any]],
    max_actions: int,
    goal: str | None,
) -> list[dict[str, Any]]:
    if max_actions <= 0:
        return []
    if _is_schema_question(goal):
        overview = [action for action in candidate_actions if action.get("tool") == "dataset_overview"]
        if overview:
            return overview[:1]
    return _rank_actions_for_goal(goal, candidate_actions)[:max_actions]


def _rank_actions_for_goal(
    goal: str | None,
    candidate_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = str(goal or "").lower()
    if not text:
        return candidate_actions
    scored = [
        (_score_action(action, text), index, action)
        for index, action in enumerate(candidate_actions)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [action for _, _, action in scored]


def _score_action(action: dict[str, Any], goal_text: str) -> int:
    tool = str(action.get("tool") or "")
    reason = str(action.get("reason") or "").lower()
    args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
    score = 0
    if _is_relationship_question(goal_text):
        if tool in {"relationship_analysis", "target_relationship_analysis", "global_relationship_analysis"}:
            score += 120
        if tool in {"correlation_analysis", "simple_linear_regression", "top_correlation_plots"}:
            score += 90
        if tool in {"missing_analysis", "categorical_summary"}:
            score -= 20
    if _is_group_question(goal_text):
        if tool in {"group_summary", "group_comparison_analysis", "t_test_by_group"}:
            score += 120
        if tool in {"missing_analysis", "categorical_summary"}:
            score -= 20
    if _is_missing_question(goal_text) and tool in {"missing_analysis", "missingness_analysis"}:
        score += 120
    if _is_distribution_question(goal_text) and tool in {"distribution_analysis", "numeric_summary", "numeric_distribution_plot"}:
        score += 90
    if any(str(value).lower() in goal_text for value in _arg_values(args)):
        score += 25
    if reason and any(word in reason for word in goal_text.split()):
        score += 5
    return score


def _arg_values(args: dict[str, Any]) -> list[str]:
    values = []
    for value in args.values():
        if isinstance(value, list):
            values.extend(str(item).replace("_", " ") for item in value)
            values.extend(str(item) for item in value)
        else:
            values.append(str(value).replace("_", " "))
            values.append(str(value))
    return [value for value in values if value]


def _is_schema_question(goal: str | None) -> bool:
    text = str(goal or "").lower()
    return any(phrase in text for phrase in ["what columns", "what variables", "variables available", "what fields", "describe this dataset"])


def _is_relationship_question(goal_text: str) -> bool:
    return any(
        phrase in goal_text
        for phrase in [
            "correlation",
            "correlate",
            "correlates",
            "related",
            "relationship",
            "associated",
            "affect",
            "affects",
            "predict",
            "strongest",
        ]
    )


def _needs_goal_correction(goal: str | None) -> bool:
    return _is_relationship_question(str(goal or "").lower())


def _is_group_question(goal_text: str) -> bool:
    return any(
        phrase in goal_text
        for phrase in [
            " by ",
            "compare",
            "across",
            "group",
            "department",
            "segment",
            "region",
            "remote work",
        ]
    )


def _is_missing_question(goal_text: str) -> bool:
    return "missing" in goal_text or "null" in goal_text


def _is_distribution_question(goal_text: str) -> bool:
    return "distribution" in goal_text or "summar" in goal_text or "histogram" in goal_text
