"""Session-level analysis memory and audit trace helpers for KAIROS."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.final_reporter import summarize_final_report


def create_log_entry(
    step: int | None = None,
    user_question: str | None = "",
    dataset_profile: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    planning_trace: dict[str, Any] | None = None,
    selected_actions: list[dict[str, Any]] | None = None,
    execution_results: list[dict[str, Any]] | None = None,
    final_report: dict[str, Any] | None = None,
    timestamp: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Create a compact audit entry for one analysis request."""
    actions = selected_actions or _plan_actions(plan)
    results = execution_results or []
    trace = planning_trace or {}
    planner = plan or {}
    return {
        "step": int(step or 0),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "user_question": str(user_question or ""),
        "dataset_summary": _dataset_summary(dataset_profile),
        "planner": {
            "mode": _planner_mode(planner),
            "analysis_type": str(trace.get("analysis_type") or ""),
            "analysis_focus": str(trace.get("analysis_focus") or ""),
            "target_column": str(trace.get("target_column") or trace.get("target") or ""),
            "selected_tools": _selected_tools(actions),
            "reason": str(planner.get("reason") or ""),
        },
        "verification": _verification_summary(results),
        "execution": _execution_summary(results),
        "final_report_summary": summarize_final_report(final_report),
    }


def append_log_entry(
    log_entries: list[dict[str, Any]],
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Append an entry, assigning the next session step number when needed."""
    next_step = _next_step(log_entries)
    stored = dict(entry)
    if not stored.get("step"):
        stored["step"] = next_step
    log_entries.append(stored)
    return stored


def summarize_log_entries(log_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact display rows for the analysis memory UI."""
    rows = []
    for entry in log_entries:
        planner = _as_dict(entry.get("planner"))
        verification = _as_dict(entry.get("verification"))
        execution = _as_dict(entry.get("execution"))
        rows.append(
            {
                "step": int(entry.get("step") or 0),
                "user_question": str(entry.get("user_question") or ""),
                "planner_mode": str(planner.get("mode") or "deterministic"),
                "analysis_type": str(planner.get("analysis_type") or ""),
                "selected_tools": _as_list(planner.get("selected_tools")),
                "verification_status": str(verification.get("status") or "failed"),
                "execution_status": str(execution.get("status") or "failed"),
                "result_summary": str(execution.get("result_summary") or ""),
            }
        )
    return rows


def export_log_jsonl(
    log_entries: list[dict[str, Any]],
    path: str | Path = "logs/analysis_log.jsonl",
) -> Path:
    """Append compact log entries to a JSONL file and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for entry in log_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return target


def _dataset_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
    data = profile if isinstance(profile, dict) else {}
    shape = _as_dict(data.get("shape"))
    column_types = _as_dict(data.get("column_types"))
    return {
        "rows": int(data.get("row_count") or shape.get("rows") or 0),
        "columns": int(data.get("column_count") or shape.get("columns") or 0),
        "numeric_columns": _as_str_list(column_types.get("numeric")),
        "categorical_columns": _as_str_list(column_types.get("categorical")),
        "datetime_columns": _as_str_list(
            column_types.get("datetime") or column_types.get("datetime_like")
        ),
    }


def _planner_mode(plan: dict[str, Any]) -> str:
    mode = str(plan.get("mode") or "deterministic")
    return "deterministic" if mode == "fallback" else mode


def _plan_actions(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    actions = plan.get("selected_actions")
    return actions if isinstance(actions, list) else []


def _selected_tools(actions: list[dict[str, Any]]) -> list[str]:
    return [
        str(action.get("tool"))
        for action in actions
        if isinstance(action, dict) and action.get("tool")
    ]


def _verification_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    warnings = []
    valid_values = []
    for result in results:
        if not isinstance(result, dict):
            continue
        verification = _as_dict(result.get("verification"))
        valid_values.append(bool(verification.get("valid")))
        warnings.extend(_as_str_list(verification.get("warnings")))
        warnings.extend(_as_str_list(result.get("warnings")))
    warnings = list(dict.fromkeys(warnings))
    if valid_values and all(valid_values):
        status = "warning" if warnings else "passed"
    elif any(valid_values):
        status = "warning"
    else:
        status = "failed"
    return {"status": status, "warnings": warnings}


def _execution_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    tools_run = [
        str(result.get("tool"))
        for result in results
        if isinstance(result, dict) and result.get("executed") and result.get("tool")
    ]
    executed_count = len(tools_run)
    total_count = len([result for result in results if isinstance(result, dict)])
    if total_count == 0 or executed_count == 0:
        status = "failed"
    elif executed_count == total_count:
        status = "success"
    else:
        status = "partial"
    return {
        "tools_run": tools_run,
        "status": status,
        "result_summary": _result_summary(results),
    }


def _result_summary(results: list[dict[str, Any]]) -> str:
    summaries = []
    for result in results:
        if not isinstance(result, dict) or not result.get("executed"):
            continue
        payload = _as_dict(result.get("result"))
        summary = payload.get("summary")
        if summary:
            summaries.append(str(summary))
    if summaries:
        return " ".join(summaries)[:500]
    if any(isinstance(result, dict) and result.get("errors") for result in results):
        return "One or more selected tools failed before producing a result."
    return "No executed result summary was available."


def _next_step(log_entries: list[dict[str, Any]]) -> int:
    existing = [int(entry.get("step") or 0) for entry in log_entries if isinstance(entry, dict)]
    return max(existing, default=0) + 1


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
