"""Deterministic final report synthesis for KAIROS analysis runs."""

from __future__ import annotations

from typing import Any

from agent.result_interpreter import interpret_result


MAX_FINDINGS = 6
MAX_LIMITATIONS = 6
MAX_SUGGESTIONS = 5


def build_final_report(
    user_question: str | None = "",
    dataset_profile: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    planning_trace: dict[str, Any] | None = None,
    selected_actions: list[dict[str, Any]] | None = None,
    execution_results: list[dict[str, Any]] | None = None,
    audit_log_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact final report from existing run metadata."""
    actions = selected_actions or _plan_actions(plan)
    results = execution_results or []
    trace = _as_dict(planning_trace)
    analysis_type = _analysis_type(trace, actions, results, audit_log_entry)
    target_column = _target_column(trace, actions, results)
    verification_notes = _verification_notes(results)

    return {
        "question_answered": str(user_question or "Explore this dataset."),
        "analyses_run": summarize_tools_run(actions, results, trace),
        "key_findings": extract_key_findings(results, user_question=user_question),
        "limitations": generate_limitations(analysis_type, results),
        "suggested_next_analyses": suggest_next_analyses(
            analysis_type,
            dataset_profile=dataset_profile,
            target_column=target_column,
            verification_warnings=verification_notes,
        ),
    }


def summarize_tools_run(
    selected_actions: list[dict[str, Any]] | None,
    execution_results: list[dict[str, Any]] | None,
    planning_trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a short description of the tools involved in this run."""
    actions = selected_actions or []
    results = execution_results or []
    trace = _as_dict(planning_trace)
    tools = _unique(
        [
            str(item.get("tool"))
            for item in list(actions) + list(results)
            if isinstance(item, dict) and item.get("tool")
        ]
    )
    if not tools:
        return []

    return [
        {
            "analysis_type": _analysis_type(trace, actions, results),
            "tools": tools,
            "target_column": _target_column(trace, actions, results),
            "group_column": _first_arg_value(actions, results, "group_col"),
            "value_column": _first_arg_value(actions, results, "value_col"),
            "status": _run_status(results),
        }
    ]


def extract_key_findings(
    execution_results: list[dict[str, Any]] | None,
    user_question: str | None = None,
) -> list[str]:
    """Extract concise findings from deterministic result interpretations."""
    findings = []
    for result in execution_results or []:
        if not isinstance(result, dict) or not result.get("executed"):
            continue
        payload = result.get("result")
        if payload is None or _is_chart_result(payload):
            continue
        missing_findings = _missingness_findings_for_report(payload)
        if missing_findings:
            findings.extend(missing_findings)
            continue
        interpretation = interpret_result(result.get("tool"), payload, user_goal=user_question)
        summary = str(interpretation.get("summary") or "")
        if summary:
            findings.append(summary)
        findings.extend(str(item) for item in interpretation.get("key_findings", []) if item)
    if not findings and _has_blocked_result(execution_results or []):
        findings.append("No factual findings were produced because the selected analysis was blocked or failed.")
    return _unique_similar(findings)[:MAX_FINDINGS]


def describe_analysis_run(analysis: dict[str, Any]) -> str:
    """Return a natural-language summary of one analysis run."""
    data = _as_dict(analysis)
    analysis_type = str(data.get("analysis_type") or "analysis")
    status = str(data.get("status") or "unknown")
    target = str(data.get("target_column") or "")
    group = str(data.get("group_column") or "")
    value = str(data.get("value_column") or target or "")

    if analysis_type == "dataset_overview":
        sentence = "We generated a dataset overview to summarize the dataset structure, column types, and quality notes."
    elif analysis_type in {"missing_analysis", "missingness_analysis"}:
        sentence = "We ran a missing-value analysis to identify which columns contain incomplete data."
    elif analysis_type in {"targeted_relationship_analysis", "target_relationship_analysis"}:
        focus = f" focused on {target}" if target else ""
        sentence = f"We ran a targeted relationship analysis{focus}."
    elif analysis_type == "group_comparison_analysis":
        if value and group:
            sentence = f"We compared {value} across {group} groups."
        else:
            sentence = "We ran a group comparison analysis."
    elif analysis_type in {"relationship_analysis", "global_relationship_analysis"}:
        sentence = "We ran a relationship analysis to look for associations between suitable variables."
    elif analysis_type == "distribution_analysis":
        focus = f" for {target}" if target else ""
        sentence = f"We ran a distribution analysis{focus}."
    else:
        sentence = "We ran the selected analysis."

    if status == "success":
        return f"{sentence} The analysis completed successfully."
    if status == "partial":
        return f"{sentence} The analysis completed partially; some selected steps did not finish."
    if status in {"blocked", "failed"}:
        return f"{sentence.replace('We ran', 'We attempted')} It could not be completed because the selected inputs were not suitable."
    return f"{sentence} The analysis status was recorded as {status}."


def generate_limitations(
    analysis_type: str | None,
    execution_results: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Generate deterministic caveats for the run type and verification outcome."""
    templates = {
        "dataset_overview": [
            "This describes dataset structure only; it does not test relationships.",
        ],
        "targeted_relationship_analysis": [
            "Correlation does not imply causation.",
            "Only suitable numeric variables were included.",
            "Identifier-like columns may be excluded.",
        ],
        "target_relationship_analysis": [
            "Correlation does not imply causation.",
            "Only suitable numeric variables were included.",
            "Identifier-like columns may be excluded.",
        ],
        "relationship_analysis": [
            "Correlation does not imply causation.",
            "Only suitable numeric variables were included.",
        ],
        "global_relationship_analysis": [
            "Correlation does not imply causation.",
            "Only suitable numeric variables were included.",
        ],
        "group_comparison_analysis": [
            "Group summaries describe differences but do not prove causation.",
            "High-cardinality groups may be excluded or summarized.",
        ],
        "missing_analysis": [
            "Missingness patterns do not explain why values are missing.",
        ],
        "missingness_analysis": [
            "Missingness patterns do not explain why values are missing.",
        ],
    }
    notes = list(templates.get(str(analysis_type or ""), []))
    for result in execution_results or []:
        if not isinstance(result, dict):
            continue
        verification = _as_dict(result.get("verification"))
        notes.extend(str(item) for item in verification.get("errors", []) if item)
        notes.extend(str(item) for item in verification.get("warnings", []) if item)
        notes.extend(str(item) for item in result.get("errors", []) if item)
        notes.extend(str(item) for item in result.get("warnings", []) if item)
    if not notes:
        notes.append("This report summarizes executed deterministic tools only.")
    return _unique(notes)[:MAX_LIMITATIONS]


def suggest_next_analyses(
    analysis_type: str | None,
    dataset_profile: dict[str, Any] | None = None,
    target_column: str | None = "",
    verification_warnings: list[str] | None = None,
) -> list[str]:
    """Suggest lightweight next analyses using templates and valid profile columns."""
    templates = {
        "dataset_overview": [
            "Ask about missing values.",
            "Ask what correlates with a numeric variable.",
            "Compare a numeric value across a category.",
        ],
        "targeted_relationship_analysis": [
            "Compare the target variable across a categorical group.",
            "Check missing values in the target variable.",
        ],
        "target_relationship_analysis": [
            "Compare the target variable across a categorical group.",
            "Check missing values in the target variable.",
        ],
        "group_comparison_analysis": [
            "Run a statistical test if the group sizes are suitable.",
            "Check distribution of the numeric variable within each group.",
        ],
        "missing_analysis": [
            "Inspect rows with missing values.",
            "Check whether missingness is concentrated in a group.",
        ],
        "missingness_analysis": [
            "Inspect rows with missing values.",
            "Check whether missingness is concentrated in a group.",
        ],
    }
    suggestions = list(templates.get(str(analysis_type or ""), []))
    numeric_columns, categorical_columns = _profile_columns(dataset_profile)
    if verification_warnings:
        if categorical_columns and numeric_columns:
            suggestions.insert(
                0,
                f"Try a safer query such as comparing {numeric_columns[0]} by {categorical_columns[0]}.",
            )
        elif numeric_columns:
            suggestions.insert(0, f"Try a safer query using valid numeric column {numeric_columns[0]}.")
        elif categorical_columns:
            suggestions.insert(0, f"Try a safer query using valid categorical column {categorical_columns[0]}.")
    if target_column and categorical_columns and str(analysis_type or "") in {
        "targeted_relationship_analysis",
        "target_relationship_analysis",
    }:
        suggestions.append(f"Compare {target_column} across {categorical_columns[0]}.")
    if not suggestions:
        suggestions.extend(["Ask about missing values.", "Request a dataset overview."])
    return _unique(suggestions)[:MAX_SUGGESTIONS]


def summarize_final_report(final_report: dict[str, Any] | None) -> dict[str, Any]:
    """Create a tiny audit-log summary without duplicating the full report."""
    report = _as_dict(final_report)
    if not report:
        return {}
    return {
        "question_answered": str(report.get("question_answered") or ""),
        "top_findings": _as_str_list(report.get("key_findings"))[:2],
        "limitations": _as_str_list(report.get("limitations"))[:2],
        "suggested_next_analyses": _as_str_list(report.get("suggested_next_analyses"))[:2],
    }


def _analysis_type(
    planning_trace: dict[str, Any],
    selected_actions: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
    audit_log_entry: dict[str, Any] | None = None,
) -> str:
    trace_type = planning_trace.get("analysis_type")
    if trace_type:
        return str(trace_type)
    planner = _as_dict(_as_dict(audit_log_entry).get("planner"))
    if planner.get("analysis_type"):
        return str(planner["analysis_type"])
    for result in execution_results:
        payload = _as_dict(_as_dict(result).get("result"))
        if payload.get("analysis_type"):
            return str(payload["analysis_type"])
    for action in selected_actions:
        tool = _as_dict(action).get("tool")
        if tool:
            return _tool_analysis_type(str(tool))
    return "analysis"


def _target_column(
    planning_trace: dict[str, Any],
    selected_actions: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
) -> str:
    for key in ("target_column", "target_col", "target"):
        if planning_trace.get(key):
            return str(planning_trace[key])
    for item in list(selected_actions) + list(execution_results):
        args = _as_dict(_as_dict(item).get("args"))
        for key in ("target_col", "target_column", "value_col", "column"):
            if args.get(key):
                return str(args[key])
        payload = _as_dict(_as_dict(item).get("result"))
        for key in ("target_col", "target_column", "value_col", "column"):
            if payload.get(key):
                return str(payload[key])
    return ""


def _first_arg_value(
    selected_actions: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
    key: str,
) -> str:
    for item in list(selected_actions) + list(execution_results):
        args = _as_dict(_as_dict(item).get("args"))
        if args.get(key):
            return str(args[key])
        payload = _as_dict(_as_dict(item).get("result"))
        if payload.get(key):
            return str(payload[key])
    return ""


def _tool_analysis_type(tool: str) -> str:
    mapping = {
        "dataset_overview": "dataset_overview",
        "missing_analysis": "missing_analysis",
        "missingness_analysis": "missingness_analysis",
        "target_relationship_analysis": "targeted_relationship_analysis",
        "group_comparison_analysis": "group_comparison_analysis",
        "group_summary": "group_comparison_analysis",
        "numeric_summary": "distribution_analysis",
        "numeric_distribution_plot": "distribution_analysis",
    }
    return mapping.get(tool, tool)


def _run_status(results: list[dict[str, Any]]) -> str:
    if not results:
        return "not_run"
    executed = [bool(_as_dict(result).get("executed")) for result in results]
    if all(executed):
        return "success"
    if any(executed):
        return "partial"
    if _has_blocked_result(results):
        return "blocked"
    return "failed"


def _has_blocked_result(results: list[dict[str, Any]]) -> bool:
    return any(not _as_dict(_as_dict(result).get("verification")).get("valid", False) for result in results)


def _verification_notes(results: list[dict[str, Any]]) -> list[str]:
    notes = []
    for result in results:
        data = _as_dict(result)
        verification = _as_dict(data.get("verification"))
        notes.extend(str(item) for item in verification.get("errors", []) if item)
        notes.extend(str(item) for item in verification.get("warnings", []) if item)
        notes.extend(str(item) for item in data.get("errors", []) if item)
        notes.extend(str(item) for item in data.get("warnings", []) if item)
    return _unique(notes)


def _profile_columns(dataset_profile: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    profile = _as_dict(dataset_profile)
    column_types = _as_dict(profile.get("column_types"))
    return _as_str_list(column_types.get("numeric")), _as_str_list(column_types.get("categorical"))


def _plan_actions(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    actions = _as_dict(plan).get("selected_actions")
    return actions if isinstance(actions, list) else []


def _is_chart_result(value: Any) -> bool:
    data = _as_dict(value)
    return {"tool_name", "chart_type", "data"}.issubset(data.keys())


def _missingness_findings_for_report(payload: Any) -> list[str]:
    data = _as_dict(payload)
    analysis_type = str(data.get("analysis_type") or "")
    if analysis_type not in {"missingness_analysis", "missing_analysis"} and not (
        data.get("ranked_missing_columns") or data.get("table")
    ):
        return []
    rows = _as_list(data.get("ranked_missing_columns")) or _as_list(data.get("table"))
    parsed = []
    for row in rows:
        details = _as_dict(row)
        column = details.get("column")
        count = details.get("missing_count")
        percent = details.get("missing_percent")
        if not column or count in (None, 0) or percent is None:
            continue
        parsed.append((str(column), percent, count))
    if not parsed:
        return []
    findings = []
    for index, (column, percent, count) in enumerate(parsed):
        column_text = column[:1].lower() + column[1:]
        if index == 0:
            findings.append(
                f"{column_text} has the highest missingness at {_format_percent_value(percent)} ({count} rows)."
            )
        else:
            findings.append(
                f"{column_text} has {_format_percent_value(percent)} missing values ({count} rows)."
            )
    return findings


def _format_percent_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".") + "%"
    return f"{value}%"


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return unique_values


def _unique_similar(values: list[str]) -> list[str]:
    unique_values = []
    seen = set()
    for value in values:
        text = str(value).strip()
        key = _similarity_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        unique_values.append(text)
    return unique_values


def _similarity_key(value: str) -> str:
    lowered = value.lower().replace("_", " ")
    for phrase in [
        " has the most missing values",
        " has the highest missingness",
        " missing values",
        " missing",
    ]:
        lowered = lowered.replace(phrase, "")
    return " ".join(lowered.split())[:80]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
