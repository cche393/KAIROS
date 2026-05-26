"""Streamlit UI for manually testing the KAIROS pipeline."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from agent.executor import execute_action
from agent.final_reporter import build_final_report, describe_analysis_run
from agent.groq_client import get_llm_config
from agent.llm_planner import plan_with_llm
from agent.memory_log import (
    append_log_entry,
    create_log_entry,
    export_log_jsonl,
    summarize_log_entries,
)
from agent.observer import inspect_dataset, load_csv
from agent.planner_helper import describe_planning_scope, recommend_actions
from agent.result_interpreter import interpret_result
from ui.styles import dashboard_css


DEFAULT_GOAL = "Explore this dataset."

ACTION_LABELS = {
    "dataset_overview": "Dataset overview",
    "distribution_analysis": "Distribution analysis",
    "relationship_analysis": "Relationship analysis",
    "target_relationship_analysis": "Target relationship analysis",
    "global_relationship_analysis": "Strongest relationships",
    "group_comparison_analysis": "Group comparison",
    "outlier_analysis": "Outlier analysis",
    "missingness_analysis": "Missingness analysis",
    "missing_analysis": "Check missing values",
    "numeric_summary": "Summarise numeric columns",
    "categorical_summary": "Summarise categories",
    "correlation_analysis": "Check numeric relationships",
    "group_summary": "Compare groups",
    "target_group_summary": "Summarise by target",
    "simple_linear_regression": "Fit simple linear relationship",
    "chi_square_test": "Test relationship between categories",
    "t_test_by_group": "Compare two group means",
    "numeric_distribution_plot": "Visualise numeric distribution",
    "scatter_plot": "Visualise numeric relationship",
    "top_correlation_plots": "Visualise strongest relationships",
    "group_mean_bar_chart": "Visualise group averages",
    "missing_value_bar_chart": "Visualise missing values",
    "regression_plot": "Visualise fitted relationship",
    "outlier_detection": "Detect potential outliers",
}

ARG_LABELS = {
    "column": "Column",
    "columns": "Columns",
    "cols": "Columns",
    "group_col": "Group column",
    "value_col": "Value column",
    "target_col": "Target column",
    "feature_col": "X column",
    "x_col": "X column",
    "y_col": "Y column",
    "col_a": "First category column",
    "col_b": "Second category column",
    "top_n": "Maximum items",
    "max_points": "Maximum points",
    "bins": "Bins",
    "include_zero": "Include complete columns",
}


def main() -> None:
    st.set_page_config(page_title="KAIROS", layout="wide")
    _inject_css()
    _ensure_analysis_memory_state()

    left, right = st.columns([0.9, 3.0], gap="medium")
    with left:
        uploaded_file, effective_goal = _agent_panel()

    with right:
        _workspace_panel(uploaded_file, effective_goal)


def _agent_panel() -> tuple[Any, str]:
    st.markdown('<div class="agent-panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="kairos-hero">
          <p class="kairos-subtitle">Guarded LLM-Guided Data Analysis Agent</p>
          <h1 class="kairos-title">KAIROS</h1>
          <p class="kairos-description">Upload a dataset, ask an analytical question, and KAIROS plans, verifies, executes, and summarizes the analysis.</p>
          <div class="badge-row">
            <span class="status-badge">LLM-guided planner</span>
            <span class="status-badge success">Verified tools</span>
            <span class="status-badge">Dataset-aware</span>
            <span class="status-badge muted">Audit trace</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="control-card"><div class="section-kicker">Planner status</div>', unsafe_allow_html=True)
    llm_config = get_llm_config()
    status_kind, status_text = _planner_availability_status(llm_config)
    badge_class = "success" if status_kind == "success" else "muted"
    st.markdown(f'<span class="status-badge {badge_class}">{status_text}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="control-card"><div class="section-kicker">Run configuration</div>', unsafe_allow_html=True)
    if status_kind == "success":
        st.caption("Hosted planner is available for action selection.")
    else:
        st.caption("KAIROS can still use deterministic planning and guarded tools.")
    max_actions = st.number_input(
        "Maximum analyses",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        key="max_selected_actions",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="question-card"><div class="section-kicker">Dataset and question</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    _sync_memory_to_uploaded_file(uploaded_file)
    question = st.text_area(
        "Analysis question",
        placeholder="Ask something like: What correlates with age?",
        height=130,
    )
    effective_goal = question.strip() or DEFAULT_GOAL
    if not question.strip():
        st.caption(f"Default question: {DEFAULT_GOAL}")

    if st.button("Generate and run analysis", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Upload a CSV dataset before running analysis.")
        else:
            _generate_and_run(uploaded_file, effective_goal, int(max_actions))
    st.markdown("</div>", unsafe_allow_html=True)

    _show_agent_explanation()

    with st.expander("Other possible analyses", expanded=False):
        bundle = st.session_state.get("analysis_bundle")
        if bundle:
            _show_action_cards(bundle["candidate_actions"], empty_message="No candidate analyses were generated.")
        else:
            st.caption("Candidate analyses will appear after KAIROS profiles the dataset.")

    _show_analysis_memory_log()

    st.caption("Follow-up conversational analysis can be added here later.")
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file, effective_goal


def _workspace_panel(uploaded_file: Any, effective_goal: str) -> None:
    st.markdown('<div class="workspace-panel"><div class="section-kicker">Operational intelligence workspace</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="workspace-title">Analysis workspace</h2>', unsafe_allow_html=True)

    if uploaded_file is None:
        _empty_workspace()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    bundle = st.session_state.get("analysis_bundle")
    if not bundle or bundle.get("file_key") != _file_key(uploaded_file):
        st.info("Enter a question, then select Generate and run analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if bundle.get("goal") != effective_goal:
        st.info("The question has changed. Select Generate and run analysis to refresh the analysis.")

    _show_dataset_overview(bundle["profile"], bundle["df"])
    _show_analysis_results(
        bundle["selected_actions"],
        bundle["execution_results"],
        bundle.get("goal"),
    )
    _show_final_report(bundle.get("final_report"))
    st.markdown("</div>", unsafe_allow_html=True)


def _generate_and_run(uploaded_file: Any, goal: str, max_actions: int) -> None:
    try:
        uploaded_file.seek(0)
        df = load_csv(uploaded_file)
    except ValueError as exc:
        st.error(f"Could not load CSV: {exc}")
        st.session_state.pop("analysis_bundle", None)
        return
    except Exception as exc:
        st.error(f"Unexpected CSV loading error: {exc}")
        st.session_state.pop("analysis_bundle", None)
        return

    profile = inspect_dataset(df)
    if df.empty and len(df.columns) == 0:
        st.warning("The uploaded CSV is empty or has no readable columns.")

    candidate_actions = recommend_actions(df, goal=goal, dataset_profile=profile)
    planning_trace = describe_planning_scope(goal, df, dataset_profile=profile)
    plan = plan_with_llm(goal, profile, candidate_actions, max_actions=max_actions)
    selected_actions = plan["selected_actions"]
    execution_results = [execute_action(df, action, dataset_profile=profile) for action in selected_actions]
    final_report = build_final_report(
        user_question=goal,
        dataset_profile=profile,
        plan=plan,
        planning_trace=planning_trace,
        selected_actions=selected_actions,
        execution_results=execution_results,
    )
    memory_entry = create_log_entry(
        user_question=goal,
        dataset_profile=profile,
        plan=plan,
        planning_trace=planning_trace,
        selected_actions=selected_actions,
        execution_results=execution_results,
        final_report=final_report,
    )
    stored_memory_entry = append_log_entry(st.session_state["analysis_memory_log"], memory_entry)
    _export_memory_entry(stored_memory_entry)

    st.session_state["analysis_bundle"] = {
        "file_key": _file_key(uploaded_file),
        "df": df,
        "goal": goal,
        "profile": profile,
        "candidate_actions": candidate_actions,
        "planning_trace": planning_trace,
        "plan": plan,
        "selected_actions": selected_actions,
        "execution_results": execution_results,
        "final_report": final_report,
    }


def _ensure_analysis_memory_state() -> None:
    if "analysis_memory_log" not in st.session_state:
        st.session_state["analysis_memory_log"] = []
    if "analysis_memory_file_key" not in st.session_state:
        st.session_state["analysis_memory_file_key"] = None


def _sync_memory_to_uploaded_file(uploaded_file: Any) -> None:
    _ensure_analysis_memory_state()
    current_key = _file_key(uploaded_file) if uploaded_file is not None else None
    previous_key = st.session_state.get("analysis_memory_file_key")
    if current_key != previous_key:
        st.session_state["analysis_memory_log"] = []
        st.session_state["analysis_memory_file_key"] = current_key
        if previous_key is not None:
            st.session_state.pop("analysis_bundle", None)


def _export_memory_entry(entry: dict[str, Any]) -> None:
    try:
        export_log_jsonl([entry])
    except Exception as exc:
        st.session_state.setdefault("analysis_memory_export_warnings", []).append(str(exc))


def _show_agent_explanation() -> None:
    bundle = st.session_state.get("analysis_bundle")
    if not bundle:
        st.markdown("### KAIROS response")
        st.info("Ask a question about the uploaded dataset. KAIROS will profile the data, choose suitable analyses, and run them through the guarded executor.")
        return

    plan = bundle["plan"]
    llm_config = get_llm_config()
    provider_name = str(llm_config["provider"]).upper()
    status = _planner_status_messages(plan, provider_name)
    mode_text = status["mode_text"]
    st.markdown("### KAIROS response")
    if plan.get("mode") == "llm":
        st.success(mode_text)
    else:
        st.info(mode_text)
        if status.get("warning"):
            st.warning(status["warning"])

    st.markdown("**Interpreted question**")
    st.write(bundle["goal"])
    trace = bundle.get("planning_trace", {})
    if trace.get("target_column"):
        st.write(f"Detected target variable: {trace['target_column']}")
        target_type = _profile_column_type(bundle.get("profile", {}), trace["target_column"])
        if target_type:
            st.caption(f"{trace['target_column']} was detected as a {target_type} variable.")
    if trace.get("analysis_type") == "targeted_relationship_analysis":
        st.write("Analysis mode: targeted relationship analysis")
    elif trace.get("analysis_focus"):
        st.write(f"Analysis focus: {trace['analysis_focus']}")

    st.markdown("**Chosen analyses**")
    selected = bundle["selected_actions"]
    if selected:
        for action in selected:
            st.write(f"- {_action_label(action)}: {action.get('reason', 'Selected for this dataset.')}")
    else:
        st.caption("No analyses were selected.")

    _show_messages("Planner warning", _visible_planner_warnings(plan.get("warnings", [])), st.warning)
    _show_messages("Planner error", plan.get("errors", []), st.error)

    with st.expander("Technical planning details", expanded=False):
        st.write("Question sent to planner")
        st.code(bundle["goal"], language="text")
        st.write(f"Planner mode: `{plan.get('mode', 'unknown')}`")
        st.write("Deterministic scope trace")
        st.json(bundle.get("planning_trace", {}))
        st.write("Selected candidate indexes")
        st.json(_selected_candidate_indexes(bundle["candidate_actions"], selected))
        tech_tabs = st.tabs(["Candidate action list", "Planner response"])
        with tech_tabs[0]:
            st.json(bundle["candidate_actions"])
        with tech_tabs[1]:
            st.json(plan)


def _empty_workspace() -> None:
    with st.container(border=True):
        st.markdown("### Dataset preview")
        st.caption("Upload a CSV dataset to begin.")
    with st.container(border=True):
        st.markdown("### Dataset overview")
        st.caption("The dataset profile will appear after analysis is generated.")
    with st.container(border=True):
        st.markdown("### Selected analyses and results")
        st.caption("KAIROS will show selected analyses and execution results here.")


def _show_dataset_overview(profile: dict[str, Any], df: pd.DataFrame) -> None:
    st.markdown("### Dataset overview")
    shape = profile["shape"]
    column_types = profile.get("column_types", {})
    missing_values = profile.get("missing_values", {})

    cols = st.columns(6)
    cols[0].metric("Rows", shape["rows"])
    cols[1].metric("Columns", shape["columns"])
    cols[2].metric("Numeric", len(column_types.get("numeric", [])))
    cols[3].metric("Categorical", len(column_types.get("categorical", [])))
    cols[4].metric("Datetime", len(column_types.get("datetime", column_types.get("datetime_like", []))))
    missing_columns = [
        column for column, values in missing_values.get("columns", {}).items()
        if values.get("missing_count", 0)
    ]
    cols[5].metric("Missing cols", len(missing_columns))

    st.markdown("**Column types**")
    type_summary_cols = st.columns(3)
    with type_summary_cols[0]:
        _pill_list("Numeric", column_types.get("numeric", []))
    with type_summary_cols[1]:
        _pill_list("Categorical", column_types.get("categorical", []))
    with type_summary_cols[2]:
        _pill_list("Datetime", column_types.get("datetime", column_types.get("datetime_like", [])))

    tabs = st.tabs(["Preview", "Columns", "Detected types", "Missing values", "Technical details"])
    with tabs[0]:
        if df.empty:
            st.info("No rows are available to preview.")
        else:
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    with tabs[1]:
        _pill_list("Column names", profile.get("columns", []))
    with tabs[2]:
        type_cols = st.columns(2)
        with type_cols[0]:
            _pill_list("Numeric columns", column_types.get("numeric", []))
            _pill_list("Categorical columns", column_types.get("categorical", []))
            _pill_list("Boolean columns", column_types.get("boolean", []))
        with type_cols[1]:
            _pill_list("Datetime-like columns", column_types.get("datetime_like", column_types.get("datetime", [])))
            _pill_list("Text-like columns", column_types.get("text_like", []))
    with tabs[3]:
        rows = [
            {
                "Column": column,
                "Missing count": values["missing_count"],
                "Missing percent": values["missing_percent"],
            }
            for column, values in missing_values.get("columns", {}).items()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No missing-value summary is available.")
        high_missing = missing_values.get("high_missingness_columns", [])
        if high_missing:
            st.warning(f"High-missingness columns: {', '.join(high_missing)}")
    with tabs[4]:
        st.json(profile)


def _show_action_cards(actions: list[dict[str, Any]], empty_message: str) -> None:
    if not actions:
        st.info(empty_message)
        return

    for index, action in enumerate(actions, start=1):
        friendly_name = _action_label(action)
        with st.container(border=True):
            st.markdown(f"**{index}. {friendly_name}**")
            if action.get("reason"):
                st.write(action["reason"])
            _show_friendly_uses(action.get("args", {}))
            with st.expander("Technical details", expanded=False):
                st.write(f"Tool name: `{action.get('tool', 'unknown_tool')}`")
                if action.get("args"):
                    st.json(action["args"])
                else:
                    st.caption("No parameters are required for this analysis.")


def _show_analysis_memory_log() -> None:
    log_entries = st.session_state.get("analysis_memory_log", [])
    with st.expander("Analysis Memory / Audit Log", expanded=False):
        if not log_entries:
            st.caption("No analysis steps have been recorded for this dataset yet.")
            return

        if st.button("Clear log", use_container_width=True):
            st.session_state["analysis_memory_log"] = []
            st.rerun()

        export_warnings = st.session_state.get("analysis_memory_export_warnings", [])
        _show_messages("Audit log export warning", export_warnings, st.warning)

        for row, raw_entry in zip(summarize_log_entries(log_entries), log_entries):
            title = row["analysis_type"] or "analysis step"
            st.markdown(f"**Step {row['step']} - {title.replace('_', ' ').title()}**")
            st.write(f"Question: {row['user_question']}")
            st.write(f"Planner: {_planner_mode_label(row['planner_mode'])}")
            tools = ", ".join(row["selected_tools"]) if row["selected_tools"] else "None"
            st.write(f"Tools: {tools}")
            st.write(f"Verification: {row['verification_status']}")
            st.write(f"Summary: {row['result_summary']}")
            with st.expander(f"Show raw log entry for step {row['step']}", expanded=False):
                st.json(raw_entry)


def _planner_mode_label(mode: str) -> str:
    if mode == "llm":
        return "LLM planner"
    return "deterministic fallback"


def _show_friendly_uses(args: dict[str, Any]) -> None:
    if not args:
        return
    st.markdown("**Uses**")
    for arg_name, value in args.items():
        label = ARG_LABELS.get(arg_name, arg_name.replace("_", " ").title())
        if isinstance(value, list):
            value_text = ", ".join(str(item) for item in value)
        else:
            value_text = str(value)
        st.write(f"{label}: {value_text}")


def _show_analysis_results(
    actions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    user_goal: str | None = None,
) -> None:
    st.markdown("### Selected analyses and results")
    sections = _cohesive_analysis_sections(actions, results)
    if not sections:
        st.info("No analyses were selected for this dataset.")
        return

    for index, section in enumerate(sections, start=1):
        action = section["action"]
        result = section["result"]
        chart_action = section.get("chart_action")
        chart_result = section.get("chart_result")
        with st.container(border=True):
            st.markdown(f"**{index}. {_section_title(action, chart_result)}**")
            if action.get("reason"):
                st.write(action["reason"])
            _show_friendly_uses(action.get("args", {}))

            if result is None:
                st.warning("This analysis did not return an execution result.")
                continue

            execution_check = _execution_check_message(result)
            st.markdown("**Execution check**")
            st.write(execution_check["summary"])
            for detail in execution_check["details"]:
                if result.get("executed"):
                    st.warning(detail)
                else:
                    st.error(detail)

            interpretation = None
            if result.get("result") is not None:
                interpretation = interpret_result(
                    result.get("tool"),
                    result.get("result"),
                    user_goal=user_goal,
                )
                _show_interpretation_summary(interpretation)

            _show_result(result.get("tool"), result["result"])

            if chart_result is not None:
                st.markdown("**Chart**")
                _show_result(chart_result.get("tool"), chart_result.get("result"))
                _show_messages("Chart warning", chart_result.get("warnings", []), st.warning)

            if interpretation is not None:
                _show_interpretation_details(interpretation)

            with st.expander("Technical details", expanded=False):
                tab_names = ["Analysis request", "Verification", "Executor response"]
                if chart_result is not None:
                    tab_names.append("Chart response")
                tech_tabs = st.tabs(tab_names)
                with tech_tabs[0]:
                    st.write(f"Tool name: `{result.get('tool', 'unknown_tool')}`")
                    raw_args = result.get("args") or action.get("args", {})
                    if raw_args:
                        st.json(raw_args)
                    else:
                        st.caption("No parameters were required.")
                with tech_tabs[1]:
                    st.json(result["verification"])
                with tech_tabs[2]:
                    st.json(result)
                if chart_result is not None:
                    with tech_tabs[3]:
                        st.write(f"Chart helper: `{chart_action.get('tool', 'unknown_tool')}`")
                        st.json(chart_result)


def _show_final_report(final_report: dict[str, Any] | None) -> None:
    if not isinstance(final_report, dict) or not final_report:
        return

    st.markdown("### Final Analysis Report")
    with st.container(border=True):
        question = final_report.get("question_answered") or DEFAULT_GOAL
        st.markdown("**Question answered:**")
        st.write(str(question))

        analyses = final_report.get("analyses_run", [])
        if analyses:
            st.markdown("**Analyses run:**")
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    continue
                st.write(f"- {describe_analysis_run(analysis)}")

        _show_report_list("Key findings", final_report.get("key_findings", []))
        with st.expander("Limitations and suggested next analyses", expanded=True):
            _show_report_list("Limitations", final_report.get("limitations", []))
            _show_report_list(
                "Suggested next analyses",
                final_report.get("suggested_next_analyses", []),
            )


def _show_report_list(title: str, values: Any) -> None:
    items = values if isinstance(values, list) else []
    st.markdown(f"**{title}:**")
    if not items:
        st.caption("No items available.")
        return
    for item in items:
        st.write(f"- {item}")


def _visible_dataset_quality_notes(profile: dict[str, Any]) -> list[str]:
    return []


def _execution_check_message(result: dict[str, Any]) -> dict[str, Any]:
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    warnings = _unique_texts(
        list(verification.get("warnings", []) or []) + list(result.get("warnings", []) or [])
    )
    errors = _unique_texts(
        list(verification.get("errors", []) or []) + list(result.get("errors", []) or [])
    )
    if not result.get("executed"):
        return {
            "summary": "The analysis could not be executed because the verifier found an incompatible tool or column selection.",
            "details": errors,
        }
    if warnings:
        return {
            "summary": "This analysis was executed with verification warnings. The verifier checked column types, dataset structure, and tool compatibility before execution.",
            "details": warnings,
        }
    return {
        "summary": "This analysis was successfully executed after the verifier confirmed that the selected columns and analysis type were compatible with the dataset structure. No verification warnings were detected.",
        "details": [],
    }


def _unique_texts(values: list[Any]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return unique_values


def _analysis_result_pairs(
    actions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    return [
        (action, results[index] if index < len(results) else None)
        for index, action in enumerate(actions)
    ]


def _cohesive_analysis_sections(
    actions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = _analysis_result_pairs(actions, results)
    consumed = set()
    sections = []
    for index, (action, result) in enumerate(pairs):
        if index in consumed:
            continue
        section = {"action": action, "result": result, "chart_action": None, "chart_result": None}
        chart_index = _matching_chart_index(index, pairs)
        if chart_index is not None:
            section["chart_action"] = pairs[chart_index][0]
            section["chart_result"] = pairs[chart_index][1]
            consumed.add(chart_index)
        sections.append(section)
    return sections


def _matching_chart_index(
    current_index: int,
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]],
) -> int | None:
    action, _ = pairs[current_index]
    args = action.get("args", {})
    for index, (candidate, _) in enumerate(pairs):
        if index == current_index:
            continue
        candidate_args = candidate.get("args", {})
        if _actions_bond(action, candidate, args, candidate_args):
            return index
    return None


def _actions_bond(
    action: dict[str, Any],
    candidate: dict[str, Any],
    args: dict[str, Any],
    candidate_args: dict[str, Any],
) -> bool:
    if action.get("tool") == "group_summary" and candidate.get("tool") == "group_mean_bar_chart":
        return (
            candidate_args.get("group_col") == args.get("group_col")
            and candidate_args.get("value_col") == args.get("value_col")
        )
    if action.get("tool") == "correlation_analysis" and candidate.get("tool") == "scatter_plot":
        columns = args.get("columns", [])
        return len(columns) == 2 and {candidate_args.get("x_col"), candidate_args.get("y_col")} == set(columns)
    if action.get("tool") == "simple_linear_regression" and candidate.get("tool") == "regression_plot":
        return (
            candidate_args.get("x_col") == args.get("feature_col")
            and candidate_args.get("y_col") == args.get("target_col")
        )
    if action.get("tool") == "numeric_summary" and candidate.get("tool") == "numeric_distribution_plot":
        columns = args.get("columns", [])
        return len(columns) == 1 and candidate_args.get("column") == columns[0]
    return False


def _section_title(action: dict[str, Any], chart_result: dict[str, Any] | None = None) -> str:
    if chart_result and isinstance(chart_result.get("result"), dict):
        title = chart_result["result"].get("title")
        if title:
            return str(title)
    return _action_label(action)


def _show_result(tool_name: str | None, result: Any) -> None:
    if result is None:
        st.info("No result was returned.")
    elif isinstance(result, dict):
        _show_dict_result(tool_name, result)
    elif isinstance(result, list):
        st.dataframe(pd.DataFrame(result), use_container_width=True, hide_index=True)
    else:
        st.code(str(result), language="text")


def _show_dict_result(tool_name: str | None, result: dict[str, Any]) -> None:
    if result.get("analysis_type"):
        _show_cohesive_result(result)
        return

    if _is_chart_spec(result):
        _show_chart_spec_result(result)
        return

    if tool_name == "correlation_analysis":
        rows = []
        for label, direction in [("strongest_positive", "Positive"), ("strongest_negative", "Negative")]:
            for pair in result.get(label, []):
                columns = pair.get("columns", [])
                if len(columns) == 2:
                    rows.append(
                        {
                            "Relationship": f"{columns[0]} and {columns[1]}",
                            "Correlation": pair.get("correlation"),
                            "Direction": direction,
                        }
                    )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("Correlation matrix", expanded=False):
                st.json(result.get("correlation_matrix", {}))
            return

    if tool_name == "chi_square_test" and isinstance(result.get("contingency_table"), dict):
        table = pd.DataFrame(result["contingency_table"]).fillna(0)
        if not table.empty:
            st.dataframe(table, use_container_width=True)
        stats = {
            "Chi-square statistic": result.get("chi_square_statistic"),
            "Degrees of freedom": result.get("degrees_of_freedom"),
            "P-value": result.get("p_value"),
        }
        st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)
        return

    if tool_name == "simple_linear_regression":
        stats = {
            "Feature": result.get("feature_col"),
            "Target": result.get("target_col"),
            "Slope": result.get("slope"),
            "Intercept": result.get("intercept"),
            "R-squared": result.get("r_squared"),
            "Rows used": result.get("n"),
        }
        st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)
        return

    if tool_name == "categorical_summary" and isinstance(result.get("columns"), dict):
        rows = []
        for column, values in result["columns"].items():
            details = values if isinstance(values, dict) else {}
            for item in details.get("top_values", []):
                rows.append(
                    {
                        "Column": column,
                        "Category": item.get("value"),
                        "Count": item.get("count"),
                        "Proportion": item.get("proportion"),
                        "Unique values": details.get("unique_values"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("Technical result details", expanded=False):
                st.json(result)
            return

    if tool_name == "missing_analysis" and isinstance(result.get("table"), list) and result["table"]:
        st.dataframe(pd.DataFrame(result["table"]), use_container_width=True, hide_index=True)
        with st.expander("Technical result details", expanded=False):
            st.json(result)
        return

    if "columns" in result and isinstance(result["columns"], dict):
        rows = []
        for column, values in result["columns"].items():
            if isinstance(values, dict):
                rows.append({"Column": column, **_title_keys(values)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("Technical result details", expanded=False):
                st.json(result)
            return

    if "groups" in result and isinstance(result["groups"], dict):
        rows = []
        for group, values in result["groups"].items():
            if isinstance(values, dict):
                rows.append({"Group": group, **_title_keys(values)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("Technical result details", expanded=False):
                st.json(result)
            return

    if "class_distribution" in result and isinstance(result["class_distribution"], dict):
        rows = [
            {"Class": label, **_title_keys(values)}
            for label, values in result["class_distribution"].items()
            if isinstance(values, dict)
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if isinstance(result.get("numeric_by_target"), dict) and result["numeric_by_target"]:
                with st.expander("Numeric summaries by target", expanded=False):
                    st.json(result["numeric_by_target"])
            with st.expander("Technical result details", expanded=False):
                st.json(result)
            return

    with st.expander("Result details", expanded=False):
        st.json(result)


def _show_cohesive_result(result: dict[str, Any]) -> None:
    chart = result.get("chart")
    relationships = result.get("relationships")

    if isinstance(chart, dict) and _is_chart_spec(chart):
        _show_chart_spec_result(chart)

    if isinstance(relationships, list) and relationships:
        rows = []
        for item in relationships:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "Variable": item.get("predictor_col") or item.get("x_col"),
                    "Compared with": item.get("target_col") or item.get("y_col"),
                    "Association": item.get("association") or item.get("correlation"),
                    "Type": item.get("association_type") or "correlation",
                }
            )
            embedded_chart = item.get("chart")
            if isinstance(embedded_chart, dict) and _is_chart_spec(embedded_chart):
                st.markdown(f"**{embedded_chart.get('title', 'Relationship chart')}**")
                _show_chart_spec_result(embedded_chart)
        if rows:
            with st.expander("Relationship summary table", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    table = result.get("table")
    if isinstance(table, list) and table:
        with st.expander("Detailed result table", expanded=False):
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    elif isinstance(result.get("statistics"), dict):
        statistics = result["statistics"]
        if result.get("analysis_type") == "distribution_analysis":
            st.dataframe(pd.DataFrame([_title_keys(statistics)]), use_container_width=True, hide_index=True)
        elif result.get("analysis_type") == "group_comparison_analysis" and isinstance(statistics.get("groups"), dict):
            rows = [{"Group": group, **_title_keys(values)} for group, values in statistics["groups"].items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    inferential_test = result.get("inferential_test")
    if isinstance(inferential_test, dict) and inferential_test:
        display = {
            key.replace("_", " ").title(): value
            for key, value in inferential_test.items()
            if key not in {"groups", "warnings"}
        }
        if display:
            with st.expander("Statistical test details", expanded=False):
                st.dataframe(pd.DataFrame([display]), use_container_width=True, hide_index=True)


def _is_chart_spec(result: dict[str, Any]) -> bool:
    return {"tool_name", "chart_type", "data"}.issubset(result.keys())


def _show_chart_spec_result(result: dict[str, Any]) -> None:
    if result.get("topic"):
        st.write(result["topic"])
    if result.get("finding"):
        st.caption(result["finding"])

    data = result.get("data", [])
    if not data:
        st.info("No chart data was returned.")
        return

    if result.get("tool_name") == "top_correlation_plots":
        frames = _top_correlation_chart_frames(result)
        if frames:
            for frame in frames:
                st.markdown(f"**{frame['title']}**")
                st.scatter_chart(frame["dataframe"], x=frame["x_col"], y=frame["y_col"])
            with st.expander("Underlying chart table", expanded=False):
                st.json(result.get("data", []))
            return

    if result.get("chart_type") in {"bar", "histogram"}:
        chart_df = _chart_dataframe(result)
        if not chart_df.empty:
            st.bar_chart(chart_df)
            with st.expander("Underlying chart table", expanded=False):
                st.dataframe(pd.DataFrame(result.get("table") or data), use_container_width=True, hide_index=True)
            return

    if result.get("chart_type") in {"scatter", "scatter_with_line"}:
        point_df = pd.DataFrame(data)
        x_col = result.get("x_col") or result.get("x")
        y_col = result.get("y_col") or result.get("y")
        if x_col in point_df.columns and y_col in point_df.columns:
            st.scatter_chart(point_df, x=x_col, y=y_col)
            with st.expander("Underlying chart table", expanded=False):
                st.dataframe(point_df, use_container_width=True, hide_index=True)
            return

    if result.get("tool_name") == "top_correlation_plots":
        rows = [
            {
                "Chart": graph.get("title"),
                "X": graph.get("x"),
                "Y": graph.get("y"),
                "Correlation": graph.get("metadata", {}).get("correlation"),
                "Points": len(graph.get("data", [])),
            }
            for graph in data
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("Chart-ready point data", expanded=False):
            st.json(data)
        return

    if isinstance(data, list) and data and isinstance(data[0], dict):
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.json(data)


def _top_correlation_chart_frames(result: dict[str, Any]) -> list[dict[str, Any]]:
    frames = []
    for graph in result.get("data", []):
        if not isinstance(graph, dict):
            continue
        data = graph.get("data", [])
        if not data:
            continue
        df = pd.DataFrame(data)
        x_col = graph.get("x_col") or graph.get("x")
        y_col = graph.get("y_col") or graph.get("y")
        if x_col in df.columns and y_col in df.columns:
            frames.append(
                {
                    "title": graph.get("title", f"{x_col} vs {y_col}"),
                    "x_col": x_col,
                    "y_col": y_col,
                    "dataframe": df,
                }
            )
    return frames


def _chart_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    data = result.get("data", [])
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if result.get("chart_type") == "histogram" and {"bin_start", "bin_end", "count"}.issubset(df.columns):
        frame = pd.DataFrame(
            {
                "bin": [
                    f"{row['bin_start']}-{row['bin_end']}"
                    for _, row in df.iterrows()
                ],
                "count": df["count"],
            }
        )
        return frame.set_index("bin")
    x_col = result.get("x_col") or result.get("x")
    y_col = result.get("y_col") or result.get("y")
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()
    return df[[x_col, y_col]].set_index(x_col)


def _show_interpretation_summary(interpretation: dict[str, Any]) -> None:
    summary = interpretation.get("summary", "")
    findings = interpretation.get("key_findings", [])

    if summary:
        st.markdown("**Result summary**")
        st.write(summary)
    if findings:
        st.markdown("**Key findings**")
        for finding in findings:
            st.write(f"- {finding}")


def _show_interpretation_details(interpretation: dict[str, Any]) -> None:
    cautions = interpretation.get("cautions", [])
    method_note = interpretation.get("method_note", "")

    if cautions:
        st.markdown("**Cautions**")
        for caution in cautions:
            st.warning(caution)
    if method_note:
        st.markdown("**Method note**")
        st.caption(method_note)


def _title_keys(values: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("_", " ").title(): value for key, value in values.items()}


def _pill_list(label: str, values: list[str]) -> None:
    st.markdown(f"**{label}**")
    if not values:
        st.caption("None detected")
        return
    html = " ".join(f'<span class="pill">{value}</span>' for value in values)
    st.markdown(html, unsafe_allow_html=True)


def _show_messages(label: str, messages: list[str], renderer: Callable[[str], Any]) -> None:
    for message in messages:
        renderer(f"{label}: {message}")


def _action_label(action: dict[str, Any]) -> str:
    return ACTION_LABELS.get(action.get("tool", ""), "Analysis step")


def _planner_status_messages(plan: dict[str, Any], provider_name: str) -> dict[str, str]:
    provider_label = _provider_label(provider_name)
    provider_warning_label = _provider_warning_label(provider_name)
    if plan.get("mode") == "llm":
        return {"mode_text": f"Planner mode: {provider_label} LLM", "warning": ""}

    fallback_cause = plan.get("fallback_cause", "")
    if fallback_cause == "deterministic_mode" or provider_label == "deterministic":
        return {
            "mode_text": "Using deterministic planner mode.",
            "warning": "",
        }
    if fallback_cause == "no_api_key":
        return {
            "mode_text": "Planner mode: deterministic fallback",
            "warning": f"{provider_warning_label} planner unavailable; using deterministic fallback.",
        }
    if fallback_cause in {"llm_error", "llm_invalid_json"}:
        return {
            "mode_text": "Planner mode: deterministic fallback",
            "warning": f"{provider_warning_label} planner unavailable; using deterministic fallback.",
        }
    if fallback_cause == "llm_empty_selection":
        return {
            "mode_text": f"Planner mode: {provider_label} LLM checked; deterministic fallback used",
            "warning": "LLM planner returned no runnable action; using deterministic fallback.",
        }
    if fallback_cause == "llm_invalid_selection":
        return {
            "mode_text": f"Planner mode: {provider_label} LLM checked; deterministic fallback used",
            "warning": "LLM planner returned invalid action indexes; using deterministic fallback.",
        }
    return {
        "mode_text": "Planner mode: deterministic fallback",
        "warning": "",
    }


def _planner_availability_status(llm_config: dict[str, Any]) -> tuple[str, str]:
    provider_label = _provider_label(str(llm_config.get("provider") or ""))
    if provider_label == "deterministic":
        return "info", "Using deterministic planner mode."
    if llm_config.get("api_key_configured"):
        return "success", f"{provider_label} API configured"
    return "info", f"{provider_label} API not configured; deterministic fallback available."


def _provider_label(provider_name: str) -> str:
    normalized = str(provider_name or "").strip().lower()
    if normalized == "groq":
        return "GROQ"
    if normalized == "openai":
        return "OpenAI"
    if normalized in {"deterministic", "fallback", "local", "no-llm", "none", "off"}:
        return "deterministic"
    return str(provider_name or "LLM").strip() or "LLM"


def _provider_warning_label(provider_name: str) -> str:
    normalized = str(provider_name or "").strip().lower()
    if normalized == "groq":
        return "Groq"
    if normalized == "openai":
        return "OpenAI"
    return _provider_label(provider_name)


def _visible_planner_warnings(messages: list[str]) -> list[str]:
    visible = []
    for message in messages or []:
        text = str(message)
        if _is_raw_provider_detail(text):
            continue
        visible.append(text)
    return visible


def _is_raw_provider_detail(message: str) -> bool:
    lowered = message.lower()
    return (
        "api_key" in lowered
        or "kairos_llm_model" in lowered
        or "sdk import failed" in lowered
    )


def _selected_candidate_indexes(
    candidate_actions: list[dict[str, Any]],
    selected_actions: list[dict[str, Any]],
) -> list[int]:
    indexes = []
    for selected in selected_actions:
        for index, candidate in enumerate(candidate_actions):
            if selected is candidate or selected == candidate:
                indexes.append(index)
                break
    return indexes


def _profile_column_type(profile: dict[str, Any], column: str) -> str:
    column_profiles = profile.get("column_profiles", {})
    if isinstance(column_profiles, dict):
        details = column_profiles.get(column, {})
        if isinstance(details, dict) and details.get("type"):
            return str(details["type"])
    for type_name, columns in profile.get("column_types", {}).items():
        if isinstance(columns, list) and column in columns:
            return str(type_name)
    return ""


def _file_key(uploaded_file: Any) -> tuple[str, int]:
    return (getattr(uploaded_file, "name", "uploaded.csv"), int(getattr(uploaded_file, "size", 0)))


def _inject_css() -> None:
    st.markdown(f"<style>{dashboard_css()}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
