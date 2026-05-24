"""Streamlit UI for manually testing the KAIROS pipeline."""

from __future__ import annotations

import os
from typing import Any, Callable

import pandas as pd
import streamlit as st

from agent.executor import execute_action
from agent.llm_planner import plan_with_llm
from agent.observer import inspect_dataset, load_csv
from agent.planner_helper import describe_planning_scope, recommend_actions
from agent.result_interpreter import interpret_result


DEFAULT_GOAL = "Explore this dataset."

ACTION_LABELS = {
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

    left, right = st.columns([0.9, 3.0], gap="medium")
    with left:
        uploaded_file, effective_goal = _agent_panel()

    with right:
        _workspace_panel(uploaded_file, effective_goal)


def _agent_panel() -> tuple[Any, str]:
    st.markdown('<div class="agent-panel">', unsafe_allow_html=True)
    st.markdown('<h1 class="kairos-title">KAIROS</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="kairos-subtitle">Guarded data-analysis agent</p>',
        unsafe_allow_html=True,
    )
    st.markdown("**Pipeline**")
    st.code("Observer -> Planner -> Verifier -> Executor -> Tools", language="text")

    st.markdown("**Planner mode**")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI API configured")
    else:
        st.info("OpenAI API not configured")

    max_actions = st.number_input(
        "Maximum analyses",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        key="max_selected_actions",
    )

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    question = st.text_area(
        "Analysis question",
        placeholder="What factors affect salary?",
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

    _show_agent_explanation()

    with st.expander("Other possible analyses", expanded=False):
        bundle = st.session_state.get("analysis_bundle")
        if bundle:
            _show_action_cards(bundle["candidate_actions"], empty_message="No candidate analyses were generated.")
        else:
            st.caption("Candidate analyses will appear after KAIROS profiles the dataset.")

    st.caption("Follow-up conversational analysis can be added here later.")
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file, effective_goal


def _workspace_panel(uploaded_file: Any, effective_goal: str) -> None:
    st.markdown('<div class="workspace-panel">', unsafe_allow_html=True)
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

    candidate_actions = recommend_actions(df, goal=goal)
    planning_trace = describe_planning_scope(goal, df)
    plan = plan_with_llm(goal, profile, candidate_actions, max_actions=max_actions)
    selected_actions = plan["selected_actions"]
    execution_results = [execute_action(df, action) for action in selected_actions]

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
    }


def _show_agent_explanation() -> None:
    bundle = st.session_state.get("analysis_bundle")
    if not bundle:
        st.markdown("### KAIROS response")
        st.info("Ask a question about the uploaded dataset. KAIROS will profile the data, choose suitable analyses, and run them through the guarded executor.")
        return

    plan = bundle["plan"]
    mode_text = "Planner mode: OpenAI LLM" if plan.get("mode") == "llm" else "Planner mode: deterministic fallback"
    st.markdown("### KAIROS response")
    if plan.get("mode") == "llm":
        st.success(mode_text)
    else:
        st.info(mode_text)
        if plan.get("errors") or plan.get("warnings"):
            st.warning("LLM planner unavailable; using deterministic fallback.")

    st.markdown("**Interpreted question**")
    st.write(bundle["goal"])

    st.markdown("**Chosen analyses**")
    selected = bundle["selected_actions"]
    if selected:
        for action in selected:
            st.write(f"- {_action_label(action)}: {action.get('reason', 'Selected for this dataset.')}")
    else:
        st.caption("No analyses were selected.")

    _show_messages("Planner warning", plan.get("warnings", []), st.warning)
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

    cols = st.columns(4)
    cols[0].metric("Rows", shape["rows"])
    cols[1].metric("Columns", shape["columns"])
    cols[2].metric("Duplicate rows", profile.get("duplicate_rows", 0))
    cols[3].metric("Missing cells", missing_values.get("total_missing_cells", 0))

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
            _pill_list("Datetime-like columns", column_types.get("datetime_like", []))
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

            status_text = "Executed" if result["executed"] else "Not executed"
            verification_text = "Verified" if result["verification"]["valid"] else "Blocked during verification"
            st.markdown("**Execution status**")
            st.caption(f"{status_text}; {verification_text}. Warnings: {len(result['warnings'])}.")

            _show_messages("Error", result["errors"], st.error)
            _show_messages("Warning", result["warnings"], st.warning)

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


def _file_key(uploaded_file: Any) -> tuple[str, int]:
    return (getattr(uploaded_file, "name", "uploaded.csv"), int(getattr(uploaded_file, "size", 0)))


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-size: 20px;
            line-height: 1.7;
        }
        .block-container {
            padding-top: 0.85rem;
            padding-right: 0.9rem;
            padding-bottom: 2.4rem;
            padding-left: 0.9rem;
            max-width: 1800px;
        }
        .agent-panel {
            position: sticky;
            top: 1rem;
        }
        .workspace-panel {
            padding-bottom: 2rem;
        }
        .kairos-title {
            font-size: 4.1rem;
            line-height: 1.05;
            margin-bottom: 0.3rem;
            color: #111827;
            font-weight: 760;
        }
        .kairos-subtitle {
            font-size: 1.28rem;
            color: #4b5563;
            margin-bottom: 1.2rem;
        }
        .workspace-title {
            font-size: 2.65rem;
            margin-top: 0.35rem;
            margin-bottom: 1rem;
            color: #111827;
        }
        h2, h3 {
            color: #111827;
            margin-top: 1.65rem;
            margin-bottom: 0.85rem;
        }
        h3 {
            font-size: 1.85rem;
        }
        label, .stTextInput label, .stTextArea label, .stFileUploader label, .stNumberInput label {
            font-size: 1.18rem !important;
            font-weight: 650 !important;
            color: #111827 !important;
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.08rem;
            color: #4b5563;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.05rem;
            color: #111827;
        }
        [data-testid="stDataFrame"] {
            font-size: 1.08rem;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-size: 1.12rem;
            line-height: 1.7;
        }
        div[data-testid="stButton"] button {
            font-size: 1.12rem;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e5e7eb !important;
            background: #ffffff;
        }
        div[data-testid="stExpander"] {
            border-color: #e5e7eb !important;
            background: #ffffff;
        }
        .pill {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            margin: 0.18rem 0.24rem 0.18rem 0;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            background: #f9fafb;
            color: #111827;
            font-size: 0.96rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
