"""Fixed deterministic tool registry for the future KAIROS planner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tools.cohesive_analysis import (
    distribution_analysis,
    global_relationship_analysis,
    group_comparison_analysis,
    missingness_analysis,
    outlier_analysis,
    relationship_analysis,
    target_relationship_analysis,
)
from tools.dataset_profile import dataset_overview
from tools.eda_tools import (
    anova_by_group,
    categorical_summary,
    chi_square_test,
    correlation_analysis,
    group_summary,
    missing_analysis,
    numeric_summary,
    outlier_detection,
    simple_linear_regression,
    t_test_by_group,
    target_group_summary,
)
from tools.viz_tools import (
    group_mean_bar_chart,
    missing_value_bar_chart,
    numeric_distribution_plot,
    regression_plot,
    scatter_plot,
    top_correlation_plots,
)


TOOL_REGISTRY = {
    "dataset_overview": {
        "description": "Return a lightweight schema, type, missingness, and quality overview for the dataset.",
        "args": {},
        "function": dataset_overview,
    },
    "distribution_analysis": {
        "description": "Cohesive single-variable numeric analysis with summary statistics and embedded chart data.",
        "args": {"column": "numeric column"},
        "function": distribution_analysis,
    },
    "relationship_analysis": {
        "description": "Cohesive relationship analysis for one explicit numeric variable pair with statistics and scatter data.",
        "args": {"x_col": "numeric x-axis column", "y_col": "numeric y-axis column"},
        "function": relationship_analysis,
    },
    "target_relationship_analysis": {
        "description": "Rank variables associated with one explicit target and include deterministic statistics and chart data.",
        "args": {"target_col": "target column", "top_n": "optional number of relationships"},
        "function": target_relationship_analysis,
    },
    "global_relationship_analysis": {
        "description": "Find strongest non-identifier numeric relationships and include chart-ready scatter data.",
        "args": {"cols": "optional list of numeric columns", "top_n": "optional number of relationships"},
        "function": global_relationship_analysis,
    },
    "group_comparison_analysis": {
        "description": "Cohesive group comparison with grouped statistics and an embedded ranked bar chart.",
        "args": {"group_col": "grouping column", "value_col": "numeric value column"},
        "function": group_comparison_analysis,
    },
    "outlier_analysis": {
        "description": "Cohesive numeric outlier analysis using the IQR rule plus distribution chart data.",
        "args": {"column": "numeric column", "method": "optional outlier method"},
        "function": outlier_analysis,
    },
    "missingness_analysis": {
        "description": "Cohesive missing-value diagnostics without default graph rendering.",
        "args": {},
        "function": missingness_analysis,
    },
    "missing_analysis": {
        "description": "Summarize missing counts, missing percentages, and high-missingness columns.",
        "args": {},
        "function": missing_analysis,
    },
    "numeric_summary": {
        "description": "Summarize numeric columns with count, mean, standard deviation, quartiles, range, and skewness.",
        "args": {"columns": "optional list of numeric column names"},
        "function": numeric_summary,
    },
    "categorical_summary": {
        "description": "Summarize categorical columns with value counts, proportions, and unique counts.",
        "args": {
            "columns": "optional list of categorical column names",
            "top_n": "maximum number of values to return per column",
        },
        "function": categorical_summary,
    },
    "correlation_analysis": {
        "description": "Compute numeric Pearson correlations and strongest positive/negative column pairs.",
        "args": {"columns": "optional list of numeric column names"},
        "function": correlation_analysis,
    },
    "group_summary": {
        "description": "Compare a numeric value column across categories of a grouping column.",
        "args": {"group_col": "categorical grouping column", "value_col": "numeric value column"},
        "function": group_summary,
    },
    "target_group_summary": {
        "description": "Summarize class balance and numeric columns by a categorical target column.",
        "args": {"target_col": "categorical or binary target column"},
        "function": target_group_summary,
    },
    "simple_linear_regression": {
        "description": "Fit a one-feature linear regression using deterministic pandas arithmetic.",
        "args": {"target_col": "numeric target column", "feature_col": "numeric feature column"},
        "function": simple_linear_regression,
    },
    "chi_square_test": {
        "description": "Compute a chi-square statistic and p-value for two categorical columns.",
        "args": {"col_a": "first categorical column", "col_b": "second categorical column"},
        "function": chi_square_test,
    },
    "t_test_by_group": {
        "description": "Compute a two-group Welch t-statistic and p-value for a numeric value column.",
        "args": {"group_col": "binary categorical group column", "value_col": "numeric value column"},
        "function": t_test_by_group,
    },
    "anova_by_group": {
        "description": "Compute one-way ANOVA for a numeric value column across three or more groups.",
        "args": {"group_col": "categorical grouping column", "value_col": "numeric value column"},
        "function": anova_by_group,
    },
    "outlier_detection": {
        "description": "Detect potential numeric outliers with the IQR rule.",
        "args": {"column": "numeric column", "method": "optional outlier method"},
        "function": outlier_detection,
    },
    "numeric_distribution_plot": {
        "description": "Return histogram or box-style chart data for one numeric column.",
        "args": {"column": "numeric column", "bins": "optional number of histogram bins"},
        "function": numeric_distribution_plot,
    },
    "scatter_plot": {
        "description": "Return chart-ready scatter data for two numeric columns.",
        "args": {
            "x_col": "numeric x-axis column",
            "y_col": "numeric y-axis column",
            "max_points": "optional maximum deterministic sample size",
        },
        "function": scatter_plot,
    },
    "top_correlation_plots": {
        "description": "Return scatter chart specs for the strongest numeric correlations, optionally centered on one target column.",
        "args": {
            "cols": "optional list of numeric columns",
            "target_col": "optional numeric target column",
            "top_n": "optional number of correlation charts",
            "max_points": "optional maximum points per scatter chart",
        },
        "function": top_correlation_plots,
    },
    "group_mean_bar_chart": {
        "description": "Return ranked bar chart data showing mean numeric value by group.",
        "args": {
            "group_col": "categorical grouping column",
            "value_col": "numeric value column",
            "top_n": "optional maximum number of groups",
        },
        "function": group_mean_bar_chart,
    },
    "missing_value_bar_chart": {
        "description": "Return ranked bar chart data for missing values by column.",
        "args": {"include_zero": "optional boolean to include columns with no missing values"},
        "function": missing_value_bar_chart,
    },
    "regression_plot": {
        "description": "Return scatter data and fitted line data for two numeric columns.",
        "args": {
            "x_col": "numeric x-axis column",
            "y_col": "numeric y-axis column",
            "max_points": "optional maximum deterministic sample size",
        },
        "function": regression_plot,
    },
}


def list_available_tools() -> list[dict[str, Any]]:
    """Return planner-safe tool specs without callable objects."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "args": spec["args"],
        }
        for name, spec in TOOL_REGISTRY.items()
    ]


def get_tool_spec(tool_name: str) -> dict[str, Any] | None:
    """Return one planner-safe tool spec by name."""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return None
    return {"name": tool_name, "description": spec["description"], "args": spec["args"]}


def run_tool(tool_name: str, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Run an allowed deterministic tool and return a structured result."""
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return {
            "tool": tool_name,
            "status": "error",
            "result": None,
            "warnings": [f"Tool {tool_name} is not allowed"],
        }

    try:
        result = spec["function"](df, **kwargs)
    except TypeError as exc:
        return {
            "tool": tool_name,
            "status": "error",
            "result": None,
            "warnings": [f"Invalid arguments for {tool_name}: {exc}"],
        }

    return {
        "tool": tool_name,
        "status": "ok",
        "result": result,
        "warnings": result.get("warnings", []) if isinstance(result, dict) else [],
    }
