# KAIROS

A lightweight agentic system for adaptive data exploration and analytical reasoning.

KAIROS is an experimental COMPSCI 767 Assignment 2 prototype. The project is intended to keep reasoning and execution separate: future LLM components will choose analysis actions, while deterministic Python tools perform factual computation.

## Current Milestone: Deterministic Observer, EDA Tools, and Verifier

The first implemented module is `agent/observer.py`. It provides deterministic CSV and DataFrame inspection utilities that produce compact, JSON-serializable summaries for a later LLM planner.

Observer capabilities:

- Load CSV files safely with `load_csv`.
- Inspect DataFrames with row and column counts, column names, duplicate rows, sample rows, missing-value summaries, inferred column types, and likely target columns.
- Detect numeric, categorical, datetime-like, boolean, and text-like columns.
- Flag high-missingness columns.
- Handle empty DataFrames, one-column datasets, datasets without numeric or categorical columns, duplicate rows, and mixed/weird column values.

KAIROS also has deterministic EDA/statistical tools in `tools/eda_tools.py`, exposed through a fixed allowed-tool registry in `agent/tool_registry.py`.

Available tools:

- `missing_analysis`
- `numeric_summary`
- `categorical_summary`
- `correlation_analysis`
- `group_summary`
- `target_group_summary`
- `simple_linear_regression`
- `chi_square_test`
- `t_test_by_group`

All tool outputs are compact dictionaries intended to be JSON-serializable. Tools do not create plots, call an LLM, or execute arbitrary code. Chi-square and t-test functions currently return test statistics without p-values because scipy is not a project dependency. Simple logistic regression is future work.

`agent/verifier.py` validates proposed tool actions before execution. It checks the requested tool against the fixed registry, validates required arguments, catches missing columns and invalid parameter types, and applies basic column-type rules such as numeric-only inputs for numeric summaries and binary grouping for t-tests. The verifier returns structured `valid`, `tool`, `args`, `errors`, and `warnings` fields and does not execute tools.

The LLM planner, Streamlit UI, memory, reflection, plotting, and final reporter are not implemented yet.

## Tests

Run all tests with:

```bash
python -m unittest discover
```

Or run focused test modules with:

```bash
python -m unittest tests.test_observer
python -m unittest tests.test_eda_tools tests.test_tool_registry
python -m unittest tests.test_verifier
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```
