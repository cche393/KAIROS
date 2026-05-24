# KAIROS

A lightweight agentic system for adaptive data exploration and analytical reasoning.

KAIROS is an experimental COMPSCI 767 Assignment 2 prototype. The project is intended to keep reasoning and execution separate: future LLM components will choose analysis actions, while deterministic Python tools perform factual computation.

## Current Milestone: Observer, Planner Helper, Optional LLM Planner, EDA Tools, Verifier, Executor, and Result Interpretation

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

All tool outputs are compact dictionaries intended to be JSON-serializable. Tools do not create plots, call an LLM, or execute arbitrary code. Chi-square and t-test functions use `scipy` to return p-values. Simple logistic regression is future work.

`agent/verifier.py` validates proposed tool actions before execution. It checks the requested tool against the fixed registry, validates required arguments, catches missing columns and invalid parameter types, and applies basic column-type rules such as numeric-only inputs for numeric summaries and binary grouping for t-tests. The verifier returns structured `valid`, `tool`, `args`, `errors`, and `warnings` fields and does not execute tools.

`agent/executor.py` connects validation to deterministic tool execution. It accepts an action proposal, calls the verifier first, blocks invalid actions before they reach a tool, dispatches valid actions through the fixed registry, and catches runtime tool errors into a structured response.

`agent/planner_helper.py` provides a small deterministic recommender for possible next actions based on DataFrame schema and simple data properties. It recommends action dictionaries such as `missing_analysis`, `numeric_summary`, `categorical_summary`, grouped summaries, t-tests, chi-square tests, correlations, and simple linear regression when the needed column types are present. It does not validate deeply or execute tools; recommended actions still go through the verifier and executor.

`agent/llm_planner.py` adds an optional LLM selector layer. It receives a user goal, dataset profile, and candidate actions, then asks the model to return JSON containing selected candidate indexes. The Python code maps those indexes back to the original candidate action objects, so the model cannot rewrite args, invent tools, or bypass the guarded execution pipeline. Planning is goal-conditioned: specific questions about correlation, prediction, group differences, categorical relationships, or missing data influence which existing candidate actions are selected.

`agent/result_interpreter.py` turns executed tool outputs into deterministic, human-readable summaries, key findings, cautions, and method notes. These interpretations are grounded only in structured tool results. They do not inspect the raw DataFrame, call an LLM, or invent findings beyond the executed analysis output.

Current guarded execution pipeline:

```text
Observer -> planner_helper -> LLM planner -> Verifier -> Executor -> Tools
```

The verifier decides whether an action is allowed for the current DataFrame schema. The executor is responsible for running a verified tool action and packaging the result, errors, and warnings for the next workflow step. The result interpreter then formats those tool outputs into readable findings for the UI.

## Optional LLM Planner

The system can run without an API key. If `OPENAI_API_KEY` is missing, the LLM planner returns a deterministic fallback that ranks candidate actions using the user question. For example, a correlation question prioritizes `correlation_analysis`, while a vague request such as `Explore this dataset` keeps broad EDA actions first.

Environment variables:

- `OPENAI_API_KEY`: OpenAI API key used only for optional action selection.
- `KAIROS_LLM_MODEL`: model name, defaulting to `gpt-4o-mini`.

Copy `.env.example` or set these variables in your shell when you want API-backed planning. API mode can be shown in a demo when a key is configured; fallback mode keeps the submitted system runnable without one.

Example usage:

```python
from agent.llm_planner import plan_with_llm
from agent.planner_helper import recommend_actions

candidate_actions = recommend_actions(df)
plan = plan_with_llm("Compare income by segment", dataset_profile, candidate_actions)

for action in plan["selected_actions"]:
    # Still pass actions through execute_action, which calls the verifier first.
    ...
```

## Streamlit UI

`app.py` provides a local two-panel Streamlit dashboard for manually testing the full guarded pipeline:

1. Upload a CSV file.
2. Ask a natural-language analysis question, such as `What factors affect salary?`.
3. Select `Generate and run analysis`.
4. KAIROS profiles the dataset, chooses analyses, runs them through the guarded executor, and displays the results.

The left panel contains the conversational control area and a short explanation of the selected analyses. The right panel contains the dataset preview, dataset overview, verification status, formatted results, and deterministic result interpretations. The UI does not execute tools directly. Every selected analysis goes through `execute_action`, which calls the verifier before dispatching to the tool registry.

User-facing results are shown as tables or compact summaries followed by key findings, interpretation, cautions, and method notes. Raw verification data, executor responses, tool names, and full structured payloads remain available in collapsed technical details sections for transparency.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with Streamlit defaults:

```bash
streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

Run on an explicit host and port:

```bash
streamlit run app.py --server.address localhost --server.port 8501
```

If port `8501` is busy, use another port:

```bash
streamlit run app.py --server.port 8502
```

Current planner modes:

1. OpenAI API planner, enabled when `OPENAI_API_KEY` is set and the API call succeeds. The UI shows `Planner mode: OpenAI LLM`.
2. Deterministic fallback planner, used when no key is set or the API fails. The UI shows `Planner mode: deterministic fallback`.

API keys must be supplied through environment variables and must not be committed. The UI displays only whether the OpenAI API is configured; it never displays the key value.

Ollama or local model support is a planned extension only and is not currently implemented.

To test OpenAI API mode in PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:KAIROS_LLM_MODEL="gpt-4o-mini"
streamlit run app.py
```

To run without an API key:

```bash
streamlit run app.py
```

This uses deterministic fallback planning.

The memory, reflection, plotting, and final reporter components are not implemented yet.

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
python -m unittest tests.test_executor
python -m unittest tests.test_planner_helper
python -m unittest tests.test_llm_planner
python -m unittest tests.test_result_interpreter
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```
