# KAIROS

A lightweight agentic system for adaptive data exploration and analytical reasoning.

KAIROS is an experimental COMPSCI 767 Assignment 2 prototype. The project is intended to keep reasoning and execution separate: future LLM components will choose analysis actions, while deterministic Python tools perform factual computation.

## Current Milestone: Scope-Aware Cohesive Analysis Objects

The first implemented module is `agent/observer.py`. It provides deterministic CSV and DataFrame inspection utilities that produce compact, JSON-serializable summaries for a later LLM planner.

Observer capabilities:

- Load CSV files safely with `load_csv`.
- Inspect DataFrames with row and column counts, column names, duplicate rows, sample rows, missing-value summaries, inferred column types, and likely target columns.
- Detect numeric, categorical, datetime-like, boolean, and text-like columns.
- Flag high-missingness columns.
- Handle empty DataFrames, one-column datasets, datasets without numeric or categorical columns, duplicate rows, and mixed/weird column values.

KAIROS now plans around cohesive analysis objects: one inferred user intent produces one analysis result containing statistics, explanation, warnings, and renderable chart data together. This replaces the earlier UI pattern where a statistical tool and a graph helper appeared as separate analysis sections.

Current cohesive analyses:

- `distribution_analysis`: single-variable numeric summary with count, missing count, min, max, mean, median, standard deviation, quartiles, IQR, skewness when available, and histogram/bar-style chart data rendered as a Streamlit bar chart.
- `relationship_analysis`: one explicit numeric pair with Pearson correlation, simple regression statistics, explanation, and scatter chart data.
- `target_relationship_analysis`: ranks variables associated with an explicit target column, excluding identifier-like predictors by default.
- `global_relationship_analysis`: returns the strongest non-identifier numeric relationships for broad relationship questions.
- `group_comparison_analysis`: compares one numeric value across one grouping column with grouped statistics, ranked means, explanation, and bar chart data. It adds a t-test for exactly two groups and ANOVA for three or more groups.
- `outlier_analysis`: detects potential numeric outliers with the IQR rule and includes distribution chart data plus cautions.
- `missingness_analysis`: reports missing-value counts and percentages without a default chart.

Bonded text/statistical analysis and graph behavior:

- `distribution_analysis` bonds descriptive numeric statistics with a histogram or discrete-count chart rendered in the UI.
- `relationship_analysis` bonds correlation/regression statistics with one scatter plot for the explicit variable pair.
- `target_relationship_analysis` bonds ranked target associations with scatter plots or group charts for the top related variables.
- `group_comparison_analysis` bonds grouped descriptive statistics with a ranked bar chart and adds a t-test for exactly two groups or ANOVA for three or more groups. For ANOVA, the report includes the F statistic, p-value, highest mean group, lowest mean group, and cautious statistical-notability wording.
- `outlier_analysis` bonds IQR thresholds, outlier counts, and example flagged rows with distribution chart data.
- `missingness_analysis` remains text/table only by default.

The older deterministic EDA/statistical tools in `tools/eda_tools.py` and graph helpers in `tools/viz_tools.py` remain available through the fixed registry for compatibility and for building the cohesive objects. The planner now prefers cohesive analyses for user-goal-driven requests.

Available tools:

- `distribution_analysis`
- `relationship_analysis`
- `target_relationship_analysis`
- `global_relationship_analysis`
- `group_comparison_analysis`
- `outlier_analysis`
- `missingness_analysis`
- `missing_analysis`
- `numeric_summary`
- `categorical_summary`
- `correlation_analysis`
- `group_summary`
- `target_group_summary`
- `simple_linear_regression`
- `chi_square_test`
- `t_test_by_group`
- `anova_by_group`
- `outlier_detection`
- `numeric_distribution_plot`
- `scatter_plot`
- `top_correlation_plots`
- `group_mean_bar_chart`
- `missing_value_bar_chart`
- `regression_plot`

All tool outputs are compact dictionaries intended to be JSON-serializable. Tools do not call an LLM or execute arbitrary code. Chi-square and t-test functions use `scipy` to return p-values. Simple logistic regression is future work.

`tools/cohesive_analysis.py` builds the new scope-aware analyses from deterministic pandas/statistical functions and chart-ready helper data. The LLM may choose an analysis object, but it does not calculate chart values, edit columns, or execute code. The verifier and executor still guard every selected action.

Planner recommendations now classify the user request before selecting actions:

- `one_variable`: distribution or outlier analysis for one numeric column.
- `explicit_pair`: relationship analysis for exactly the two mentioned variables.
- `target_driven`: target relationship analysis for questions such as `What predicts promotion?`, `What affects salary?`, or `What variables are most strongly related to salary?`.
- `group_comparison`: one grouped comparison such as salary by department.
- `global_relationships`: strongest non-identifier numeric relationships.
- `missingness`: missing-value diagnostics only.
- `fallback_overview`: distribution first, then relationships or group comparison, with missingness later.

If a user asks about salary, pay, compensation, or income, KAIROS prefers columns such as `salary`, `annual_salary`, `monthly_salary`, `income`, or `pay`. Similar lightweight matching exists for age, sales/revenue, performance, promotion, experience, satisfaction, and engagement-style metrics.

Identifier-like columns such as IDs, indexes, row numbers, serials, codes, keys, UUIDs, and high-uniqueness employee-code fields are excluded by default from grouping, correlation, and regression-style planning. If the user explicitly asks for an ID-based comparison, KAIROS allows it but adds a warning that the result may not be meaningful.

Intent robustness:

- Relationship and correlation wording outranks group fallback. If wording mentions correlation, correlated, relationship, related to, associated with, variables related to, factors related to, strongest relationships, most strongly related, what variables, or what factors, KAIROS chooses target or global relationship analysis before considering group comparison.
- Explicit variables are prioritized over generic defaults.
- If relationship-style wording plus one target column is present, KAIROS uses `target_relationship_analysis` instead of defaulting to department/group comparisons.
- If relationship-style wording has no explicit target, KAIROS uses `global_relationship_analysis` for the top non-ID numeric relationships.
- Group comparison requires group-style wording such as by department, across groups, compare across categories, vary by, differ by, or which group/department/region.
- If two valid numeric variables are explicitly mentioned, KAIROS uses one `relationship_analysis` and does not add unrelated global summaries even when the UI maximum analyses setting is high.
- If a request is vague, KAIROS uses a conservative fallback hierarchy: distributions for two or three important numeric variables first, then non-ID numeric relationships, then missingness, with group comparison only when it is clearly useful.
- Remaining limitations: the matching is lightweight and deterministic, not a full semantic parser; ambiguous column names may still require the user to ask a more specific question.

Example question-to-analysis behavior:

- `Is salary related to years_experience?` selects one `relationship_analysis` for that pair only.
- `What predicts promotion?` selects `target_relationship_analysis` with `promotion` as the target.
- `What variables are most strongly related to salary?` selects `target_relationship_analysis` with `salary` as the target, excluding ID-like predictors.
- `What are the strongest correlations?` selects `global_relationship_analysis` and excludes ID-like columns.
- `Which groups are paid the most?` selects `group_comparison_analysis` using a salary-like value column and a sensible group column.
- `Show salary distribution` selects `distribution_analysis`.
- `Are there outliers in salary?` selects `outlier_analysis`.
- `Which columns have missing values?` selects `missingness_analysis` without a missingness graph.
- `Analyze this dataset` uses the fallback hierarchy, starting with two or three useful numeric distributions, then relationship checks, then missingness. Missingness replaces low-value generic grouping when the group comparison would not add much.

`agent/verifier.py` validates proposed tool actions before execution. It checks the requested tool against the fixed registry, validates required arguments, catches missing columns and invalid parameter types, and applies basic column-type rules such as numeric-only inputs for numeric summaries and binary grouping for t-tests. The verifier returns structured `valid`, `tool`, `args`, `errors`, and `warnings` fields and does not execute tools.

`agent/executor.py` connects validation to deterministic tool execution. It accepts an action proposal, calls the verifier first, blocks invalid actions before they reach a tool, dispatches valid actions through the fixed registry, and catches runtime tool errors into a structured response.

`agent/planner_helper.py` provides a small deterministic recommender for possible next actions based on DataFrame schema, inferred scope, and simple column-name matching. It does not validate deeply or execute tools; recommended actions still go through the verifier and executor.

`agent/llm_planner.py` adds an optional LLM selector layer. It receives a user goal, dataset profile, and candidate actions, then asks the model to return JSON containing selected candidate indexes. The Python code maps those indexes back to the original candidate action objects, so the model cannot rewrite args, invent tools, or bypass the guarded execution pipeline. Planning is goal-conditioned: specific questions about correlation, prediction, group differences, categorical relationships, or missing data influence which existing candidate actions are selected.

`agent/result_interpreter.py` turns executed tool outputs into deterministic, human-readable summaries, key findings, cautions, and method notes. These interpretations are grounded only in structured tool results. They do not inspect the raw DataFrame, call an LLM, or invent findings beyond the executed analysis output.

Current guarded execution pipeline:

```text
Observer -> planner_helper -> optional LLM planner -> Verifier -> Executor -> Cohesive analysis tools
```

The verifier decides whether an action is allowed for the current DataFrame schema. The executor is responsible for running a verified tool action and packaging the result, errors, and warnings for the next workflow step. The result interpreter then formats those tool outputs into readable findings for the UI.

## Optional LLM Planner

The system can run without an API key. If `OPENAI_API_KEY` is missing, the LLM planner returns a deterministic fallback that ranks candidate actions using the user question. For example, a specific pair question prioritizes `relationship_analysis`, while a vague request such as `Explore this dataset` uses the scope-aware fallback hierarchy.

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

The left panel contains the conversational control area and a short explanation of the selected analyses. The right panel contains the dataset preview, dataset overview, verification status, formatted results, actual Streamlit charts for graph-helper outputs, and deterministic result interpretations. The UI does not execute tools directly. Every selected analysis goes through `execute_action`, which calls the verifier before dispatching to the tool registry.

User-facing results are shown as one coherent mini-report per selected analysis: explanation, key findings, embedded chart where useful, detailed table, cautions, and method note. Raw verification data, executor responses, tool names, and full structured payloads remain available in collapsed technical details sections for transparency.

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

The memory, reflection, and final reporter components are not implemented yet.

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
python -m unittest tests.test_viz_tools
python -m unittest tests.test_viz_registry_planner
python -m unittest tests.test_cohesive_analysis
python -m unittest tests.test_app_cohesive_rendering
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```
