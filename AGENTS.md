# Codex Prompt: Create Initial Project Guidance for KAIROS

Use Superpowers workflow. First inspect the current repo structure. Then create an `AGENTS.md` file at the project root. Do not implement application code yet.

The project is called **KAIROS**.

## Project purpose

KAIROS is an experimental reasoning agent for autonomous exploratory data analysis. It is for COMPSCI 767 Assignment 2. The goal is to build a small but clearly agentic prototype, not a production system.

The system should allow a user to upload a CSV dataset, provide an optional analysis goal, and then let an agent inspect the dataset, decide what analysis step to perform next, execute deterministic Python tools, observe the result, and continue until it produces a final analysis summary.

## Core architecture

Use this rough architecture:

```text
User Goal + CSV File
        ↓
Streamlit UI
        ↓
Dataset Observer
        ↓
LLM Planner / Decider
        ↓
Verifier / Guard
        ↓
Tool Executor
        ↓
Observation Result
        ↓
Memory + Reflection
        ↓
Continue or Finish
        ↓
Final Analysis Report
```

## Key design principle

Separate reasoning from execution.

The LLM should decide what to do next, but deterministic Python tools should perform all factual computation.

Do:

* use Python/pandas for dataset inspection, statistics, missing-value checks, and plotting
* use the LLM for planning, action selection, reflection, and final explanation
* pass compact structured observations to the LLM
* restrict the LLM to a fixed list of allowed tools/actions

Do not:

* let the LLM read or reason over the entire raw CSV
* let the LLM execute arbitrary Python code
* build a giant chatbot
* overengineer multi-agent systems
* duplicate similar logic in multiple files

## Planned modules

Suggested structure:

```text
KAIROS/
  app.py
  agent/
    observer.py
    planner.py
    executor.py
    verifier.py
    memory.py
    reporter.py
  tools/
    eda_tools.py
    plot_tools.py
  memory/
    lessons.json
    run_logs.jsonl
  outputs/
    plots/
  tests/
  README.md
  AGENTS.md
  requirements.txt
```

## Main functionality

The system should eventually support:

1. CSV upload through Streamlit
2. Dataset inspection:

   * row/column count
   * column names
   * inferred numeric/categorical/date/text columns
   * missing values
   * duplicate rows
   * sample rows
   * possible target columns
3. LLM planner:

   * receives user goal, dataset summary, previous observations, and relevant memory
   * chooses one next action from a fixed tool list
4. Tool executor:

   * runs safe predefined Python functions
5. Verifier:

   * checks whether an action is valid before execution
   * for example, histogram requires numeric column; bar plot requires categorical column
6. Memory:

   * stores run logs and compressed lessons
   * retrieves only a few relevant lessons rather than feeding full history to the LLM
7. Reflection:

   * records whether a step was useful, failed, or should influence future decisions
8. Final report:

   * summarizes findings, plots, limitations, and possible next analyses

## Example scenarios

The system should work well for these dataset types:

1. Customer churn dataset

   * target column such as `churn`
   * agent should prioritize target-group comparison and class balance

2. Retail sales dataset

   * date, region, product, sales, profit
   * agent should prioritize trends and grouped comparisons

3. Sensor or IoT dataset

   * timestamp, temperature, pressure, status
   * agent should prioritize missingness, anomaly checks, and trends

4. Messy dataset

   * missing values, mixed types, duplicates, invalid values
   * agent should detect quality issues before deeper analysis

## Edge cases to consider

The code should eventually handle:

* empty uploaded file
* non-CSV file
* CSV with only one column
* CSV with no numeric columns
* CSV with no categorical columns
* high missingness columns
* duplicate rows
* invalid selected column for a plot
* tool execution failure
* LLM returns invalid action
* user gives vague goal
* user gives no goal

## Development rules

Before implementing any feature:

1. Inspect whether similar functionality already exists.
2. Avoid duplicate implementations.
3. Keep changes minimal and modular.
4. Prefer simple readable Python over complex abstractions.
5. Keep the agent workflow explicit and easy to explain in the report.
6. Update README.md when setup, usage, architecture, or functionality changes.
7. Update requirements.txt if dependencies are added.
8. Add or update basic tests where practical.
9. Do not introduce unnecessary frameworks unless they clearly help the assignment.
10. Keep the project small enough to demo in 2 minutes.

## Current first milestone

Do not build the full agent yet.

The first implementation milestone after this guidance file should be:

* `agent/observer.py`
* deterministic dataset inspection functions
* basic tests for observer behavior
* minimal README update explaining the architecture
