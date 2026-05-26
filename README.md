# KAIROS

A guarded LLM-guided intelligent data analysis agent for CSV datasets.

KAIROS is a Streamlit-based COMPSCI 767 Assignment 2 prototype. It lets a user upload a CSV file, ask an analytical question in natural language, and receive verified deterministic analysis results with charts, summaries, an audit log, and a final report.

## What the System Can Do

- Upload CSV datasets through a Streamlit interface.
- Profile dataset structure, column types, missing values, duplicates, and quality issues.
- Accept natural-language analysis questions.
- Use an LLM planner through the Groq API.
- Use `llama-3.3-70b-versatile` as the default Groq planning model.
- Select deterministic analysis actions from fixed candidate actions.
- Verify tool names, arguments, columns, and data-type compatibility before execution.
- Run allow-listed deterministic EDA and statistical tools.
- Generate charts, structured summaries, cautions, and final reports.
- Maintain and export a transparent audit log.

## Why It Is Agentic

KAIROS follows an explicit agent workflow rather than acting as a general chatbot. It perceives the uploaded dataset and user query, profiles the data environment, plans suitable next analysis actions, verifies whether those actions are valid, executes deterministic tools, interprets structured results, and records an audit trace of what happened.

The LLM is used for planning and action selection only. Factual computation is performed by Python tools, and every selected action passes through verifier safeguards before execution.

## Technology Stack

- Python
- Streamlit
- pandas
- scipy
- Groq API
- `llama-3.3-70b-versatile`
- pytest

## Setup

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your own Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
KAIROS_LLM_PROVIDER=groq
KAIROS_LLM_MODEL=llama-3.3-70b-versatile
```

Markers and users can create a free Groq API key from Groq, then paste it into their local `.env` file. Do not commit real API keys.

Run the Streamlit app:

```powershell
streamlit run app.py
```

KAIROS can also run in deterministic fallback mode if no hosted planner is configured, but the intended assignment configuration uses Groq.

## Testing

Run the test suite:

```powershell
pytest
```

If `pytest` is unavailable, run:

```powershell
python -m unittest discover tests
```

## Sample Dataset for Testing

A sample CSV dataset is included in the repository:

`sample_data/employee_demo.csv`

Markers can:

1. Run the Streamlit app.
2. Upload `sample_data/employee_demo.csv`.
3. Try example questions such as:

- `What columns are in this dataset?`
- `What variable correlates with salary?`
- `Does salary vary by departments?`

## Example Capabilities

Dataset overview: KAIROS can report row and column counts, column names, inferred data types, missingness, duplicates, likely ID columns, and quality notes.

Targeted relationship analysis: KAIROS can identify variables most related to a named target such as salary, age, performance score, or promotion.

Group comparison analysis: KAIROS can compare a numeric variable across categorical groups, such as salary by department, with appropriate warnings and basic statistical checks.

Verification safeguards: KAIROS checks that requested tools exist, required arguments are present, referenced columns exist, and column types are compatible before executing tools.

Audit logging: KAIROS records the user question, dataset profile summary, planner mode, selected actions, verification results, execution status, key result summary, and timestamp.

Final report synthesis: KAIROS combines verified results into a concise final analysis report with findings, limitations, and suggested next analyses.

## Security and Safety Notes

- Raw datasets are not directly sent to the LLM.
- The LLM selects from deterministic candidate actions rather than writing arbitrary code.
- Tools are allow-listed in a fixed registry.
- The verifier checks actions before execution.
- API keys should stay in `.env`, not GitHub.
- `.env.example` contains placeholders only and is safe to commit.

## Limitations

- KAIROS is a prototype/research system, not a production analytics platform.
- There is no automatic replanning after failed verification.
- Statistical checks are basic and tool-specific rather than expert-level.
- The system works best on structured CSV datasets.
- Hosted LLM planning depends on valid Groq API configuration.
