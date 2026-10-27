# LLM Experiment Lab

A local app for running repeatable, auditable experiments across multiple LLMs via [OpenRouter](https://openrouter.ai/).

You define an experiment once, send it to as many models as you like, and get back a report where every number can be traced to a recorded response. The point is to replace "I asked a chatbot and it said X" with evidence you can rerun, check and share.

## How it works

Every experiment goes through the same five stages.

**1. Prompt**
Each experiment is a set of prompts, often with controlled variations (for example, the same scenario with only a name changed). The app sends every prompt to every selected model through OpenRouter.

**2. Collect**
Raw responses are stored exactly as returned, along with the model, prompt and run details. Nothing is summarised or discarded at this stage, so you can always go back to the source.

**3. Classify**
A separate evaluator model reads each response and labels it against a structured schema. Keeping the evaluator independent from the models under test avoids a model marking its own homework.

**4. Review**
A human checks each classification and can accept or correct it. Corrections are recorded, so the final results reflect reviewed judgements, not just the evaluator's first pass.

**5. Report**
The dashboard and exported report are built from the reviewed data. Every figure is computed from recorded responses. No conclusions are hardcoded anywhere.

## Built-in experiments

### Nurse gender assumptions
Tests whether models default to assuming a nurse is a woman. Prompts describe a nurse without stating gender, and the evaluator records which pronouns or gender cues the model introduces.

### Sentencing bias
Tests whether a model's sentencing recommendation changes when only one detail changes:

- the defendant's name, or
- the dialect of the defendant's statement.

Everything else in the case stays identical, so any difference in the recommended sentence can be tied to that single change.

## Supporting other kinds of study

The app isn't limited to these two experiments. Each experiment has an `evaluation_domain` field that decides:

- which evaluator is used
- which classification schema applies
- how results are analysed
- how the dashboard displays them

To add a new type of study, you define a new domain and point an experiment at it. The existing ones are good templates to copy from.

## Getting started

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
cp .env.example .env

streamlit run app.py
```

Opens at `http://localhost:8501`. The SQLite database is created automatically at `data/ai_verdict.db` on first run.

You'll need an OpenRouter API key. Keep it in a local environment file and never commit it.

## Project structure

```
app.py                    Home page: list / duplicate / open experiments
pages/
  1_Create_Experiment.py  Define prompts, models, repetitions, params
  2_Run_Experiment.py     Plan and execute runs; pause/resume
  3_Results_Dashboard.py  Filterable results with 95% confidence intervals
  4_Human_Review.py       Review and correct automated classifications
  5_Export.py             CSV / Markdown report / JSON archive downloads
  8_Settings.py           API key status, model catalog cache
core/
  db.py                   SQLite schema and all persistence
  openrouter_client.py    API calls, retry/backoff
  evaluator.py            Independent evaluator for the nurse-gender domain
  sentencing_evaluator.py Independent evaluator for the sentencing domain
  runner.py               Turns an experiment config into runs and executes them
  analysis.py             Shared statistics engine (Wilson CI, grouping, etc.)
  export.py               CSV / Markdown report / JSON archive generation
  charts.py               Dashboard chart builders
  seed.py / sentencing_seed.py   The two built-in experiments
tests/                    139 tests, no live API calls required to run them
```

## Contributing

Issues and pull requests are welcome, particularly new experiment domains, evaluator improvements and better analysis views.
