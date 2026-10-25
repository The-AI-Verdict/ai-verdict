# The AI Verdict - LLM Experiment Lab

A local, working application for running repeatable, auditable experiments against multiple LLMs
(via [OpenRouter](https://openrouter.ai)), collecting their raw responses, classifying them with an
independent structured evaluator, human-reviewing that classification, and exporting an evidence-based
report + YouTube production brief.

Built for **The AI Verdict** YouTube channel's first experiment: *"Do LLMs Assume Nurses Are Women?"* -
but the app itself is general-purpose; that experiment is just the built-in sample.

No conclusions are hardcoded anywhere. Every number shown on the dashboard, in the exported report, or in
the video brief is computed from actually recorded model responses and their (human-reviewable)
classifications.

## What's in here

```
app.py                         Home page: list / duplicate / open experiments
pages/
  1_Create_Experiment.py       Define an experiment: prompts, models, repetitions, params
  2_Run_Experiment.py          Plan runs, add/remove models, execute with live progress, pause/resume
  3_Results_Dashboard.py       Filterable results, proportions with 95% CI, disagreements
  4_Human_Review.py            Review raw response + automated classification side by side
  5_Export_and_Video_Brief.py  CSV / Markdown report / JSON archive / YouTube brief downloads
  6_Settings.py                API key status, model catalog cache, integrity notes
core/
  db.py                        SQLite schema + all persistence
  openrouter_client.py         Chat completions + model discovery, retry/backoff
  evaluator.py                 Independent gender-classification evaluator (prompt + orchestration)
  schema.py                    Pydantic JSON schema + validation for evaluator output
  runner.py                    Turns experiment config into runs; executes; pause/resume/crash-safe
  analysis.py                  Proportions, Wilson confidence intervals, filters, disagreements
  export.py                    CSV / Markdown report / video brief / JSON archive generation
  charts.py                    Matplotlib chart builders for the dashboard
  seed.py                      The built-in "Do LLMs Assume Nurses Are Women?" experiment
  ui_common.py                 Shared Streamlit helpers (env bootstrap, experiment selector, etc.)
tests/                         52 tests - see "Tests" below
```

## Setup

Requires Python 3.11+ (tested on 3.13).

```bash
cd "AI Verdict"
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

Get an OpenRouter API key at https://openrouter.ai/keys. OpenRouter gives access to models from OpenAI,
Anthropic, Google, Meta, Mistral, and many others through one API and one key.

## Running

```bash
streamlit run app.py
```

Opens at http://localhost:8501. The SQLite database is created automatically at `data/ai_verdict.db` on
first run.

## Running the sample experiment

1. On the home page, click **Load built-in sample experiment**. This seeds "Do LLMs Assume Nurses Are
   Women?" with three immutable prompt variants: `baseline` (the exact prompt from the brief),
   `role_reversed` (doctor/nurse swap which action they perform), and `alternative_profession`
   (pilot/flight attendant, to check whether any pattern is specific to doctor/nurse).
2. Go to **Run Experiment**, select the experiment, and add one or more models (either via **Discover
   models from OpenRouter**, which lists everything currently available through your key, or by typing a
   model id manually, e.g. `openai/gpt-4o-mini`).
3. Click **Plan runs** - this materialises every (prompt variant x model x repetition) combination as a
   row in the database and locks the prompt versions.
4. Click **Start / Resume**. Progress updates live; **Pause** stops after the current in-flight call
   finishes (nothing is left half-written). Resuming later picks up exactly where it left off - already
   completed runs are never re-sent.
5. Each successful response is automatically evaluated by an independent model (default
   `anthropic/claude-3.5-haiku`, configurable via `AI_VERDICT_EVALUATOR_MODEL`) using a structured
   rubric that never infers gender from profession, name, or stereotype - only from actual gendered
   language in the response, with a verbatim supporting quote required.
6. Go to **Human Review** to confirm or correct each automated classification. The original automated
   result is always preserved alongside your revision.
7. Go to **Results Dashboard** for proportions (with 95% Wilson confidence intervals and explicit sample
   sizes), filterable by model / prompt variant / review status.
8. Go to **Export and Video Brief** to download the CSV, the Markdown report, the YouTube production
   brief, and the full JSON archive.

## Environment variables

See `.env.example`. All are read via `python-dotenv` from a local `.env` file - never hardcoded, never
logged, never shown in the UI (the Settings page deliberately never displays even a masked key, since
this app is designed to be screen-recorded for video production).

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes, to run experiments | Your OpenRouter API key |
| `OPENROUTER_SITE_URL` | No | Sent as `HTTP-Referer` for OpenRouter's attribution |
| `OPENROUTER_APP_NAME` | No | Sent as `X-Title` for OpenRouter's attribution |
| `AI_VERDICT_DB_PATH` | No | Overrides the SQLite file location (default `./data/ai_verdict.db`) |
| `AI_VERDICT_EVALUATOR_MODEL` | No | Default evaluator model (default `anthropic/claude-3.5-haiku`) |

## Analysis methodology (what the numbers mean)

- **Denominator = valid classified responses only.** A run only counts toward a percentage if the API
  call succeeded *and* there is either a valid automated evaluation or a human review that resolved it.
  Failed API calls and unresolved-invalid evaluations are excluded and reported separately as counts, not
  silently dropped.
- **Explicit and implied assignments are always reported separately**, never combined into one
  unexplained score. A clearly-labelled "any assignment (explicit + implied)" figure is also shown,
  explicitly defined as their sum.
- **No cross-model "bias score" is computed anywhere.** The spec for this project explicitly prohibits
  inventing one without a defensible method, so none exists in this codebase.
- **Confidence intervals use the Wilson score interval** (valid at small n, unlike the naive normal
  approximation), implemented from scratch with no extra statistics dependency.
- **Groups with fewer than 20 valid responses are labelled "exploratory"** everywhere they appear
  (dashboard, report, brief) - they are never presented as if they were statistically conclusive.
- **Repeated samples from the same model are not independent observations.** This is stated explicitly in
  every generated report. A single prompt, however many times it's run, cannot establish a general
  stereotype - that's why the sample experiment ships with role-reversed and alternative-profession
  variants, and the generated report/brief actively encourage checking all of them before concluding
  anything.

## The evaluator

The full evaluator system prompt lives in `core/evaluator.py` (`EVALUATOR_SYSTEM_PROMPT`) and is
completely separate from whatever prompt is sent to the model under test - the evaluator instructions are
never seen by, or mixed into, the tested model's call. By default a different model evaluates than
whatever is under test (configurable per-experiment).

Every evaluation must satisfy a Pydantic schema (`core/schema.py`) before being accepted:
- `gender_classification`: `explicit_male` / `explicit_female` / `explicit_other` / `unspecified` /
  `ambiguous`
- `assignment_type`: `explicit` / `implied` / `none`
- a **verbatim** supporting quote is required whenever `assignment_type != "none"` - the quote is checked
  as an actual substring of the response being evaluated, so a fabricated or paraphrased quote fails
  validation
- `ambiguous` can never pair with `explicit`/`implied` (an ambiguous reference is never counted as a
  confirmed assignment) - enforced as a hard schema rule, not a convention
- a top-level `uncertain` flag (with required `uncertain_reason` when true) lets the evaluator flag its
  own low confidence independent of any single role's classification

An invalid evaluator response gets one automatic corrective retry (the validation error is sent back to
the evaluator model); if still invalid, it is stored as `invalid_needs_review` and surfaced on the
dashboard and in Human Review rather than silently accepted or discarded.

## Experimental integrity safeguards

- The same prompt text and generation parameters are sent to every selected model; if OpenRouter's model
  catalog says a model doesn't support a requested parameter, that's recorded per-run (not silently
  dropped) in each run's `unsupported_params`.
- Prompt versions lock the moment the first run against them is planned. Editing a prompt after that
  creates a new version (`core/db.py: edit_prompt_version`); the old version is never mutated.
- Every run stores: experiment id, prompt version, model id, run number, full prompt, full raw response,
  timestamp, generation parameters, token usage, cost (when OpenRouter reports it), latency, and any
  error - successes and failures are different `status` values, never conflated.
- Retries use exponential backoff and only re-send the HTTP request when no response was already
  received for that attempt, to avoid double-charging on a slow-but-successful call.
- Runs can be paused and safely resumed; a process crash mid-call leaves a run in `running` with no
  stored response, which is detected and safely re-queued (never skipped, never duplicated) - see
  `tests/test_runner_resume.py`.
- Run execution order can be randomised (`core/runner.py: _order_pending_runs`) to avoid confounding
  model/prompt effects with time-of-run effects; the originally *planned* order is kept separately from
  the *executed* order for auditability.
- OpenRouter does not always guarantee a fixed, reproducible model snapshot behind a model id - this is
  stated explicitly in every generated report, and the exact `model` string OpenRouter actually returned
  for each call is recorded and exported.

## Tests

```bash
pytest
# or, for a coverage summary:
pytest -v
```

52 tests, no live API calls (everything that would call OpenRouter is mocked via `monkeypatch`). Covers:

- prompt immutability / versioning (`test_db.py`)
- response persistence, success vs. failure distinction (`test_db.py`)
- API retry/backoff behaviour, which errors are/aren't retried, retry exhaustion (`test_openrouter_client.py`)
- evaluator JSON schema validation, including every gender-classification edge case - ambiguous rows,
  fabricated quotes, missing uncertainty reasons (`test_schema_evaluation.py`)
- the evaluator's repair-retry loop and markdown-fence stripping (`test_evaluator.py`)
- percentage/denominator correctness, Wilson CI behaviour, exploratory-sample flagging, disagreement
  detection (`test_analysis.py`)
- human-review overrides and revision history, preserving the original automated result (`test_db.py`)
- CSV / Markdown report / video brief / JSON archive generation, including the "never fabricate an
  example" fallback (`test_export.py`)
- pause/resume, crash recovery, idempotent run planning, no duplicate API calls across a pause/resume
  boundary (`test_runner_resume.py`)

Fixtures for testing the evaluator without live calls live in `tests/fixtures/sample_responses.py` -
hand-written exemplar model outputs and matching evaluator JSON, used to exercise the full pipeline
deterministically.

## Known limitations

- **No live model call has been made against this code.** I do not have an OpenRouter API key in this
  environment, so the full run -> evaluate pipeline is verified end-to-end with mocked HTTP responses
  (52 passing tests) and the Streamlit UI was smoke-tested live in a browser (experiment creation, model
  discovery UI, run planning, pause/resume controls, dashboard rendering, report/brief generation, all
  with an empty/zero-run dataset) - but not an actual OpenRouter request/response. Before recording real
  results, run a small experiment (e.g. 1 model x 1 variant x 1 repetition) first and check the raw
  response and evaluation on the Human Review page.
- Model discovery, cost reporting, and the exact set of `supported_parameters` all depend on what
  OpenRouter's `/models` endpoint currently returns for a given model - this can change over time and
  varies by provider.
- The evaluator is itself an LLM and can be wrong; that is exactly why Human Review exists and why
  automated output is never treated as ground truth anywhere in the dashboard, exports, or reports.
- This is a local, single-user tool (SQLite, short-lived connections per Streamlit rerun). It is not
  designed for concurrent multi-user access.
