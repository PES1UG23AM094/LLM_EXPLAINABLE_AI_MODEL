# ExplainAI — Decision Justification System

An explainable AI assistant that analyzes decision questions and returns structured, transparent reasoning — including confidence scores, assumption tracking, hallucination risk assessment, and alternative viewpoints.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# Edit .env and replace "your_api_key_here" with your actual key

# 4. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
MINI_PROJECT/
├── models.py              # Pydantic data models (shared schema)
├── llm_pipeline.py        # LLM call and structured output parsing
├── explanation_engine.py  # Post-processing and UI helper functions
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── .env.example           # API key template (safe to commit)
└── .env                   # Your actual API key (gitignored, never commit)
```

---

## File-by-File Breakdown

### `models.py` — Data Models

Defines the Pydantic v2 schemas that represent the AI's structured output. All models use `ConfigDict(extra="forbid")` so OpenAI's Structured Outputs feature (which requires `additionalProperties: false`) works correctly.

**`ReasoningStep`**
Represents a single step in the chain-of-thought reasoning chain.

| Field | Type | Purpose |
|---|---|---|
| `step_number` | int | Step index (1, 2, 3…) |
| `description` | str | Text of the reasoning step |
| `is_weak` | bool | Whether the step is flagged as uncertain (default: False) |
| `weakness_reason` | str | Explanation of why it's weak (default: empty) |

**`DecisionAnalysis`**
The full analysis returned by the AI for a given decision question.

| Field | Type | Purpose |
|---|---|---|
| `answer` | str | Main recommendation/conclusion |
| `reasoning_steps` | List[ReasoningStep] | Step-by-step chain of thought |
| `assumptions` | List[str] | Key assumptions the analysis relies on |
| `confidence_score` | float | How confident the model is (0.0–1.0) |
| `uncertainty_highlights` | List[str] | Specific areas the model is unsure about |
| `alternative_viewpoints` | List[str] | Opposing or different perspectives |
| `counter_argument` | str | Strongest argument against the main answer |
| `hallucination_risk` | "low"\|"medium"\|"high" | Risk that the model fabricated information |
| `hallucination_reasons` | List[str] | Why that hallucination risk level was assigned |

---

### `llm_pipeline.py` — LLM Pipeline

Handles all communication with the OpenAI API. This is the core AI layer.

**`analyze_decision(question, model)`**
Takes a decision question string and a model name, returns a fully populated `DecisionAnalysis` object.

How it works:
1. Creates an `OpenAI` client (reads `OPENAI_API_KEY` from environment automatically)
2. Sends the question to `client.beta.chat.completions.parse()` — OpenAI's structured outputs endpoint
3. Passes `response_format=DecisionAnalysis` so the API validates and parses the response directly into the Pydantic model
4. Clamps `confidence_score` to [0.0, 1.0] in case the model drifts out of range
5. Raises `ValueError` if the model refuses to answer

**System prompt** (defined at module level as `SYSTEM_PROMPT`): instructs the model to be an "Explainable AI Decision Assistant", produce numbered reasoning steps, be honest about uncertainty, and assess its own hallucination risk.

**Supported models:**
- `gpt-4o` — most capable, default
- `gpt-4o-mini` — faster and cheaper

---

### `explanation_engine.py` — Explanation Engine

Post-processes the AI output and provides UI helper functions. This layer adds an extra quality check on top of what the model self-reports.

**`detect_weak_steps(analysis)`**
Scans every reasoning step for signs of uncertain or vague language — even if the model didn't self-flag it. Returns a new `DecisionAnalysis` with updated step annotations.

A step is flagged as weak if it contains any of these indicator words/phrases:
> might, could, perhaps, maybe, possibly, probably, seems, appears, likely, unlikely, not sure, uncertain, depends, varies, generally, typically, often, sometimes, I think, I believe, I feel, I assume

A step is also flagged if it is fewer than 30 characters (not enough detail).

When a step is flagged, a `weakness_reason` is set explaining why (e.g., "Contains uncertain language: might, could").

**`get_reasoning_quality(analysis)`**
Computes an overall reasoning quality score as a weighted blend:
- 70% — proportion of strong (non-weak) steps
- 30% — the model's stated confidence score

Returns `(score: float, label: str)` where label is one of: `"Strong"`, `"Moderate"`, `"Weak"`, `"Very Weak"`.

**UI helpers** (pure functions, no side effects):
- `get_confidence_color(score)` → green/orange/red hex color based on score thresholds (0.7, 0.4)
- `get_hallucination_color(risk)` → green/orange/red for low/medium/high
- `get_hallucination_emoji(risk)` → ✅/⚠️/🚨 for low/medium/high

---

### `app.py` — Streamlit UI

The web interface. Renders all analysis results visually and handles user interaction.

**Page layout:**
Two tabs are shown side by side.

**Tab 1 — Single Analysis**
- Dropdown to pick an example question or type your own
- Model selector (GPT-4o or GPT-4o Mini)
- "Analyze Decision" button triggers `analyze_decision()` then `detect_weak_steps()`
- Results are stored in `st.session_state` so they persist across Streamlit rerenders

**Tab 2 — Model Comparison**
- Same question sent to two models simultaneously
- Results shown side by side in two columns
- Useful for comparing how GPT-4o and GPT-4o Mini reason differently

**`render_analysis(analysis)`**
The main rendering function. Displays a full `DecisionAnalysis` in this order:
1. **Answer box** — the main recommendation in a highlighted card
2. **Metrics row** (3 columns):
   - Confidence bar — gradient fill showing percentage
   - Hallucination risk badge — color-coded with emoji
   - Reasoning quality — label with strong/total step count
3. **Reasoning Steps** — expandable section; weak steps shown in yellow with warning note, strong steps in green
4. **Detail columns** (3 columns): Assumptions, Uncertainties, Alternative Views
5. **Counter-argument** — shown in an info box
6. **Hallucination details** — expandable section explaining the risk level

**CSS** is injected via `st.markdown(..., unsafe_allow_html=True)` to style the answer box, step cards, and risk badge. Colors are explicitly set with `!important` to work correctly in both light and dark Streamlit themes.

---

### `requirements.txt` — Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openai` | >=1.40.0 | OpenAI SDK with structured outputs support |
| `streamlit` | >=1.35.0 | Web UI framework |
| `python-dotenv` | >=1.0.0 | Loads `.env` file into environment variables |
| `pydantic` | >=2.0.0 | Data validation and schema generation |

---

### `.env.example` — API Key Template

A safe template file committed to the repo showing the required environment variable:

```
OPENAI_API_KEY=your_api_key_here
```

Copy this to `.env` and fill in your real key. The `.env` file is listed in `.gitignore` and will never be committed.

---

## How the Code Flows End-to-End

```
User types question
        │
        ▼
    app.py (button click)
        │
        ▼
llm_pipeline.analyze_decision()
  ├── Builds system + user messages
  ├── Calls OpenAI structured outputs API
  └── Returns DecisionAnalysis (validated Pydantic object)
        │
        ▼
explanation_engine.detect_weak_steps()
  ├── Scans reasoning steps for uncertain language
  └── Returns updated DecisionAnalysis with weak flags set
        │
        ▼
    app.py render_analysis()
  ├── Answer box
  ├── Confidence bar (get_confidence_color)
  ├── Hallucination badge (get_hallucination_color, get_hallucination_emoji)
  ├── Reasoning quality (get_reasoning_quality)
  ├── Reasoning steps (color-coded by is_weak)
  ├── Assumptions / Uncertainties / Alternative views
  ├── Counter-argument
  └── Hallucination reasons
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key from platform.openai.com/api-keys |

---

## Security Notes

- Never commit your `.env` file — it is gitignored by default
- Rotate your API key immediately if it is ever exposed (e.g., pasted in a chat or pushed to a public repo)
- The `.env.example` file is safe to commit; it contains no real credentials
