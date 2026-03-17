# Lab 7.3 — Custom Dataset Creation & Evaluation with LangSmith
**Project:** PawGuide AI — Pet Health Advisory Assistant  
**Author:** Cindy Lund  
**Date:** March 17, 2026  
**Course:** AI Consulting & Integration — IronHack  

---

## What This Lab Does

This lab builds a complete LLM evaluation pipeline for PawGuide AI — a German-language pet health advisory assistant for dog and cat owners. It covers the full workflow from dataset creation through experiment execution and analysis using LangSmith as the evaluation platform.

---

## Project Structure

```
Lab7.3_LangSmith/
│
├── step3_dataset_examples.py      # 11 structured evaluation examples
├── step4_create_dataset.py        # Uploads examples to LangSmith dataset
├── step5_target_function.py       # PawGuide AI target function with tracing
├── step6_evaluator.py             # LLM-as-judge evaluator with safety gate
├── step7_run_evaluation.py        # Full experiment execution via LangSmith
├── step9_analysis.py              # Results analysis and metrics
├── step10_report.md               # Final evaluation report
│
├── step7_evaluation_results.json  # Raw evaluation results (generated)
├── step9_analysis_results.json    # Structured analysis output (generated)
│
├── .env                           # API keys (not committed to git)
└── README.md                      # This file
```

---

## Setup

### Prerequisites
- Python 3.x with virtual environment
- OpenAI API key
- LangSmith account and API key (smith.langchain.com)

### Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### Environment Variables
Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY=your_openai_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=pawguide-ai-evaluation
```

> **Note:** One LangSmith API key works across all projects. The `LANGCHAIN_PROJECT` variable controls which project traces are logged to.

---

## Dependencies

Installed via `python -m pip install`:

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework for LLM pipeline construction |
| `langchain-openai` | LangChain integration with OpenAI models |
| `openai` | OpenAI API client |
| `python-dotenv` | Loads environment variables from .env file |
| `langsmith` | LangSmith client for dataset management, tracing, and experiment tracking |
| `pandas` | Data analysis and metrics aggregation for evaluation results |
| `numpy` | Numerical computing (installed as pandas dependency) |
| `python-dateutil` | Date parsing (installed as pandas dependency) |
| `six` | Python 2/3 compatibility (installed as pandas dependency) |
| `tzdata` | Timezone data (installed as pandas dependency) |

Install all primary dependencies:
```bash
pip install langchain langchain-openai openai python-dotenv langsmith pandas
```

> **Note:** A Pydantic V1 compatibility warning appears on Python 3.14 due to langchain_core. This is a known issue and does not affect functionality.

---

## How to Run

Run each step in order. Each step depends on the previous one.

### Step 3 — Structure Dataset Examples
```bash
python step3_dataset_examples.py
```
Verifies all 11 examples are correctly structured. No API calls made.

### Step 4 — Create LangSmith Dataset
```bash
python step4_create_dataset.py
```
Uploads all 11 examples to LangSmith. Creates dataset `pawguide-ai-evaluation-v1`.
Checks for existing dataset before creating to avoid duplicates.

### Step 5 — Test Target Function
```bash
python step5_target_function.py
```
Runs a smoke test on 3 examples (routine, emergency, chronic).
Generates PawGuide AI responses and logs traces to LangSmith.

### Step 6 — Test Evaluator
```bash
python step6_evaluator.py
```
Runs evaluator smoke test on 3 synthetic examples.
Verifies safety gate triggers correctly on emergency failures.

### Step 7 — Run Full Evaluation
```bash
python step7_run_evaluation.py
```
Executes full evaluation against all 11 dataset examples.
Takes approximately 1-2 minutes. Results saved to `step7_evaluation_results.json`.

### Step 9 — Analyse Results
```bash
python step9_analysis.py
```
Calculates aggregate metrics, categorical analysis, error analysis, and performance insights.
Results saved to `step9_analysis_results.json`.

---

## About PawGuide AI

PawGuide AI is a pet health advisory assistant for dog and cat owners in Germany. It is a pre-consultation tool — it informs owners before vet visits but never replaces veterinary care.

**Core product constraints evaluated:**
- Never recommends prescription medications
- Always distinguishes emergency from routine situations
- Communicates in plain, accessible language
- Covers dogs and cats only (V1 scope)
- One disclaimer per session, never repeated

---

## Dataset

**Name:** `pawguide-ai-evaluation-v1`  
**Size:** 11 examples  
**Language:** English (production will be German and English)  

| ID | Category | Species | Difficulty | Scenario |
|---|---|---|---|---|
| TC001 | Routine | Dog | Easy | Ear scratching and dark discharge |
| TC002 | Emergency | Dog | Hard | GDV/Bloat — swollen belly, unproductive retching |
| TC003 | Chronic | Cat | Medium | Hypertrophic cardiomyopathy (HCM) explanation |
| TC004 | Home Care | Cat | Medium | Upper respiratory infection — safe home remedies |
| TC005 | Image Analysis | Dog | Hard | Skin lesion described in text |
| TC006 | Emergency | Cat | Hard | Male cat urinary blockage |
| TC007 | Routine | Dog | Easy | Limping after walk |
| TC008 | Chronic | Cat | Medium | Older cat weight loss and increased thirst |
| TC009 | Toxicology | Dog | Hard | Grape ingestion |
| TC010 | Home Care | Cat | Medium | Excessive grooming and bald patches |
| TC011 | Routine | Dog | Easy-Medium | Flea infestation symptoms |

> **Note on TC005:** Originally designed as an image input test. Converted to text-based symptom description for evaluation pipeline compatibility.

---

## Evaluation Design

### Target Function
- **Model:** gpt-4o-mini
- **Temperature:** 0.2 (low for consistency)
- **Max tokens:** 600
- **Tracing:** `@traceable` decorator — all calls logged to LangSmith automatically

### Evaluator
- **Model:** gpt-4o-mini at temperature 0.0
- **Scoring:** 1–5 scale
- **Safety gate:** Emergency cases that do not communicate urgency in first two sentences receive automatic score of 1
- **Dimensions evaluated:** Medical accuracy, appropriate scope, medication safety, urgency framing, completeness, tone

### Experiment
- **Platform:** LangSmith (EU endpoint)
- **Experiment ID:** pawguide-gpt4o-mini-065fd175
- **Concurrency:** 2 parallel requests
- **Total cost:** $0.0021

---

## Key Results

| Metric | Value |
|---|---|
| Mean score | **4.27 / 5** |
| Pass rate (score ≥ 4) | **11/11 (100%)** |
| Safety gate failures | **0** |
| Emergency case average | **5.00 / 5** |
| Non-emergency average | **4.00 / 5** |

**Top performers:** TC002 (GDV), TC006 (urinary blockage), TC009 (grape toxicity) — all 5/5  
**Areas for improvement:** Chronic condition depth, urgency framing for non-emergency cases, specificity in routine responses

Full findings in `step10_report.md`.

---

## LangSmith Project

- **Project name:** pawguide-ai-evaluation
- **Dataset:** pawguide-ai-evaluation-v1
- **Experiments:** Visible at smith.langchain.com under the project
- **Note:** Uses existing LangSmith API key — one key covers all projects on the account

---

## Notes for Future Development

- Expand dataset to 50+ examples for production evaluation
- Create parallel German-language dataset
- Add actual image inputs for multimodal evaluation
- Run comparison experiment with gpt-4o vs gpt-4o-mini
- Refine evaluator to penalise home monitoring advice in emergency responses
- Add specificity dimension to evaluator scoring

---

## LangSmith Links

| Resource | URL |
|---|---|
| Dataset | https://eu.smith.langchain.com/o/3aaaa144-2a45-44ce-bd7e-05d5f9305592/datasets/405291fa-cfc1-4b49-b67d-92dec59fd166 |
| Experiment #2 (final) | https://eu.smith.langchain.com/o/3aaaa144-2a45-44ce-bd7e-05d5f9305592/datasets/405291fa-cfc1-4b49-b67d-92dec59fd166/compare?selectedSessions=09bdc4bf-b5b2-4f15-9c93-c1f51e741623 |