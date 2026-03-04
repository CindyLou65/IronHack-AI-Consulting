# 🤖 Autonomous AI Research & Executive Briefing Agent

> An autonomous AI agent that monitors weekly AI developments and delivers McKinsey-grade executive briefings — automatically, to your phone.

**Author:** Cindy Lund  
**Date:** March 2026  
**Bootcamp:** Ironhack AI Consulting & Integration Bootcamp

---

## 🎯 Problem Statement

The AI field evolves too rapidly for manual monitoring. Business leaders need synthesized, decision-ready intelligence — not raw news feeds. This agent automates the entire research-to-briefing pipeline, delivering a structured executive report every week via Telegram and Google Drive.

---

## 🚀 What It Does

1. **Researches** the latest AI developments in real-time using Tavily search
2. **Retrieves context** from past reports and foundational AI research papers (RAG)
3. **Analyzes** developments using a McKinsey-grade analyst prompt with evidence tagging
4. **Classifies** the signal into one of 7 topic categories (Model Economics, LLM Infrastructure, etc.)
5. **Writes** a structured 400–500 word executive briefing (opportunity-first, not risk-first)
6. **Generates** a clean Telegram executive signal (4 sentences, mobile-optimized)
7. **Archives** the full report as a PDF to Google Drive
8. **Delivers** the summary + PDF directly to Telegram

---

## 🏗️ Architecture

```
Telegram /report command
        │
        ▼
   n8n Workflow
        │
        ▼
FastAPI Service (api.py)
        │
        ▼
LangGraph Pipeline
   ├── 🔍 Researcher Node    — Tavily real-time web search
   ├── 🧠 RAG Node           — Pinecone memory retrieval
   │     ├── Past weekly reports (context continuity)
   │     └── Foundational AI research papers (ArXiv)
   ├── 📊 Analyst Node       — McKinsey-grade signal analysis
   ├── ✍️  Writer Node        — Executive briefing composition
   ├── 📱 Summary Node       — Telegram executive signal extraction
   └── ✅ Reviewer Node      — Quality check + Pinecone memory save
        │
        ▼
   n8n Workflow
   ├── 📱 Telegram — Clean executive summary (4 sentences)
   ├── 📄 PDF Generation — Full formatted report
   ├── ☁️  Google Drive — PDF archived with date stamp
   └── 📱 Telegram — Full PDF delivered to phone
```

---

## 📋 Report Structure

Each weekly briefing contains:

| Section | Description |
|---|---|
| 📌 Executive Summary | 4 sentences: signal, impact, watch item, strategic implication |
| 🔍 Key Developments | 3 concrete facts with implications (no invented metrics) |
| 🏃 Competitive Signal | What market leaders are doing + wait risk |
| 📊 Impact Analysis | 2 business implications with KPIs and time horizons |
| ⚠️ Risks & Considerations | Only if genuinely material (max 2) |
| ✅ Recommended Actions | This week (monitor) + Next 30 days (if confirmed) |
| 🧠 Historical Context | Trend analysis + what to watch next week |

---

## 🏷️ Topic Classification

Every briefing is automatically classified into one of 7 categories:

- `LLM Infrastructure`
- `Model Economics`
- `AI Agents`
- `Multimodal AI`
- `Compute / Hardware`
- `Regulation`
- `Enterprise Adoption`

This creates a structured intelligence feed over time:
```
🚨 AI Weekly Executive Signal | Compute / Hardware — Mar 4, 2026
🚨 AI Weekly Executive Signal | Model Economics — Mar 11, 2026
🚨 AI Weekly Executive Signal | AI Agents — Mar 18, 2026
```

---

## 🧠 RAG Knowledge Base (Pinecone)

The agent uses structured retrieval across two knowledge layers:

### Layer 1: Weekly Report Memory
- Stores every generated report as a vector
- Retrieves relevant past reports for historical context
- Enables trend detection across weeks

### Layer 2: Foundational AI Research (13 papers, 26 chunks)
Each paper is stored as 2 chunks (technical + business relevance):

| Paper | Topic |
|---|---|
| Attention Is All You Need | Transformer Architecture |
| Chinchilla Scaling Paper | Scaling Laws |
| FlashAttention | Infrastructure |
| PagedAttention / vLLM | Infrastructure |
| Switch Transformer (MoE) | Model Architecture |
| Retrieval-Augmented Generation | RAG |
| Chain-of-Thought Prompting | Reasoning |
| Self-Consistency Reasoning | Reasoning |
| CLIP | Multimodal |
| Flamingo | Multimodal |
| ReAct | Agent Reasoning |
| Toolformer | Agent Reasoning |
| Scaling Laws for Neural LMs | Scaling Laws |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph |
| **LLM** | OpenAI GPT-4o |
| **Web Search** | Tavily API |
| **Vector Memory** | Pinecone (1024 dimensions) |
| **Embeddings** | OpenAI text-embedding-3-large |
| **API Service** | FastAPI + Uvicorn |
| **Automation** | n8n (cloud) |
| **PDF Generation** | PDF Generator API (pdfgeneratorapi.com) via n8n JavaScript wrapper |
| **Delivery** | Telegram Bot API |
| **Archive** | Google Drive |
| **Observability** | LangSmith |
| **Tunnel (dev)** | ngrok — exposes local FastAPI to public URL so n8n can reach it |

> **How FastAPI + ngrok work together:** FastAPI runs locally on `localhost:8000` inside VS Code. ngrok creates a public HTTPS URL that tunnels directly to that local port — allowing n8n (cloud) to send HTTP requests to your local Python agent.
| **Version Control** | GitHub |

---

## ⚙️ Prompt Engineering Highlights

### Analyst Node
- Evidence-tagged facts (press release / benchmark / report)
- Confidence levels per fact (High / Med / Low)
- Magnitude Rule: no invented comparisons without stated baseline
- Decision triggers in If/Then format with owner + timebox
- Topic classification from fixed 7-category taxonomy

### Writer Node
- Primary Signal as the structural spine of every briefing
- Opportunity-first framing (not risk-first)
- 400–500 word discipline
- "Not reported" instead of invented metrics
- Strategic vs tactical action separation
- Forward-looking Historical Context

---

## 📦 Project Structure

```
WEEK05_New_AI_Agent/
├── agent/
│   ├── __init__.py
│   ├── state.py              # AgentState TypedDict
│   └── graph.py              # 6 nodes + LangGraph workflow
├── api.py                    # FastAPI service
├── main.py                   # CLI entry point
├── setup_research_rag.py     # One-time RAG setup script
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and documentation
├── SETUP.md                  # Technical setup guide
├── n8n_workflow.json         # n8n workflow export
└── .env                      # API keys (not committed)
```

---

## 🚀 Quick Start

### Prerequisites
See [SETUP.md](SETUP.md) for full installation instructions.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your API keys
```

### 3. Initialize RAG knowledge base (one-time)
```bash
python setup_research_rag.py
```

### 4. Start the API server
```bash
uvicorn api:app --reload
```

### 5. Start ngrok tunnel (for n8n)
```bash
ngrok http 8000
```

### 6. Trigger via CLI
```bash
python main.py
```

### 7. Trigger via Telegram
Send `/report` to your configured Telegram bot.

---

## 📊 Observability

| Metric | Tool |
|---|---|
| Token usage per run | LangSmith |
| Cost per report (~$0.03) | LangSmith |
| Latency per node | LangSmith |
| Error rate | LangSmith |
| API health | FastAPI /health endpoint |

---

## 🗺️ Roadmap

- [ ] LinkedIn post generation node
- [ ] Benchmark context RAG (Stanford AI Index, Epoch AI)
- [ ] Enterprise context RAG (McKinsey State of AI)
- [ ] Weekly signals rotation (SemiAnalysis, Import AI, Ben's Bites)
- [ ] ArXiv live feed for breaking research papers
- [ ] Scheduled weekly delivery (no manual trigger)
- [ ] Multi-topic parallel research
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)

---

## 🎓 About This Project

Built independently as a capstone project during the Ironhack AI Consulting & Integration Bootcamp (March 2026). All architecture decisions, prompt engineering, and implementation were self-directed.
