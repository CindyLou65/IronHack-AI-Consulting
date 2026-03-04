# agent/graph.py

from multiprocessing import context
import os
import langsmith  # LangSmith auto-detects env variables
import time # adding Retry to make agent more resilient when APIs are slow or rate-limited or fail temporarily
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from agent.state import AgentState

# ─────────────────────────────────────────────
# RETRY HELPER
# Automatically retries failed API calls
# ─────────────────────────────────────────────
def retry_api_call(func, max_retries=3, wait_seconds=5):
    """
    Retries a function up to max_retries times.
    Waits wait_seconds between each attempt.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"⏳ Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"❌ API call failed after {max_retries} attempts: {e}")
                raise


# ─────────────────────────────────────────────
# NODE 1: RESEARCHER
# Real-time web search via Tavily
# ─────────────────────────────────────────────
def researcher_node(state: AgentState) -> dict:
    print("--- NODE: RESEARCHER (Live Tavily Search) ---")
    topic = state["target_topic"]

    # Real-time web search with retry
    search_result = retry_api_call(
        lambda: tavily.search(
            query=topic,
            search_depth="advanced",
            max_results=3
        )
    )

    # Format results for the next node
    context = "\n".join([
        f"- {res['title']}: {res['content']}"
        for res in search_result['results']
    ])
    return {"research_data": context}


# ─────────────────────────────────────────────
# NODE 2: RAG (Pinecone Memory)
# Retrieves past reports AND foundational research
# ─────────────────────────────────────────────
def rag_node(state: AgentState) -> dict:
    print("--- NODE: RAG (Pinecone Memory) ---")
    topic = state["target_topic"]
    
    try:
        # Connect to our index
        index = pinecone_client.Index("ai-agent-reports")
        
        # Get embeddings for our topic using OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Convert topic to a vector
        embedding_response = client.embeddings.create(
            input=topic,
            model="text-embedding-3-large",
            dimensions=1024
        )
        topic_vector = embedding_response.data[0].embedding
        
        # ─────────────────────────────────────────────
        # RETRIEVAL 1: Past weekly reports
        # ─────────────────────────────────────────────
        past_results = index.query(
            vector=topic_vector,
            top_k=3,
            include_metadata=True,
            filter={"type": {"$ne": "research"}}  # Exclude research chunks
        )
        
        past_insights = ""
        if past_results["matches"]:
            past_insights = "\n".join([
                f"- {match['metadata'].get('summary', 'No summary available')}"
                for match in past_results["matches"]
                if match["score"] > 0.7
            ])

        # ─────────────────────────────────────────────
        # RETRIEVAL 2: Foundational research chunks
        # Structured retrieval: 1 technical + 1 business
        # ─────────────────────────────────────────────
        tech_results = index.query(
            vector=topic_vector,
            top_k=1,
            include_metadata=True,
            filter={
                "type": {"$eq": "research"},
                "chunk_type": {"$eq": "technical"}
            }
        )

        biz_results = index.query(
            vector=topic_vector,
            top_k=1,
            include_metadata=True,
            filter={
                "type": {"$eq": "research"},
                "chunk_type": {"$eq": "business"}
            }
        )

        # Extract research chunks
        research_context = ""
        if tech_results["matches"]:
            tech_match = tech_results["matches"][0]
            research_context += f"\nFOUNDATIONAL RESEARCH (Technical):\n"
            research_context += f"{tech_match['metadata'].get('text', '')}\n"
            research_context += f"Confidence: {tech_match['metadata'].get('confidence', 'high')}\n"

        if biz_results["matches"]:
            biz_match = biz_results["matches"][0]
            research_context += f"\nFOUNDATIONAL RESEARCH (Business):\n"
            research_context += f"{biz_match['metadata'].get('text', '')}\n"
            research_context += f"Confidence: {biz_match['metadata'].get('confidence', 'high')}\n"

        # ─────────────────────────────────────────────
        # COMBINE: Past reports + Research context
        # ─────────────────────────────────────────────
               
        context_parts = []

        if past_insights:
            context_parts.append(f"PAST WEEKLY REPORTS:\n{past_insights}")
        else:
            context_parts.append("PAST WEEKLY REPORTS: No highly relevant past reports found.")

        if research_context:
            context_parts.append(f"FOUNDATIONAL RESEARCH CONTEXT:{research_context}")
        else:
            context_parts.append("FOUNDATIONAL RESEARCH CONTEXT: No relevant research found.")

        context = "\n\n".join(context_parts)

    except Exception as e:
        print(f"Pinecone error: {e}")
        context = "Historical memory (Pinecone) not yet initialized."
        
    print(f"📚 Past context found: {context[:300]}...")
    print(f"🔬 Research context found: '{research_context[:200] if research_context else 'EMPTY'}'")

    return {"past_context": context}


# ─────────────────────────────────────────────
# NODE 3: ANALYST
# GPT-4o analyzes and ranks the research (improved prompt with structured output and clearer instructions)
# ─────────────────────────────────────────────
def analyst_node(state: AgentState) -> dict:
    print("--- NODE: ANALYST (AI Reasoning) ---")
    prompt = f"""
You are a senior AI industry analyst writing for C-suite executives (CEO/CIO/CTO/CISO/COO).
Your job is to convert noisy research into a decision-ready assessment.

TOPIC: {state['target_topic']}

RESEARCH DATA (may include duplicates, hype, partial claims):
{state['research_data']}

ANALYST GOAL
Produce a brief an executive can act on in <3 minutes:
- What changed (verifiable)
- Why it matters (business impact)
- What to watch (signals)
- When to act (decision triggers)

METHOD
1) Extract only the 3–6 most material facts (release/capability/pricing/partnership/regulation/security/benchmarks).
2) De-duplicate. If sources conflict, state the conflict and pick the best-supported claim.
3) Quantify when available. If missing, write “not reported” (do NOT invent).
4) Keep claims falsifiable. Avoid hype words.

OUTPUT FORMAT — follow EXACTLY:

## Primary Signal of the Week
**Topic:** [choose EXACTLY ONE: LLM Infrastructure | Model Economics | AI Agents | Multimodal AI | Compute / Hardware | Regulation | Enterprise Adoption]
**Signal:** [1 sentence: the single most important development this week]
**Why it matters:** [1 sentence: cost/revenue/risk/competitive impact + time horizon]
**Confidence:** [High/Med/Low]

## What Changed (Evidence-Backed)
- **Fact 1:** [1 sentence, specific] **Evidence:** [press release/docs/benchmark/incident report/analysis] **Metric:** [number or “not reported”] **Confidence:** [High/Med/Low]
- **Fact 2:** ...
- **Fact 3:** ...
(Optional: up to 6 facts total)

## Impact Scores
**Technical Impact:** [0-100]/100 — [1 sentence rationale]
SCORING GUIDE: Most weekly developments score 50-75. Only paradigm shifts score 85+. Incremental updates score 30-50. Be honest — do not default to 85.
**Business Materiality:** [1-5]/5 — [1 sentence: revenue/cost/risk impact within timeframe]
**Time Horizon:** [Now / 30–90d / 6–12m]
**Confidence Level:** [High/Med/Low] — [1 sentence on evidence quality]

## Top 2 Business Implications (Decision-Relevant)
1) **[Implication Title]:** [2–3 sentences: who is affected, KPI change, horizon, dependencies/constraints]
2) **[Implication Title]:** [same]

## Competitive Signal
[2 sentences: what leading players are doing + what “wait” risks]

## Key Risks (Practical, Not Theoretical)
- **Risk:** [1 sentence] **Likelihood:** [H/M/L] **Severity:** [H/M/L] **Mitigation lever:** [policy/process/architecture choice]
- **Risk:** ...
(2–4 risks)

## Decision Triggers (If/Then)
- If [observable signal + threshold], then [action + owner + timebox].
- If [observable signal + threshold], then [action + owner + timebox].
(2–4 triggers)

CONSTRAINTS
- No drastic weekly strategy shifts; prefer monitoring, pilots, governance, vendor evaluation, sequencing.
- Do not repeat research verbatim; synthesize.
- MAGNITUDE RULE
--Do not use comparative claims like "10x cheaper", "dramatic reduction", or
--"significant improvement" unless the comparison baseline is explicitly stated
--in the research data. If the baseline is unclear, describe the change
--qualitatively (e.g., "lower pricing relative to prior models") instead.
"""
    response = retry_api_call(lambda: llm.invoke(prompt))
    return {"analysis": response.content}


# ─────────────────────────────────────────────
# NODE 4: WRITER
# GPT-4o compiles the final executive briefing (improved prompt with clearer structure and style rules)
# ─────────────────────────────────────────────
def writer_node(state: AgentState) -> dict:
    print("--- NODE: WRITER (AI Composition) ---")
    prompt = f"""
You are a senior business intelligence writer producing a Weekly AI Executive Briefing
for C-suite and business leaders (CEO, CIO, CTO, CISO, COO). Write crisp, decision-oriented prose.

TOPIC: {state['target_topic']}

INPUTS
Research (raw): {state['research_data']}
Analysis (structured): {state['analysis']}
Historical context (prior weeks): {state['past_context']}

GOAL
Deliver an executive briefing that enables a decision in <3 minutes:
- what changed, why it matters, what to watch, and when to act.

IMPORTANT CONTEXT
- AI strategy changes on quarterly cycles, not weekly.
- Weekly actions should be MONITORING, PILOTS, and EVALUATION.
- Never recommend changing AI strategy based on a single week’s developments.

CRITICAL INSTRUCTION
- The entire briefing MUST be centered on the **Primary Signal of the Week** from the analysis.
- All other developments should support or contextualize that primary signal.
- Aim for 400–500 words (max 600).
- Each section must appear EXACTLY ONCE — never repeat any section or bullet point.
- If you find yourself repeating content, stop and move to the next section.
- Use the **Topic** from the analyst's "Primary Signal of the Week" in the Telegram summary header.
- The ## Telegram Executive Signal section is for Telegram ONLY — it must appear at the very end of 
  your response after all other sections, clearly separated.


BRIEFING STRUCTURE — include these exact sections:

# Weekly AI Intelligence Briefing

## Executive Summary
Write 3–4 sentences:
1) Restate the Primary Signal of the Week (what changed)
2) Why it matters (business impact + time horizon)
3) The one signal to monitor this week
4) Strategic implication over the next 30–90 days

## Key Developments
Exactly 3 bullets. Each bullet MUST include:
- a concrete development (release/benchmark/pricing/regulation/partnership)
- a measurable detail if available — omit the metric line entirely if not available, do NOT write "Metric: not reported"
- a short "Implication:" clause

## Competitive Signal
2 sentences:
- what leading players are doing
- what risk businesses face if they wait

## Impact Analysis
Rewrite the **Top 2 Business Implications** for non-technical leaders.
For each implication include:
- who is affected
- what KPI could change
- time horizon: Now / 30–90d / 6–12m

Also include:
- **Technical Impact:** [x/100]
- **Business Materiality:** [x/5]
- **Who should care:** [Product / Security / Legal / Finance / Operations]

## Risks & Considerations
Maximum 2 bullets. Use this format:
- **Risk:** [1 sentence] **Likelihood:** [H/M/L] **Severity:** [H/M/L] **Mitigation:** [1 phrase]

## Recommended Actions
### This Week (Monitor & Evaluate)
- **Watch:** [specific signal/metric]
- **Share with:** [team/function]

### Next 30 Days (If Signal Confirms)
- **Action:** [verb + outcome]
- **Owner:** [role]
- **Trigger:** [observable confirmation — NEVER invent percentages or numbers; 
    if no benchmark exists in the research data write: 
    "Observable improvement vs current baseline (establish baseline before piloting)"]
- **Success metric:** [measurable KPI]

## Historical Context
2–3 sentences comparing this week to prior weeks:
- escalating trend / stable pattern / one-off signal
- what to watch next week

## Telegram Executive Signal
Generate exactly 4 lines in this format:

🚨 AI Weekly Executive Signal | [Topic] — {{ date }}

Primary signal:
[1 sentence from Executive Summary sentence 1]

Why it matters:
[1 sentence from Executive Summary sentence 2]

What to watch:
[1 sentence from Executive Summary sentence 3]

Strategic implication:
[1 sentence from Executive Summary sentence 4]

📄 Full briefing attached.

STYLE RULES
- Plain language, short paragraphs, no jargon without explanation.
- No hype words (“game-changing”, “revolutionary”).
- Prefer numbers when available; never invent.
- Under 600 words total.
"""
    response = retry_api_call(lambda: llm.invoke(prompt))
    return {"final_report": response.content}

# ─────────────────────────────────────────────
# NODE 4.5: SUMMARY (Telegram Executive Signal)
# Extracts the Telegram summary from the report
# ─────────────────────────────────────────────
def summary_node(state: AgentState) -> dict:
    print("--- NODE: SUMMARY (Telegram Executive Signal) ---")
    report = state["final_report"]
    
    # Extract Telegram Executive Signal section from report
    try:
        if "Telegram Executive Signal" in report:
            # Extract everything after "## Telegram Executive Signal"
            telegram_part = report.split("## Telegram Executive Signal")[1].strip()
            # Clean up any trailing sections
            if "\n## " in telegram_part:
                telegram_part = telegram_part.split("\n## ")[0].strip()
        else:
            # Fallback if section not found
            telegram_part = f"🚨 AI Weekly Executive Signal\n\n📄 Full briefing attached."
            
    except Exception as e:
        print(f"Summary extraction error: {e}")
        telegram_part = f"🚨 AI Weekly Executive Signal\n\n📄 Full briefing attached."
    
    print(f"📱 Telegram summary: {telegram_part[:100]}...")
    # Remove Telegram section from final report
    clean_report = state["final_report"]
    if "## Telegram Executive Signal" in clean_report:
        clean_report = clean_report.split("## Telegram Executive Signal")[0].strip()

    return {
        "telegram_summary": telegram_part,
        "final_report": clean_report
    }

# ─────────────────────────────────────────────
# NODE 5: REVIEWER
# Approves report and saves it to Pinecone memory
# ─────────────────────────────────────────────
def reviewer_node(state: AgentState) -> dict:
    print("--- NODE: REVIEWER (Final Check & Memory Save) ---")
    
    try:
        # Connect to Pinecone index
        index = pinecone_client.Index("ai-agent-reports")
        
        # Get embeddings for the topic
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        embedding_response = client.embeddings.create(
            input=state["target_topic"],
            model="text-embedding-3-large",
            dimensions=1024
        )
        topic_vector = embedding_response.data[0].embedding
        
        # Save report to Pinecone with metadata
        import time
        index.upsert(vectors=[{
            "id": f"report_{int(time.time())}",
            "values": topic_vector,
            "metadata": {
                "topic":   state["target_topic"],
                "summary": state["final_report"][:1000],  # Store first 1000 chars
                "date":    time.strftime("%Y-%m-%d")
            }
        }])
        print("✅ Report saved to Pinecone memory!")
        
    except Exception as e:
        print(f"Could not save to Pinecone: {e}")
    
    # Auto-approve for the build phase
    return {"is_approved": True}

# ─────────────────────────────────────────────
# ASSEMBLE THE GRAPH
# ─────────────────────────────────────────────
def build_graph():
    # Initialize clients here AFTER load_dotenv() has run
    global tavily, pinecone_client, llm
    tavily          = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    llm             = ChatOpenAI(model="gpt-4o", temperature=0.7)

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("rag",        rag_node)
    workflow.add_node("analyst",    analyst_node)
    workflow.add_node("writer",     writer_node)
    workflow.add_node("summary",    summary_node)
    workflow.add_node("reviewer",   reviewer_node)

    # Define flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "rag")
    workflow.add_edge("rag",        "analyst")
    workflow.add_edge("analyst",    "writer")
    workflow.add_edge("writer",     "summary")
    workflow.add_edge("summary",    "reviewer")
    workflow.add_edge("reviewer",   END)

    return workflow.compile()
