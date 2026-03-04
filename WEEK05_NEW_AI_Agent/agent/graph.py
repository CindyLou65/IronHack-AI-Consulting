# agent/graph.py

import os
import langsmith  # LangSmith auto-detects env variables
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from agent.state import AgentState

# ─────────────────────────────────────────────
# NODE 1: RESEARCHER
# Real-time web search via Tavily
# ─────────────────────────────────────────────
def researcher_node(state: AgentState) -> dict:
    print("--- NODE: RESEARCHER (Live Tavily Search) ---")
    topic = state["target_topic"]

    # Real-time web search
    search_result = tavily.search(
        query=topic,
        search_depth="advanced",
        max_results=3
    )

    # Format results for the next node
    context = "\n".join([
        f"- {res['title']}: {res['content']}"
        for res in search_result['results']
    ])
    return {"research_data": context}


# ─────────────────────────────────────────────
# NODE 2: RAG (Pinecone Memory)
# Checks if we covered this topic before
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
        
        # Convert topic to a vector (numbers Pinecone understands)
        embedding_response = client.embeddings.create(
            input=topic,
            model="text-embedding-3-large",
            dimensions=1024
        )
        topic_vector = embedding_response.data[0].embedding
        
        # Search Pinecone for similar past reports
        results = index.query(
            vector=topic_vector,
            top_k=3,
            include_metadata=True
        )
        
        # If we found past reports, extract them
        if results["matches"] and len(results["matches"]) > 0:
            past_insights = "\n".join([
                f"- {match['metadata']['summary']}"
                for match in results["matches"]
                if match["score"] > 0.7  # Only use highly relevant matches
            ])
            context = past_insights if past_insights else "No highly relevant past reports found."
        else:
            context = "No previous reports found in Pinecone memory yet."
            
    except Exception as e:
        print(f"Pinecone error: {e}")
        context = "Historical memory (Pinecone) not yet initialized."
    
    print(f"📚 Past context found: {context[:100]}...")
    return {"past_context": context}


# ─────────────────────────────────────────────
# NODE 3: ANALYST
# GPT-4o analyzes and ranks the research (improved prompt with structured output and clearer instructions)
# ─────────────────────────────────────────────
def analyst_node(state: AgentState) -> dict:
    print("--- NODE: ANALYST (AI Reasoning) ---")
    prompt = f"""
    You are a senior AI industry analyst writing for C-suite executives (CEO/CIO/CTO/CISO/COO).
    Your job is to turn noisy research into a decision-ready assessment.

    TOPIC: {state['target_topic']}

    RESEARCH DATA (may include duplicates, hype, or partial claims):
    {state['research_data']}

    ANALYST TASK
    1) Extract ONLY the 3-6 most material facts (product release, capability change, pricing, partnerships, regulation, security events, benchmarks).
    2) De-duplicate and resolve contradictions. If uncertain, explicitly mark uncertainty.
    3) Translate technical changes into business impact with time horizon and decision triggers.

    OUTPUT FORMAT - follow EXACTLY:

    ## Technical Impact Score: [0-100]/100
    **Score meaning:** 0=no meaningful change, 100=paradigm shift likely to alter competitive dynamics.
    **Rationale (2 sentences):** Include (a) what changed and (b) why it is material.

    ## Business Materiality Score: [0-5]
    Rate how likely this affects revenue/cost/risk within 90 days.
    **Rationale (1 sentence).**

    ## What Changed (Evidence-Backed)
    - **Fact 1:** [1 sentence] **Evidence:** [source type: press release / doc / benchmark / report] **Confidence:** [High/Med/Low]
    - **Fact 2:** ...
    - **Fact 3:** ...

    ## Top 2 Business Implications (Decision-Relevant)
    1. **[Implication Title]:** [2-3 sentences: who is affected (function), what KPI shifts (cost/time/risk/revenue), and expected time horizon: Now / 30-90d / 6-12m]
    2. **[Implication Title]:** [same]

    ## Key Risks (Practical, Not Theoretical)
    - **Risk:** [1 sentence] **Likelihood:** [H/M/L] **Severity:** [H/M/L] **Mitigation lever:** [1 phrase]
    - **Risk:** ...

    ## Decision Triggers (If/Then)
    - If [observable signal], then [recommended decision/action].
    - If [observable signal], then [recommended decision/action].

    RULES
    - Be specific and falsifiable; avoid vague claims ("game changer", "significant").
    - Prefer numbers (price, latency, context length, eval results, adoption metrics) when present; otherwise give bounded estimates and mark them as estimates.
    - Do not repeat the research verbatim; synthesize.
    """
    response = llm.invoke(prompt)
    return {"analysis": response.content}


# ─────────────────────────────────────────────
# NODE 4: WRITER
# GPT-4o compiles the final executive briefing (improved prompt with clearer structure and style rules)
# ─────────────────────────────────────────────
def writer_node(state: AgentState) -> dict:
    print("--- NODE: WRITER (AI Composition) ---")
    prompt = f"""
    You are a professional business intelligence writer producing a Weekly Executive Briefing
    for C-suite technology leaders. Write crisp, decision-oriented prose.

    TOPIC: {state['target_topic']}

    INPUTS
    Research (raw): {state['research_data']}
    Analysis (structured): {state['analysis']}
    Historical context (last weeks): {state['past_context']}

    GOAL
    Deliver an executive briefing that enables a decision in <3 minutes:
    - what changed, why it matters, what we should do this week, what to watch next.

    BRIEFING STRUCTURE - include these exact sections and keep total under 600 words:

    # Weekly AI Intelligence Briefing

    ## 📌 Executive Summary
    Write 3-4 sentences:
    1) The single most important change
    2) Why it matters (business impact + time horizon)
    3) The one action to take this week
    4) The key risk/unknown to monitor

    ## 🔍 Key Developments
    3-5 bullets. Each bullet must include:
    - a concrete detail (metric, capability, pricing, release name, partnership, policy)
    - a short "So what" clause

    ## 📊 Impact Analysis
    Include:
    - **Technical Impact Score** and **Business Materiality Score**
    - The **Top 2 Business Implications** rewritten for non-technical leaders
    - A 1-sentence "who should care" mapping (e.g., Product / Security / Legal / Finance)

    ## ⚠️ Risks & Considerations
    2-3 bullets with:
    - likelihood/severity (H/M/L)
    - one mitigation lever per risk

    ## ✅ Recommended Actions (This Week)
    Exactly 3 actions.
    Each action must be written as:
    - **Action:** [verb + outcome]
    - **Owner:** [role]
    - **Timebox:** [7 days / 30 days]
    - **Success metric:** [measurable KPI]
    - **Trigger:** [what would change the plan]

    ## 🧠 Historical Context
    2-3 sentences: compare to prior weeks using {state['past_context']} (trend, escalation, or reversal).

    STYLE RULES
    - Write for non-technical executives: plain language, no jargon without explanation.
    - Prioritize decision utility over completeness.
    - Do not overclaim. If uncertain, say what would confirm it.
    - No filler, no hype, no generic advice.
    """
    response = llm.invoke(prompt)
    return {"final_report": response.content}


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
    workflow.add_node("reviewer",   reviewer_node)

    # Define flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "rag")
    workflow.add_edge("rag",        "analyst")
    workflow.add_edge("analyst",    "writer")
    workflow.add_edge("writer",     "reviewer")
    workflow.add_edge("reviewer",   END)

    return workflow.compile()

