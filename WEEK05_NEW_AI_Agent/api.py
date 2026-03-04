# api.py
# This file defines a FastAPI web service that exposes an endpoint to run the AI agent and can not trigger agent via HTTP requests.

import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from agent.graph import build_graph

# Load API keys
load_dotenv()

# Initialize FastAPI app and names it
app = FastAPI(
    title="AI Research & Briefing Agent",
    description="Runs a full AI intelligence cycle and returns an executive briefing",
    version="1.0.0"
)

# Define the REQUEST body - defines what the caller must send to trigger the agent
class AgentRequest(BaseModel):
    target_topic: str

# Define the RESPONSE body - defines what the API sends back after the agent finishes
class AgentResponse(BaseModel):
    topic:        str
    final_report: str
    is_approved:  bool

# ─────────────────────────────────────────────
# POST /run-agent
# Main Endpoint: Triggers the full LangGraph workflow - Main door: when someone calls POST /run-agent with a target topic, it builds and runs the 
# graph, then returns the final report and approval status.
# ─────────────────────────────────────────────
@app.post("/run-agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    print(f"\n>>> API CALL RECEIVED: {request.target_topic}\n")

    # Build and run the graph
    graph = build_graph()
    result = graph.invoke({
        "target_topic": request.target_topic
    })

    return AgentResponse(
        topic=        request.target_topic,
        final_report= result["final_report"],
        is_approved=  result["is_approved"]
    )

# ─────────────────────────────────────────────
# GET /health
# HEALTHCHECK:Quick check that the API is running
# async: The agent takes 30-60 seconds to run. async means FastAPI can handle other requests while waiting — it doesn't freeze.
# ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "online", "agent": "AI Briefing Agent v1.0"}