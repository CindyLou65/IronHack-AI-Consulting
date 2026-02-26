## LAB 4.1 Normal Objects - Strict Complaint Processor
Cindy Lund

## Step 6 — Comparison: Creative Agent (Lab 1) vs Structured Workflow (Lab 2)

# Control of Execution:
In Lab 1, the LangChain agent dynamically chose which tools to use and in what order. In Lab 2, LangGraph enforces a fixed workflow (intake → validate → investigate → resolve → close) with no step skipping.

# Predictability:
The creative agent could vary its reasoning and tool chaining between runs. The LangGraph workflow produces consistent, rule-based behavior for the same input.

# Tool Usage vs Rule Enforcement:
Lab 1 emphasized flexible tool chaining (e.g., records → demogorgon → spells). Lab 2 enforces business rules in code (e.g., no investigation without validation).

# Handling Missing Information:
The creative agent might attempt to answer even with incomplete data. The structured workflow explicitly blocks progress and requests clarification when required fields are missing.

# Traceability:
Lab 1 required callback tracking to analyze tool usage. Lab 2 inherently tracks execution history and logs through the workflow state.

# Best Use Cases:
The freeform agent is ideal for exploratory, creative, and ambiguous tasks. The structured LangGraph workflow is better suited for compliance-driven systems requiring reliability and auditability.



## Comparison: LangGraph vs LangChain Agent

1️⃣ Architecture & Control
LangGraph (Structured Workflow)
Deterministic state machine
Explicit nodes and conditional edges
Fixed step order: intake → validate → investigate → resolve → close
No step skipping possible
LangChain Agent (Creative Approach)
LLM-driven reasoning loop
Dynamic tool selection
Execution path decided by the model
Flexible but less controlled

2️⃣ Determinism & Reliability
LangGraph
Business rules enforced in Python
Same input → same workflow path
Easy to debug and test
LangChain Agent
Non-deterministic reasoning
Path may vary between runs
Harder to enforce strict rules

3️⃣ Auditability & Compliance
LangGraph
Full execution history stored in state
Timestamped logs
Clear stop reasons
Ideal for compliance-heavy systems
LangChain Agent
Reasoning hidden inside LLM
Harder to trace decisions
More prompt-dependent

4️⃣ When to Use Each
Use LangGraph when:
Process order must be guaranteed
Audit trail is required
Business rules must be enforced
Compliance matters
Use LangChain Agent when:
Creative reasoning is needed
Tasks are open-ended
Flexible tool use is required

5️⃣ Key Insight from This Lab
This project used a hybrid approach:
LLM for categorization
LangGraph for strict workflow enforcement
This reflects real-world best practice:
Use LLMs for interpretation
Use structured workflows for control