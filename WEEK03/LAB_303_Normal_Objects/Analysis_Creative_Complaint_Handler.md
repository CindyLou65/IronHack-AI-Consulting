#Lab Week3 03 Normal Objects
Cindy Lund

# NormalObjects Lab 1 — Agent Behavior Analysis

Self Reflection and Challenges: for this lab I had alot of difficulties and retrying very many approaches
as the version of LangChain I used no longer included the older AgentExecutor and create_openai_tools_agent APIs, which led me through several import attempts and workarounds. Ultimately, the solution was to switch to LangChain v1’s create_agent function, which allowed me to build a flexible agent with tool support and achieve the desired behavior.

## What I built
I built a freeform LangChain agent with four custom tools (Demogorgon perspective, Hawkins records lookup, interdimensional spell suggestions, and D&D party wisdom). The agent can choose and chain tools in any order to respond to open-ended complaints.

## When the agent used tools creatively
The agent used tools as “sources” that provide different styles of reasoning:
- Hawkins records: pseudo-factual grounding (patterns, hotspots, EM activity)
- Demogorgon: surreal/chaotic interpretation that reframes assumptions
- Party wisdom: collaborative, practical hints (what details to gather, pattern spotting)
- Spells: imaginative actions that feel like “solutions” in the fictional universe

Tool chaining often looked like: records → demogorgon → (optional) party/spell → final response.

## Tool usage patterns and chaining
I tracked tool usage via a callback handler. The counts and call sequence show which tools the agent prefers and how it chains them. This helps identify:
- tool preference (overused tools vs unused tools)
- repetition (same tool called multiple times)
- common chains (e.g., records then demogorgon)

## Comparison with structured approaches (LangGraph)
Freeform agent (this lab):
- Flexible ordering: can chain any tools in any order
- Less predictable: may repeat tools, skip tools, or vary behavior between runs
- Harder to enforce rules (e.g., “always consult X before Y”)

Structured workflow (LangGraph):
- Enforced steps and guardrails (must-do checks, allowed transitions)
- More consistent outputs and easier debugging
- Better for production systems needing reliability and compliance

## Recommendations: when to use each approach
Use a freeform agent when:
- tasks are exploratory, creative, or ambiguous
- you want emergent tool use and flexible reasoning
- prototyping and ideation matter more than consistency

Use a structured workflow when:
- steps must be followed in a strict order
- you need predictable behavior, safety, or compliance guarantees
- you want robust debugging and reproducibility