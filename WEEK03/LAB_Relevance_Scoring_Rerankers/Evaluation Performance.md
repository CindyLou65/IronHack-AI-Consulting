Evaluation Performance

### Baseline Vector Retrieval + RAG
# Observations
Top retrieved chunks included:
Article 21 (Cooperation with competent authorities)
Cybersecurity obligations
Registration requirements (EU database)
Provider responsibility clauses
Some broader contextual recital material
# The generated answer included:
Compliance with Section 2
Quality management system (Article 17)
Cooperation with authorities
Registration requirements
Confidentiality obligations
Provider responsibility when modifying systems

# Strengths
✔ Good overall coverage
✔ Mentions registration requirement
✔ Captures multiple types of obligations

# Weaknesses
✖ Slightly broader and less focused
✖ Includes some contextual provisions not strictly core obligations


2️⃣ LLM Relevance Scoring (Step 3)
# Method
Retrieved 30 candidates
Scored each using LLM (0–10 relevance scale)
Combined:
 - 65% LLM relevance
 - 35% vector similarity
Re-ranked accordingly
# Observations
Promoted cybersecurity obligations and Article 21
Reduced ranking of recital-style content
Improved semantic prioritization of explicit duties

# Strengths
✔ More obligation-focused ranking
✔ Strong semantic relevance
✔ Tunable weighting between similarity and relevance

# Weaknesses
✖ More expensive (LLM call per chunk)
✖ Prompt-dependent
✖ Slower and less scalable

3️⃣ Cohere Rerank (Step 4)
# Method
Retrieved 30 candidates (EU-only)
Applied Cohere rerank-v3.5
Selected top 10
# Observations
Top ranked passages emphasized:
Cybersecurity obligations
Article 21 (authority cooperation)
Provider obligations under Article 16
Authority communication procedures
Manufacturer considered provider
# Generated answer:
Focused strongly on explicit provider duties
More concise and structured
Did not explicitly include registration requirement

# Strengths
✔ Clean and stable ranking
✔ Strong focus on explicit legal duties
✔ Faster and scalable
✔ No prompt engineering required

# Weaknesses
✖ Slightly reduced coverage (missed some registration details)
✖ Less tunable than LLM-based scoring

Direct Comparison
Criterion	Baseline	LLM Scoring	Cohere Rerank
Precision	Medium	High	High
Coverage	High	High	Medium-High
Stability	Medium	Medium	High
Cost	Low	High	Medium
Scalability	High	Low	High
Winner by Category

Precision:
Cohere ≈ LLM Scoring > Baseline

Coverage:
Baseline ≥ LLM > Cohere

Overall (Legal RAG system context):
👉 Cohere Rerank provides the best tradeoff between precision, scalability, and cost.

Final Recommendation
For legal-tech RAG systems querying structured regulatory documents:
Apply metadata filtering first
Retrieve a larger candidate set (k = 30–50)
Apply a dedicated reranker (Cohere)
Generate answer using reranked context
LLM-based relevance scoring is useful for:
Experimental setups
Domain-specific tuning
Research environments
However, for production environments:
Dedicated rerankers such as Cohere provide strong precision improvements while maintaining scalability and operational efficiency.
