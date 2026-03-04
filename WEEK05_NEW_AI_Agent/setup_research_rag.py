# setup_research_rag.py
# Run once to populate Pinecone with foundational AI research papers
# Uses 2 chunks per paper: Technical + Business Relevance

import os
import arxiv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

FOUNDATIONAL_PAPERS = [
    {"id": "1706.03762", "topic": "transformer_architecture",  "name": "Attention Is All You Need"},
    {"id": "2001.08361", "topic": "scaling_laws",              "name": "Scaling Laws for Neural LMs"},
    {"id": "2203.15556", "topic": "scaling_laws",              "name": "Chinchilla Scaling Paper"},
    {"id": "2205.14135", "topic": "infrastructure",            "name": "FlashAttention"},
    {"id": "2309.06180", "topic": "infrastructure",            "name": "PagedAttention / vLLM"},
    {"id": "2101.03961", "topic": "model_architecture",        "name": "Switch Transformer (MoE)"},
    {"id": "2005.11401", "topic": "rag",                       "name": "Retrieval-Augmented Generation"},
    {"id": "2201.11903", "topic": "reasoning",                 "name": "Chain-of-Thought Prompting"},
    {"id": "2203.11171", "topic": "reasoning",                 "name": "Self-Consistency Reasoning"},
    {"id": "2103.00020", "topic": "multimodal",                "name": "CLIP"},
    {"id": "2204.14198", "topic": "multimodal",                "name": "Flamingo"},
    {"id": "2210.03629", "topic": "agent_reasoning",           "name": "ReAct"},
    {"id": "2302.04761", "topic": "agent_reasoning",           "name": "Toolformer"},
]

def extract_chunks_with_gpt(llm, paper_name, topic, abstract):
    prompt = f"""
    You are extracting structured information from an AI research paper abstract.
    Use ONLY information explicitly stated in the abstract.
    If a field is not clearly stated, write "not reported".
    Never invent or infer beyond what is written.

    PAPER: {paper_name}
    TOPIC: {topic}
    ABSTRACT: {abstract}

    Extract EXACTLY this structure:

    TECHNICAL CHUNK:
    Key Innovation: [1 sentence: what technical problem was solved]
    Method: [1 sentence: how it was solved]
    Benchmark Improvement: [specific metric if reported, or "not reported"]
    Limitation: [1 sentence if mentioned, or "not reported"]

    BUSINESS CHUNK:
    Industry Relevance: [1 sentence: why this matters for AI systems today]
    Adoption Signal: [1 sentence: evidence of real-world adoption, or "not reported"]
    Business Impact: [1 sentence: cost/speed/capability impact, or "not reported"]
    Related Technologies: [comma-separated list of technologies this enables]
    """
    response = llm.invoke(prompt)
    return response.content

def parse_chunks(raw_text, paper_name, topic, year):
    lines = raw_text.strip().split('\n')
    technical_lines = []
    business_lines = []
    current = None

    for line in lines:
        if 'TECHNICAL CHUNK:' in line:
            current = 'technical'
        elif 'BUSINESS CHUNK:' in line:
            current = 'business'
        elif current == 'technical' and line.strip():
            technical_lines.append(line.strip())
        elif current == 'business' and line.strip():
            business_lines.append(line.strip())

    technical_chunk = f"""FOUNDATIONAL AI RESEARCH — TECHNICAL REFERENCE
Paper: {paper_name}
Topic: {topic.replace('_', ' ').title()}
Year: {year}
Confidence: High (peer-reviewed)

{chr(10).join(technical_lines)}
"""

    business_chunk = f"""FOUNDATIONAL AI RESEARCH — BUSINESS RELEVANCE
Paper: {paper_name}
Topic: {topic.replace('_', ' ').title()}
Year: {year}
Confidence: High (peer-reviewed)

{chr(10).join(business_lines)}
"""
    return technical_chunk, business_chunk

def setup_research_rag():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("ai-agent-reports")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

# ← ADD CLEANUP HERE
    print("🧹 Cleaning up old chunks...")
    for paper in FOUNDATIONAL_PAPERS:
        try:
            index.delete(ids=[f"research_{paper['id']}"])
        except:
            pass
    print("✅ Cleanup complete\n")

    print(f"📚 Processing {len(FOUNDATIONAL_PAPERS)} foundational papers...")
    print(f"📊 Will create {len(FOUNDATIONAL_PAPERS) * 2} chunks in Pinecone\n")

    success_count = 0

    for paper in FOUNDATIONAL_PAPERS:
        try:
            print(f"⬇️  Fetching: {paper['name']}...")

            search = arxiv.Search(id_list=[paper["id"]])
            result = next(arxiv.Client().results(search))
            year = str(result.published.year)

            print(f"🤖 Extracting structured chunks with GPT-4o...")

            raw_extraction = extract_chunks_with_gpt(
                llm,
                paper["name"],
                paper["topic"],
                result.summary
            )

            technical_chunk, business_chunk = parse_chunks(
                raw_extraction,
                paper["name"],
                paper["topic"],
                year
            )

            tech_vector = embeddings.embed_query(technical_chunk)
            biz_vector = embeddings.embed_query(business_chunk)

            index.upsert(vectors=[
                {
                    "id": f"research_tech_{paper['id']}",
                    "values": tech_vector,
                    "metadata": {
                        "type": "research",
                        "chunk_type": "technical",
                        "topic": paper["topic"],
                        "name": paper["name"],
                        "year": year,
                        "confidence": "high",
                        "text": technical_chunk[:1000]
                    }
                },
                {
                    "id": f"research_biz_{paper['id']}",
                    "values": biz_vector,
                    "metadata": {
                        "type": "research",
                        "chunk_type": "business",
                        "topic": paper["topic"],
                        "name": paper["name"],
                        "year": year,
                        "confidence": "high",
                        "text": business_chunk[:1000]
                    }
                }
            ])

            success_count += 1
            print(f"✅ Uploaded 2 chunks: {paper['name']}\n")

        except Exception as e:
            print(f"❌ Failed: {paper['name']} — {e}\n")

    print("=" * 50)
    print(f"🎉 Research RAG setup complete!")
    print(f"📊 {success_count}/{len(FOUNDATIONAL_PAPERS)} papers processed")
    print(f"📦 {success_count * 2} chunks uploaded to Pinecone")
    print("=" * 50)

if __name__ == "__main__":
    setup_research_rag()