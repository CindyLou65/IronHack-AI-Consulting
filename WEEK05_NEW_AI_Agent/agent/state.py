# agent/state.py


from typing import TypedDict

class AgentState(TypedDict):
    target_topic:  str    # The topic we are researching
    research_data: str    # Raw news gathered by the Researcher node
    analysis:      str    # Filtered and ranked analysis by Analyst node
    past_context:  str    # Retrieved memory from Pinecone (RAG node)
    final_report:  str    # The finished report produced by the Writer node
    is_approved:   bool   # Reviewer approval flag