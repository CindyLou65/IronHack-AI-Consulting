# main.py

from dotenv import load_dotenv
from agent.graph import build_graph

# Load API keys from .env
load_dotenv()

def main():
    print("\n>>> STARTING LIVE AGENT WORKFLOW <<<\n")

    # Build and compile the graph
    app = build_graph()

    # Kick off the agent with a real topic
    initial_input = {
        "target_topic": "Latest LLM releases and efficiency breakthroughs March 2026"
    }

    result = app.invoke(initial_input)

    print("\n" + "="*50)
    print("FINAL LIVE REPORT (MARKDOWN)")
    print("="*50 + "\n")
    print(result["final_report"])

if __name__ == "__main__":
    main()