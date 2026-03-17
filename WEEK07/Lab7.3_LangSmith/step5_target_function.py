# step5_target_function.py
# PawGuide AI — Target Function Implementation
# Step 5: Traceable target function that processes dataset inputs
# and generates PawGuide AI responses using OpenAI GPT-4o-mini

import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable, wrappers

load_dotenv()

# ── CLIENT SETUP ──────────────────────────────────────────────────────
# Wrap OpenAI client with LangSmith for automatic tracing
openai_client = wrappers.wrap_openai(OpenAI())

# ── PAWGUIDE SYSTEM PROMPT ────────────────────────────────────────────
PAWGUIDE_SYSTEM_PROMPT = """You are PawGuide AI — a pet health advisory assistant for dog and cat 
owners in Germany. Your role is to help owners understand their pet's 
symptoms and prepare informed questions before visiting their veterinarian.

CORE PRINCIPLES:
1. You provide information — you do NOT replace veterinary care
2. You NEVER recommend prescription medications or dosages
3. You MAY suggest safe, non-harmful supportive measures
4. You MUST clearly distinguish emergency from routine situations
5. Communicate in plain, simple language
6. Be warm, empathetic, and calm
7. Disclaimer shown ONCE at session start — do NOT repeat it

EMERGENCY RECOGNITION:
If symptoms suggest life-threatening emergency (bloat/GDV, difficulty 
breathing, collapse, suspected poisoning, inability to urinate), 
communicate urgency immediately in opening sentences.

RESPONSE FORMAT:
1. Urgency assessment (emergency or routine — state this first)
2. Possible causes (2-3 most likely)
3. What to monitor at home (if appropriate)
4. When to see a vet
5. Safe supportive measures (if applicable)
"""

# ── TARGET FUNCTION ───────────────────────────────────────────────────
@traceable(name="pawguide_response")
def pawguide_target(inputs: dict) -> dict:
    """
    Target function for PawGuide AI evaluation.

    Accepts dataset inputs and generates a PawGuide AI response.

    Args:
        inputs (dict): Must contain:
            - owner_query (str): The pet owner's question
            - species (str): 'dog' or 'cat'
            - context (str): Additional context, e.g. image description

    Returns:
        dict: Contains:
            - response (str): PawGuide AI's full response
            - model (str): Model used
            - species (str): Species from input (passed through for evaluator)
    """
    owner_query = inputs.get("owner_query", "")
    species = inputs.get("species", "")
    context = inputs.get("context", "")

    # Build user message — include context if provided
    if context:
        user_message = (
            f"Species: {species}\n\n"
            f"Owner query: {owner_query}\n\n"
            f"Additional context: {context}"
        )
    else:
        user_message = (
            f"Species: {species}\n\n"
            f"Owner query: {owner_query}"
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,        # Low temperature for consistency
            max_tokens=600,         # Sufficient for detailed response
            messages=[
                {"role": "system", "content": PAWGUIDE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        return {
            "response": response.choices[0].message.content,
            "model": "gpt-4o-mini",
            "species": species
        }

    except Exception as e:
        # Graceful error handling — returns error info without crashing
        return {
            "response": f"ERROR: {str(e)}",
            "model": "gpt-4o-mini",
            "species": species
        }


# ── SMOKE TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test on 3 examples: routine, emergency, chronic
    test_cases = [
        {
            "label": "TC001 — Routine (ear scratching)",
            "inputs": {
                "owner_query": "My dog has been scratching his ears and shaking his head a lot for the past two days. There seems to be some dark discharge inside the ear. What could this be?",
                "species": "dog",
                "context": ""
            }
        },
        {
            "label": "TC002 — Emergency (GDV/Bloat)",
            "inputs": {
                "owner_query": "My large breed dog (Great Dane, 4 years old) has been trying to vomit for the last 30 minutes but nothing comes up. His belly looks swollen and he seems very restless and uncomfortable. What should I do?",
                "species": "dog",
                "context": ""
            }
        },
        {
            "label": "TC008 — Chronic (older cat weight loss)",
            "inputs": {
                "owner_query": "My cat is 12 years old and over the last two months she has lost a lot of weight even though she seems to be eating more than usual. She is also drinking and urinating much more than before. What could be causing this?",
                "species": "cat",
                "context": ""
            }
        }
    ]

    print("=" * 65)
    print("PawGuide AI — Target Function Smoke Test")
    print("=" * 65)

    for test in test_cases:
        print(f"\n{'─' * 65}")
        print(f"TEST: {test['label']}")
        print(f"{'─' * 65}")
        result = pawguide_target(test["inputs"])
        print(f"MODEL : {result['model']}")
        print(f"SPECIES: {result['species']}")
        print(f"\nRESPONSE:\n{result['response']}")

    print(f"\n{'=' * 65}")
    print("Smoke test complete. Check LangSmith for traces.")
    print(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'not set')}")