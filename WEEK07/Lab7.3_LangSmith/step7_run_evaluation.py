# step7_run_evaluation.py
# PawGuide AI — Run Full Evaluation Experiment
# Step 7: Execute evaluation against all 11 dataset examples using LangSmith

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client

# Import target function and evaluator from previous steps
from step5_target_function import pawguide_target
from step6_evaluator import pawguide_evaluator

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────────
DATASET_NAME = "pawguide-ai-evaluation-v1"
EXPERIMENT_PREFIX = "pawguide-gpt4o-mini"
MAX_CONCURRENCY = 2  # Conservative — avoids rate limit issues

# ── CLIENT SETUP ──────────────────────────────────────────────────────
client = Client()

# ── RUN EVALUATION ────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 65)
    print("PawGuide AI — Full Evaluation Experiment")
    print("=" * 65)
    print(f"Dataset    : {DATASET_NAME}")
    print(f"Experiment : {EXPERIMENT_PREFIX}")
    print(f"Model      : gpt-4o-mini")
    print(f"Concurrency: {MAX_CONCURRENCY}")
    print(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─' * 65}")
    print("Running evaluation — this will take 1-2 minutes...")
    print()

    results = client.evaluate(
        pawguide_target,
        data=DATASET_NAME,
        evaluators=[pawguide_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=MAX_CONCURRENCY,
        metadata={
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "dataset_version": "v1",
            "lab": "Lab7.3_LangSmith",
            "description": "PawGuide AI full evaluation — 11 examples"
        }
    )

    return results


# ── COLLECT AND DISPLAY RESULTS ───────────────────────────────────────
def collect_results(results):
    print(f"{'─' * 65}")
    print("Collecting results...")
    print()

    collected = []

    for result in results:
        # Extract run information
        run = result.get("run", {})
        example = result.get("example", {})
        evaluation_results = result.get("evaluation_results", {})

        # Get inputs and outputs
        inputs = run.inputs if hasattr(run, 'inputs') else {}
        outputs = run.outputs if hasattr(run, 'outputs') else {}

        # Get metadata from example
        metadata = example.metadata if hasattr(example, 'metadata') else {}
        tc_id = metadata.get("test_case_id", "N/A")
        category = metadata.get("category", "N/A")
        species = metadata.get("species", "N/A")
        difficulty = metadata.get("difficulty", "N/A")

        # Get evaluator score
        score = None
        reasoning = ""
        safety_gate = False

        eval_results = evaluation_results.get("results", [])
        for eval_result in eval_results:
            if hasattr(eval_result, 'score'):
                score = eval_result.score
            if hasattr(eval_result, 'comment'):
                reasoning = eval_result.comment or ""

        collected.append({
            "test_case_id": tc_id,
            "category": category,
            "species": species,
            "difficulty": difficulty,
            "score": score,
            "reasoning": reasoning,
        })

        # Print result
        score_display = f"{score}/5" if score is not None else "N/A"
        print(f"  {tc_id} | {species:<4} | {category:<14} | {difficulty:<12} | Score: {score_display}")

    return collected


# ── SAVE RESULTS ──────────────────────────────────────────────────────
def save_results(collected):
    output_file = "step7_evaluation_results.json"

    # Calculate summary stats
    scores = [r["score"] for r in collected if r["score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else 0
    pass_count = sum(1 for s in scores if s >= 4)
    fail_count = sum(1 for s in scores if s == 1)

    summary = {
        "experiment": EXPERIMENT_PREFIX,
        "dataset": DATASET_NAME,
        "model": "gpt-4o-mini",
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(collected),
        "average_score": round(avg_score, 2),
        "pass_rate": f"{pass_count}/{len(scores)}",
        "automatic_fails": fail_count,
        "results": collected
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"{'─' * 65}")
    print("SUMMARY")
    print(f"{'─' * 65}")
    print(f"  Total examples evaluated : {len(collected)}")
    print(f"  Average score            : {avg_score:.2f}/5")
    print(f"  Pass rate (score >= 4)   : {pass_count}/{len(scores)}")
    print(f"  Automatic fails (score 1): {fail_count}")
    print()
    print(f"Results saved to: {output_file}")
    print(f"View experiment in LangSmith:")
    print(f"  Project  : {os.getenv('LANGCHAIN_PROJECT', 'not set')}")
    print(f"  Experiment: {EXPERIMENT_PREFIX}")
    print(f"{'=' * 65}")
    print("Step 7 complete.")

    return summary


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_evaluation()
    collected = collect_results(results)
    save_results(collected)