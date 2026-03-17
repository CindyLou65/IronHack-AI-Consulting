# ============================================================
# PawGuide AI — Step 10: Metrics Collection & Analysis
# Reads evaluation_results_raw.json from Step 9
# Calculates aggregate statistics and saves final report
# ============================================================

import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# ============================================================
# COST CONSTANTS
# gpt-4o-mini pricing as of March 2026
# Input:  $0.150 per 1M tokens
# Output: $0.600 per 1M tokens
# ============================================================
COST_PER_1M_INPUT_TOKENS  = 0.150   # USD
COST_PER_1M_OUTPUT_TOKENS = 0.600   # USD

# Approximate token counts per call (since gpt-4o-mini
# doesn't always return usage in LangChain — we estimate)
# These are conservative estimates based on prompt lengths
ESTIMATED_TOKENS = {
    "production_input":  350,   # System prompt + user prompt
    "production_output": 250,   # Typical PawGuide response
    "judge_input":      1200,   # System + evaluation prompt + RAG + response
    "judge_output":      600,   # Structured JSON output
}


def estimate_cost_usd(n_production_calls: int, n_judge_calls: int) -> dict:
    """
    Estimate total API cost based on number of calls.
    
    Args:
        n_production_calls: Number of PawGuide response generations
        n_judge_calls: Number of judge evaluations
    
    Returns:
        Dictionary with cost breakdown
    """
    prod_input_tokens  = n_production_calls * ESTIMATED_TOKENS["production_input"]
    prod_output_tokens = n_production_calls * ESTIMATED_TOKENS["production_output"]
    judge_input_tokens  = n_judge_calls * ESTIMATED_TOKENS["judge_input"]
    judge_output_tokens = n_judge_calls * ESTIMATED_TOKENS["judge_output"]

    total_input_tokens  = prod_input_tokens  + judge_input_tokens
    total_output_tokens = prod_output_tokens + judge_output_tokens

    input_cost  = (total_input_tokens  / 1_000_000) * COST_PER_1M_INPUT_TOKENS
    output_cost = (total_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
    total_cost  = input_cost + output_cost

    return {
        "estimated_total_input_tokens":  total_input_tokens,
        "estimated_total_output_tokens": total_output_tokens,
        "estimated_input_cost_usd":      round(input_cost,  6),
        "estimated_output_cost_usd":     round(output_cost, 6),
        "estimated_total_cost_usd":      round(total_cost,  6),
        "estimated_total_cost_eur":      round(total_cost * 0.92, 6),
        "cost_per_test_case_usd":        round(total_cost / max(n_production_calls, 1), 6),
        "projected_cost_per_1000_queries_usd": round(
            (total_cost / max(n_production_calls, 1)) * 1000, 4
        )
    }


def load_raw_results(filepath: str = "evaluation_results_raw.json") -> list:
    """Load raw results from Step 9."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find {filepath}. "
            "Please run step9_test_dataset.py first."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(results: list) -> dict:
    """
    Calculate all metrics from raw evaluation results.
    
    Args:
        results: List of result dictionaries from Step 9
    
    Returns:
        Complete metrics dictionary
    """

    # --- Individual metrics per test case ---
    individual_metrics = []

    for result in results:
        judge = result.get("judge_evaluation", {})
        criteria = judge.get("criteria_met", {})

        # Count criteria met (excluding null values)
        criteria_values = [
            v for v in criteria.values()
            if v is not None
        ]
        criteria_met_count   = sum(1 for v in criteria_values if v is True)
        criteria_total_count = len(criteria_values)
        criteria_pct = round(
            (criteria_met_count / criteria_total_count * 100)
            if criteria_total_count > 0 else 0, 1
        )

        # RAG faithfulness summary
        rag_details  = judge.get("rag_faithfulness_details", {})
        supported    = len(rag_details.get("claims_supported_by_rag", []))
        missing      = len(rag_details.get("claims_missing_from_response", []))
        contradicted = len(rag_details.get("claims_contradicting_rag", []))

        individual_metrics.append({
            "test_case_id":              result["test_case_id"],
            "title":                     result["title"],
            "difficulty":                result["difficulty"],
            "score":                     judge.get("score"),
            "safety_gate_passed":        judge.get("safety_gate_passed"),
            "automatic_fail_triggered":  judge.get("automatic_fail_triggered"),
            "generation_time_seconds":   result.get("generation_time_seconds"),
            "judge_time_seconds":        judge.get("judge_time_seconds"),
            "total_time_seconds":        round(
                (result.get("generation_time_seconds") or 0) +
                (judge.get("judge_time_seconds") or 0), 2
            ),
            "criteria_met_count":        criteria_met_count,
            "criteria_total_count":      criteria_total_count,
            "criteria_met_percentage":   criteria_pct,
            "rag_claims_supported":      supported,
            "rag_claims_missing":        missing,
            "rag_claims_contradicted":   contradicted,
            "key_strength":              judge.get("key_strength"),
            "key_improvement":           judge.get("key_improvement"),
            "reasoning_summary": {
                "safety_gate":        judge.get("reasoning", {}).get("safety_gate"),
                "rag_faithfulness":   judge.get("reasoning", {}).get("rag_faithfulness"),
                "communication":      judge.get("reasoning", {}).get("communication_quality"),
                "constraints":        judge.get("reasoning", {}).get("constraint_compliance"),
            }
        })

    # --- Aggregate statistics ---
    scores      = [m["score"] for m in individual_metrics if m["score"] is not None]
    gen_times   = [m["generation_time_seconds"] for m in individual_metrics
                   if m["generation_time_seconds"] is not None]
    judge_times = [m["judge_time_seconds"] for m in individual_metrics
                   if m["judge_time_seconds"] is not None]
    total_times = [m["total_time_seconds"] for m in individual_metrics
                   if m["total_time_seconds"] is not None]

    safety_gates_passed = sum(
        1 for m in individual_metrics if m["safety_gate_passed"] is True
    )
    auto_fails = sum(
        1 for m in individual_metrics if m["automatic_fail_triggered"] is True
    )

    # Score distribution
    score_distribution = {str(i): scores.count(i) for i in range(1, 6)}

    # Criteria performance across all test cases
    all_criteria = {}
    for result in results:
        criteria = result.get("judge_evaluation", {}).get("criteria_met", {})
        for criterion, value in criteria.items():
            if value is not None:
                if criterion not in all_criteria:
                    all_criteria[criterion] = {"met": 0, "not_met": 0, "total": 0}
                all_criteria[criterion]["total"] += 1
                if value:
                    all_criteria[criterion]["met"] += 1
                else:
                    all_criteria[criterion]["not_met"] += 1

    criteria_performance = {
        k: {
            "met":       v["met"],
            "not_met":   v["not_met"],
            "total":     v["total"],
            "pass_rate": round(v["met"] / v["total"] * 100, 1) if v["total"] > 0 else 0
        }
        for k, v in all_criteria.items()
    }

    # Cost estimation
    n_cases = len(results)
    cost_metrics = estimate_cost_usd(n_cases, n_cases)

    aggregate = {
        "total_test_cases":          n_cases,
        "scores": {
            "average":               round(sum(scores) / len(scores), 2) if scores else 0,
            "minimum":               min(scores) if scores else 0,
            "maximum":               max(scores) if scores else 0,
            "distribution":          score_distribution,
        },
        "safety": {
            "safety_gates_passed":   safety_gates_passed,
            "safety_gates_failed":   n_cases - safety_gates_passed,
            "safety_gate_pass_rate": round(safety_gates_passed / n_cases * 100, 1),
            "automatic_fails":       auto_fails,
        },
        "timing": {
            "total_generation_time_seconds": round(sum(gen_times), 2),
            "total_judge_time_seconds":      round(sum(judge_times), 2),
            "total_pipeline_time_seconds":   round(sum(total_times), 2),
            "average_generation_time_seconds": round(
                sum(gen_times) / len(gen_times), 2) if gen_times else 0,
            "average_judge_time_seconds":    round(
                sum(judge_times) / len(judge_times), 2) if judge_times else 0,
            "average_total_time_per_case_seconds": round(
                sum(total_times) / len(total_times), 2) if total_times else 0,
        },
        "cost_estimation":           cost_metrics,
        "criteria_performance":      criteria_performance,
    }

    return {
        "individual_metrics": individual_metrics,
        "aggregate":          aggregate
    }


def print_metrics_report(metrics: dict) -> None:
    """Print a formatted metrics report to the terminal."""

    agg = metrics["aggregate"]
    ind = metrics["individual_metrics"]

    print("\n" + "=" * 65)
    print("📊 PAWGUIDE AI — EVALUATION METRICS REPORT")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # --- Scores ---
    print("\n📈 SCORES")
    print(f"   Average Score:  {agg['scores']['average']}/5")
    print(f"   Minimum Score:  {agg['scores']['minimum']}/5")
    print(f"   Maximum Score:  {agg['scores']['maximum']}/5")
    print(f"   Distribution:   ", end="")
    for score, count in agg["scores"]["distribution"].items():
        icons = {
            "1": "🔴", "2": "🟠", "3": "🟡", "4": "🟢", "5": "✅"
        }
        print(f"{icons[score]}{score}:{count}  ", end="")
    print()

    # --- Individual Results ---
    print("\n📋 INDIVIDUAL RESULTS")
    print(f"   {'ID':<8} {'Title':<40} {'Score':<8} {'Safety':<10} {'Time(s)'}")
    print(f"   {'-'*8} {'-'*40} {'-'*8} {'-'*10} {'-'*8}")
    for m in ind:
        safety = "✅ Pass" if m["safety_gate_passed"] else "❌ FAIL"
        title  = m["title"][:38] + ".." if len(m["title"]) > 40 else m["title"]
        print(f"   {m['test_case_id']:<8} {title:<40} "
              f"{m['score']}/5{'':<5} {safety:<10} {m['total_time_seconds']}")

    # --- Safety ---
    print(f"\n🛡️  SAFETY METRICS")
    print(f"   Safety Gate Pass Rate:  "
          f"{agg['safety']['safety_gate_pass_rate']}% "
          f"({agg['safety']['safety_gates_passed']}/{agg['total_test_cases']})")
    print(f"   Automatic Fails:        {agg['safety']['automatic_fails']}")

    # --- Timing ---
    print(f"\n⏱️  TIMING METRICS")
    print(f"   Total Pipeline Time:         "
          f"{agg['timing']['total_pipeline_time_seconds']}s")
    print(f"   Avg Generation Time/Case:    "
          f"{agg['timing']['average_generation_time_seconds']}s")
    print(f"   Avg Judge Time/Case:         "
          f"{agg['timing']['average_judge_time_seconds']}s")
    print(f"   Avg Total Time/Case:         "
          f"{agg['timing']['average_total_time_per_case_seconds']}s")

    # --- Cost ---
    cost = agg["cost_estimation"]
    print(f"\n💶 COST ESTIMATION (gpt-4o-mini)")
    print(f"   Estimated Total Cost:        "
          f"${cost['estimated_total_cost_usd']} USD "
          f"(~€{cost['estimated_total_cost_eur']})")
    print(f"   Cost Per Test Case:          "
          f"${cost['cost_per_test_case_usd']} USD")
    print(f"   Projected per 1,000 queries: "
          f"${cost['projected_cost_per_1000_queries_usd']} USD")
    print(f"   Est. Input Tokens:           "
          f"{cost['estimated_total_input_tokens']:,}")
    print(f"   Est. Output Tokens:          "
          f"{cost['estimated_total_output_tokens']:,}")

    # --- Criteria Performance ---
    print(f"\n✅ CRITERIA PASS RATES (across all test cases)")
    criteria = agg["criteria_performance"]
    for criterion, data in sorted(
        criteria.items(), key=lambda x: x[1]["pass_rate"]
    ):
        bar_filled = int(data["pass_rate"] / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        name = criterion.replace("_", " ").title()[:35]
        print(f"   {name:<35} [{bar}] {data['pass_rate']}% "
              f"({data['met']}/{data['total']})")

    # --- Key Findings ---
    print(f"\n🔍 KEY FINDINGS PER TEST CASE")
    for m in ind:
        score_icons = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "✅"}
        icon = score_icons.get(m["score"], "⚪")
        print(f"\n   {icon} {m['test_case_id']} — {m['title']}")
        print(f"      Score: {m['score']}/5 | "
              f"Criteria: {m['criteria_met_count']}/{m['criteria_total_count']} "
              f"({m['criteria_met_percentage']}%)")
        if m["key_strength"]:
            print(f"      💪 {m['key_strength']}")
        if m["key_improvement"]:
            print(f"      ⚠️  {m['key_improvement']}")

    print("\n" + "=" * 65)


def save_metrics(metrics: dict,
                 filepath: str = "evaluation_results.json") -> None:
    """Save complete metrics to JSON file."""
    output = {
        "report_generated":  datetime.now().isoformat(),
        "model_used":        "gpt-4o-mini",
        "evaluation_scope":  "PawGuide AI V1 — Dogs & Cats, German Market",
        "veterinary_advisor": "Dr. Lund",
        "metrics":           metrics
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Complete metrics saved to: {filepath}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("🐾 PawGuide AI — Step 10: Metrics Collection & Analysis")
    print(f"   Loading results from Step 9...\n")

    # Load raw results from Step 9
    raw_results = load_raw_results("evaluation_results_raw.json")
    print(f"   ✅ Loaded {len(raw_results)} test case results")

    # Calculate all metrics
    print(f"   ⏳ Calculating metrics...")
    metrics = calculate_metrics(raw_results)
    print(f"   ✅ Metrics calculated")

    # Print formatted report
    print_metrics_report(metrics)

    # Save complete metrics
    save_metrics(metrics, "evaluation_results.json")

    print("\n✅ Step 10 complete.")
    print("   → evaluation_results.json saved")
    print("   → Proceed to Step 11 for visualization")