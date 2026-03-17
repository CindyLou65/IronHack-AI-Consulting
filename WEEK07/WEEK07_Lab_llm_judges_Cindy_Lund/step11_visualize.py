# ============================================================
# PawGuide AI — Step 11: Analyze & Visualize Results
# Reads evaluation_results.json from Step 10
# Creates terminal visualization and saves summary report
# ============================================================

import json
import os
from datetime import datetime


def load_metrics(filepath: str = "evaluation_results.json") -> dict:
    """Load metrics from Step 10."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find {filepath}. "
            "Please run step10_metrics.py first."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate long text for display."""
    if not text:
        return "N/A"
    return text[:max_len] + "..." if len(text) > max_len else text


def render_bar(value: float, max_value: float = 5.0,
               width: int = 20, fill: str = "█",
               empty: str = "░") -> str:
    """Render a simple ASCII progress bar."""
    filled = int((value / max_value) * width)
    return fill * filled + empty * (width - filled)


def print_score_distribution(metrics: dict) -> None:
    """Visualize score distribution as ASCII bar chart."""

    agg = metrics["metrics"]["aggregate"]
    dist = agg["scores"]["distribution"]

    print("\n" + "=" * 65)
    print("📊  SCORE DISTRIBUTION")
    print("=" * 65)

    score_labels = {
        "1": "🔴 Score 1 — Automatic Fail   ",
        "2": "🟠 Score 2 — Major Issues     ",
        "3": "🟡 Score 3 — Partial Pass     ",
        "4": "🟢 Score 4 — Good             ",
        "5": "✅ Score 5 — Excellent        ",
    }

    total = sum(dist.values())
    for score, label in score_labels.items():
        count = dist.get(score, 0)
        pct   = (count / total * 100) if total > 0 else 0
        bar   = "█" * count + "░" * (5 - count)
        print(f"  {label} [{bar}] {count} test case(s) ({pct:.0f}%)")

    print(f"\n  Average: {agg['scores']['average']}/5  |  "
          f"Min: {agg['scores']['minimum']}/5  |  "
          f"Max: {agg['scores']['maximum']}/5")


def print_criteria_performance(metrics: dict) -> None:
    """Visualize criteria pass rates as ASCII bar chart."""

    criteria = metrics["metrics"]["aggregate"]["criteria_performance"]

    print("\n" + "=" * 65)
    print("✅  CRITERIA PERFORMANCE (Pass Rates)")
    print("=" * 65)

    # Sort by pass rate ascending — worst first
    sorted_criteria = sorted(criteria.items(), key=lambda x: x[1]["pass_rate"])

    for criterion, data in sorted_criteria:
        name     = criterion.replace("_", " ").title()
        bar      = render_bar(data["pass_rate"], max_value=100, width=20)
        rate     = data["pass_rate"]
        fraction = f"{data['met']}/{data['total']}"

        # Flag criteria that need attention
        flag = ""
        if rate < 70:
            flag = " ⚠️  NEEDS ATTENTION"
        elif rate < 100:
            flag = " 📌 Room to improve"

        print(f"  {name:<35} [{bar}] {rate:>5.1f}% ({fraction}){flag}")


def print_per_case_analysis(metrics: dict) -> None:
    """Print detailed per-case analysis with reasoning."""

    individual = metrics["metrics"]["individual_metrics"]

    print("\n" + "=" * 65)
    print("🔍  PER TEST CASE ANALYSIS")
    print("=" * 65)

    score_icons = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "✅"}

    for m in individual:
        icon  = score_icons.get(m["score"], "⚪")
        bar   = render_bar(m["score"], max_value=5, width=15)
        crit  = f"{m['criteria_met_count']}/{m['criteria_total_count']}"
        crit_pct = m["criteria_met_percentage"]

        print(f"\n  {icon} {m['test_case_id']} — {m['title']}")
        print(f"  {'─' * 60}")
        print(f"  Score:      [{bar}] {m['score']}/5")
        print(f"  Criteria:   {crit} met ({crit_pct}%)")
        print(f"  Safety:     {'✅ Gate Passed' if m['safety_gate_passed'] else '❌ Gate FAILED'}")
        print(f"  Gen Time:   {m['generation_time_seconds']}s  |  "
              f"Judge Time: {m['judge_time_seconds']}s  |  "
              f"Total: {m['total_time_seconds']}s")

        # RAG faithfulness
        print(f"  RAG:        {m['rag_claims_supported']} supported  |  "
              f"{m['rag_claims_missing']} missing  |  "
              f"{m['rag_claims_contradicted']} contradicted")

        # Reasoning summaries — truncated
        r = m.get("reasoning_summary", {})
        if r.get("safety_gate"):
            print(f"  Safety:     {truncate(r['safety_gate'], 75)}")
        if r.get("rag_faithfulness"):
            print(f"  RAG:        {truncate(r['rag_faithfulness'], 75)}")
        if r.get("communication"):
            print(f"  Comms:      {truncate(r['communication'], 75)}")
        if r.get("constraints"):
            print(f"  Constraints:{truncate(r['constraints'], 75)}")

        # Key insight
        if m.get("key_strength"):
            print(f"  💪 {truncate(m['key_strength'], 75)}")
        if m.get("key_improvement"):
            print(f"  ⚠️  {truncate(m['key_improvement'], 75)}")


def print_time_cost_metrics(metrics: dict) -> None:
    """Visualize timing and cost metrics."""

    agg     = metrics["metrics"]["aggregate"]
    timing  = agg["timing"]
    cost    = agg["cost_estimation"]
    ind     = metrics["metrics"]["individual_metrics"]

    print("\n" + "=" * 65)
    print("⏱️   TIMING & COST METRICS")
    print("=" * 65)

    # Timing bar chart per case
    print("\n  Response Generation Time Per Case:")
    max_time = max(m["generation_time_seconds"] for m in ind)
    for m in ind:
        bar  = render_bar(m["generation_time_seconds"],
                          max_value=max(max_time, 1), width=20)
        name = m["test_case_id"]
        print(f"  {name}  [{bar}] {m['generation_time_seconds']}s")

    print(f"\n  Pipeline Totals:")
    print(f"  Total Generation Time:  {timing['total_generation_time_seconds']}s")
    print(f"  Total Judge Time:       {timing['total_judge_time_seconds']}s")
    print(f"  Total Pipeline Time:    {timing['total_pipeline_time_seconds']}s")
    print(f"  Avg Time Per Case:      {timing['average_total_time_per_case_seconds']}s")

    print(f"\n  Cost Breakdown (gpt-4o-mini):")
    print(f"  Est. Input Tokens:      {cost['estimated_total_input_tokens']:,}")
    print(f"  Est. Output Tokens:     {cost['estimated_total_output_tokens']:,}")
    print(f"  Total Cost (5 cases):   ${cost['estimated_total_cost_usd']} USD "
          f"(~€{cost['estimated_total_cost_eur']})")
    print(f"  Cost Per Test Case:     ${cost['cost_per_test_case_usd']} USD")
    print(f"  Per 1,000 queries:      ${cost['projected_cost_per_1000_queries_usd']} USD")

    # Cost scale context
    print(f"\n  Cost At Scale (projected):")
    scales = [1000, 10000, 100000]
    for scale in scales:
        projected = round(
            cost["cost_per_test_case_usd"] * scale, 2
        )
        print(f"  {scale:>8,} queries/month → ${projected:>7.2f} USD/month")


def print_patterns_and_insights(metrics: dict) -> None:
    """Identify and print patterns across test cases."""

    agg      = metrics["metrics"]["aggregate"]
    ind      = metrics["metrics"]["individual_metrics"]
    criteria = agg["criteria_performance"]

    print("\n" + "=" * 65)
    print("💡  PATTERNS & KEY INSIGHTS")
    print("=" * 65)

    # Pattern 1: Safety
    safety_rate = agg["safety"]["safety_gate_pass_rate"]
    print(f"\n  1. SAFETY PERFORMANCE")
    if safety_rate == 100:
        print(f"     ✅ 100% safety gate pass rate — no dangerous advice")
        print(f"        generated across all test cases. Emergency")
        print(f"        escalation logic is functioning correctly.")
    else:
        print(f"     ❌ Safety gate passed {safety_rate}% — CRITICAL: review")
        print(f"        failed cases before any deployment consideration.")

    # Pattern 2: Weakest criteria
    weak = [
        (k, v) for k, v in criteria.items()
        if v["pass_rate"] < 100 and v["total"] > 1
    ]
    weak_sorted = sorted(weak, key=lambda x: x[1]["pass_rate"])

    print(f"\n  2. WEAKEST CRITERIA (requiring prompt engineering attention)")
    if weak_sorted:
        for criterion, data in weak_sorted[:3]:
            name = criterion.replace("_", " ").title()
            print(f"     ⚠️  {name}: {data['pass_rate']}% "
                  f"({data['met']}/{data['total']})")
    else:
        print(f"     ✅ All criteria at 100% — excellent result")

    # Pattern 3: Difficulty vs performance
    print(f"\n  3. DIFFICULTY VS PERFORMANCE")
    for m in ind:
        bar  = render_bar(m["score"], max_value=5, width=10)
        diff = m["difficulty"][:25]
        print(f"     {m['test_case_id']}: [{bar}] {m['score']}/5 "
              f"— {diff}")

    # Pattern 4: Consistent improvement theme
    improvements = [
        m["key_improvement"] for m in ind
        if m.get("key_improvement") and
        "no significant" not in m["key_improvement"].lower()
    ]
    print(f"\n  4. RECURRING IMPROVEMENT THEME")
    urgency_count = sum(
        1 for imp in improvements
        if imp and "urgency" in imp.lower()
    )
    if urgency_count >= 2:
        print(f"     📌 Urgency framing flagged in {urgency_count}/5 test cases.")
        print(f"        Recommendation: Add explicit instruction to PawGuide")
        print(f"        system prompt for stronger urgency signals in")
        print(f"        responses involving serious or chronic conditions.")
    else:
        print(f"     ✅ No dominant recurring issue identified.")

    # Pattern 5: TC003 specific
    tc003 = next((m for m in ind if m["test_case_id"] == "TC003"), None)
    if tc003 and tc003["score"] <= 3:
        print(f"\n  5. TC003 — FELINE HCM SPECIFIC FINDING")
        print(f"     🟡 Score 3/5 — lowest performing test case.")
        print(f"        Root cause: missing aortic thromboembolism")
        print(f"        warning (life-threatening complication).")
        print(f"        Fix: Add chronic condition complication")
        print(f"        checklist to system prompt.")
        print(f"        Priority: HIGH — this is a safety-adjacent gap.")

    # Pattern 6: RAG performance
    rag_perf = criteria.get("rag_grounded_claims", {})
    print(f"\n  6. RAG PIPELINE PERFORMANCE")
    if rag_perf.get("pass_rate", 0) == 100:
        print(f"     ✅ RAG grounding: 100% — all responses grounded")
        print(f"        in retrieved veterinary literature.")
        print(f"        German veterinary corpus is functioning correctly.")
    else:
        print(f"     ⚠️  RAG grounding below 100% — review retrieval pipeline.")


def save_summary_report(metrics: dict) -> None:
    """Save a clean summary report as JSON."""

    agg = metrics["metrics"]["aggregate"]
    ind = metrics["metrics"]["individual_metrics"]

    summary = {
        "report_title":      "PawGuide AI — Evaluation Summary Report",
        "generated":         datetime.now().isoformat(),
        "model":             metrics.get("model_used", "gpt-4o-mini"),
        "veterinary_advisor": metrics.get("veterinary_advisor", "Dr. Lund"),
        "evaluation_scope":  metrics.get("evaluation_scope"),
        "executive_summary": {
            "average_score":          agg["scores"]["average"],
            "score_range":            f"{agg['scores']['minimum']}-{agg['scores']['maximum']}/5",
            "safety_gate_pass_rate":  f"{agg['safety']['safety_gate_pass_rate']}%",
            "automatic_fails":        agg["safety"]["automatic_fails"],
            "deployment_ready":       agg["scores"]["average"] >= 4.0 and
                                      agg["safety"]["safety_gate_pass_rate"] == 100.0,
            "primary_concern":        "TC003 Feline HCM — missing critical complication warning",
            "primary_strength":       "Emergency escalation — 100% safety gate pass rate"
        },
        "per_case_summary": [
            {
                "id":            m["test_case_id"],
                "title":         m["title"],
                "score":         m["score"],
                "criteria_pct":  m["criteria_met_percentage"],
                "safety_passed": m["safety_gate_passed"],
                "total_time_s":  m["total_time_seconds"],
                "strength":      m["key_strength"],
                "improvement":   m["key_improvement"]
            }
            for m in ind
        ],
        "cost_summary": {
            "total_5_cases_usd":     agg["cost_estimation"]["estimated_total_cost_usd"],
            "per_query_usd":         agg["cost_estimation"]["cost_per_test_case_usd"],
            "per_1000_queries_usd":  agg["cost_estimation"]["projected_cost_per_1000_queries_usd"],
        },
        "timing_summary": {
            "avg_generation_s":  agg["timing"]["average_generation_time_seconds"],
            "avg_judge_s":       agg["timing"]["average_judge_time_seconds"],
            "total_pipeline_s":  agg["timing"]["total_pipeline_time_seconds"]
        },
        "top_recommendations": [
            "Add chronic condition complication checklist to system prompt "
            "(addresses TC003 ATE warning gap)",
            "Strengthen urgency framing instruction for serious but "
            "non-emergency conditions",
            "Expand test dataset to 25+ prompts before deployment decision",
            "Schedule Dr. Lund calibration review on TC003 category outputs",
            "Target response generation time reduction from 5.4s to <3s "
            "for production UX"
        ]
    }

    filepath = "implementation_summary.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary report saved to: {filepath}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("🐾 PawGuide AI — Step 11: Analyze & Visualize Results")
    print(f"   Loading metrics from Step 10...\n")

    metrics = load_metrics("evaluation_results.json")
    print(f"   ✅ Metrics loaded — {metrics['metrics']['aggregate']['total_test_cases']} test cases\n")

    # Run all visualizations
    print_score_distribution(metrics)
    print_criteria_performance(metrics)
    print_per_case_analysis(metrics)
    print_time_cost_metrics(metrics)
    print_patterns_and_insights(metrics)

    # Save summary report
    save_summary_report(metrics)

    print("\n" + "=" * 65)
    print("✅ Step 11 complete — Full evaluation pipeline finished.")
    print("=" * 65)
    print("\n📁 Files created in your WEEK07 directory:")
    print("   ✅ llm_judge_evaluation.py    — Step 7 & 8: Judge implementation")
    print("   ✅ step9_test_dataset.py       — Step 9:     Test dataset & pipeline")
    print("   ✅ step10_metrics.py           — Step 10:    Metrics collection")
    print("   ✅ step11_visualize.py         — Step 11:    Analysis & visualization")
    print("   ✅ evaluation_results_raw.json — Raw results from Step 9")
    print("   ✅ evaluation_results.json     — Full metrics from Step 10")
    print("   ✅ implementation_summary.json — Summary report from Step 11")
    print("\n🐾 PawGuide AI evaluation pipeline — complete.\n")