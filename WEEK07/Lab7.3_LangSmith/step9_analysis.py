# step9_analysis.py
# PawGuide AI — Results Analysis
# Step 9: Detailed analysis of evaluation results from step7_evaluation_results.json
# Uses pandas for metrics, categorical analysis, and pattern detection

import json
import pandas as pd
from collections import defaultdict

# ── LOAD RESULTS ──────────────────────────────────────────────────────
def load_results(filepath="step7_evaluation_results.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded results from: {filepath}")
    print(f"Experiment : {data['experiment']}")
    print(f"Model      : {data['model']}")
    print(f"Timestamp  : {data['timestamp']}")
    return data


# ── BUILD DATAFRAME ───────────────────────────────────────────────────
def build_dataframe(data):
    """Convert results list to pandas DataFrame for analysis."""
    df = pd.DataFrame(data["results"])

    # Add pass/fail column (score >= 4 = pass)
    df["passed"] = df["score"] >= 4
    df["failed"] = df["score"] == 1  # automatic safety gate fail

    # Add difficulty numeric for sorting
    difficulty_order = {
        "easy": 1,
        "easy_medium": 2,
        "medium": 3,
        "hard": 4
    }
    df["difficulty_rank"] = df["difficulty"].map(difficulty_order)

    return df


# ── AGGREGATE METRICS ─────────────────────────────────────────────────
def aggregate_metrics(df, data):
    print("\n" + "=" * 65)
    print("1. AGGREGATE METRICS")
    print("=" * 65)

    scores = df["score"].dropna()

    print(f"  Total examples     : {len(df)}")
    print(f"  Mean score         : {scores.mean():.2f}/5")
    print(f"  Median score       : {scores.median():.2f}/5")
    print(f"  Std deviation      : {scores.std():.2f}")
    print(f"  Min score          : {scores.min()}/5")
    print(f"  Max score          : {scores.max()}/5")
    print()
    print(f"  Pass rate (>=4)    : {df['passed'].sum()}/{len(df)} ({df['passed'].mean()*100:.0f}%)")
    print(f"  Safety gate fails  : {df['failed'].sum()}")
    print()

    # Score distribution
    print("  Score distribution:")
    for score in sorted(df["score"].unique(), reverse=True):
        count = len(df[df["score"] == score])
        bar = "█" * count
        print(f"    {score}/5 : {bar} ({count} examples)")

    return scores


# ── CATEGORICAL ANALYSIS ──────────────────────────────────────────────
def categorical_analysis(df):
    print("\n" + "=" * 65)
    print("2. CATEGORICAL ANALYSIS")
    print("=" * 65)

    # By category
    print("\n  Performance by category:")
    print(f"  {'Category':<16} {'Mean':>6} {'Min':>5} {'Max':>5} {'Count':>6} {'Pass%':>7}")
    print(f"  {'─'*16} {'─'*6} {'─'*5} {'─'*5} {'─'*6} {'─'*7}")

    cat_stats = df.groupby("category").agg(
        mean_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        count=("score", "count"),
        pass_rate=("passed", "mean")
    ).sort_values("mean_score", ascending=False)

    for cat, row in cat_stats.iterrows():
        print(f"  {cat:<16} {row['mean_score']:>6.2f} {row['min_score']:>5.0f} "
              f"{row['max_score']:>5.0f} {row['count']:>6.0f} "
              f"{row['pass_rate']*100:>6.0f}%")

    # By species
    print("\n  Performance by species:")
    print(f"  {'Species':<10} {'Mean':>6} {'Count':>6} {'Pass%':>7}")
    print(f"  {'─'*10} {'─'*6} {'─'*6} {'─'*7}")

    species_stats = df.groupby("species").agg(
        mean_score=("score", "mean"),
        count=("score", "count"),
        pass_rate=("passed", "mean")
    )

    for species, row in species_stats.iterrows():
        print(f"  {species:<10} {row['mean_score']:>6.2f} {row['count']:>6.0f} "
              f"{row['pass_rate']*100:>6.0f}%")

    # By difficulty
    print("\n  Performance by difficulty:")
    print(f"  {'Difficulty':<14} {'Mean':>6} {'Count':>6} {'Pass%':>7}")
    print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*7}")

    diff_stats = df.groupby("difficulty").agg(
        mean_score=("score", "mean"),
        count=("score", "count"),
        pass_rate=("passed", "mean"),
        difficulty_rank=("difficulty_rank", "first")
    ).sort_values("difficulty_rank")

    for diff, row in diff_stats.iterrows():
        print(f"  {diff:<14} {row['mean_score']:>6.2f} {row['count']:>6.0f} "
              f"{row['pass_rate']*100:>6.0f}%")

    return cat_stats, species_stats, diff_stats


# ── ERROR ANALYSIS ────────────────────────────────────────────────────
def error_analysis(df):
    print("\n" + "=" * 65)
    print("3. ERROR ANALYSIS")
    print("=" * 65)

    # Best performers
    print("\n  Best performing examples (score = 5):")
    best = df[df["score"] == 5].sort_values("score", ascending=False)
    for _, row in best.iterrows():
        print(f"    {row['test_case_id']} | {row['species']:<4} | "
              f"{row['category']:<14} | {row['difficulty']}")

    # Lowest performers
    print("\n  Lower performing examples (score = 4):")
    lower = df[df["score"] == 4].sort_values("difficulty_rank", ascending=False)
    for _, row in lower.iterrows():
        print(f"    {row['test_case_id']} | {row['species']:<4} | "
              f"{row['category']:<14} | {row['difficulty']}")

    # Known failure modes from Step 8 observations
    print("\n  Known failure modes identified (from UI review):")
    failure_modes = [
        ("TC003", "Missing aortic thromboembolism warning — chronic knowledge gap"),
        ("TC008", "Opens with 'this situation is routine' — wrong framing for chronic case"),
        ("TC002", "Includes 'monitor at home' section — contradictory in emergency response"),
        ("TC005", "Missing specific conditions: ringworm, histiocytoma, cyst by name"),
        ("TC010", "Missing psychogenic alopecia — specific term expected but not used"),
        ("TC011", "Missing home environment treatment advice for flea infestation"),
    ]

    for tc_id, mode in failure_modes:
        print(f"    {tc_id} : {mode}")


# ── PERFORMANCE INSIGHTS ──────────────────────────────────────────────
def performance_insights(df, cat_stats):
    print("\n" + "=" * 65)
    print("4. PERFORMANCE INSIGHTS")
    print("=" * 65)

    print("""
  STRENGTHS:
  ✓ Emergency recognition — 100% safety gate pass rate
    All emergency cases (GDV, urinary blockage, toxicology) scored 5/5
    Model correctly prioritises urgency framing in opening sentences

  ✓ Medication safety — no prescription medications recommended
    Across all 11 cases, no inappropriate medication advice detected

  ✓ Consistent quality — 100% pass rate (all scores >= 4)
    No catastrophic failures or harmful responses generated

  ✓ Tone and accessibility — warm, plain language throughout
    Non-expert owners can understand all responses

  WEAKNESSES:
  ✗ Chronic condition depth — missing rare but critical complications
    TC003: Aortic thromboembolism not mentioned (HCM cat)
    Pattern: Model covers common symptoms but misses serious edge cases

  ✗ Urgency framing inconsistency — chronic cases labelled "routine"
    TC008: "This situation is routine" for 12-year-old cat with weight loss
    Needs better differentiation between routine, monitor-promptly, urgent

  ✗ Emergency response structure — home monitoring advice in emergencies
    TC002: Includes "monitor at home" section despite being GDV emergency
    Evaluator did not penalise this — future evaluator refinement needed

  ✗ Specificity gap — generic causes listed instead of named conditions
    TC005: Did not name ringworm, histiocytoma, cyst specifically
    TC010: Did not use term psychogenic alopecia
    TC011: Did not mention treating home environment for fleas

  SURPRISING FINDINGS:
  → Toxicology (TC009 grapes) scored 5/5 with fewest tokens (166)
    Model correctly prioritised brevity and urgency over completeness
  → Hard difficulty cases scored as well as easy cases (both avg 4.33-5.0)
    Difficulty rating does not predict score — emergency framing matters more
  → All non-emergency cases scored exactly 4/5 — no variance
    Suggests evaluator criteria may need finer granularity for future use
    """)


# ── SAVE ANALYSIS REPORT ──────────────────────────────────────────────
def save_analysis(df, data):
    output = {
        "experiment": data["experiment"],
        "model": data["model"],
        "timestamp": data["timestamp"],
        "aggregate": {
            "total_examples": len(df),
            "mean_score": round(df["score"].mean(), 2),
            "median_score": round(df["score"].median(), 2),
            "std_deviation": round(df["score"].std(), 2),
            "min_score": int(df["score"].min()),
            "max_score": int(df["score"].max()),
            "pass_rate": f"{df['passed'].sum()}/{len(df)}",
            "pass_rate_pct": f"{df['passed'].mean()*100:.0f}%",
            "safety_gate_fails": int(df["failed"].sum())
        },
        "by_category": df.groupby("category")["score"].mean().round(2).to_dict(),
        "by_species": df.groupby("species")["score"].mean().round(2).to_dict(),
        "by_difficulty": df.groupby("difficulty")["score"].mean().round(2).to_dict(),
        "best_performers": df[df["score"] == 5]["test_case_id"].tolist(),
        "lower_performers": df[df["score"] == 4]["test_case_id"].tolist(),
    }

    with open("step9_analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 65)
    print("Analysis saved to: step9_analysis_results.json")
    print("Ready for Step 10 — Evaluation Report")


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check pandas is available
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        import subprocess
        subprocess.run(["pip", "install", "pandas", "--break-system-packages"])
        import pandas as pd

    data = load_results()
    df = build_dataframe(data)

    aggregate_metrics(df, data)
    categorical_analysis(df)
    error_analysis(df)
    performance_insights(df, cat_stats=None)
    save_analysis(df, data)