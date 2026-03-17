# MEMORANDUM

**TO:** Chief Marketing Officer  
**FROM:** [Your Name]  
**DATE:** March 2026  
**RE:** Marketing Channel ROI Analysis — Budget Allocation Recommendation  
**DATASET:** Multi-Brand Marketing Campaign Performance Dataset (Kaggle)  
**PERIOD:** July 2024 – June 2025 (358 days)

---

## Executive Summary

Analysis of 166,665 campaigns across 5 marketing channels (Email, Influencer, 
Paid Ads, SEO, Social Media) for three beauty brands (Nykaa, Tira, Purplle) 
reveals no statistically meaningful performance differences between channels. 
Despite rigorous statistical testing, all channels perform at essentially 
equivalent levels. Budget allocation recommendations are based on marginal 
metric advantages only and should be treated with appropriate caution.

---

## Key Findings

### 1. Channel Performance
All 5 channels perform within a narrow range across all metrics:

| Channel | CPA ($) | ROAS | Conv Rate |
|---|---|---|---|
| Paid Ads | 2.02 | 6,487 | 21.95% |
| Social Media | 2.05 | 6,532 | 21.87% |
| SEO | 2.09 | 6,378 | 22.00% |
| Email | 2.11 | 6,479 | 22.00% |
| Influencer | 2.10 | 6,319 | 21.91% |

### 2. Statistical Test Results

**t-tests (CPA, ROAS, Conversion Rate):**
- 0 out of 30 pairwise comparisons significant after FDR correction
- All Cohen's d values < 0.02 — negligible effect sizes
- Conclusion: No meaningful CPA or ROAS differences between channels

**Fisher's Exact Test (Conversion Rate):**
- 9 out of 10 pairs statistically significant (p<0.0001)
- However all odds ratios between 0.997–1.006 — essentially no difference
- Significance is a statistical artifact of extremely large sample size (150M+ clicks)
- Conclusion: Not practically meaningful for budget decisions

### 3. Data Adequacy Assessment
Power analysis confirms our dataset is more than sufficient:

| Effect Size | Min Days Needed | Days Available | Status |
|---|---|---|---|
| 5% CPA difference | 180 days | 358 days | ✅ Sufficient |
| 10% CPA difference | 60 days | 358 days | ✅ Sufficient |
| 15% CPA difference | 30 days | 358 days | ✅ Sufficient |
| 20% CPA difference | 30 days | 358 days | ✅ Sufficient |

**Critical insight:** We had sufficient statistical power to detect any 
meaningful difference — and found none. This is a strong null result, 
not a consequence of insufficient data.

---

## Recommendations

### Budget Allocation ($500K Monthly)

Based on composite scoring (40% CPA, 40% ROAS, 20% Conversion Rate):

| Channel | Allocation | Budget ($) | Rationale |
|---|---|---|---|
| Paid Ads | 30.0% | $150,192 | Lowest CPA, 2nd highest ROAS |
| Social Media | 24.7% | $123,408 | 2nd lowest CPA, highest ROAS |
| SEO | 17.0% | $84,843 | Balanced performance |
| Email | 16.0% | $79,852 | Highest conversion rate |
| Influencer | 12.3% | $61,704 | Highest CPA, lowest ROAS |
| **TOTAL** | **100%** | **$500,000** | |

### Strategic Actions

1. **Do not make dramatic budget shifts** — channel differences are negligible
2. **Investigate Paid Ads and Social Media** further — marginal CPA advantage 
   warrants deeper creative and audience analysis
3. **Monitor Email channel** — highest conversion rate suggests strong 
   audience engagement worth nurturing
4. **Review Influencer ROI** — highest CPA and lowest ROAS suggests 
   reviewing influencer selection and contract terms
5. **Run controlled A/B tests** — with specific creative variations per 
   channel to identify what drives performance differences

---

## Statistical Caveats

### Dataset Limitations
- Dataset is synthetic (generated) — patterns may not reflect real market dynamics
- Source: Kaggle (sshriya08) — not proprietary first-party data
- Revenue figures appear unrealistically large relative to costs (ROAS ~6,400x)
- Absolute dollar values should not be used for financial planning

### Multiple Comparisons
- 40 total statistical tests performed (30 t-tests + 10 Fisher's exact)
- Both Bonferroni and Benjamini-Hochberg FDR corrections applied
- Results consistent across both correction methods

### Confidence Intervals
All 95% confidence intervals for CPA overlap substantially:
- Paid Ads: [$1.91, $2.14]
- Social Media: [$1.95, $2.16]
- SEO: [$1.99, $2.19]
- Email: [$2.00, $2.23]
- Influencer: [$2.00, $2.22]

Overlapping intervals confirm no statistically distinguishable differences.

### Statistical vs. Practical Significance
Fisher's exact test found 9/10 pairs statistically significant — but with 
150M+ observations, even a 0.05% difference becomes detectable. Odds ratios 
of 1.003–1.006 have no practical meaning for budget allocation.

### Power Analysis Limitations
- Simulations assume normally distributed CPA data
- Real CPA distributions may be skewed or have heavier tails
- Power estimates are approximations based on 1,000 simulations per condition

### External Factors Not Captured
- Seasonality effects across the 358-day period
- Brand-specific audience differences (Nykaa vs Tira vs Purplle)
- Creative quality variations across campaigns
- Platform algorithm changes during the study period

---

## Next Steps

1. **Collect real first-party data** — replace synthetic dataset with 
   actual campaign performance data
2. **Run channel-specific A/B tests** — isolate creative, audience, 
   and bidding variables
3. **Analyze by brand** — Nykaa, Tira and Purplle may show different 
   channel preferences
4. **Analyze by audience segment** — Premium Shoppers vs College Students 
   may respond differently to channels
5. **Re-analyze quarterly** — monitor for emerging channel differences 
   as market conditions evolve
6. **Expand metrics** — include Customer Lifetime Value (CLV) and 
   attribution data for deeper insights

---

*This analysis was conducted using rigorous statistical methods including 
independent t-tests, Fisher's exact test, Bonferroni and Benjamini-Hochberg 
FDR corrections, bootstrap confidence intervals, and empirical power analysis. 
All findings should be interpreted in the context of the synthetic dataset limitations noted above.*