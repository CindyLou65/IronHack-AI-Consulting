# Reflection: Lab 7.2 Marketing Channel ROI Analysis

## What surprised you about the results?

I chose this dataset because of my strong interest in analyzing ad spend, 
conversions and revenue across competing brands. The dataset appealed to me 
because it captured the complete marketing funnel — Impressions → Clicks → 
Leads → Conversions → Revenue — across 5 distinct campaign channel types 
(Social Media, Paid Ads, Influencer, Email, SEO). Having 3 competitive beauty 
brands (Nykaa, Tira, Purplle) and 5 target audience segments made it feel 
close to a real-world consulting scenario, even though the data is synthetic.

The most surprising finding was that despite having 166,665 campaigns and 
150 million+ clicks across 5 channels, no meaningful performance differences 
existed between channels. Going into the analysis, I expected at least one 
channel to clearly outperform the others — particularly given the different 
nature of each channel type (paid vs organic vs influencer vs email). 
The complete funnel data I was excited to analyze ultimately showed 
remarkably consistent performance at every stage across all channels.

The Fisher's exact test result was especially striking — 9 out of 10 pairs 
were "statistically significant" with p=0.0000, yet the actual differences 
were less than 0.1% in conversion rate. This was a powerful demonstration 
of how massive sample sizes can make trivial differences look important.

## How did multiple comparisons correction change your conclusions?

For the t-tests, corrections made no difference — zero significant results 
remained zero after both Bonferroni and FDR correction.

For Fisher's exact test, corrections also made no difference — 9 significant 
results survived both methods. However this revealed an important lesson: 
multiple comparisons correction cannot fix the problem of an overpowered 
study. With 150M observations, the real issue was not false positives from 
running many tests — it was that the sample size was so large that 
statistically detectable differences had no practical meaning.

The key insight: corrections control for running too many tests, but they 
cannot tell you whether a statistically significant result is 
practically meaningful. That requires effect sizes and business judgment.

## What are the limitations of this analysis?

1. **Synthetic data** — the dataset was artificially generated, which explains 
   why all channels perform almost identically. Real marketing data would show 
   more meaningful variation between channels.

2. **Unrealistic financials** — ROAS of 6,400x and revenue figures in the 
   billions are not realistic for a beauty brand. Absolute dollar 
   recommendations should not be taken at face value.

3. **No attribution modeling** — the analysis treats each campaign 
   independently without accounting for cross-channel effects. A customer 
   might see a Social Media ad, click an Email link, and convert via SEO.

4. **No seasonality analysis** — the 358-day period likely includes seasonal 
   peaks (festivals, sales events) that could affect channel performance 
   differently.

5. **Aggregated brand data** — combining Nykaa, Tira and Purplle may mask 
   brand-specific channel preferences that would be visible in separate analyses.

6. **CPA as primary metric** — Acquisition_Cost in this dataset may not 
   capture full media spend, making CPA comparisons potentially incomplete.

## How would you communicate these findings to non-technical stakeholders?

I would avoid statistical terminology entirely and focus on the business 
implications:

**What I would say:**
> "We analyzed a full year of campaign data across all five marketing channels. 
> The good news is that all channels are performing consistently — none is 
> wasting money. The data does not support making dramatic budget shifts. 
> Paid Ads and Social Media show marginal cost efficiency advantages, so we 
> recommend modestly increasing their share. We should run focused A/B tests 
> to identify what specific creative and audience strategies drive the best 
> results within each channel."

**What I would NOT say:**
> "Fisher's exact test found 9 statistically significant pairwise differences 
> after Benjamini-Hochberg FDR correction, however Cohen's d values below 0.02 
> indicate negligible practical effect sizes..."

The goal is to give stakeholders confidence in the recommendation while being 
honest that the differences are small — without undermining trust in the 
analysis by overwhelming them with statistical caveats.

## Personal Takeaways

1. **Real data matters** — synthetic datasets teach methodology but real 
   first-party data is essential for actual business decisions. Next time 
   I would prioritize finding a real-world dataset with genuine channel 
   variation, even if smaller in size, to make the business recommendations 
   more meaningful and actionable.

2. **Statistical significance is not business significance** — the most 
   important lesson of this lab. A p-value tells you IF a difference exists, 
   not WHETHER it matters.

3. **Power analysis should come before data collection** — knowing you need 
   180 days to detect a 5% difference would change how you design a study.
   Next time I would run power analysis first to determine the minimum 
   sample size needed before choosing a dataset.

4. **Effect sizes are essential** — Cohen's d and odds ratios gave more 
   actionable information than p-values in this analysis.

5. **Null results are valuable** — finding no difference with sufficient 
   statistical power is a strong, reliable conclusion — not a failure.