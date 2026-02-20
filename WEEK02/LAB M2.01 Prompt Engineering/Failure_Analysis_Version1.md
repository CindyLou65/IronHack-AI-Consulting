# Failure Analysis – Version 1 Prompts (15-Run Test)

Generated: 2026-02-20 11:51

## Summary Metrics

| Task | Runs | Unique Outputs | Consistency (%) |
|---|---:|---:|---:|
| Sentiment Analysis | 15 | 2 | 93.33 |
| Product Description | 15 | 15 | 6.67 |
| Data Extraction | 15 | 6 | 66.67 |

---

## 1) Sentiment Analysis (v1)

**Most common outputs:**  
- (8x) `The sentiment of the customer message is positive.`
- (7x) `The sentiment of the message is positive.`

**Observed failure patterns:**
- Output is often a full sentence instead of a single-word label (format not controlled).
- Zero-shot prompt does not enforce strict output constraints, making parsing fragile.

---

## 2) Product Description Generation (v1)

**Most common outputs (previews):**  
- (1x) `### AeroBrew Travel Mug: Your Perfect Companion for On-the-Go Living  Elevate your daily commute or travel adventures with the **AeroBrew Tr...`
- (1x) `### AeroBrew Travel Mug  **Price: $29.99**  Elevate your on-the-go experience with the **AeroBrew Travel Mug**—your perfect companion for ev...`
- (1x) `### AeroBrew Travel Mug  **Price:** $29.99  Elevate your on-the-go beverage experience with the **AeroBrew Travel Mug**—the ultimate compani...`

**Observed failure patterns:**
- High variability in structure (headings/bold/markdown), length, and tone across runs.
- Creative additions beyond the provided product info can appear.
- No structure or length constraints -> unpredictable output.

---

## 3) Data Extraction (v1)

**Most common outputs (previews):**  
- (5x) ````json {   "order_id": "A12345",   "product_name": "AeroBrew Travel Mug",   "issues": "The lid started leaking after two days.",   "positiv...`
- (4x) ````json {   "order_id": "A12345",   "product_name": "AeroBrew Travel Mug",   "issues": "The lid started leaking after two days.",   "positiv...`
- (3x) ````json {   "order_id": "A12345",   "product_name": "AeroBrew Travel Mug",   "issues": "lid started leaking after two days",   "positive_not...`

**Detected extraction-specific patterns (counts out of 15 runs):**
- Code block wrapped: 15
- Extra explanation text: 1
- Issues field as list: 0

**Observed failure patterns:**
- JSON is sometimes wrapped in markdown code fences and/or preceded by explanatory text.
- Field type inconsistency can appear (e.g., issues as string vs list).
- Not reliably machine-parseable without post-processing.

---

## Overall Diagnosis

- Classification is content-stable but format-loose without strict constraints.
- Generation shows extreme variability without structure and length rules.
- Extraction is structurally unreliable (format drift, inconsistent JSON), unsafe for production parsing.

## Recommended Next Improvements

- Add explicit output constraints (single label; fixed schemas).
- Add structured formatting requirements (JSON only; no markdown fences).
- Add few-shot examples to lock in format.
- Use more advanced prompting techniques in later iterations.
