# Benchmark Audit
## PawGuide AI — LLM Evaluation Lab
### Scenario: AI-Powered Pet Health Advisory App (Dogs & Cats, German Market)

---

## Client Scenario

PawGuide AI is a German-language pet health advisory application for dog and cat owners. The app helps owners understand their pet's symptoms and prepare informed questions before visiting their veterinarian. It is a pre-consultation informational tool — not a diagnostic replacement. The app must provide accurate, medically grounded information translated into plain language, analyze images of visible symptoms, apply appropriate urgency framing (distinguishing emergencies from routine situations), suggest safe supportive home care where appropriate, and never recommend prescription medications. A one-time disclaimer per chat session informs users that the tool does not replace veterinary care. The knowledge base is grounded in a RAG corpus of German veterinary curriculum literature, validated by Dr. Lund, a licensed German veterinarian.

---

## Benchmark Evaluation Card 1: MedQA (USMLE-style)

**Benchmark Name:** MedQA
**Year:** 2021 (ongoing)
**Source:** paperswithcode.com / Jin et al. 2021 — arxiv.org/abs/2009.13081

**Why it seemed relevant:**
PawGuide requires strong underlying medical knowledge. MedQA tests clinical reasoning through exam-style questions, providing a baseline signal for whether a model knows enough medicine to give informed guidance — even if adapted to veterinary rather than human medicine.

**Contamination risk:** High
MedQA questions are widely published and almost certainly present in the training data of any major frontier LLM. Scores therefore reflect memorization as much as reasoning.

**Saturation risk:** High
GPT-5 reached 95.84% accuracy on MedQA, making it essentially a solved benchmark for frontier models. It no longer meaningfully differentiates between models.

**Format:** Multiple Choice

**Verdict:** Reject
Too saturated and contaminated to be meaningful. More critically, it tests human medical knowledge in a multiple-choice format — PawGuide requires veterinary knowledge in open-ended conversation. The gap is too large. Useful only as a rough sanity check that the base model has general medical literacy.

---

## Benchmark Evaluation Card 2: HealthBench

**Benchmark Name:** HealthBench
**Year:** 2025
**Source:** OpenAI — arxiv.org/abs/2505.08775

**Why it seemed relevant:**
HealthBench is structurally closest to the PawGuide use case. It consists of 5,000 multi-turn conversations between a model and individual users or healthcare professionals, evaluated using conversation-specific rubrics created by 262 physicians, spanning behavioral dimensions such as accuracy, instruction following, communication quality, and emergency referral behavior. The multi-turn, open-ended format mirrors how a pet owner would actually interact with PawGuide.

**Contamination risk:** Medium
Published May 2025 — recent enough that older models will not have seen it, but newer models trained after mid-2025 may have partial exposure.

**Saturation risk:** Low
Overall scores range from 0.16 (GPT-3.5 Turbo) to 0.60 (o3), meaning even the best frontier models are far from perfect. Meaningful differentiation between models remains.

**Format:** Multi-turn free-form conversation with rubric scoring

**Verdict:** Adapt
The format and evaluation philosophy are exactly right for PawGuide. However, all conversations are human medical — none are veterinary. The rubric criteria and conversation themes would need to be adapted to pet health scenarios. The emergency referral theme is directly applicable and particularly valuable for testing PawGuide's R7 requirement (urgency framing).

---

## Benchmark Evaluation Card 3: Veterinary Undergraduate MCQ Benchmark

**Benchmark Name:** Veterinary Undergraduate Multiple-Choice Examination Benchmark
**Year:** 2025
**Source:** Frontiers in Veterinary Science — doi.org/10.3389/fvets.2025.1616566

**Why it seemed relevant:**
This is the only benchmark found that directly tests veterinary knowledge in LLMs. It evaluates nine advanced LLMs on 250 multiple-choice questions from a veterinary undergraduate final qualifying examination, spanning various species, clinical topics, reasoning stages, and both text-based and image-based formats. The inclusion of image-based questions is particularly relevant given PawGuide's R3 requirement (image analysis of visible symptoms).

**Contamination risk:** Low
Published in 2025 from a specific university examination — very unlikely to be present in the training data of any current model.

**Saturation risk:** Medium
ChatGPT o1Pro and ChatGPT 4.5 achieved correct response rates of 90.4% and 90.8% respectively, while performance consistently declined with increased question difficulty. Top models score well on easy questions but still struggle on harder ones — some meaningful differentiation between models remains.

**Format:** Multiple Choice + Image-based questions

**Verdict:** Adapt
Most directly relevant to the PawGuide domain of all four benchmarks audited. Main limitation is shared with MedQA — the multiple-choice format does not reflect the conversational, open-ended nature of the app. It also tests veterinary student knowledge breadth rather than the specific skills PawGuide requires: plain language explanation, safe escalation behavior, appropriate supportive home care suggestions, and image analysis with appropriate uncertainty framing. Use as a veterinary knowledge baseline and image analysis signal, not as a complete evaluation.

---

## Benchmark Evaluation Card 4: TruthfulQA

**Benchmark Name:** TruthfulQA
**Year:** 2021
**Source:** Lin, Hilton & Evans — arxiv.org/abs/2109.07958 / github.com/sylinrl/TruthfulQA

**Why it seemed relevant:**
PawGuide's single greatest danger is not ignorance but confident wrongness — a model stating incorrect medical information in fluent, reassuring language. TruthfulQA comprises 817 questions across 38 topics including health, law, finance, and politics, specifically crafted to provoke common misconceptions or untruthful responses. The health category directly maps to the C1 concern (hallucination). It also tests whether a model knows when to say "I don't know" — critical for PawGuide's escalation requirements.

**Contamination risk:** High
TruthfulQA has been publicly available since 2021 and is almost certainly present in the training data of any major frontier model. Recent research has also identified incorrect gold answers in the dataset, further reducing its reliability.

**Saturation risk:** High
At original publication, the best LLM scored only 58% compared to 94% for humans. That gap has largely closed for modern frontier models, reducing its ability to meaningfully differentiate between them.

**Format:** Multiple Choice (MC1 and MC2 modes) + open-ended generation

**Verdict:** Adapt with caution
Do not use TruthfulQA scores as a primary metric — saturation and contamination make raw scores unreliable. Use it in two specific ways: as a behavioral signal (run on the health-related question subset and inspect how the model handles misconceptions, not just whether it scores correctly), and as inspiration for custom evaluation design (the adversarial design philosophy of crafting questions to trigger plausible-sounding wrong answers is exactly the approach to apply to veterinary-specific prompts). Relying solely on TruthfulQA scores creates a false sense of readiness for domain-specific applications.

---

## Summary

| Benchmark | Relevance | Contamination | Saturation | Verdict |
|---|---|---|---|---|
| MedQA | Low — human medicine MCQ | High | High | ❌ Reject |
| HealthBench | High — multi-turn, rubric-based | Medium | Low | ✅ Adapt |
| Vet Undergrad MCQ | Medium — veterinary MCQ + images | Low | Medium | ✅ Adapt |
| TruthfulQA | Medium — hallucination signal | High | High | ⚠️ Adapt with caution |

**Note on scope:** Four benchmarks were audited. MedQA was rejected, leaving three active benchmarks. This is intentional — a rejected benchmark with documented reasoning is a stronger audit than forcing an unsuitable benchmark to fit the scenario.

**Key conclusion:** No existing benchmark fully covers the PawGuide scenario. HealthBench provides the right format philosophy and conversational structure. The Veterinary Undergraduate MCQ (Frontiers, 2025) provides domain knowledge relevance and image analysis signal. TruthfulQA provides a hallucination tendency signal as a general behavioral check. Together they offer three complementary but partial signals — none cover the critical requirements of plain language translation, safe escalation behavior, appropriate supportive home care suggestions, image analysis with uncertainty framing, or the one-disclaimer-per-chat rule. This gap directly justifies the custom evaluation suite designed in Steps 3-4.
