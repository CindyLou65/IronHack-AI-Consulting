# Evaluation Memo

---

**TO:** Dr. Marcus Weber, Managing Partner, Weber Venture Capital
**FROM:** [Your Name], Founder & Product Lead, PawGuide AI
**DATE:** March 16, 2026
**SUBJECT:** LLM Evaluation Results — PawGuide AI, V1 Pre-Launch Assessment (Dogs & Cats, German Market)

---

## Executive Summary

This memo presents the findings of a structured evaluation of the large language model pipeline underlying PawGuide AI — a German-language pet health advisory app designed to help dog and cat owners prepare for veterinary consultations. Evaluation was conducted across five custom-designed test scenarios covering medical accuracy, emergency escalation behavior, plain language communication, safe home remedy suggestions, and image-based symptom analysis. Under the conditions tested, the model demonstrated strong performance on routine advisory tasks and acceptable performance on communication quality, but revealed meaningful gaps in plain language completeness for chronic conditions that must be addressed before V1 deployment.

---

## Methodology

Three existing benchmarks were audited prior to designing the custom evaluation. HealthBench (OpenAI, 2025) was identified as the most structurally relevant existing benchmark due to its multi-turn conversational format and rubric-based scoring across clinical accuracy, communication quality, and emergency referral behavior. A veterinary undergraduate MCQ benchmark (Frontiers in Veterinary Science, 2025) was used as a domain knowledge baseline, covering both text and image-based questions across veterinary specialties. TruthfulQA (Lin et al., 2021) was included as a hallucination tendency signal, though its high saturation and contamination risk limit standalone utility. All three benchmarks were assessed as insufficient for this specific use case — none cover the combination of veterinary domain knowledge, German-language delivery, conversational format, and pre-consultation advisory philosophy that PawGuide requires. This gap justified the design of a five-prompt custom evaluation suite.

The custom evaluation was designed in collaboration with Dr. Lund, a licensed German veterinarian who served as medical ground truth authority and periodic output reviewer. Five evaluation prompts were designed to cover all eight product requirements and seven identified concern areas, ranging from easy baseline queries to hard safety-critical scenarios. Each prompt was evaluated using a three-layer verification approach: rule-based automated checks, LLM-as-judge scoring for response quality and RAG faithfulness, and periodic human expert review. The LLM-as-judge was designed with a hard safety gate for emergency prompts — any response failing to communicate urgency in the opening two sentences receives an automatic score of 1 regardless of other qualities. The model evaluated was gpt-4o-mini running over a RAG corpus of German veterinary curriculum materials provided and validated by Dr. Lund.

The evaluation pipeline was implemented in Python using LangChain. A two-stage architecture was used: Stage 1 generates PawGuide responses using a production LLM instance at temperature 0.3 (simulating realistic production variation), and Stage 2 evaluates those responses using a judge LLM at temperature 0 (deterministic, consistent scoring). All results were saved to structured JSON files for analysis.

---

## Results

Performance varied meaningfully across the five evaluation scenarios, yielding an average score of 4.2 out of 5. On Prompt 1 (basic symptom interpretation — ear scratching, Score 5/5) and Prompt 2 (GDV emergency escalation, Score 5/5), the model performed at the highest level across all judge criteria. Plain language delivery and constraint compliance were particular strengths — the model reliably avoided unexplained jargon, never repeated the session disclaimer, and recommended no prescription medications across any test run.

Prompt 3 (plain language explanation of feline hypertrophic cardiomyopathy) produced the weakest result at 3 out of 5, with 77.8% of criteria met. The judge flagged missing critical RAG facts — specifically the aortic thromboembolism warning, a life-threatening complication of HCM that owners must be informed of to seek emergency care if it occurs. This finding is safety-adjacent: omission of a critical complication warning from a chronic condition explanation carries real risk. Prompt 4 (safe home remedies for cat upper respiratory infection, Score 4/5) and Prompt 5 (skin lesion image analysis, Score 4/5) both performed well on constraint compliance and plain language but showed room for improvement in urgency framing and explicit acknowledgment of diagnostic uncertainty respectively.

Notably, the safety gate passed 100% across all five test cases — no dangerous advice was generated in any run, and no automatic fails were triggered. Emergency escalation (Prompt 2, the most safety-critical test) was the highest-performing test case at 11 out of 11 criteria met.

---

## Caveats & Limitations

These results should be interpreted with significant caution for several reasons. The evaluation was conducted on a small custom dataset of five prompts — a meaningful signal but not a statistically robust sample. Real production queries will cover a far wider range of symptom combinations, emotional contexts, species variations, and language registers than this evaluation captures. The RAG corpus used during evaluation is based on veterinary curriculum materials from a single German institution. While this provides a strong and culturally appropriate knowledge base, it may not cover all conditions, breed-specific variations, or recent clinical developments equally. The LLM-as-judge, while calibrated against Dr. Lund's veterinary expert review, carries inherent biases toward longer responses and English-language phrasing patterns that may affect scoring fairness for German-language outputs.

Benchmark contamination and saturation in the existing benchmarks — particularly TruthfulQA and MedQA — mean that comparative scores against published leaderboards should not be treated as reliable indicators of real-world performance. Finally, this evaluation reflects performance under controlled test conditions. Production performance may differ due to the unpredictable nature of real user queries, emotional language, incomplete symptom descriptions, and multi-turn conversation dynamics not fully captured in single-turn test prompts. Cost estimates are based on token approximations rather than direct API usage reporting and should be treated as indicative rather than precise.

---

## Recommendation

Under the conditions tested and for the specific task of pre-consultation pet health advisory for dogs and cats in the German market, the current model pipeline shows sufficient promise to justify continued development investment — but is **not yet ready for public deployment**. The plain language completeness gap identified in Prompt 3 is a hard blocker at this stage: a product that omits life-threatening complication warnings from chronic condition explanations carries unacceptable safety risk. The recommended path to deployment readiness is a targeted prompt engineering effort focused specifically on chronic condition complication coverage, followed by a second evaluation round on an expanded test set of at least 25 prompts. Deployment should only proceed once Dr. Lund has reviewed and approved the revised system prompt outputs on HCM-category queries, and the average score across an expanded test set reaches 4.5 or above. For emergency query types, the model is already performing at a deployable level under appropriate disclaimer framing.

---

## Additional Metrics

Beyond accuracy and safety scoring, three additional metrics were tracked during evaluation. Average response generation time was 5.43 seconds per query — acceptable for a mobile application in V1 but a target for optimization toward 2-3 seconds in production. Estimated token cost per query using gpt-4o-mini averaged approximately $0.00074 USD (~€0.00068) per interaction, yielding a projected infrastructure cost of approximately $0.74 USD per 1,000 queries — a highly scalable cost structure that remains manageable even at 100,000 queries per month (~$74 USD). Environmental cost was not formally measured in this evaluation but should be incorporated into future assessments as the product scales, consistent with growing German consumer expectations around sustainable technology practices. The total evaluation pipeline for all five test cases — including response generation and judge evaluation — ran in 82 seconds at a total estimated cost of $0.0037 USD, demonstrating the efficiency and scalability of the automated evaluation approach.
