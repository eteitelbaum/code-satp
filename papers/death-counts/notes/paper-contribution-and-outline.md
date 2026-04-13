# Paper Contribution, Literature Context, and Outline

## State of the Literature on Machine-Coded Event Counts

### The short answer

This is relatively uncharted territory in the conflict studies literature, but the framing needs one nuance: this is not the *first* NLP paper on victim count extraction in any domain, but it is likely the first to treat conflict event death count extraction as the primary research problem.

### What exists in conflict studies NLP

All prior automated event coding work focuses on *event types* — not numeric counts:

- **Dictionary/rule-based era** (TABARI, GDELT, CAMEO): fatality counts appear only as GDELT's noisy V2COUNTS field — unvalidated, not the research focus.
- **SPEED** (Nardulli 2015): NLP assists human coders; humans still extract the numbers.
- **Halterman IndiaPoliceEvents** (2021): codes a binary KILL field (did police kill *someone?*), not how many.
- **UCDP-AEC** (Simon 2025): includes fatality counts in an abstractive event extraction framework, but it is a dataset paper, count extraction is one of many coded fields, and per-field performance on counts is not broken out.
- **LEMONADE** (ACL 2025): includes fatality counts within ACLED-based event argument extraction, but as a subcomponent of a broader multilingual framework.

In short: the conflict studies community has moved from rules to encoders for *classification* tasks. Numeric count extraction has not received dedicated attention.

### What exists outside conflict studies

- **Zhong et al. 2023 (EACL)** — "Extracting Victim Counts from Text" — is the closest true precedent. Uses NT5 (numeracy-augmented T5 variant) to extract death/injury counts from humanitarian datasets (World Atrocities Dataset, NAVCO, European Media Monitor). This is the paper to cite prominently and distinguish from. Differences: general humanitarian framing rather than armed conflict/terrorism; no analysis of rare-event performance; no LLM vs. seq2seq comparison.
- **Epidemiology/disaster monitoring** (EventEpi, EpiTator): structurally analogous task for disease case counts from WHO and ProMED reports.
- **Steinert 2025 (JPR)** — "How User Language Affects Conflict Fatality Estimates in ChatGPT": shows naive LLM prompting for conflict fatality counts is unreliable and language-biased. Motivates a supervised/fine-tuned approach.

### The contribution claim

Strong and defensible: this appears to be the **first paper to treat conflict event death count extraction as the primary research problem**, specifically:

1. In the armed conflict / political violence literature (the field that actually *uses* these counts)
2. Comparing seq2seq fine-tuning against LLM prompting systematically
3. Analyzing the rare-event problem (high-fatality incidents) as a distinct sub-challenge with tailored interventions
4. Distinguishing parser reliability as a measurement quality issue separate from model accuracy

---

## Key Framing Notes

**Lead with the gap in conflict studies, not NLP broadly.** The audience cares that political violence researchers cannot currently automate count extraction reliably. The Steinert paper and GDELT's noisy V2COUNTS are the evidence that existing tools fail.

**Cite Zhong et al. early and distinguish clearly.** They work on humanitarian crises in general (atrocities, civil resistance, European media) with no specific focus on terrorism/insurgency; they don't address rare-event performance; they don't compare prompting vs. fine-tuning architectures.

**The rare-bin analysis is the most novel methodological contribution.** The insight that overall MAE obscures systematic failure on the cases that matter most — high-fatality events are precisely what conflict datasets are built to capture — is genuinely important and absent from Zhong or the other count extraction work.

**The instruction-following asymmetry (L1 helpful for GPT-4o-mini, harmful for Llama)** is a secondary but interesting finding relevant to ongoing debates about LLM reliability for social science annotation.

---

## Suggested Outline

```
1. Introduction
   - Automated conflict event coding: from rules to transformers
   - The gap: event types are now well-addressed; numeric counts are not
   - Stakes: high-fatality events drive conflict severity metrics, but
     models fail disproportionately on rare high-count cases
   - Preview: seq2seq vs. LLM comparison; rare-bin intervention study

2. Related Work
   a. Automated conflict event coding
      - Dictionary/rule-based era (TABARI, GDELT, CAMEO)
      - Supervised ML era (SPEED, CROICU et al.)
      - Transformer era (ConfliBERT, 3M-Transformers, Brandt 2025)
      - What these systems code: event types, actors, locations —
        counts treated as secondary or left to human coders
   b. NLP for numerical information extraction
      - Zhong et al. 2023: victim counts from humanitarian texts —
        closest precedent; distinguish by domain and evaluation design
      - LEMONADE / UCDP-AEC: counts as one argument among many
      - Epidemiological surveillance as structural analogue
   c. LLMs for political science annotation
      - General capabilities (Gilardi, Ornstein, Tornberg)
      - Limitations in conflict contexts: Steinert (2025) on LLM
        fatality estimate instability

3. Data and Coding Protocol
   - SATP corpus: 9,919 incidents, Maoist insurgency in India
   - Count distribution: highly skewed (71% zero; 6+ deaths = 2%)
   - Maximalist coding protocol: all reported deaths, including
     claimed attacker casualties (contrast with UCDP conservative approach)
   - Test set design and bin definitions

4. Models and Baselines
   - Seq2seq fine-tuning track: Flan-T5-Large (+ comparators)
   - LLM prompting track: Llama-3.1-8B (+ comparators)
   - Ceiling references: Flan-T5-XL-QLoRA, GPT-4o-mini
   - Evaluation metrics: MAE, exact match, within-1/2, bin-level breakdown
   - **NOTE — metric choice rationale (follow up in results section write-up):**
     Zhong et al. (2023) evaluate using exact match and token-level F1 inherited from
     the extractive QA / SQuAD paradigm. Token-level F1 measures character or word
     overlap between predicted and gold spans — it is not precision/recall over a
     binary outcome. For a single-number answer, exact match and F1 typically coincide,
     but F1 can award partial credit for partially overlapping strings (e.g., "23" vs.
     "230") in ways that are meaningless for count evaluation. More importantly, neither
     exact match nor token-level F1 penalizes errors proportionally to their magnitude:
     predicting 1 when the answer is 100 scores the same zero as predicting 23 when the
     answer is 24. MAE is the appropriate metric for count extraction because it captures
     *how wrong* the model is, not just *whether* the string matched. The bin-level
     breakdown adds a further dimension: it reveals *where* in the distribution errors
     concentrate, which aggregate MAE also obscures. Use this to motivate the metric
     choices in Section 4 and to explain why Zhong et al.'s results, while informative,
     cannot be directly compared to ours or used to assess distributional reliability.

5. Baseline Results
   - Overall and per-bin performance
   - Seq2seq vs. LLM comparison
   - Identifying the rare-bin gap as the key challenge

6. Strategies for Rare-Bin Improvement
   a. LLM track (prompt engineering, L0–L4)
      - Protocol clarification (L1): harmful for Llama; well-calibrated
        for GPT-4o-mini — instruction-following capability difference
      - Bin-balanced few-shot (L2), hard-case few-shot (L3), combined (L4)
      - L4 closes the bin 6+ gap entirely (54% MAE reduction)
      - Takeaway: examples are the active ingredient, not instructions alone
   b. Seq2seq track (training interventions, S0–S6)
      - Weighted sampling, loss weighting, targeted examples: all hurt overall
      - Back-translation S4 (Hindi/Urdu/Bengali): only intervention to
        improve bin 3-5 exact match; preserves South Asian entities
      - Bin 6+: no intervention closes the gap — capacity problem, not data
      - Takeaway: T5-Large lacks arithmetic reasoning for multi-group sums

7. Error Analysis
   - Hard floor cases shared across all models
   - Three categories: annotation errors, unconfirmed attacker deaths
     (epistemic discounting vs. protocol gap), revised counts
   - Model-specific failures: Llama hallucination on ambiguous narratives,
     T5-Large arithmetic overcounting

8. Discussion
   - Practical recommendations: which architecture for which constraints
   - Parser reliability as measurement hygiene (separate from model accuracy)
   - Coding protocol choices and replication implications
   - Limitations and future work

9. Conclusion
```

---

## Key Papers to Add to Bibliography

The following are not yet in nlp-coding.bib and should be added or checked:

- Zhong, Dhuliawala, Stoehr (2023). "Extracting Victim Counts from Text." EACL 2023. arXiv: 2302.12367. **Priority: high** — closest direct precedent.
- Simon et al. (2025). "Abstractive Event Analysis of Armed Conflicts: Introducing the UCDP-AEC Dataset." KONVENS 2025. ACL Anthology: 2025.konvens-2.8. *[already in bib as simon2025 — confirm field coverage includes counts]*
- LEMONADE (Semnani et al., 2025). "A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World." ACL 2025 Findings. arXiv: 2506.00980. **Priority: medium** — includes fatality counts in ACLED event argument extraction.
- Nardulli, Althaus, Hayes (2015). "A Progressive Supervised-learning Approach to Generating Rich Civil Strife Data." Sociological Methodology. *[already in bib as nardulli2015]*
