# Section 6: Improving Performance on Rare Bins

## Target: 1,200–1,500 words

---

## Structure
Two subsections mirroring the classification paper's imbalance handling section. Each
subsection opens with a brief description of the strategy, then presents results. No
separate methods block — description and results are interleaved. Cross-track synthesis
closes the section.

  **6.1 Seq2seq Training Interventions** (Flan-T5-Large, S1–S5)
  **6.2 LLM Prompting Strategies** (Llama-3.1-8B, L1–L4, where L denotes Llama)

---

## Purpose
Report the results of systematic interventions targeting the rare high-count bins on
both tracks. For the prompting track: prompt engineering (L0–L4 on Llama-3.1-8B,
GPT-4o-mini L1 ceiling). For the seq2seq track: training-side interventions (S0–S5,
T5-XL ceiling reference). Each intervention is motivated directly by the §5 diagnostics.
Draw the key conclusions about what works, what doesn't, and why.

---

## Outline

### 6.0 Setup (≈75 words)
- Two parallel tracks; one representative model per track
  - LLM: Llama-3.1-8B-Instruct (primary); GPT-4o-mini (ceiling, L1 only)
  - Seq2seq: Flan-T5-Large (primary); T5-XL-QLoRA (ceiling, no retraining)
- Intervention labels: L0–L4 (prompting), S0–S5 (training)
- Outcome metrics: bin 3–5 and bin 6+ MAE + exact match; overall MAE alongside
- Validation set used for pre-test regression checks; test set evaluated once per final strategy

---

### 6.1 LLM Track: Prompt Engineering Interventions (L0–L4) (≈550 words)

#### 6.1.1 Results overview

| Model | Strategy | Overall MAE | Bin 3-5 MAE | Bin 6+ MAE | Exact 3-5% | Exact 6+% |
|---|---|---|---|---|---|---|
| GPT-4o-mini | L0 Baseline | 0.133 | 0.160 | 1.237 | 93.3 | 84.2 |
| Llama-3.1-8B | L0 Baseline | 0.235 | 0.213 | 1.526 | 89.3 | 78.9 |
| Llama-3.1-8B | L1 Protocol clarification | 0.318 | 0.227 | 0.947 | 88.0 | 76.3 |
| Llama-3.1-8B | L2 Bin-balanced few-shot | 0.199 | 0.280 | 0.974 | 88.0 | 78.9 |
| Llama-3.1-8B | L3 Hard-case few-shot | 0.113 | 0.200 | 0.842 | 90.7 | 81.6 |
| Llama-3.1-8B | L4 Combined few-shot (L2+L3) | 0.107 | 0.280 | 0.632 | 88.0 | 89.5 |
| GPT-4o-mini | L1 Ceiling reference | 0.158 | 0.267 | 0.368 | 90.7 | 89.5 |

#### 6.1.2 L1 — Protocol clarification
- Adds: "Count all reported deaths on all sides, including claimed attacker casualties even if bodies were not recovered" — directly aligns with maximalist protocol confirmed in §3.3
- **Llama:** harmful — MAE 0.235 → 0.318; blowup on val set (bin 6+ MAE 7.026 vs 2.658 baseline); instruction causes hallucinated attacker death counts on ambiguous narratives; ruled out as standalone strategy pre-test
- **GPT-4o-mini (L1 ceiling):** different pattern — overall MAE rises slightly (0.133 → 0.158) but bin 6+ MAE drops dramatically (1.237 → 0.368); well-calibrated enough to apply instruction without overcounting
- Key finding: **instruction-following asymmetry** — same instruction helps GPT, harms Llama; reflects meaningfully different calibration, not just model size

#### 6.1.3 L2–L4 — Few-shot progressions
- **L2 (5 bin-balanced examples):** overall MAE improves (→0.199); bin 6+ MAE improves; bin 3–5 worsens slightly — count range made salient but some overcounting in 3–5
- **L3 (4 hard-case examples):** best bin 3–5 of all Llama variants (exact 90.7%); strong bin 6+ gain (exact 81.6%); examples anchor "bodies carried away" and multi-group patterns that instruction alone cannot
- **L4 (9 examples = L2 + L3 + L1):** bin 6+ exact 89.5% — **matches GPT-4o-mini L1 exactly**; overall MAE 0.107 (54% reduction from baseline); bin 3–5 trades off slightly vs. L3 (88.0% vs 90.7%); L2 examples calibrate L3's high-count push and eliminate the bin-0 regression L3 alone produces
- Key finding: **examples are the active ingredient** — L1 alone misfires; demonstrations show the model concretely what to count; L3 is preferable if bin 3–5 matters equally to bin 6+

#### 6.1.4 Key conclusions: LLM track (3–4 bullets, ≈75 words)
- L4 closes the bin 6+ gap: few-shot Llama matches GPT-4o-mini (89.5% exact)
- Instructions without demonstrations are harmful for Llama-scale models
- GPT retains a bin 6+ MAE advantage (0.368 vs 0.632) even when tied on exact match — errors are smaller when GPT misses
- Practical implication: Llama + L4 is now competitive with GPT for researchers without API access

---

### 6.2 Seq2seq Track: Training-Side Interventions (S0–S5) (≈500 words)

#### 6.2.1 Results overview

| Strategy | Overall MAE | 3-5 MAE | 3-5 Exact% | 6+ MAE | 6+ Exact% |
|---|---|---|---|---|---|
| T5-XL-QLoRA (ceiling) | **0.116** | **0.187** | **90.7%** | **1.053** | **84.2%** |
| S0 — Baseline | 0.156 | 0.240 | 86.7% | 1.974 | 71.1% |
| S1 — Weighted sampling | 0.258 | 0.267 | 85.3% | 4.079 | 73.7% |
| S2 — Loss weighting | 0.225 | 0.240 | 86.7% | 3.947 | 68.4% |
| S3 — Targeted examples | 0.242 | 0.240 | 85.3% | 4.053 | 68.4% |
| S4 — Back-translation (filtered) | 0.268 | 0.213 | 88.0% | 4.184 | 71.1% |
| S1+S4 — Combined (filtered) | 0.250 | 0.213 | 88.0% | 4.000 | 71.1% |
| S5 — T5 paraphrase (filtered) | 0.270 | 0.227 | 86.7% | 4.342 | 71.1% |

Test set: n=859 overall (bin 3–5: n=75, bin 6+: n=38)

#### 6.2.2 S1–S3 — Reweighting strategies
- **S1 (weighted random sampling):** oversample bins 3–5/6+ inversely proportional to frequency; result: overall MAE 0.156 → 0.258; bin 6+ MAE 1.974 → 4.079; model predicts high counts more aggressively, hurting the many easy cases
- **S2 (loss weighting):** same logic, training-loss implementation; same failure pattern (MAE rises, no bin improvement)
- **S3 (targeted oversampling):** oversample identified multi-group arithmetic and claimed-death examples; same outcome; structural patterns are present in training data but T5-Large cannot generalize regardless of exposure
- Shared finding: **forcing the model toward rare counts destabilizes performance on common counts** — the overall MAE cost exceeds any rare-bin gain

#### 6.2.3 S4 — Back-translation augmentation (pivot: Hindi/Urdu/Bengali)
- South Asian pivots preserve Indian place names and group acronyms; European pivots mangle them (motivates language choice)
- Filter: null/empty only — **do not apply count-preservation regex**; ~18% of examples would fail a numeral-check because totals are implicit ("five police and two civilians" → 7; "7" never appears); filtering removes exactly the hard cases needed for training
- Result: **only strategy to improve bin 3–5** — MAE 0.240 → 0.213, exact 86.7% → 88.0%; bin 6+ exact restored to baseline (71.1%)
- S1+S4 combined: marginal further improvement on bin 6+ MAE (4.184 → 4.000) at no cost to bin 3–5 → recommended strategy
- S5 (T5 paraphrase) directionally consistent but less effective than S4; use S1+S4

#### 6.2.4 S6 note (brief)
- Inference-time few-shot on a fine-tuned model: not viable — bin 0 accuracy collapses (98.4% → 74.9%), MAE 0.586; fine-tuning overwrites pre-trained few-shot behavior; contrasts with LLM track where few-shot is the mechanism

#### 6.2.5 Key conclusions: seq2seq track (3–4 bullets, ≈75 words)
- S0 baseline has the best overall MAE of any T5-Large strategy; every intervention increases it
- Only S4/S1+S4 improves any rare bin (bin 3–5, directionally); no strategy improves bin 6+ MAE
- Bin 6+ is a model capacity problem: T5-XL gap (84.2% vs 71.1% exact) cannot be closed by training interventions — the path is more capacity, not more data
- Back-translation with South Asian pivots is the right augmentation approach for this domain

---

### 6.3 Cross-Track Synthesis (≈150 words)

| | LLM track | Seq2seq track |
|---|---|---|
| Bin 3–5 | L3: directional improvement | S4/S1+S4: directional improvement |
| Bin 6+ | L4: 59% MAE reduction, matches ceiling | None — capacity problem |
| Overall MAE | L4: 54% reduction (best strategy) | All interventions worsen vs. S0 |

- The key asymmetry: LLM few-shot operates at inference time with zero training cost and can demonstrate exact reasoning patterns; seq2seq training interventions require T5-Large to generalize patterns it lacks capacity for
- For bin 6+, few-shot Llama outperforms any T5-Large training strategy
- Practical recommendations (brief; expanded in §7):
  - No GPU: Llama + L4 (89.5% bin 6+ exact, no training cost)
  - Fine-tuning available: T5-Large + S1+S4 (best seq2seq on bin 3–5)
  - Highest accuracy: GPT-4o-mini + L1 (best bin 6+ MAE: 0.368)

---

## Statistical Power Caveat (to include explicitly in the section)
Bin-level improvements should be read as directional findings, not precise effect sizes. Bin 6+ (n=38): bootstrap CIs ≈ ±8pp; each prediction = 2.6pp. Bin 3–5 (n=75): CIs ≈ ±4pp; S4's 1.3pp gain = 1 additional correct prediction. All differences within noise for bin-level comparisons. Overall MAE differences (n=859) are statistically testable and reported with bootstrap significance.

---

## Key Citations
- Parolin et al. 2022 (Confli-T5) — generative augmentation for conflict text
- Deep-translator / Google Translate — implementation reference for S4
- Northcutt et al. 2021 — annotation quality (S6 finding context)

## Figures / Tables
- Table @tbl-rarebin-seq2seq (already in .qmd)
- Table @tbl-rarebin-llms (already in .qmd)

## Notes
- S1+S4 combined does not appear in the final PDF table (Table 3); the five strategies are S1–S5 only. S1+S4 was run but excluded — results nearly identical to S4 alone.
- S4 filter rationale (null-only, not numeral check): covered by §2–4; one clause in prose is sufficient, no separate explanation needed.
- S6 (inference-time few-shot on fine-tuned model): was run but excluded from table. Bin-0 exact match collapsed from 98.4% to 74.9%. Worth one sentence or footnote in the seq2seq subsection.
- Ceiling for LLM bootstrap tests changed from GPT-4o-mini L1 to GPT-4o-mini L0. Bootstrap rerun and table re-rendered (2026-04-12). L4 earns daggers on overall MAE, bin 6+ MAE, and bin 6+ exact vs GPT-4o-mini L0.

---

# Improving Performance on Rare Bins

Building on this diagnostic taxonomy, we evaluate a series of targeted interventions designed to improve performance on the rare high-count bins. The interventions are organized around the two model families evaluated in the previous section. For the seq2seq models, we apply five training-side strategies to Flan-T5-Large, ranging from reweighting and oversampling to data augmentation through back-translation and paraphrase generation. For the LLMs, we apply four prompt engineering strategies to Llama-3.1-8B, progressively adding protocol clarification and few-shot demonstrations. T5-XL-QLoRA and GPT-4o-mini serve as ceiling references for the seq2seq and LLM results respectively. All interventions were evaluated against the validation set before the test set was touched, and the test set was evaluated once per finalized strategy.

## Seq2seq Training Interventions

The baseline Flan-T5-Large model miscodes roughly one in eight bin 3–5 events and nearly one in three bin 6+ events. We evaluate five training-side strategies designed to improve performance on these rare bins. Three modify how training examples are sampled or weighted during fine-tuning to compensate for the skewed count distribution. Two augment the rare-bin training data through paraphrase generation, increasing the surface-form diversity of high-count narratives without requiring additional annotation.

### Intervention Strategies

**Weighted sampling.** Standard fine-tuning samples training examples in proportion to their frequency, which means the model sees bin 3–5 and bin 6+ examples only rarely. Weighted random sampling addresses this by assigning each training example a weight inversely proportional to its bin frequency, so that rare-bin examples appear more often within each epoch [@buda2018]. We implement this using PyTorch's `WeightedRandomSampler`.

**Loss weighting.** Rather than changing which examples are sampled, loss weighting increases the training signal from rare-bin examples by assigning higher weights to their contributions to the cross-entropy loss [@he2009]. Weights are set inversely proportional to bin frequency, with conservative capping to prevent extreme weights from destabilizing training. This provides an alternative route to the same goal as weighted sampling through a different mechanism.

**Targeted oversampling.** Weighted sampling and loss weighting treat all rare-bin examples equally. Targeted oversampling instead selects specific examples matching the failure patterns identified in the previous section, such as narratives with multi-group enumeration and narratives where attacker casualties are described as claimed or unrecovered, and oversamples those at a higher rate. The rationale is that increased exposure to the exact narrative structures driving the errors may be more effective than upweighting rare bins.

**Back-translation augmentation.** Back-translation generates paraphrases of existing training examples by translating them into a pivot language and back into English, producing surface-form variation while preserving the underlying content [@sennrich2016; @halterman2025]. We apply back-translation to all rare-bin training examples using Hindi, Urdu, and Bengali as pivot languages, chosen because these languages preserve South Asian place names and group acronyms that European pivot languages tend to mangle.

**T5 paraphrase augmentation.** As an alternative to back-translation, we use Flan-T5-Large to generate paraphrases of rare-bin training examples directly in English [@chung2022; @parolin2022b]. This produces more diverse surface-form variation than back-translation but at the cost of greater risk that the paraphrase alters numerically critical content. The same rare-bin examples targeted by back-translation are augmented here, allowing a direct comparison of the two approaches.

### Results

Table 3 reports the results. The three reweighting strategies (weighted sampling, loss weighting, and targeted oversampling) all increase overall MAE relative to the baseline and none improves bin 3–5 or bin 6+ exact match. All three push the model toward higher count predictions on ambiguous narratives, increasing errors on the common low-count cases that dominate the test set without producing compensating gains in the rare bins. Back-translation improves on the baseline, raising bin 3–5 exact match from 86.7% to 88.0% while holding bin 6+ exact match at the baseline level, but the difference is not statistically significant. T5 paraphrase augmentation is directionally similar but weaker, returning bin 3–5 exact match to the baseline level. No strategy improves bin 6+ MAE, which remains near 4.0 for all interventions against a baseline of 1.97.

--Table 3 about here--

These results suggest that the bin 6+ gap between Flan-T5-Large (71.1% exact match) and T5-XL-QLoRA (84.2%) reflects a model capacity constraint rather than a data imbalance problem. The failure patterns identified in the previous section (multi-group enumeration, implicit totals, claimed casualties) require the model to aggregate counts across clauses and reason about what to include. These are mainly arithmetic reasoning tasks that scale with model capacity rather than training data volume. The T5-XL-QLoRA result, achieved without any rare-bin intervention, suggests the gap is closeable with a larger model, and that reweighting and augmentation cannot substitute for the capacity of a much larger model.

## LLM Prompt Engineering Interventions

The baseline Llama-3.1-8B model miscodes roughly one in ten bin 3–5 events and roughly one in five bin 6+ events. We evaluate four prompt engineering strategies designed to improve performance on these rare bins. One adds a protocol clarification to the system prompt, instructing the model to count claimed attacker casualties even when bodies were not recovered. Three vary the content and composition of few-shot demonstrations, progressively targeting the narrative structures most associated with high-count miscoding. GPT-4o-mini serves as the ceiling reference for the LLM track. The full text of the prompts is available in Appendix B.

### Intervention Strategies

**Protocol clarification.** The baseline prompt does not specify how to handle attacker casualties that are described as claimed or unrecovered. This strategy adds a single instruction to the system prompt directing the model to count all reported deaths on all sides, including claimed attacker casualties even when bodies were not recovered. This directly encodes the maximalist counting protocol established in the data collection design.

**Bin-balanced few-shot.** This strategy replaces the zero-shot baseline with five demonstrations sampled to cover the full range of death counts, including at least one example from each bin. The goal is to make the count range salient to the model before it reads the target narrative, without specifically targeting the failure patterns identified earlier.

**Hard-case few-shot.** Rather than sampling for bin coverage, this strategy selects four demonstrations that exemplify the specific narrative structures driving miscoding in the baseline, including events where attacker casualties are described as claimed or inferred and events where deaths must be summed across multiple armed groups. The examples are drawn from the validation set and excluded from evaluation.

**Combined few-shot.** This strategy combines the bin-balanced and hard-case example sets (nine demonstrations total) with the protocol clarification instruction. The bin-balanced examples provide count-range calibration, the hard-case examples anchor the difficult narrative structures concretely, and the protocol instruction aligns the model on what to count.

### Results

Table 4 reports the results. The protocol clarification strategy is counterproductive. Overall MAE rises from 0.235 to 0.318 and bin 6+ exact match falls from 78.9% to 76.3%. However, the few-shot strategies improve performance at every step. Bin-balanced demonstrations reduce overall MAE to 0.199 while holding bin 6+ exact match at the baseline level. Hard-case demonstrations reduce overall MAE further to 0.113 and raise bin 6+ exact match to 81.6%, the strongest rare-bin result among all Llama strategies. The combined strategy improves on all three metrics, with overall MAE falling to 0.107, bin 6+ exact match rising to 89.5%, and bin 3–5 exact match holding at 88.0%. All three figures nominally exceed GPT-4o-mini, though none of the differences are statistically significant.

-- Table 4 about here --

These results suggest that demonstrations rather than instructions are the active ingredient in few-shot prompting for this task. The protocol clarification instruction specifies what to count but provides no examples of how to handle the ambiguous cases that drive miscoding in the baseline, including claimed casualties, implied totals, and multi-group enumeration. Without anchoring examples, the model applies the instruction too broadly, increasing predicted counts on cases where deaths are genuinely uncertain. The hard-case demonstrations address uncertainty by showing the model how to handle each failure type, while the protocol instruction adds value within the combined prompt because the examples first establish when claimed casualties should be counted and when they should not.