# Citation Allocation by Paper Section

A guide to which sources go where and why. The governing principle: each citation does one job in one place. Simon et al. and Zhong et al. are the only papers that appear in both intro and theory — they do different argumentative work in each location.

---

## Introduction

Purpose: motivate and preview. Establish stakes, name the gap, and state the puzzle. No architecture argument — that belongs in the theory section. Cite only to establish that (a) conflict coding matters, (b) counts have not been solved, (c) naive approaches fail, (d) some prior work exists but is incomplete.

| Key | Use |
|---|---|
| `steinert2025` | Motivating warning: naive LLM prompting for fatality counts is unreliable and language-biased. The "don't just ask ChatGPT" evidence. |
| `gilardi2023`, `tornberg2025` | LLMs viable for well-defined annotation tasks generally — sets up the contrast with count extraction |
| `leetaru2013` | GDELT as the canonical automated conflict dataset — counts present but unvalidated (V2COUNTS) |
| `nardulli2015` | SPEED as the supervised ML era — human coders still extract counts |
| `halterman2021` | IndiaPoliceEvents — binary KILL indicator, not a count; the outer boundary of current transformer work |
| `hu2022` OR `brandt2025` | Transformer era conflict NLP — use one, not both, to establish that event *types* are now well-addressed |
| `zhong2023` | Brief: closest NLP precedent for count extraction; distinguish in theory |
| `simon2025` | Brief: most recent AEE work includes counts; distinguish in theory |

---

## Theory / Background Section

Purpose: build expectations from architectural properties. Each subsection should end with a prediction that the results section later confirms or qualifies. Cite to support architectural claims, not to describe what people did.

### 2.1 Why count extraction is structurally different from classification

| Key | Use |
|---|---|
| `brandt2025` | The foil: fine-tuned extractive encoders beat generative LLMs on *classification*. This advantage does not transfer to count extraction — use to motivate the structural distinction |
| `zhong2023` | Their own ablations show generative > extractive even within their paper — direct evidence that the extractive ceiling is lower for counts |
| `alsarra2025` | Multilingual extractive QA: shows span-selection failure modes when answers are not verbatim spans |

### 2.2 Architectural expectations (the ladder)

**Dictionary/rule-based:**
| Key | Use |
|---|---|
| `schrodt2012` | CAMEO codebook — the ontology that dictionary systems operate on; why co-occurrence counting over-counts |
| `leetaru2013` | GDELT V2COUNTS as the canonical noisy dictionary-based count field |

**Encoder + regression/Poisson head:**
| Key | Use |
|---|---|
| `devlin2019` | BERT — encoder architecture reference |
| `hu2022`, `brandt2025` | ConfliBERT as domain-specific encoder; also our baseline model class |
| `nardulli2015` | SPEED uses regression-like ML-assisted coding — the supervised non-generative ceiling |

**Extractive QA:**
| Key | Use |
|---|---|
| `zhong2023` | NT5 approach is extractive QA; their own results motivate moving to generative |
| `alsarra2025` | Explicit evidence of extractive QA limits on political texts |

**Seq2seq (encoder-decoder):**
| Key | Use |
|---|---|
| `chung2022` | Flan-T5 / instruction finetuning — our primary seq2seq model family |
| `simon2025` | Their per-field breakdown (Table 5): Deaths Low at 69% for DEGREE even in their favorable setting; Text2Event collapses on Low/High — evidence the ceiling matters |

**Decoder LLMs:**
| Key | Use |
|---|---|
| `brown2020` | GPT-3 / few-shot learning foundations |
| `kojima2023` | Zero-shot reasoning capabilities |
| `wei2023` | Chain-of-thought: LLMs can do implicit arithmetic with the right prompting |
| `steinert2025` | Without structured prompting, LLM fatality estimates are unreliable — motivates the intervention study |

**Structural analogues (brief mention):**
| Key | Use |
|---|---|
| `braun2025` | Braun & Oswald: LLMs extract counts from ACLED abduction text at >90% accuracy. Distinguish: their counts are small, explicitly stated, and the distribution is not skewed — structurally easier than death counts |
| `al-garadi2025` | Mortality extraction from social media/obituaries — epidemiology structural analogue for count extraction from unstructured text |

### 2.3 The distribution problem and evaluation design

| Key | Use |
|---|---|
| `simon2025` | Aggregate Deaths score (81%) is the mean of 6 sub-fields; Deaths Low is 69% for best model; no rare-event analysis; UCDP excludes zero-death events by construction |
| `zhong2023` | No rare-event analysis; corpus less skewed than SATP |
| `semnani2025` | LEMONADE does not break out fatality-specific performance; aggregate AEAE F1 conflates counts with categoricals and booleans |
| `eck2012` | UCDP vs. ACLED comparison — stakes of getting counts right for conflict measurement |

---

## Data Section

Purpose: describe SATP corpus, coding protocol, and test set design.

| Key | Use |
|---|---|
| `fetzer2020`, `ghatak2017a`, `vandeneynde2018`, `shapiro2023`, `gomes2015b` | Maoist conflict in India context — substantive background |
| `start2021`, `lafree2007` | GTD — comparison point for conflict database design |
| `eck2012` | UCDP/ACLED data quality stakes (also usable in intro) |

---

## Models and Baseline Section

| Key | Use |
|---|---|
| `chung2022` | Flan-T5-Large and T5-XL-QLoRA architecture |
| `devlin2019`, `sanh2020`, `liu2019` | BERT/DistilBERT/RoBERTa — encoder baseline family |
| `hu2022` | ConfliBERT-Poisson baseline |
| `brown2020`, `kojima2023` | Llama/GPT-4o-mini baseline justification |

---

## Interventions / Experiments Section

Save all of these for here — none should appear in intro or theory:

| Key | Use |
|---|---|
| `sennrich2016` | Back-translation (S4 augmentation strategy) |
| `buda2018`, `he2009` | Class imbalance — oversampling / weighted sampling (S1) |
| `lin2018` | Focal loss (S2 loss weighting) |
| `parolin2022b` | Confli-T5 AutoPrompt augmentation — precedent for conflict text augmentation |
| `halterman2025` | Synthetically generated text for supervised learning — relevant to augmentation strategies |
| `pangakis2023`, `pangakis2024` | Validation and knowledge distillation — annotation quality framing |
| `egami2024` | Downstream consequences of LLM annotation errors — relevant if paper makes pipeline recommendations |
| `mosbach2021` | Fine-tuning stability — relevant to training variance discussion |

---

## Discussion / Conclusion

| Key | Use |
|---|---|
| `egami2024` | Statistical consequences of using automated annotations downstream |
| `pangakis2023` | Validation requirement — when can automated extraction replace human coders |
| `steinert2025` | Return to: the paper answers Steinert's critique by showing systematic prompting fixes the reliability problem |
| `abdurahman2025` | Best practices for LLM research in social science — methodology framing |

---

## Gap in the bib

No TABARI/KEDS system paper. If the theory section discusses dictionary-based systems substantively, add **Schrodt & Gerner (1994)** "Validity Assessment of a Machine-Coded Event Data Set for the Middle East, 1982–1992" — the standard TABARI reference. The `schrodt2012` CAMEO codebook is in the bib but is a codebook, not a system paper.
