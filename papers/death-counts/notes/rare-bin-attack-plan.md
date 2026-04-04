# Rare-Bin Improvement: Attack Plan

Working plan for the rare-bin improvement study (Part 2 of the death counts paper). Builds on the diagnostic findings in `high-count-diagnostics.md` and the strategy framework in `rare-bin-performance-improvement-concept.md`.

## Design Overview

The improvement study runs in two parallel tracks — one for prompted LLMs, one for fine-tuned seq2seq — because the intervention levers are fundamentally different for each. Each track uses a single representative model to keep the results readable.

**LLM track representative:** Llama-3.1-8B (open-source, locally reproducible). GPT-4o-mini shown as a ceiling reference in the results table but not subjected to every intervention since it is a proprietary API model.

**Seq2seq track representative:** Flan-T5-Large (most room for improvement among the capable seq2seq models; T5-XL-QLoRA is already near-ceiling and serves as an upper-bound reference).

**Primary outcome metrics:** bin-level MAE and exact match on the 3-5 and 6+ bins. Overall MAE and nonzero MAE reported alongside for completeness.

---

## Annotation Quality Note (Pre-Analysis)

Before running interventions, flag the ~3-4 test cases where all models unanimously predict 0 but the true label is 3-5. Narrative review confirms these describe injuries not fatalities — likely coding errors. Do not correct the labels (would require auditing all splits; see discussion in `high-count-diagnostics.md` and Northcutt et al. 2021 on best practice). Instead:

- Report metrics as-is on the original labels
- Report metrics again excluding the flagged cases
- Flag in the paper as LLM-assisted annotation quality audit finding

---

## LLM Track

All interventions are prompt-only. No model retraining required. Each can be tested by re-running inference on the existing test set with the modified prompt — cheap and fast.

### Models
- **Primary:** Llama-3.1-8B-Instruct
- **Reference ceiling:** GPT-4o-mini (run L1 only, given API cost)

### Interventions

**L0 — Baseline**
Current zero-shot prompt:
> "How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": <integer>}. If no fatalities are mentioned, use 0."

**L1 — Attacker deaths clarification**
Add one sentence to the instruction:
> "Count all reported deaths on all sides, including claimed attacker casualties even if bodies were not recovered."

Motivation: diagnostic analysis shows all LLMs systematically exclude unconfirmed attacker deaths (e.g. true=33, pred=18; true=14, pred=2). Human coders used a maximalist protocol counting all reported deaths. This aligns the extraction instruction with the coding protocol. Verify this does not cause overcounting on zero-death and low-count cases before reporting.

**L2 — Bin-balanced few-shot**
Prepend one example from each bin (0, 1, 2, 3-5, 6+) to the prompt, selected to be clear unambiguous cases. Examples should demonstrate the correct JSON output format and cover a range of narrative styles (single victim, multiple groups, Maoist + security forces).

Motivation: may reduce model bias toward low counts by making the full count range salient in the prompt.

**L3 — Hard-case few-shot**
Replace or supplement L2 examples with examples targeting known failure modes:
- Multi-group enumeration: "X [group A] and Y [group B] were killed → X+Y"
- Bodies carried away: "one body recovered, four carried away → total is five"
- Succumbed to injuries: "killed on spot plus succumbed to injuries both count as deaths"
- Claimed attacker deaths (if L1 prompt clarification alone is insufficient)

Motivation: targets the specific narrative structures driving the residual errors identified in the diagnostic analysis.

### Validation approach
Test L1, L2, L3 first on the validation set to check for regressions on low-count cases before evaluating on test. Specifically check that zero-death exact match does not degrade.

---

## Seq2seq Track

All interventions are training-side. Each requires a new fine-tuning run on Colab. Use the existing seq2seq notebook infrastructure and adapt the classification paper's imbalance handling code.

### Model
- **Primary:** Flan-T5-Large
- **Reference ceiling:** Flan-T5-XL-QLoRA (baseline results already saved; no retraining needed)

### Bin definitions
Consistent with existing metrics code:
- Bin 0: total_fatalities == 0 (n=319 in test)
- Bin 1: total_fatalities == 1 (n=325)
- Bin 2: total_fatalities == 2 (n=102)
- Bin 3-5: total_fatalities in [3, 4, 5] (n=75)
- Bin 6+: total_fatalities >= 6 (n=38)

High-count bins (3-5 and 6+) together are 113/859 = 13.2% of the test set.

### What transfers from the classification paper

**Direct transfer (minimal adaptation):**
- `WeightedRandomSampler` from `models/classification-models/utils/strategy_experiments.py` — drop-in for the seq2seq training loop's dataloader
- Back-translation augmentation from `models/classification-models/imbalance-handling/imbalance_handling_strategies.py` — works on text regardless of task; apply bin filter to select which rows to augment
- T5 paraphrase augmentation — same

**Needs adaptation:**
- `compute_conservative_class_weights()` — converts from per-label classification weights to per-bin count weights. Logic is the same; redefine "class" as count bin.
- Loss weighting — classification paper uses `BCEWithLogitsLoss` per-label weights. Seq2seq generation loss requires per-example weighting in the Trainer, either via a custom `compute_loss` override or by passing `sample_weight` equivalents. Moderate effort.

**New code needed:**
- Bin assignment function: maps `total_fatalities` to bin label (trivial — one line)
- Targeted example identifier for S3: script to find multi-group arithmetic and claimed/unrecovered deaths narratives in the training set via keyword/regex search

### Interventions

**S0 — Baseline**
Standard fine-tuning on the full training set with natural data distribution. Results already saved in `papers/death-counts/results/death-counts-seq2seq/`.

**S1 — Weighted random sampling**
Use `WeightedRandomSampler` to oversample high-count bin examples during training. Assign sampling weights inversely proportional to bin frequency (bins 3-5 and 6+ upweighted). Apply conservative capping to avoid extreme weights destabilizing training (following classification paper approach).

**S2 — Loss weighting by bin**
Alternative to S1 (or combined). Assign higher training loss weight to examples in bins 3-5 and 6+. Compare simple hand-set weights (e.g. 3× for 3-5 bin, 5× for 6+ bin) versus frequency-based weights. Adapt `compute_conservative_class_weights()` for this purpose.

**S3 — Targeted training examples**
Two related failure patterns identified in diagnostic analysis, both addressable through the same mechanism:

*Multi-group arithmetic* (T5-Large unique failures): narratives listing deaths across heterogeneous groups where the model overcounts. Find existing training examples with this structure; oversample them.

*Claimed/unrecovered attacker deaths* (universal failure shared with LLMs): narratives where attacker casualties are reported with uncertainty ("claimed," "bodies taken away"). The training labels already encode the maximalist count — the model just needs more exposure to these cases. Find and oversample them. T5-XL-QLoRA already gets at least one such case right (true=24), confirming the pattern is learnable from training data.

Practical steps:
1. Search training set for multi-group keywords ("and X [group] were killed", "cadres were killed", "including X... and Y...")
2. Search for claimed-death keywords ("claimed", "bodies taken away", "bodies not recovered", "suspected")
3. Verify labels on flagged cases are correct
4. Add to fine-tuning data with oversampling weight (2-5×)

**S4 — Back-translation augmentation**
Generate paraphrases of rare-bin training examples (bins 3-5 and 6+) via back-translation. Use existing `BackTranslationAugmenter` from classification paper. Apply count-preservation check: verify that the augmented narrative still supports the original label before adding to training data (critical constraint — augmentation must not alter the true count).

**S5 — T5 paraphrase augmentation**
Alternative to S4 using T5 paraphrase generation. Existing `T5ParaphraseAugmenter` in classification paper. Same count-preservation constraint applies.

Note: S4 and S5 are higher friction than S1-S3. Run only if S1-S3 gains plateau. The classification paper found these useful for very rare classes; whether the gain justifies the complexity for count extraction is an empirical question.

**S6 — Inference-time few-shot on fine-tuned model (exploratory)**
After fine-tuning T5-Large via S1-S3, test whether prepending few-shot examples to the inference prompt adds further improvement. The fine-tuned model was trained on plain inputs with no examples, so this is out-of-distribution at inference time — the result is genuinely uncertain. If it helps, that is an interesting finding about residual in-context learning capacity in fine-tuned Flan-T5. If it does not, that confirms fine-tuning overwrites the pre-trained few-shot behavior.

Run on the validation set first. Use the same hard-case examples as LLM track L3 for comparability.

### Recommended experimental sequence

Run in order, evaluate bin-level metrics after each step, stop if gains plateau:

1. S1 (weighted sampling) — lowest friction, strong prior from classification paper
2. S2 (loss weighting) — run alongside S1 as an alternative; pick the better performer
3. S3 (targeted examples) — add after confirming S1/S2 gains; tests whether specific pattern coverage adds to general oversampling
4. S4/S5 (augmentation) — only if S1-S3 plateau
5. S6 (inference-time few-shot) — exploratory, run last

---

## Results Table Structure

Two tables in the paper, one per track, showing cumulative effect of each intervention. Columns:

| Strategy | Overall MAE | Nonzero MAE | Bin 3-5 MAE | Bin 6+ MAE | Exact 3-5 (%) | Exact 6+ (%) |
|---|---|---|---|---|---|---|
| Baseline | | | | | | |
| + intervention | | | | | | |
| ... | | | | | | |

Include a row for the reference ceiling model (GPT-4o-mini / T5-XL-QLoRA) to show how much headroom remains after interventions.

---

## Relationship to Existing Notes

- `high-count-diagnostics.md` — source of the error patterns motivating S3 and L1-L3
- `rare-bin-performance-improvement-concept.md` — original strategy brainstorm; this plan narrows and prioritizes it
- `parser-analysis.md` — parser fix already applied; `parse_fatalities()` updated with metadata support before new experiments run
- `paper-framing-notes.md` — overall paper structure; this plan covers Part 2 of the paper
