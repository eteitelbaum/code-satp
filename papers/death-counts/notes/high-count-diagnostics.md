# High-Count Case Diagnostics

Working notes from qualitative and quantitative analysis of model errors on high-count death events (true_label >= 3) across all models.

## Setup

All analysis uses the test set (n=859). High-count cases are defined as true_label >= 3 (n=113, ~13% of the test set). Models compared:

**LLMs (zero-shot inference):** GPT-4o-mini, Llama-3.1-8B, Mistral-7B, Mixtral-8×7B

**Seq2seq (fine-tuned):** Flan-T5-XL-QLoRA, Flan-T5-Large, Flan-T5-Base, mT5-Base, NT5-Small, IndicBART, ConfliBERT-Poisson

## Overall High-Count Performance

| Model | Type | Exact | Pct | Over | Under | MAE |
|---|---|---|---|---|---|---|
| GPT-4o-mini | LLM | 102 | 90.3% | 0 | 11 | 0.52 |
| Flan-T5-XL-QLoRA | seq2seq | 100 | 88.5% | 2 | 11 | 0.48 |
| Mixtral-8×7B | LLM | 99 | 87.6% | 0 | 14 | 0.58 |
| Llama-3.1-8B | LLM | 98 | 86.7% | 4 | 11 | 0.65 |
| Flan-T5-Large | seq2seq | 92 | 81.4% | 6 | 15 | 0.82 |
| Mistral-7B | LLM | 95 | 84.1% | 3 | 15 | 0.66 |
| Flan-T5-Base | seq2seq | 89 | 78.8% | 5 | 19 | 1.70 |
| NT5-Small | seq2seq | 86 | 76.1% | 3 | 24 | 1.62 |
| mT5-Base | seq2seq | 80 | 70.8% | 0 | 33 | 1.19 |
| IndicBART | seq2seq | 62 | 54.9% | 5 | 46 | 1.65 |
| ConfliBERT-Poisson | seq2seq | 19 | 16.8% | 16 | 78 | 2.49 |

Key finding: GPT-4o-mini and Flan-T5-XL-QLoRA are essentially tied. Flan-T5-Large and the open-source LLMs (Llama, Mistral, Mixtral) form a second tier that is broadly comparable to each other.

## The Shared Hard Floor

Error overlap analysis between GPT-4o-mini, Flan-T5-XL-QLoRA, and Flan-T5-Large reveals a shared floor of ~10-12 cases that every competent model gets wrong. These errors are driven by the text itself, not model limitations, and fall into two categories:

### Category 1: Likely ground-truth labeling errors (n≈3-4)

Cases where the narrative clearly describes injuries/wounds but the true label is nonzero. All models return 0; the true label appears to be a coding error where injured personnel were entered as fatalities.

Example (true=5, all models predict 0):
> "Five CRPF personnel were **wounded** in an ambush by armed Maoists..."

Example (true=3, all models predict 0):
> "Three District Police personnel, including a Head Constable, were **injured** in an encounter..."

These cases penalize all models equally for being correct. They are worth flagging as potential annotation errors in the dataset.

### Category 2: Unconfirmed attacker deaths (n≈5-6)

Narratives where security force deaths are precise and confirmed but Maoist/attacker casualties are reported with uncertainty ("claimed," "bodies taken away," "suspected"). Human coders summed both; all models discount the unconfirmed deaths.

Example (true=33, all models predict 18):
> "killed at least 18 Policemen... **Police sources said that about 15 Maoists were also killed**"

Example (true=14, all models predict 2):
> "Two CRPF personnel were killed... **Police sources claimed** that over a dozen Maoists were killed, but their colleagues **managed to take away the bodies**"

This is not a model error per se — it reflects a defensible epistemic choice. Whether to count unconfirmed attacker deaths depends on how `total_fatalities` was defined in the coding protocol. If the intended coding is "all reported deaths including claimed attacker casualties," the prompt should say so explicitly.

### Category 3: Revised counts not in the narrative (n≈1-2)

The true label exceeds any number mentioned in the narrative, suggesting the label was updated from a follow-up report after the initial summary was written.

Example (true=84, all models predict 76):
> "75 CRPF personnel and a State Policeman were killed..." (= 76; source of remaining 8 unknown)

No prompt or model change can recover these.

## T5-Large vs Open-Source LLMs

T5-Large and the three open-source LLMs are on the same performance tier but fail in characteristically different ways.

**Error overlap (high-count cases):**
- Shared hard floor (all models wrong): 12 cases
- T5-Large fails, all OSS LLMs correct: 4 cases
- Some OSS LLM fails, T5-Large correct: 5 cases

### T5-Large-specific failures: multi-group arithmetic overcount

All 4 T5-Large-only errors involve narratives listing deaths across two or more victim groups. T5-Large overcounts by 1-2 in each case.

Example (true=7, T5-Large predicts 9):
> "Four CPI-Maoist cadres, a Policeman and two villagers were killed..."

Example (true=8, T5-Large predicts 9):
> "Seven Police personnel and a civilian are killed..."

The LLMs handle these correctly. T5-Large appears to double-count one group or pick up an extra number from context (e.g., weapon counts, distances). This is a learnable pattern addressable through targeted training examples.

### OSS LLM-specific failures: idiosyncratic and model-specific

The 5 cases where T5-Large is correct but an OSS LLM fails are not shared across LLMs — each is a single-model quirk:

- **Llama**: returns 0 on a short clear sentence ("killed at least six Maoists"); overcounts by adding weapon counts or other numbers; +1 overcount on a list of named individuals
- **Mistral**: discounts bodies "carried away" (counts only recovered body); slight overcount on "X killed on spot, Y succumbed to injuries" (counts a survivor as dead)
- **Mixtral**: off-by-one on a sentence with mixed security force and Maoist counts

These do not share a single root cause and are less addressable with targeted interventions than T5-Large's consistent overcount pattern.

## GPT-4o-mini's Advantage

GPT-4o-mini makes 0 overcounts on high-count cases and commits only 11 undercounts, all of which fall into the shared hard floor categories above. It does not make the multi-group arithmetic errors that T5-Large makes, nor the idiosyncratic errors of the smaller LLMs.

On cases where the true label appears to be a coding error (injured != killed), GPT-4o-mini is arguably *more* accurate than the ground truth. This is a meaningful qualifier on interpreting its performance metrics: some of its "errors" are correct extractions penalized by mislabeled data.

## Compound Narrative Special Case

Case 8 in the OSS LLM analysis (true=4, Llama predicts 6, Mistral and Mixtral predict 0) involves a narrative with two separate incidents stitched together, with the second incident text duplicated verbatim. This is a data quality issue in the incident summaries themselves. Mistral and Mixtral appear to fail on the parse (the echoed text is the same narrative that caused the two silent parse failures noted in parser-analysis.md). Llama overcounts to 6. GPT-4o-mini and T5-Large correctly extract 4.

## Implications for Recovery Strategies

The error analysis suggests the following:

1. **The hard floor (~12 cases) is not recoverable by any model or prompt improvement.** It requires fixing ground-truth labels (for the injury/death confusion cases) or clarifying the coding protocol (for the unconfirmed attacker death cases).

2. **T5-Large's 4 unique failures are structurally similar and likely addressable** through targeted training examples demonstrating multi-group enumeration. This is training-side intervention, not prompting.

3. **OSS LLM idiosyncratic failures are harder to address systematically** because they don't share a root cause. A few targeted few-shot examples might close some (e.g., "bodies carried away still count"; "critically injured ≠ dead") but the gains would be small (1-3 cases per model).

4. **The unconfirmed attacker deaths category** (true=33/14/24/18/18) is the largest source of high-magnitude errors. Prompt engineering could address this by explicitly instructing models to include claimed attacker deaths — but whether that is the right behavior depends on the intended coding protocol, not model capability.

See `rare-bin-performance-improvement-concept.md` for the broader strategy agenda and `parser-analysis.md` for parser-related findings.

---

## Full Dataset Distribution and Coding Protocol Analysis

Analysis of the full `data/deaths.csv` (n=9,919) to understand the high-count distribution and confirm the human coding protocol.

### Count distribution across the full dataset

| Bin | N | % | Cumulative % |
|---|---|---|---|
| 0 | 7,040 | 71.0% | 71.0% |
| 1 | 1,749 | 17.6% | 88.6% |
| 2 | 539 | 5.4% | 94.0% |
| 3–5 | 395 | 4.0% | 98.0% |
| 6+ | 196 | 2.0% | 100.0% |

Further breakdown of the tail:
- 3+ cases: 591 (6.0% of dataset)
- 6+ cases: 196 (2.0%)
- 10+ cases: 87 (0.9%)
- 20+ cases: 21 (0.2%)
- Maximum: 148

The 6+ bin thins out rapidly above 13 — single-digit counts for most values above that, with only 21 cases at 20 or above. This has direct implications for training data coverage: with ~8,200 training examples and the same proportional distribution, there are roughly 330 cases in the 3-5 bin and 160 in the 6+. These are not negligible counts, but the heterogeneity within those bins (multi-group arithmetic, claimed deaths, single large events, revised counts) means the model needs to see each structural variant repeatedly to generalize.

### Coding protocol confirmed as maximalist

`deaths.csv` contains five component columns alongside `total_fatalities`:
- `govt_official_fatalities`
- `civilian_fatalities`
- `security_fatalities`
- `maoist_fatalities`
- `other_armed_grp_fatalities`

Key findings from the component analysis:

**`total_fatalities` equals the sum of components in 9,908 of 9,919 cases (99.9%).** The 11 exceptions are minor data entry issues (components not filled in) rather than a different counting protocol.

**Maoist/attacker deaths are systematically included in `total_fatalities`.** In the 6+ bin:
- 48% of cases have `maoist_fatalities > 0`
- Maoist deaths account for 33.8% of all deaths in the 6+ bin
- Security force deaths account for 41.5%
- Civilian deaths account for 21.9%

This confirms that `total_fatalities` is consistently defined as all deaths on all sides — not security force deaths only. The coding protocol is unambiguously maximalist throughout the dataset, not just in the test cases examined qualitatively.

**Specific test cases confirmed against component data:**
- true=84 (Dantewada): `security=76, maoists=8` → 76+8=84 ✓
- true=33 (Gadchiroli): corresponding full-dataset case shows `security=3, maoists=30` → same coding logic
- true=44 (Dantewada): `security=24, maoists=20` → 24+20=44 ✓
- true=13 (Chintagufa): `security=10, maoists=3` → 10+3=13 ✓

**Implication for the LLM prompt (Strategy L1):** The attacker deaths clarification is not a guess at coder intent — it is confirmed by the component column data across the full 9,919-row dataset. Adding "Count all reported deaths on all sides, including claimed attacker casualties" to the LLM prompt directly aligns the extraction instruction with the documented coding protocol.

**Implication for seq2seq training (Strategy S3):** The ~160 training examples in the 6+ bin include a substantial share where `maoist_fatalities > 0`. The training labels already encode the maximalist protocol. The issue is that models have not learned to generalize this to cases where the attacker death count is expressed with uncertainty ("claimed," "bodies taken away") rather than as a confirmed figure. Targeted oversampling of those specific cases is the correct intervention.
