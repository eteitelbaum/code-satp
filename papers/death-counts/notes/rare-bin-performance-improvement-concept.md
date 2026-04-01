# Concept Note: Improving Performance on Rare Death Count Bins

Working concept note on strategies to improve extraction performance for rare but substantively important death-count bins (especially `4+`, `6+`) in SATP conflict event narratives.

## Motivation

Overall metrics (e.g., MAE) can obscure poor performance in the tail of the count distribution.

For conflict research, the higher-count events are often analytically central even when they are rare. A model that performs well on `0`, `1`, and `2` fatalities but fails on `4+` may look strong in aggregate while missing the events of greatest substantive interest.

This motivates a targeted methods agenda for improving rare-bin extraction performance and reporting those improvements transparently.

## Core Research Question

Which interventions most improve performance on rare/high-fatality bins (`4+`, `6+`) without degrading performance on common bins?

## Strategy Families to Evaluate

### 1. Prompting-Side Interventions (Decoder/API LLMs and Prompted T5)

These are relatively low-friction experiments and useful for code/presentation demos.

#### A. Targeted Few-Shot Prompting with High-Count Examples

- Include exemplars with higher fatality counts (`4+`, `6+`) in the prompt.
- Emphasize examples with multiple numbers (injuries + deaths) to teach the model which number to extract.
- Include examples demonstrating:
  - “injured X, killed Y”
  - “no one was killed” despite many injuries
  - number words (e.g., “two,” “sixteen”)

Rationale:
- May reduce model bias toward common low counts.
- Improves task framing and output consistency on hard cases.

#### B. Bin-Balanced Few-Shot Prompt Sets

- Construct few-shot prompts with one example from each bin:
  - `0`, `1`, `2`, `3-5`, `6+`
- Compare against random few-shot examples.

Rationale:
- Makes the prompt distribution less skewed than the training corpus / natural data frequency.

#### C. Format-Reinforced Few-Shot Prompting

- Keep strict JSON output instruction and include examples showing exact JSON output for high-count cases.

Rationale:
- May improve parseability even if counting improvements are modest.

#### D. Hard-Case Prompt Sets (Failure-Mode-Specific)

- Build prompts around known failure modes:
  - multiple numeric mentions
  - injuries vs fatalities confusion
  - compound events in one narrative
  - indirect fatality language

Rationale:
- Targets the actual error distribution rather than generic examples.

### 2. Training-Side Interventions (Fine-Tuned Seq2Seq / Supervised Models)

These are likely the most important interventions for durable improvements in rare bins.

#### A. Weighted Sampling / Oversampling by Rare Count Bin

- Oversample examples from higher-fatality bins (`3-5`, `6+`) in training.
- Keep validation and test distributions unchanged.

Rationale:
- Directly increases exposure to scarce outcomes.
- Closely analogous to weighted random sampling used in classification imbalance work.

#### B. Per-Example Loss Weighting by Count Bin

- Increase training loss weight for examples in rare/high-count bins.
- Compare simple hand-set weights vs frequency-based weights.

Rationale:
- Encourages the model to attend more to rare high-cost errors.

#### C. Data Augmentation for Rare Bins (Back-Translation / Paraphrase)

- Generate paraphrases for rare-bin examples only.
- Candidate approaches:
  - back-translation (contextually relevant languages)
  - T5/FLAN-T5 paraphrase augmentation

Critical constraint:
- Must preserve fatality count exactly.
- Add filtering/validation checks so synthetic examples do not alter the true count label.

Rationale:
- Expands training signal for rare bins without requiring new manual annotations.

#### D. Synthetic Data Generation for Rare Bins (LLM-Assisted)

- Generate additional rare-bin narratives (`4+`, `6+`) with explicit count labels.
- Use strict quality control and human spot-checking.

Risks:
- Label noise
- unrealistic language
- distribution shift away from SATP reporting style

Rationale:
- Could be high-upside when natural rare-bin sample sizes are very small.

#### E. Curriculum / Phase-Based Sampling (Optional)

- Start with natural distribution, then introduce stronger rare-bin oversampling later in training.

Rationale:
- May improve stability relative to aggressive oversampling from the beginning.

## What Transfers From the Classification Imbalance Strategies

From the imbalance-handling strategies used in the classification paper/workflow, several ideas transfer well to rare-bin count extraction:

### Strongly transferable

- Weighted sampling / oversampling
- Back-translation augmentation
- T5 paraphrase augmentation
- (Potentially) LLM-based synthetic generation, with stronger factual consistency checks

### Partially transferable / needs adaptation

- Class weights / focal loss
  - natural for classification
  - less direct for seq2seq generation
  - may require reformulating the task (e.g., bin classification auxiliary objective) or applying per-example weighting in training loops

### Not directly transferable

- Threshold tuning (from multi-label classification calibration)
  - not directly applicable to generated count extraction outputs

## Important Measurement Issue: Parser Reliability

Improvements in rare bins can be masked or distorted if parser failures are silently mapped to `0`.

This is especially important because:

- high-count examples often contain multiple numbers
- some models output number words rather than digits
- current parser behavior can confound parse failure with true zero predictions

Therefore, any rare-bin improvement study should include parser reliability diagnostics alongside predictive metrics.

## Recommended Evaluation Focus for Rare-Bin Experiments

In addition to overall MAE/RMSE:

- bin-level MAE (especially `3-5` and `6+`)
- within-1 / within-2 by bin
- nonzero MAE
- exact match in high-count bins
- parser success / parse failure rates
- ambiguity rate (multiple candidate numbers in output)

## Practical Experimental Sequence (Low-to-High Effort)

1. Improve parser robustness and track parser diagnostics.
2. Test targeted/bin-balanced few-shot prompting for decoder/API LLMs.
3. Try weighted sampling / oversampling in fine-tuned seq2seq training.
4. Add rare-bin-focused augmentation (back-translation / paraphrases) with count-preservation checks.
5. Explore synthetic LLM-generated rare-bin data only if earlier interventions plateau.

## Potential Contribution to a Death Counts Paper

This line of work supports a strong contribution beyond “which model wins”:

- how to improve tail performance for substantively important rare events
- how parser reliability interacts with apparent model performance
- which imbalance strategies transfer (or fail to transfer) from classification to extraction tasks

That framing should be of broad interest in computational social science, especially for event data workflows where rare but high-impact observations matter disproportionately.
