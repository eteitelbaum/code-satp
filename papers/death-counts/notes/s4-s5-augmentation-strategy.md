# S4/S5 Augmentation Strategy: Back-Translation and T5 Paraphrase

Notes for implementing S4 (back-translation) and S5 (T5 paraphrase) augmentation for
rare-bin improvement. To be revisited after S6 results are in.

## Status

S1–S3 and S6 complete. S4/S5 deferred per the attack plan: "Run only if S1–S3 gains
plateau." S1 improved bin 6+ by 5.3pp (68.4% → 73.7%). Whether that is sufficient
depends on how the full results table reads.

---

## S4 — Back-Translation

### Pivot languages

The classification paper's `BackTranslationAugmentation` uses Hindi (`hi`), Urdu (`ur`),
and Bengali (`bn`) as pivot languages — the same should be used here. The rationale from
the classification work applies: SATP incident summaries contain Indian place names
(district names, forest areas), group names (CPI-Maoist, CoBRA, STF, CRPF), and
narrative conventions from South Asian journalism. South Asian pivot languages preserve
these better than European pivots (French, German) which may mangle or drop unfamiliar
proper nouns.

The tradeoff is that Hindi/Urdu/Bengali translation quality is lower than French/German
overall, which could introduce noise. For short factual sentences this is acceptable.

### Implementation

The `BackTranslationAugmentation` class is already implemented in:
`models/classification-models/imbalance-handling/imbalance_handling_strategies.py`

It uses `deep-translator` or `googletrans` (whichever is available). Requires internet
access on Colab — works fine in standard runtime.

Adapt for count extraction by:
1. Filtering training set to bins 3–5 and 6+ only
2. Running back-translation on each example (all three pivot languages → 3 augmented
   versions per example)
3. Running count-preservation check on every augmented example (see below)
4. Adding passing examples to the training set with appropriate bin weights

### Count-preservation check

This is the critical constraint. After back-translation, run `parse_prediction` on the
augmented text and compare to the original label. Discard any example where the extracted
number does not match. A reasonable threshold is 85%+ pass rate across the augmented set
— if it's lower, the augmentation is adding more noise than signal.

```python
def count_preserved(original_label, augmented_text):
    extracted = parse_prediction(augmented_text)
    return extracted == original_label
```

Run this diagnostic on a sample of ~50 rare-bin examples before committing to full
augmentation. Log the pass rate per pivot language — one language may be more reliable
than the others for numeric content.

### Expected outcome

The classification paper found back-translation gave +20–30% F1 for rare classes. For
count extraction the gain will likely be smaller since the task is harder and the
augmentation only affects ~13% of training examples (bins 3–5 and 6+). A reasonable
expectation is 2–4pp improvement on bin 6+ exact match, potentially with a smaller
overall MAE penalty than S1.

---

## S5 — T5 Paraphrase

### Approach

Uses a T5 model fine-tuned for paraphrase generation (e.g. `Vamsi/T5_Paraphrase_Paws`)
to generate surface-form rewrites of rare-bin examples.

### Why this is riskier than S4

T5 paraphrase models are not constrained to preserve numbers. They may:
- Convert numerals to words or vice versa ("three" → "several")
- Drop casualty groups in long compound sentences
- Reorder clauses in ways that change what the count refers to

The count-preservation check is even more important here than for back-translation.
Empirically, T5 paraphrase is likely to have a lower pass rate than back-translation on
numeric factual content.

### Recommendation

Try S4 first. If the count-preservation rate for back-translation is high (85%+) and the
augmented training set is large enough, S4 alone may be sufficient. S5 is worth trying
only if S4 gives insufficient coverage (e.g. too many examples fail the preservation
check across all three pivot languages).

---

## Diagnostic notebook plan

Before implementing either strategy in the training pipeline, run a standalone diagnostic
notebook:

1. Load the rare-bin training examples (bins 3–5 and 6+, ~300–400 examples)
2. Run back-translation through Hindi, Urdu, Bengali
3. For each augmented example, run `parse_prediction` and compare to original label
4. Report pass rate per language and overall
5. Manually inspect 10–20 failing examples to understand failure modes
6. If pass rate is acceptable, also run T5 paraphrase on the same sample for comparison

This notebook can reuse `BackTranslationAugmentation` from the classification codebase
directly with minimal adaptation.

---

## Integration with training pipeline

If S4 is approved after the diagnostic:
- Add augmented examples to `train_df` before tokenization
- Assign same bin weights as the originals (they are members of the same bin)
- Could combine with S1 (oversampling) for a combined S1+S4 run — oversampling ensures
  augmented rare-bin examples are seen frequently; augmentation increases their diversity
