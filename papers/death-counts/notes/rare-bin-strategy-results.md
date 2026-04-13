# Rare-Bin Strategy Results — Seq2Seq Track (Flan-T5-Large)

Final test-set results for all training-side interventions (S0–S6) on Flan-T5-Large,
targeting rare death-count bins (3–5 and 6+). T5-XL-QLoRA included as upper-bound
comparison. All S4/S5 results use the count-preservation filter (numeral + word form).

## Results Table

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
| S6 — Few-shot on S1+S4 | 0.586 | 0.227 | 88.0% | 4.158 | 65.8% |

Test set: N=859 overall (bin 3-5: n=75, bin 6+: n=38).

---

## Key Findings

### What worked

- **S4 and S1+S4 are the only strategies that improve bin 3-5** on both exact match
  (+1.3pp, 86.7%→88.0%) and MAE (0.240→0.213). The improvement is directionally
  consistent across both strategies.
- Adding S1 weighted sampling on top of S4 (S1+S4) gives a marginally better bin 6+
  MAE (4.184→4.000) and ±1% (76.3%→81.6%) at no cost to bin 3-5.
- The count-preservation filter (numeral + word form regex) is necessary for S4:
  unfiltered S4 degraded bin 3-5 (85.3%) and bin 6+ exact% (63.2%); filtered S4
  restores bin 6+ exact% to baseline and improves bin 3-5.

### What did not work

- **S0 baseline has the best overall MAE (0.156) of any T5-Large strategy.** Every
  intervention increases overall MAE because all hurt bin 6+ MAE substantially.
- **No intervention improves bin 6+ MAE.** S0 (1.974) is the best on that metric by
  a wide margin — interventions roughly double it (3.8–4.3). Strategies that restore
  bin 6+ exact% to baseline (S4, S1+S4, S5) do so by making the model more aggressive
  about predicting high counts, which causes large errors when wrong.
- **S1, S2, S3 all degrade or hold flat on bin 3-5** while making bin 6+ MAE worse.
  No pure reweighting or oversampling strategy improves rare bins.
- **S6 few-shot is not viable.** Prepending rare-bin examples collapses bin 0 accuracy
  (98.4%→74.9%), dominating overall MAE (0.586). The few-shot examples push the model
  toward non-zero predictions even for zero-death events.

### The count-preservation filter and S5

The filter helps S4 (back-translation) but had a mixed effect on S5 (T5 paraphrase):
filtered S5 returned bin 3-5 to baseline (86.7%) vs unfiltered S5 (89.3%). The
difference is ~1 correct prediction on n=75 and may be noise. For consistency and
principled reasons (the filter argument applies equally to both strategies), the
filtered results are used as the canonical S5 result.

### The T5-XL-QLoRA gap

The ceiling model is 3.5pp ahead on bin 3-5 and 13.1pp ahead on bin 6+ exact%, with
roughly half the bin 6+ MAE. This gap is not meaningfully closed by any training-side
intervention on T5-Large. Bin 6+ appears to be a model capacity problem.

---

## Statistical Power

**The bin-level improvements are almost certainly not statistically significant.**

- Bin 3-5 (n=75): the 1.3pp improvement from S4/S1+S4 = 1 additional correct
  prediction. Bootstrap CIs on a proportion ~87% with n=75 are approximately ±4pp —
  the observed difference is well within noise.
- Bin 6+ (n=38): each prediction = 2.6pp. Bootstrap CIs are ±8pp or more. No
  strategy difference here is distinguishable from random variation.
- Overall (n=859): most power here, but overall differences are ~1pp and dominated
  by bins 0–2 where all strategies perform similarly.

Bootstrap significance testing (see `bootstrap-ci-todo.md`) should be run before
reporting bin-level differences in the paper. The paper should acknowledge explicitly
that rare-bin test sets are underpowered to detect realistic effect sizes from
training-side interventions.

---

## Recommendation for Paper

- **Primary finding:** Training-side interventions on Flan-T5-Large produce modest,
  directionally consistent bin 3-5 improvements (S4, S1+S4) but do not solve bin 6+.
- **Best performing strategy:** S1+S4 (filtered) — best bin 3-5 on MAE, ties baseline
  on bin 6+ exact%, best bin 6+ ±1% among augmentation strategies.
- **Framing:** Bin 6+ is a model capacity problem. The T5-XL-QLoRA gap (84.2% vs
  71.1% exact) suggests the necessary capacity for multi-group arithmetic inference
  is not present in T5-Large regardless of training strategy.
- **Caveat:** All bin-level improvements should be reported with bootstrap CIs and
  the small-n limitation noted explicitly.
