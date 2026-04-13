# Bootstrap CIs — TODO

After all test-set predictions are saved (LLM track + seq2seq track), run paired
bootstrap confidence intervals on the key comparisons.

## Why

Bin 6+ has only n=38 on the test set. Differences like the L0→L4 improvement
(76.3% → 84.2% exact match = ~3 additional correct predictions) need CIs to
establish whether they are signal or noise. Overall metrics on n=859 are precise
without bootstrapping; the bin-specific comparisons are where CIs are essential.

## Design

**Paired bootstrap** — resample example tuples (true_label, pred_model_A, pred_model_B)
together with replacement. The test statistic is the difference in metric between
models. This accounts for correlation across variants (hard cases are hard for all
models) and gives more power than independent bootstraps.

**Metrics to report:** exact match and MAE for bins 3-5 and 6+. Overall MAE for
completeness.

## Key Comparisons

1. Llama L0 vs Llama L4 — headline LLM intervention result
2. Llama L4 vs GPT-4o-mini L1 — open-source vs proprietary ceiling
3. T5-Large S0 vs S1+S4 (filtered) — headline seq2seq intervention result
4. Best seq2seq vs best LLM — cross-track comparison

**Note (April 2026):** All S0–S6 results now available. The bin 3-5 improvement from
S1+S4 (86.7%→88.0% exact, n=75) is ~1 additional correct prediction. Bootstrap CIs
are expected to be wide enough (±4pp) to render this non-significant. This should be
confirmed and reported explicitly in the paper — see `rare-bin-strategy-results.md`.

## Implementation

Python script is most natural — all predictions are already saved as CSVs.
Existing R bootstrap code at `papers/location-extraction/data-viz/bootstrap_significance_tables.R`
could also be adapted if preferred.

A single script that loads all prediction CSVs and outputs a formatted CI table
in one pass is cleaner than running bootstraps piecemeal.
