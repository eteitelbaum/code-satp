# S2 Loss Weighting: Implementation Notes

Technical notes on the `LossWeightedSeq2SeqTrainer` implementation in
`models/count-models/utils/training_utils.py`.

## The Core Problem

HuggingFace's `Seq2SeqTrainer` computes loss as the mean cross-entropy over all
non-padding tokens across the entire batch. This flattens all examples together —
there is no per-example loss handle to attach a weight to. To apply per-example
weights, we need to recompute the loss manually.

## Mechanism

Weights are embedded as a `sample_weight` column directly in the tokenized
HuggingFace dataset before passing it to the trainer:

```python
weights = compute_bin_weights(train_df['total_fatalities'].tolist())
train_tokenized = train_tokenized.add_column("sample_weight", weights)
```

`DataCollatorForSeq2Seq` stacks scalar columns into a `(batch_size,)` tensor, so the
weights arrive in `inputs` automatically at each training step. `compute_loss` pops
them before the model forward pass (the model doesn't accept that argument), then
applies them at the sequence level.

## Per-Token → Per-Example Math

Rather than using `outputs.loss` (which averages over all non-padding tokens across
the whole batch without tracking which example each token belongs to), we recompute
manually:

1. `CrossEntropyLoss(reduction="none")` → per-token losses, shape `(batch, seq_len)`
2. Mask padding tokens (label == -100), sum and divide by non-padding token count
   → per-example mean loss, shape `(batch,)`
3. Multiply each example's scalar loss by its bin weight
4. Average across the batch → final scalar loss for backprop

```python
loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
token_losses = loss_fct(...).view(shift_labels.size())
mask = (shift_labels != -100).float()
per_example_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
loss = (per_example_loss * w.float().to(per_example_loss.device)).mean()
```

## Why Weight at the Sequence Level, Not the Token Level

Weighting at the token level would have an unintended side effect: long narratives
would receive more total weight than short ones, independently of their bin
membership. A 200-token high-count narrative would dominate a 20-token high-count
narrative even if both belong to the same bin with the same intended weight. Sequence-
level weighting (weight the mean per-example loss, not individual tokens) avoids this.

## Fallback Behavior

If `sample_weight` is not present in `inputs` — e.g. because `DataCollatorForSeq2Seq`
drops unknown columns — `w` will be `None` and the trainer silently falls back to
`outputs.loss`, producing identical results to the S0 baseline. This failure mode is
silent and would not raise an error.

**Verification step:** add a sanity check in the first training batch to confirm
weights are arriving:

```python
# Temporary debug — add to compute_loss, remove after confirming
if w is None:
    print("WARNING: sample_weight not found in inputs — check DataCollator behavior")
else:
    print(f"sample_weight batch: min={w.min():.3f} max={w.max():.3f}")  # remove after 1 batch
```

If `w` is consistently `None`, the fix is to use a custom DataCollator that
explicitly passes through the `sample_weight` column, or to fall back to S1
(WeightedRandomSampler) which avoids the DataCollator issue entirely.

## Relationship to S1

S1 (WeightedRandomSampler) and S2 (loss weighting) apply pressure at different points
in training:

- S1 changes **which examples the model sees** in each epoch — rare-bin examples are
  drawn more frequently, so they appear more times in total.
- S2 changes **how much each example contributes to the gradient** when it is seen —
  rare-bin examples are seen at the natural frequency but exert more influence per
  occurrence.

In expectation over many batches they have similar effects, but they can differ in
practice: S1 may cause the model to overfit to the small set of rare-bin examples
(seeing them many times); S2 keeps the full training distribution while amplifying
the rare-bin signal. Which works better for this task is an empirical question, which
is why we run both.
