# Death Count Extraction: Why Filter to Armed Assault + Bombing?

The death count extraction models (seq2seq and LLM) are trained and evaluated on
Armed Assault and Bombing incidents only (`first_action` in these two categories),
drawn from `satp_clean.csv`. This gives ~4,300 incidents vs. ~9,900 in the full
dataset.

## Rationale

1. **Fatality concentration.** Deaths in SATP are overwhelmingly concentrated in
   Armed Assault and Bombing events. Other event types (arrests, surrenders, property
   damage, political events) occasionally have fatalities but they are rare and
   incidental — including them would mostly add zero-death noise without covering
   meaningfully different narrative structures.

2. **Pipeline alignment.** The intended deployment pipeline runs the classification
   model first (perpetrator, action type) and then routes events to the count
   extractor. If the extractor is only called on events classified as Armed Assault
   or Bombing, then training on only those events keeps train and deployment
   distributions matched.

3. **Task difficulty.** Violent events have the hardest extraction problems —
   multi-party confrontations, claimed vs. confirmed casualties, multi-group
   arithmetic. Focusing the model on that harder distribution is appropriate.

## Paper language

> "We restrict count extraction training and evaluation to Armed Assault and Bombing
> incidents, which account for the overwhelming majority of fatalities in the dataset
> and represent the deployment scope of the extraction pipeline."

## Open questions

- The filter uses `first_action` only. Events coded as "Arson, Armed Assault"
  (Armed Assault in `second_action`) are excluded. Whether this is intentional or
  an oversight is worth confirming.
- If the classifier routes non-Armed-Assault/Bombing events with deaths to the count
  extractor, those events are out-of-distribution at inference time. Low risk in
  practice given fatality concentration, but worth noting as a limitation.

## Contrast with location extraction

Location models use all event types (no action-type filter) with a temporal 80/10/10
split. Death count models use a random stratified 60/20/20 split on the filtered
subset. The val/test sets are shared across seq2seq and LLM count experiments,
making cross-model comparisons valid.
