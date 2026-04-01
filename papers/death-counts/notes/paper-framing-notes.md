# Death Counts Paper Framing Notes

Working notes on whether the death count extraction work should stand alone as a paper and what the likely contribution is.

## Recommendation

The death count extraction work appears strong enough to support a standalone paper rather than being combined with location extraction in a single manuscript.

## Why It Can Stand Alone

- The task is substantively important for conflict event data (fatality counts are central outcome variables).
- The output structure (scalar count extraction) is methodologically distinct from location extraction.
- The evaluation framework is already substantial:
  - MAE
  - RMSE
  - exact / within-k style metrics
  - nonzero MAE
  - bin-level performance (0, 1, 2, 3-5, 6+)
- Multiple model families are compared:
  - fine-tuned seq2seq models
  - decoder/API LLMs
  - supervised baseline (e.g., ConfliBERT + Poisson head)

## Potential Contribution to CSS / Conflict Methods

- Demonstrates practical performance tradeoffs across extraction architectures for a high-value conflict variable.
- Highlights that extraction accuracy is not uniform across count ranges (especially higher-fatality events).
- Provides a useful benchmark design for count extraction from event narratives in applied settings.
- Opens a methodological contribution around parser reliability and measurement validity:
  - parsing failures can be mistaken for valid zero predictions
  - prompt compliance and parseability should be evaluated alongside predictive accuracy

## Strong Angles to Emphasize

- Real-world extraction constraints (messy narratives, multiple numbers, ambiguity between injuries and deaths).
- Error decomposition:
  - model reasoning/counting mistakes
  - formatting noncompliance
  - parser failures
- Reliability of extraction pipelines as a measurement issue, not just an engineering issue.

## Risks If Combined With Location Extraction

- Too much methodological surface area in one paper.
- Harder to give adequate depth to parser/evaluation issues in both tasks.
- The unifying “T5” theme is useful for a talk, but likely weaker as the main scholarly contribution in a paper.

## Possible Positioning

- Applied conflict-methods paper on automated fatality count extraction from incident summaries.
- Comparative methods paper on extraction architectures + parser reliability in conflict event coding.
