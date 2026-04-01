# Location Extraction Paper Framing Notes

Working notes on whether the location extraction/autocoding work should stand alone as a paper and what the likely contribution is.

## Recommendation

The location extraction work appears strong enough to support a standalone paper separate from death count extraction.

## Why It Can Stand Alone

- Location coding is a major bottleneck in conflict event data production and analysis.
- The task is structurally different from count extraction:
  - hierarchical outputs (state, district, village, other locations)
  - partial correctness is common and substantively meaningful
  - geocoding implications create downstream stakes
- The model comparison space is already rich:
  - seq2seq models
  - span-based / decoder-style extraction approaches
  - LLM prompting baselines
  - baseline location extraction tools (e.g., GLiNER / geocoding pipeline components)

## Potential Contribution to CSS / Conflict Methods

- Provides a systematic comparison of approaches for hierarchical location extraction from conflict incident narratives.
- Shows how evaluation choice matters (level-specific exact match vs full hierarchy exact match).
- Offers practical lessons for autocoding geographic information in event datasets.
- Bridges extraction quality to downstream geocoding/spatial analysis reliability.

## Strong Angles to Emphasize

- Hierarchical evaluation design:
  - per-level exact match
  - core-level exact match
  - all-place exact match
- Error taxonomy:
  - missed entities
  - boundary/normalization issues
  - partial hierarchy recovery
  - spelling/variant place names
- Applied pipeline implications (matching, geocoding, and analysis-ready event data).

## Why It Should Not Be Forced Into the Death Counts Paper

- Different target structure and error mechanisms require different methods discussion.
- Different evaluation logic deserves dedicated explanation and interpretation.
- Combining both can dilute the contribution of each and create an overly broad methods paper.

## Possible Positioning

- Computational social science paper on automated hierarchical location coding in conflict event data.
- Methods paper on comparing seq2seq, span extraction, and LLM prompting for event-location autocoding.
