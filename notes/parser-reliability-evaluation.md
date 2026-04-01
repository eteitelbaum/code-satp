# Parser Reliability Evaluation for Death Count Extraction

Notes on how to evaluate and report parser success/failure separately from model predictive performance in the count extraction workflows.

## Motivation

Current count extraction results can conflate:

- true model prediction errors
- output formatting noncompliance
- parser failures (especially silent fallback to `0`)

This is particularly important in the LLM workflow, where `parse_fatalities(...)` can return `0` when it fails to find a digit, making parser failures look like valid zero-death predictions.

## Current Risk in the LLM Parser

In `models/count-models/utils/llm_utils.py`, `parse_fatalities(...)`:

- tries JSON key extraction (`"fatalities": <int>`)
- otherwise extracts digits via regex
- returns `0` if no digits are found

Implications:

- outputs like `"Two"` are currently parsed as `0`
- empty/malformed outputs can be silently treated as `0`
- parser failures may bias MAE and nonzero-error metrics

## Recommendation: Track Parser Reliability as a Separate Layer

Add parser diagnostics so we can distinguish:

- model response failure
- format noncompliance
- parser failure
- successful parse with fallback heuristics

This should be reported alongside accuracy metrics (MAE, RMSE, exact match, etc.).

## Suggested Parser Evaluation Metrics

### Core metrics

- `parse_success_rate`
  - Fraction of outputs where the parser extracted a count confidently.

- `parse_failure_rate`
  - Fraction of outputs where parsing failed.

- `zero_due_to_parse_failure_rate`
  - Fraction of outputs whose final prediction was `0` because parsing failed (not because model clearly predicted zero).

- `format_compliance_rate`
  - Fraction of outputs that match the requested output format (e.g., exact JSON for decoder/API LLMs, or number-only for T5 prompts).

### Diagnostic metrics (helpful for analysis)

- `parse_method_distribution`
  - How often parsing succeeded via:
  - JSON key match
  - direct integer parse
  - digit regex fallback
  - number-word parse
  - fallback to zero

- `ambiguity_rate`
  - Fraction of outputs containing multiple numeric candidates (parser had to choose).

- `number_word_rate`
  - Fraction of outputs using number words (`"one"`, `"two"`, etc.) instead of digits.

- `empty_output_rate`
  - Fraction of outputs that are empty/blank after stripping.

- `exception_rate`
  - Fraction of outputs triggering parser exceptions (should be near zero if parser is robust).

## Suggested Data to Save Per Prediction

In each per-model output CSV (or a companion diagnostics CSV), store parser metadata columns:

- `<model>_raw`
- `<model>_prediction`
- `<model>_parse_success`
- `<model>_parse_method`
- `<model>_format_compliant`
- `<model>_num_candidates`
- `<model>_had_digits`
- `<model>_had_number_words`
- `<model>_error_type`
- `<model>_defaulted_to_zero_due_to_parse_failure`

This allows both aggregate reporting and case-level audit.

## Suggested Parser Return Shape (Future Refactor)

Instead of returning only an `int`, add a structured parser function, e.g.:

```python
{
  "prediction": 2,
  "parse_success": True,
  "parse_method": "json_fatalities",
  "format_compliant": False,
  "num_candidates": 2,
  "had_digits": False,
  "had_number_words": True,
  "error_type": None,
  "defaulted_to_zero_due_to_parse_failure": False,
}
```

The notebook can still use `prediction` for metrics while also summarizing parser reliability.

## Minimal Viable Improvement (Low Friction)

If a full refactor is too much immediately, add just:

- `parse_success` (bool)
- `parse_method` (string)
- `defaulted_to_zero_due_to_parse_failure` (bool)

This would already make the death-count LLM results much more interpretable.

## Research Value

Reporting parser reliability improves the credibility of results by:

- separating parsing artifacts from model capability
- clarifying whether prompting changes improve reasoning or just output formatting
- making comparisons across model families (seq2seq vs decoder/API LLMs) more defensible
- highlighting where parser improvements may yield larger gains than prompt tuning

## Related Follow-Up (Parsing Consistency Across Pipelines)

The repo currently uses different parsing functions:

- LLM notebook path: `parse_fatalities(...)` (`llm_utils.py`)
- seq2seq notebook path: `extract_number(...)` / `parse_prediction(...)` (`extraction_utils.py`)

These are not equivalent (notably, `extract_number(...)` handles some number words but `parse_fatalities(...)` currently does not).

Potential future improvement:

- unify shared parsing logic into one core parser
- allow LLM-specific JSON-first parsing as a wrapper around that shared core
