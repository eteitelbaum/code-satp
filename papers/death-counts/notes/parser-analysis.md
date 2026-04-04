# Parser Analysis: Output Parsing in the Death Count Extraction Pipeline

Working notes on parser design, format compliance, and the decision not to pursue parser experiments in the rare-bin analysis.

## Two Parsers, Two Pipelines

The death count extraction workflow uses two separate parsing functions matched to their respective inference setups:

- **LLM pipeline:** `parse_fatalities()` in `models/count-models/utils/llm_utils.py`
  - Tries JSON key extraction first (`"fatalities": <int>`)
  - Falls back to first plausible digit (filtered to 0–200)
  - Returns 0 on failure (silent)

- **Seq2seq pipeline:** `extract_number()` + `parse_prediction()` in `models/count-models/utils/extraction_utils.py`
  - Tries direct integer conversion, then digit regex, then number-word lookup (zero through ten)
  - Returns `None` on failure; `parse_prediction()` converts `None` to 0

These are not inconsistent — each is matched to the output format its models are trained or prompted to produce. The LLM prompt requests JSON (`{"fatalities": <integer>}`); the seq2seq training targets are plain digit strings (e.g. `"3"`).

## Apparent Differences That Are Not Real Issues

Three surface differences between the parsers turn out not to matter in practice:

**JSON key extraction.** `parse_fatalities()` checks for `"fatalities": N` explicitly; `extract_number()` does not. But seq2seq models are trained to output plain numbers, not JSON, so the check would be irrelevant for that pipeline.

**Number-word handling.** `extract_number()` maps words ("two") to digits; `parse_fatalities()` does not. Inspection of all 859 × 4 = 3,436 LLM raw outputs found zero instances of word-only outputs. The JSON format instruction (`"Return JSON exactly as: {"fatalities": <integer>}"`) is sufficient to prevent this.

**Plausibility filter.** `parse_fatalities()` filters extracted digits to the range 0–200. `extract_number()` has no such filter. For seq2seq models this is appropriate: they are fine-tuned to output only a count, so any digit in the output is a genuine model prediction, not a stray number grabbed from context. The filter earns its keep in the LLM pipeline because verbose outputs may contain years, IDs, or other large numbers.

## Observed Format Compliance

Compliance was measured against the expected output format for each pipeline.

### LLM models (expected: strict JSON `{"fatalities": N}`)

| Model | Strict JSON | JSON + trailing | No JSON |
|---|---|---|---|
| GPT-4o-mini | 100% | 0% | 0% |
| Mistral-7B | 99.8% | 0.1% | 0.1% |
| Mixtral-8×7B | 99.9% | 0.0% | 0.1% |
| Llama-3.1-8B | 0% | **100%** | 0% |

Llama-3.1-8B always produces valid JSON as the first element of its output but appends reasoning commentary on every prediction (e.g. `{"fatalities": 6}  # 6 people were killed. 8 people were injured...`). The parser correctly extracts the JSON value in all cases, so this does not cause errors. It is nonetheless a meaningful behavioral difference: Llama behaves more like a chain-of-thought model than a format-compliant extractor.

### Seq2seq models (expected: number only)

| Model | Number only | Number + extra | No number |
|---|---|---|---|
| Flan-T5-base/large/XL-QLoRA | 100% | 0% | 0% |
| mT5-base, NT5-small | 100% | 0% | 0% |
| IndicBART | 0% | 99.9% | 0.1% |
| ConfliBERT-Poisson | 0% | 100% | 0% |

IndicBART outputs tokenizer artifacts around the number (e.g. `[CLS][CLS] 3[SEP]`); the digit regex in `extract_number()` recovers the correct value in all but one case. ConfliBERT-Poisson outputs raw floats (regression model), handled separately via the `model_type='regression'` path in `parse_prediction()`.

## Actual Parse Failures

Failures (outputs from which no count could be extracted) are extremely rare:

- **LLM pipeline:** 2 failures out of 3,436 predictions (0.06%) — the same incident echoed as input text by both Mistral-7B and Mixtral-8×7B. True label was 4; both recorded as 0.
- **Seq2seq pipeline:** 0 true failures. The one IndicBART edge case (word output `[CLS][CLS] two people were killed?...`) was correctly parsed to 2 via number-word lookup. True label was 3.

## The Silent Failure Issue

`parse_fatalities()` returns 0 without any signal when it fails. This makes the 2 LLM failures indistinguishable from genuine zero-death predictions in the saved results. The failures are too rare to affect any reported metric, but the silent behavior is worth noting as a measurement hygiene issue.

**Decision for subsequent experiments:** Before running the rare-bin improvement experiments, `parse_fatalities()` will be updated to return parse metadata alongside the prediction (at minimum: a `parse_success` flag and `parse_method` string). This allows failure cases to be tracked explicitly without changing the numerical results of the baseline comparison. No experiments will be conducted on the parser itself; it is treated as infrastructure to get right before the substantive analysis.

## Implications for the Model Comparison

The parser analysis does not invalidate the existing model comparison. The two parsers are each correctly specified for their pipeline. Format compliance is near-perfect for all models except Llama-3.1-8B (which complies in substance but appends commentary) and IndicBART (which wraps output in tokenizer tokens). Neither causes measurement errors in the current results.

The comparison between LLM and seq2seq pipelines is methodologically sound. The Flan-T5-XL entry in the LLM results is a special case: it is a seq2seq encoder-decoder model run through a decoder-LLM inference path with the wrong parser, producing catastrophically wrong metrics (overall MAE 5.8, bin-0 MAE 14.6). It should be excluded from the LLM comparison table and treated solely as the fine-tuned Flan-T5-XL-QLoRA entry in the seq2seq results.
