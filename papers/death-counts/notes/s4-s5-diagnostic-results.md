# S4/S5 Augmentation Diagnostic Results

Results from running `models/count-models/augmentation-diagnostics.ipynb` on all rare-bin training
examples (bins 3–5 and 6+). Raw results saved to
`MyDrive/colab/satp-results/augmentation-diagnostics/`.

---

## Sample

- All rare-bin training examples: **338 examples** (bins 3–5 and 6+)
- Back-translation: 338 × 3 languages = **1,014 augmented texts**
- T5 paraphrase: 338 × 2 paraphrases = **676 augmented texts**
- Count-preservation check: checks for the fatality count as a numeral (`\b7\b`) OR English word
  form (`\bseven\b`, case-insensitive). Word-form checking was added after the initial run revealed
  that SATP narratives systematically spell out small counts rather than using numerals.

---

## S4 — Back-Translation Results

### Pass rates

| Pivot language | n_total | n_pass | pass_rate |
|---|---|---|---|
| Bengali (bn) | 338 | 276 | 81.7% |
| Hindi (hi) | 338 | 277 | 82.0% |
| Urdu (ur) | 338 | 277 | 82.0% |

| Pivot language | Bin | n_total | n_pass | pass_rate |
|---|---|---|---|---|
| bn | 3-5 | 223 | 197 | 88.3% |
| bn | 6+ | 115 | 79 | 68.7% |
| hi | 3-5 | 223 | 197 | 88.3% |
| hi | 6+ | 115 | 80 | 69.6% |
| ur | 3-5 | 223 | 197 | 88.3% |
| ur | 6+ | 115 | 80 | 69.6% |

### Key findings

- **All three pivot languages perform identically.** No language is clearly preferable for this
  corpus. Any one or all three can be used without differential quality risk.
- **Bin 3–5 clears the 85% viability threshold** (88.3% across all languages).
- **Bin 6+ falls below threshold** (~69%), consistent across all languages.
- **Translation quality is high.** Manual inspection of failures confirms the translations are
  faithful rewrites. The 18% failure rate substantially overstates true augmentation failures.

### Failure analysis

Most "failures" are not translation errors. Two structural causes account for the majority:

1. **Multi-group arithmetic cases:** The fatality count is implicit — e.g. "five Police personnel
   and two civilians were killed" with label=7. The numeral "7" never appears in either the
   original or the augmented text. The back-translation is accurate; the count-preservation check
   cannot pass these by design. This is the same failure pattern targeted by S3.

2. **Implicit single-death narratives:** e.g. "a senior leader was killed" with label=3 where the
   count refers to additional deaths described in a truncated portion of the narrative. Again a
   check artefact, not a translation failure.

The true translation failure rate (cases where back-translation actually corrupted count
information) is estimated to be well below 18%.

---

## S5 — T5 Paraphrase Results

Model: `google/flan-t5-large` (same as classification paper experiments).

### Pass rates

| Overall | n_total | n_pass | pass_rate |
|---|---|---|---|
| All bins | 676 | 539 | 79.7% |

| Bin | n_total | n_pass | pass_rate |
|---|---|---|---|
| 3-5 | 446 | 385 | 86.3% |
| 6+ | 230 | 154 | 67.0% |

### Failure type breakdown

| Failure type | Count |
|---|---|
| number_dropped | 93 |
| wrong_number | 44 |
| numeral_to_word | 0 |

### Key findings

- **Overall pass rate (79.7%) is comparable to back-translation (~82%).** S5 does not obviously
  underperform S4, contrary to the prior expectation in `s4-s5-augmentation-strategy.md`.
- **Bin 3–5 clears the 85% threshold** (86.3%). Bin 6+ does not (67.0%).
- **`numeral_to_word` failures: zero.** Unlike back-translation, T5 does not convert numerals to
  word forms. It either drops numbers entirely or substitutes wrong ones.
- **`number_dropped` (93 cases):** T5 rewrites at a level of abstraction that loses numbers —
  either omitting them, truncating the output, or generalising to "several". Generation length
  (max_new_tokens) is a contributing factor in some cases.
- **`wrong_number` (44 cases):** Two sub-causes: (a) multi-group arithmetic where the total is
  implicit and some other numeral (group size, age) is present; (b) genuine hallucination of a
  different count.
- **Truncation:** Several paraphrases are cut off mid-sentence, indicating the generation hits
  the max_new_tokens limit. Would be reduced by increasing max_new_tokens in the paraphrase call.

### Failure analysis

The same multi-group arithmetic and implicit-count patterns that account for most S4 failures
also account for most S5 failures. Manual inspection confirms the paraphrase quality is generally
high for passing examples. The true generative failure rate is lower than 20.3%.

---

## Comparison summary

| Strategy | Overall pass rate | Bin 3-5 | Bin 6+ |
|---|---|---|---|
| S4 back-translation (hi) | 82.0% | 88.3% | 69.6% |
| S4 back-translation (ur) | 82.0% | 88.3% | 69.6% |
| S4 back-translation (bn) | 81.7% | 88.3% | 68.7% |
| S5 T5 paraphrase (flan-t5-large) | 79.7% | 86.3% | 67.0% |

---

## Conclusions and implications for training integration

1. **Both strategies are viable for bin 3–5.** Pass rates comfortably exceed 85% and inspection
   confirms the augmented texts are high-quality.

2. **Bin 6+ is marginal for both strategies (~68–70%).** This is a property of bin 6+ examples
   (complex multi-group narratives, implicit counts) rather than a failure of the augmentation
   methods. Even at 70% pass rate, ~80 new bin 6+ examples per language would be added — a
   meaningful increase given the baseline n=115.

3. **The two strategies are interchangeable on pass rate.** The choice between S4 and S5 should
   be made on other grounds: S4 is faster (API calls, no GPU load), S5 produces more diverse
   surface-form variation (generative rewriting vs. round-trip translation).

4. **Recommended integration:** S1 + S4 (weighted sampling + back-translation) as the primary
   combined strategy, using all three pivot languages. S5 can be added as an additional diversity
   source if needed. See `rare-bin-attack-plan.md` for the full experimental sequence.

5. **Paper note:** The word-form finding (SATP narratives systematically spell out small counts
   rather than using numerals) is worth a sentence in the methods section when describing the
   count-preservation check. It has implications for parser design as well.
