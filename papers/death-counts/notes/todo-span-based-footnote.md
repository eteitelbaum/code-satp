# TO-DO: Span-Based Architecture Footnote

## What this is

A footnote to be placed in Section 4 (Models and Evaluation), likely attached to the
first mention of the seq2seq framing, briefly noting the architectural limitation of
span-based extractive QA for count extraction tasks.

## What the footnote should say

Span-based extractive QA models (e.g. BERT fine-tuned for SQuAD-style tasks) predict
start and end token positions within the input, so the answer must appear as a verbatim
span. This fails by design when the correct count is an implicit aggregate — e.g.
"five police and two civilians were killed" with a label of 7, where the number 7
does not appear in the text. The architectural argument for seq2seq is therefore
independent of the empirical comparison with generative models. Cite @alsarra2025 for
an application of extractive QA to conflict texts using ConfliBERT variants.

## What NOT to claim

Do NOT claim Simon et al. (2025) provide empirical evidence that extractive models fail
on death counts. Their results show DEGREE (an input-anchored but BART-based model)
achieving 81.3% exact match on Deaths, matching Flan-T5-Large (81.0%). The
extractive-vs-generative comparison in Simon et al. is not a clean win for seq2seq.

## Placement decision

The generative paradigm is already the consensus in the nascent literature (Simon et al.,
Zhong et al.). The footnote should acknowledge the span-selection limitation briefly
without making it the main argument. Main text should focus on the two-track experimental
design, not on architectural justification.
