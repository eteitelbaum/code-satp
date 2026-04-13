# Section 8: Appendix

---

## Purpose
Supplementary materials supporting the main text. Four sections covering hard cases,
full prompting strategy, parser reliability, and ancillary results. Material for each
section already exists in various forms; this document tracks what needs to be assembled
and formatted.

---

## Appendix A — Hard Cases

Cases where all models fail systematically, illustrating the limits of automated
extraction regardless of architecture.

**Content:** A table of representative hard cases with incident text, true label, and
model predictions. Cases organized by failure category: annotation errors, unconfirmed
attacker deaths (epistemic discounting vs. protocol gap), and revised or delayed counts.

**Status:** Source material in hard_cases_appendix.csv and associated analysis notes.
**To do:** Format as a readable appendix table with brief narrative framing. Confirm
failure categories match the error analysis discussion in §7.

---

## Appendix B — Full Prompting Strategy

Complete documentation of all LLM prompts, enabling full reproducibility without
API access.

**Content:**
- L0 baseline prompt (already quoted in §4)
- L1 protocol clarification prompt (full text)
- L2 bin-balanced few-shot prompt (full prompt with all 5 examples)
- L3 hard-case few-shot prompt (full prompt with all 4 examples)
- L4 combined prompt (full 9-shot prompt)
- GPT-4o-mini L1 variant if text differs from Llama L1

**Status:** All prompt text is in models/count-models/utils/llm_utils.py.
**To do:** Extract and format for the appendix. Add brief framing sentence for each
variant noting its design rationale.

---

## Appendix C — Parser Reliability Analysis

Analysis of prompt compliance rates and parsing failures across models and strategies.
Establishes that parsing failures are not a significant source of error and are
distinguishable from genuine model prediction errors.

**Content:** Brief prose summary of findings. Compliance rate table by model if
available.

**Status:** Analysis exists in notes/parser-analysis.md and
notes/parser-reliability-evaluation.md.
**To do:** Summarize key findings. The footnote in §4 and any reference in §7
point here.

---

## Appendix D — Ancillary Results (provisional)

Figures and tables that support the main results but are too detailed for the main
body.

**Candidates:**
- Bootstrap confidence interval plots for overall MAE comparisons
- Per-bin CI plots for rare-bin intervention results
- Full results table including within-1 and within-2 metrics

**Status:** Provisional. Confirm which results belong here once §5 and §6 drafts
are complete. Some CI plots may need to be generated.

---

## Cross-references
- §4 footnote on parser reliability → Appendix C
- §6 rare-bin intervention prompts → Appendix B
- §7 error analysis hard cases → Appendix A
- Appendix D cross-references TBD once main results sections are drafted
