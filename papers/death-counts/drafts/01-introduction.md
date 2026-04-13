# Section 1: Introduction

## Target: 500–700 words

---

## Purpose
Motivate the problem, establish the gap in conflict studies, and preview the paper's contributions. Lead with the stakes for conflict measurement research, not with NLP methodology.

---

## Outline

### 1.1 Opening hook + gap statement (≈150 words)
- Conflict researchers increasingly rely on automated event coding; transformer-era tools (ConfliBERT, IndiaPoliceEvents) now code event types, actors, locations reliably at scale
- But numeric count extraction — how many people died — has not received the same treatment
- GDELT's V2COUNTS is the only automated count in wide use; acknowledged as noisy and unvalidated
- Recent abstractive event extraction (Simon et al. UCDP-AEC; LEMONADE) includes fatality counts but buries them in aggregate metrics and excludes zero-count events by design
- **The gap:** no paper has treated conflict event death count extraction as the primary research problem

### 1.2 Why it matters: the rare-event problem (≈150 words)
- 71% of incidents in our corpus have zero deaths; 2% involve six or more — but high-fatality events drive conflict severity metrics, escalation codings, and policy attention
- Getting high-count events wrong is not a marginal error; it is a systematic bias in automated conflict measurement
- Steinert (2025, JPR): naive LLM prompting for fatality counts is unreliable and language-biased — existing tools are not a solution

### 1.3 This paper (≈200 words)
- ~10,000 SATP incidents; two parallel tracks: seq2seq fine-tuning (Flan-T5-Large; T5-XL-QLoRA ceiling) and LLM prompting (Llama-3.1-8B; GPT-4o-mini ceiling)
- Methodological contribution: **bin-level evaluation** decomposes overall MAE into performance across the count distribution — reveals systematic failure that aggregate metrics mask
- Systematic **intervention study** on both tracks targeting rare high-count bins
- Findings preview (4 bullets max):
  - Few-shot prompting (L4) closes the bin 6+ gap for Llama: 59% MAE reduction, matches GPT-4o-mini
  - Training interventions help T5-Large on bin 3–5 (back-translation S4) but cannot close bin 6+ — a capacity problem, not a data problem
  - Instruction-following asymmetry (L1 helps GPT, harms Llama) has practical model-selection implications
  - A three-category error taxonomy identifies the irrecoverable hard floor and motivates each intervention

### 1.4 Roadmap (≈50 words, can be a single sentence or brief list)
- §2 Related work → §3 Data → §4 Models → §5 Baseline results + diagnostics → §6 Rare-bin interventions → §7 Discussion

---

## Key Citations
- Steinert 2025 (JPR); Simon et al. 2025 (UCDP-AEC); LEMONADE; Halterman 2021; ConfliBERT; GDELT V2COUNTS

## Tone Notes
- Lead with conflict measurement stakes, not NLP methodology
- Zhong et al. 2023 (EACL): mention briefly here or in §2 as closest NLP precedent; distinguish by domain and what this paper adds
- Keep the roadmap to one sentence or a compact list — don't over-explain the structure
