# Section 7: Discussion

## Target: 700–900 words

---

## Purpose
Synthesize findings into practical recommendations, address broader implications for the field, and note limitations. Avoid restating results; assume the reader has just finished §6. Focus on the "so what" for conflict researchers and the methodological takeaways for automated event coding.

---

## Outline

### 7.1 Practical Recommendations (≈200 words)
Lead with the actionable table, then brief prose on the key decision points:

| Scenario | Recommendation |
|---|---|
| No GPU; reproducibility required | Llama-3.1-8B + L4 (89.5% bin 6+ exact; no training cost) |
| Fine-tuning available; bin 3–5 critical | T5-Large + S1+S4 (best seq2seq on moderate counts) |
| Highest overall accuracy | GPT-4o-mini + L1 (best bin 6+ MAE: 0.368) |
| High-volume pipeline | T5-Large + S0 (best overall MAE; avoid reweighting) |

- For Llama-scale models: use demonstrations, not instructions alone; L1 without examples harms performance
- Fine-tuning is worth the cost for overall MAE but does NOT help bin 6+; if high-fatality events are the priority, L4 prompting is the better investment

### 7.2 Evaluation Design Implication (≈150 words)
- Bin-level decomposition is a methodological contribution, not just a reporting choice
- Simon et al.'s "deaths is easy" finding is a data-construction artifact (UCDP excludes zero-death events); aggregate MAE masks the rare-bin failure that this paper identifies
- Recommendation for the field: automated event coding papers should report distributional performance on count fields, not just aggregate MAE; skewed distributions make aggregate metrics misleading

### 7.3 Coding Protocol Transparency (≈150 words)
- The L1 asymmetry (helps GPT, harms Llama) makes explicit what prior work left implicit: automated systems apply an evidentiary standard that may not match the human coding protocol
- Different LLMs have internalized different evidentiary defaults — opaque unless explicitly tested
- Practical recommendation: specify which protocol the system follows (maximalist vs. conservative), ground it in the data (as L1 is grounded in component-column analysis), and verify on a validation set before deployment
- Automated systems enter the ACLED vs. UCDP methodological debate whether researchers intend it or not — better to make the choice explicit

### 7.4 Limitations (≈200 words)

**Single corpus:** All results from the Indian Maoist conflict / SATP; generalizability requires validation on other conflicts and reporting conventions. Counter: the focused corpus enables diagnostic precision — the goal is understanding *why* models fail, not building a general-purpose system.

**Underpowered bin-level tests:** Bin 6+ n=38 → effects as large as 13pp are not statistically distinguishable; a larger test set with stratified rare-bin sampling would allow sharper inference on intervention effects.

**LLM version drift:** Results tied to specific model versions; API models update without notice; pin versions in replication packages.

**Per-side decomposition not analyzed:** Component columns (security/maoist/civilian) are used to verify protocol but not as separate extraction targets — prevents direct testing of the evidentiary asymmetry hypothesis and direct comparison with UCDP-AEC's per-field schema.

### 7.5 Future Work (≈100 words, 4–5 bullets)
- Per-side decomposition: test evidentiary asymmetry hypothesis quantitatively; compare with UCDP-AEC 6-field schema
- Other count fields: injuries, arrests, surrenders — same distributional challenge; same framework applies
- Larger seq2seq models: does T5-XL full fine-tuning close the bin 6+ gap? Capacity interpretation predicts yes
- Chain-of-thought prompting: intermediate steps (identify groups → count per group → sum) for multi-group cases
- Low/high bound estimation: analogous to UCDP Deaths Low / Deaths High; adds uncertainty quantification compatible with UCDP methodology

---

## Key Citations
- Steinert 2025 (JPR) — LLM instability on fatality estimates; motivates protocol transparency
- Northcutt et al. 2021 — annotation quality auditing
- Simon et al. 2025 (UCDP-AEC) — "deaths is easy" artifact; aggregate metric limitation
- UCDP/ACLED methodology — protocol comparison
- Parolin et al. 2022 (Confli-T5) — augmentation precedent

## Tone Notes
- Keep practical recommendations concrete — name specific models and strategies, not just principles
- The aggregate-MAE critique of Simon et al. should be collegial, not adversarial — frame as a limitation of evaluation design that this paper addresses, not a flaw in their work
- The coding protocol discussion should be framed as a contribution to an ongoing methodological debate in conflict studies, not just an NLP finding
- Limitations section: be honest about single-corpus generalizability; but lean on the diagnostic precision argument to show this is a feature not a bug for the paper's main contribution
