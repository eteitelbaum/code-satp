# Appendix

This appendix provides supplementary material for the main text. Appendix A presents representative hard cases that motivated the error taxonomy used in Sections 5 and 7. Appendix B documents the full prompting strategies used in the LLM intervention study. Appendix C summarizes parser reliability and format compliance. Appendix D presents supplementary bootstrap figures that support the main comparisons.

## Appendix A. Hard Cases Discussed in the Main Analysis

Table A1 lists the 12 hard cases that map most directly onto the diagnostic discussion in Section 5. The cases are grouped to mirror the argument in the text: two probable labeling errors, two revised-count cases where the gold label exceeds what is stated in the narrative, four cases involving unconfirmed attacker deaths, and four cases illustrating the Flan-T5-Large overcount pattern. This keeps the appendix aligned with the analysis actually developed in the paper while still documenting the specific incidents behind that discussion.

Abbreviations in the predictions column are as follows: `GPT` = GPT-4o-mini, `Llama` = Llama-3.1-8B, `Mixtral` = Mixtral-8x7B, `T5-XL` = Flan-T5-XL-QLoRA, and `T5-L` = Flan-T5-Large.

| Incident | Category | Gold | Predictions | Note | Excerpt |
|---|---|---:|---|---|---|
| 301140802 | Likely labeling error | 5 | GPT 0; Llama 0; Mixtral 0; T5-XL 0; T5-L 0 | Narrative describes wounds/injuries not deaths; all models correctly predict 0 | Five CRPF personnel were wounded in an ambush by armed Maoists in a forest in the Narayan… |
| 308241302 | Likely labeling error | 3 | GPT 0; Llama 0; Mixtral 0; T5-XL 0; T5-L 0 | Narrative describes injuries not deaths; all models correctly predict 0 | Three District Police personnel, including a Head Constable, were injured in an encounter… |
| 304061001 | Revised count (follow-up report) | 84 | GPT 76; Llama 76; Mixtral 76; T5-XL 76; T5-L 79 | Narrative: 75 CRPF + 1 SP = 76; label updated to 84 from a follow-up source (8 additional deaths) | 75 CRPF personnel and a State Policeman were killed in an attack by about 1000 CPI-Maoist… |
| 807021301 | Revised count (follow-up report) | 6 | GPT 5; Llama 5; Mixtral 5; T5-XL 5; T5-L 5 | Narrative: SP + 4 constables = 5; 6th death likely recorded in follow-up report | Pakur District SP Amarjit Balihar (45) and four other Policemen were killed in an ambush … |
| 1210080901 | Unconfirmed attacker deaths | 33 | GPT 18; Llama 18; Mixtral 18; T5-XL 21; T5-L 19 | Models count 18 confirmed police deaths; 15 Maoist deaths ('Police sources said… also killed') excluded by models | Cadres of the CPI-Maoist killed at least 18 Policemen, including Sub-Inspector C. S. Desh… |
| 1405191101 | Unconfirmed attacker deaths | 24 | GPT 16; Llama 16; Mixtral 16; T5-XL 24; T5-L 4 | Models count ~16 (explicit police + partial Maoist); '20+ Maoists killed' in two encounters undercounted; T5-XL-QLoRA correctly extracts 24 | Four Police personnel, including two SPOs and over 20 CPI-Maoist cadres were killed in tw… |
| 1402150801 | Unconfirmed attacker deaths | 18 | GPT 15; Llama 15; Mixtral 15; T5-XL 15; T5-L 15 | Models count 15 police+civilian; 3 Maoist deaths ('claimed… bodies not recovered') excluded by models | 14 Police personnel and a civilian were killed and four Policemen were wounded when aroun… |
| 802080801 | Unconfirmed attacker deaths | 14 | GPT 2; Llama 2; Mixtral 2; T5-XL 2; T5-L 2 | Models count 2 CRPF; 12+ Maoist deaths ('police sources claimed… bodies taken away') excluded by models | Two CRPF personnel were killed and four others injured during an encounter with cadres of… |
| 305171501 | T5 overcount / succumbed to injuries | 5 | GPT 5; Llama 5; Mixtral 5; T5-XL 5; T5-L 6 | Narrative includes 'succumbed to injuries'; T5-Large overcounts to 6; all other models correct at 5 | Three Policemen and two CPI-Maoist cadres were killed in an encounter in Bijapur District… |
| 1205300501 | T5-Large overcount (multi-group arithmetic) | 8 | GPT 8; Llama 8; Mixtral 8; T5-XL 8; T5-L 9 | T5-Large overcounts to 9; all LLMs correct; double-counts one group in multi-group enumeration | Seven Police personnel and a civilian are killed in a landmine blast triggered by the CPI… |
| 1204121301 | T5-Large overcount (multi-group arithmetic) | 7 | GPT 7; Llama 7; Mixtral 7; T5-XL 7; T5-L 9 | T5-Large overcounts to 9; all LLMs correct; double-counts one group in multi-group enumeration | Four CPI-Maoist cadres, a Policeman and two villagers were killed in an encounter between… |
| 306010501 | T5-Large overcount (multi-group arithmetic) | 6 | GPT 6; Llama 6; Mixtral 6; T5-XL 6; T5-L 7 | T5-Large overcounts to 7; all LLMs correct; double-counts one group in multi-group enumeration | Assistant Commander of the CRPF, R K Mishra, and five other CRPF personnel are killed whe… |

Taken together, these 12 cases capture the error structures emphasized in Section 5. Four are effectively irreducible because the narrative and gold label do not align. Four reflect the evidentiary asymmetry around claimed attacker deaths. The remaining four illustrate the recurring overcount pattern that is especially visible for Flan-T5-Large on difficult aggregation and delayed-death narratives.

## Appendix B. Full Prompting Strategy

This appendix records the exact prompt variants used in the LLM intervention study reported in Section 6. GPT-4o-mini used the same `L1` instruction text when that ceiling-reference variant was evaluated; there was no separate GPT-specific wording.

### B.1 L0 Baseline Prompt

The baseline prompt is a zero-shot extraction instruction with a strict JSON output schema.

```text
How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": <integer>}. If no fatalities are mentioned, use 0.

Text: {incident_summary}
Answer:
```

### B.2 L1 Protocol Clarification Prompt

This variant adds a single sentence aligning the model with the maximalist coding protocol used in the dataset by instructing it to include claimed attacker deaths.

```text
How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": <integer>}. If no fatalities are mentioned, use 0. Count all reported deaths on all sides, including claimed attacker casualties even if bodies were not recovered.

Text: {incident_summary}
Answer:
```

### B.3 L2 Bin-Balanced Few-Shot Prompt

This variant prepends one clear example from each count bin (`0`, `1`, `2`, `3-5`, and `6+`) to make the count range salient before the target narrative is processed.

```text
How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": <integer>}. If no fatalities are mentioned, use 0. Count all reported deaths on all sides, including claimed attacker casualties even if bodies were not recovered.

Text: Seven CRPF personnel were injured in a landmine blast in the Bijapur District.
Answer: {"fatalities": 0}

Text: CPI-Maoists killed one villager in Garwah district
Answer: {"fatalities": 1}

Text: CPI-Maoist cadres kill two farmers in the Rohtas District.
Answer: {"fatalities": 2}

Text: Three persons were killed and five others injured by the CPI-Maoist at Khaira village in Lakhisarai District.
Answer: {"fatalities": 3}

Text: Seven CPI-Maoist cadres were killed in a gun battle with SFs in Latehar District.
Answer: {"fatalities": 7}

Text: {incident_summary}
Answer:
```

### B.4 L3 Hard-Case Few-Shot Prompt

This variant uses four examples chosen to match the recurrent failure modes identified in the baseline diagnostics: multi-group arithmetic, claimed attacker deaths, and delayed deaths from injuries.

```text
How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": <integer>}. If no fatalities are mentioned, use 0. Count all reported deaths on all sides, including claimed attacker casualties even if bodies were not recovered.

Text: Three Maoists and a civilian were killed during an encounter at Bhejji locality in the Dantewada District.
Answer: {"fatalities": 4}

Text: Police claimed to have killed six cadres of the CPI-Maoist in an encounter at Bangudwa Naktaia hills in the Gaya District. The Deputy Superintendent of Police said that dead bodies of the slain Maoists could not be recovered from the encounter site as these were taken away by their colleagues.
Answer: {"fatalities": 6}

Text: Three troopers of CoBRA were killed and at least 15 others were injured in an encounter with CPI-Maoist cadres in Sukma District. Officials said while two Commandos had succumbed to bullet injuries on March 3, their colleague died on March 4. At least 15 others were injured.
Answer: {"fatalities": 3}

Text: Five security personnel, including two STF troopers, were injured when CPI-Maoist cadres ambushed a team of SFs in Sukma District. Police also claimed to have gunned down at least 15 Maoists in the encounter although no bodies were recovered from the spot.
Answer: {"fatalities": 15}

Text: {incident_summary}
Answer:
```

### B.5 L4 Combined Few-Shot Prompt

The combined prompt merges the five bin-balanced examples from `L2` with the four hard-case examples from `L3`, for a total of nine demonstrations.

```text
How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": <integer>}. If no fatalities are mentioned, use 0. Count all reported deaths on all sides, including claimed attacker casualties even if bodies were not recovered.

Text: Seven CRPF personnel were injured in a landmine blast in the Bijapur District.
Answer: {"fatalities": 0}

Text: CPI-Maoists killed one villager in Garwah district
Answer: {"fatalities": 1}

Text: CPI-Maoist cadres kill two farmers in the Rohtas District.
Answer: {"fatalities": 2}

Text: Three persons were killed and five others injured by the CPI-Maoist at Khaira village in Lakhisarai District.
Answer: {"fatalities": 3}

Text: Seven CPI-Maoist cadres were killed in a gun battle with SFs in Latehar District.
Answer: {"fatalities": 7}

Text: Three Maoists and a civilian were killed during an encounter at Bhejji locality in the Dantewada District.
Answer: {"fatalities": 4}

Text: Police claimed to have killed six cadres of the CPI-Maoist in an encounter at Bangudwa Naktaia hills in the Gaya District. The Deputy Superintendent of Police said that dead bodies of the slain Maoists could not be recovered from the encounter site as these were taken away by their colleagues.
Answer: {"fatalities": 6}

Text: Three troopers of CoBRA were killed and at least 15 others were injured in an encounter with CPI-Maoist cadres in Sukma District. Officials said while two Commandos had succumbed to bullet injuries on March 3, their colleague died on March 4. At least 15 others were injured.
Answer: {"fatalities": 3}

Text: Five security personnel, including two STF troopers, were injured when CPI-Maoist cadres ambushed a team of SFs in Sukma District. Police also claimed to have gunned down at least 15 Maoists in the encounter although no bodies were recovered from the spot.
Answer: {"fatalities": 15}

Text: {incident_summary}
Answer:
```

The logic of the prompt sequence is cumulative. `L1` attempts protocol alignment by instruction alone. `L2` adds count-range calibration. `L3` adds demonstrations of the hardest narrative structures. `L4` combines both forms of guidance and was the strongest open-source prompt variant in the final evaluation.

## Appendix C. Parser Reliability Analysis

The paper's main results depend on extraction quality rather than format artifacts only if parser failures are rare and separable from genuine model error. That condition is satisfied here. The count-extraction workflow uses different parsers for the two model families because the expected output formats differ. The LLM pipeline asks for JSON and first searches for a `"fatalities"` key, while the seq2seq pipeline expects plain numeric output and uses a simpler integer-oriented parser.

### C.1 Format Compliance

Observed format compliance was near-perfect for all models, with one notable behavioral difference: Llama-3.1-8B almost always emitted valid JSON followed by extra commentary rather than returning the requested JSON string alone. Because the JSON object appeared first, the parser recovered the intended value without difficulty.

#### Table C1. LLM output compliance

| Model | Strict JSON | JSON + trailing text | No JSON |
|---|---:|---:|---:|
| GPT-4o-mini | 100.0% | 0.0% | 0.0% |
| Mistral-7B | 99.8% | 0.1% | 0.1% |
| Mixtral-8x7B | 99.9% | 0.0% | 0.1% |
| Llama-3.1-8B | 0.0% | 100.0% | 0.0% |

#### Table C2. Seq2seq output compliance

| Model family | Number only | Number + extra text | No number |
|---|---:|---:|---:|
| Flan-T5 base/large/XL-QLoRA | 100.0% | 0.0% | 0.0% |
| mT5-base, NT5-small | 100.0% | 0.0% | 0.0% |
| IndicBART | 0.0% | 99.9% | 0.1% |

IndicBART's noncompliance is due to tokenizer artifacts around the predicted numeral rather than genuine uncertainty about the task. The digit extraction fallback recovers the correct count in all but one edge case.

### C.2 Parse Failures

Actual parse failures were extremely rare.

| Pipeline | Outputs evaluated | Parse failures | Failure rate | Notes |
|---|---:|---:|---:|---|
| LLM | 3,436 | 2 | 0.06% | Both failures were echoed-text outputs from Mistral-7B and Mixtral-8x7B on the same incident. |
| Seq2seq | evaluated seq2seq prediction outputs | 0 | 0.00% | No true parse failures observed. |

The main limitation of the original LLM parser was not unreliability but silence: when extraction failed, the parser returned `0` without recording that the zero came from a parse failure rather than from a model prediction. This has now been addressed in the code by allowing the parser to return metadata, including `parse_success`, `parse_method`, and `defaulted_to_zero_due_to_parse_failure`. The substantive results in the paper are unaffected because the failures are too rare to move any reported metric.

### C.3 Implication for the Main Results

The parser analysis supports two conclusions. First, the model comparisons in the paper are not driven by formatting artifacts. Second, prompt improvements in Section 6 should be interpreted as changes in extraction performance rather than changes in output compliance, because compliance is already near-ceiling in the baseline condition.

## Appendix D. Supplementary Bootstrap Figures

This appendix presents supplementary bootstrap figures that support the main statistical comparisons in the paper. Whereas the main text reports the key numeric results directly, the figures below visualize the uncertainty around those comparisons and make the significance patterns easier to inspect.

### D.1 Bootstrap Results for Baseline Model Comparisons

Figure D1 visualizes the bootstrap MAE comparisons across the main model families. It provides a compact view of the separation between the strongest generative models and the encoder baseline, while also showing where uncertainty overlaps among the best-performing seq2seq and LLM systems.

![Bootstrap MAE comparisons across model families](../../presentations/extraction-models-pssi/images/death-counts/bootstrap_mae_combined.png)

The corresponding overall MAE comparisons used in the main text are summarized below.

| Comparison | `p` vs ConfliBERT on overall MAE | Interpretation |
|---|---:|---|
| Flan-T5-Large vs ConfliBERT | 0.0000 | Significant improvement |
| Flan-T5-XL-QLoRA vs ConfliBERT | 0.0000 | Significant improvement |
| T5-Base vs ConfliBERT | 0.1052 | Not significant |
| mT5-Base vs ConfliBERT | 0.1132 | Not significant |
| NT5-Small vs ConfliBERT | 0.1628 | Not significant |
| Llama-3.1-8B vs ConfliBERT | 0.2220 | Not significant |
| Mistral-7B vs ConfliBERT | 0.5288 | Not significant |
| Mixtral-8x7B vs ConfliBERT | 0.5692 | Not significant |
| GPT-4o-mini vs ConfliBERT | 0.0000 | Significant improvement |

These results support the main paper's claim that generative models often improve point estimates substantially while only a subset of those improvements reach conventional significance on overall MAE.

### D.2 Bootstrap Results for Rare-Bin Interventions

Figures D2 and D3 report the bootstrap significance results for the seq2seq and LLM intervention studies respectively. Together they show the asymmetry emphasized in the main text: prompt-based interventions produce clearer gains in the LLM track than training-side interventions do in the seq2seq track.

![Bootstrap significance results for seq2seq models and interventions](../../presentations/extraction-models-pssi/images/death-counts/bootstrap_significance_seq2seq.png)

![Bootstrap significance results for LLMs and prompting interventions](../../presentations/extraction-models-pssi/images/death-counts/bootstrap_significance_llms.png)

For the LLM intervention study, three prompt variants improve overall MAE relative to the Llama-3.1-8B baseline at the `p < .05` threshold, while none is significantly different from the ceiling reference on overall MAE.

| Llama strategy | `p` vs Llama L0 on overall MAE | `p` vs ceiling on overall MAE | Interpretation |
|---|---:|---:|---|
| L1 | 0.1424 | 0.0224 | Worse than the ceiling; not a significant improvement over baseline |
| L2 | 0.0300 | 0.6576 | Significant improvement over baseline; not different from ceiling |
| L3 | 0.0424 | 0.3636 | Significant improvement over baseline; not different from ceiling |
| L4 | 0.0356 | 0.2748 | Significant improvement over baseline; not different from ceiling |

For the seq2seq intervention study, no training-side strategy improves overall MAE relative to the Flan-T5-Large baseline. The augmentation strategies (`S4` and `S5`) are significantly worse on overall MAE, which is consistent with the main text's conclusion that these interventions do not close the high-count gap even when they produce directional bin-level gains.

| Flan-T5-Large strategy | `p` vs S0 baseline on overall MAE | Interpretation |
|---|---:|---|
| S1 weighted sampling | 0.1204 | Not significantly different from baseline |
| S2 loss weighting | 0.6180 | Not significantly different from baseline |
| S3 targeted oversampling | 0.4800 | Not significantly different from baseline |
| S4 back-translation | 0.0428 | Significantly worse than baseline on overall MAE |
| S5 T5 paraphrase | 0.0308 | Significantly worse than baseline on overall MAE |
