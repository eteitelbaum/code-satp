# Improving Performance on Rare Bins

Building on this diagnostic taxonomy, we evaluate a series of targeted interventions designed to improve performance on the rare high-count bins. The interventions are organized around the two model families evaluated in the previous section. For the seq2seq models, we apply five training-side strategies to Flan-T5-Large, ranging from reweighting and oversampling to data augmentation through back-translation and paraphrase generation. For the LLMs, we apply four prompt engineering strategies to Llama-3.1-8B, progressively adding protocol clarification and few-shot demonstrations. T5-XL-QLoRA and GPT-4o-mini serve as ceiling references for the seq2seq and LLM results respectively. All interventions were evaluated against the validation set before the test set was touched, and the test set was evaluated once per finalized strategy.

## Seq2seq Training Interventions

The baseline Flan-T5-Large model miscodes roughly one in eight bin 3–5 events and nearly one in three bin 6+ events. We evaluate five training-side strategies designed to improve performance on these rare bins. Three modify how training examples are sampled or weighted during fine-tuning to compensate for the skewed count distribution. Two augment the rare-bin training data through paraphrase generation, increasing the surface-form diversity of high-count narratives without requiring additional annotation.

### Intervention Strategies

**Weighted sampling.** Standard fine-tuning samples training examples in proportion to their frequency, which means the model sees bin 3–5 and bin 6+ examples only rarely. Weighted random sampling addresses this by assigning each training example a weight inversely proportional to its bin frequency, so that rare-bin examples appear more often within each epoch [@buda2018]. We implement this using PyTorch's `WeightedRandomSampler`.

**Loss weighting.** Rather than changing which examples are sampled, loss weighting increases the training signal from rare-bin examples by assigning higher weights to their contributions to the cross-entropy loss [@he2009]. Weights are set inversely proportional to bin frequency, with conservative capping to prevent extreme weights from destabilizing training. This provides an alternative route to the same goal as weighted sampling through a different mechanism.

**Targeted oversampling.** Weighted sampling and loss weighting treat all rare-bin examples equally. Targeted oversampling instead selects specific examples matching the failure patterns identified in the previous section, such as narratives with multi-group enumeration and narratives where attacker casualties are described as claimed or unrecovered, and oversamples those at a higher rate. The rationale is that increased exposure to the exact narrative structures driving the errors may be more effective than upweighting rare bins.

**Back-translation augmentation.** Back-translation generates paraphrases of existing training examples by translating them into a pivot language and back into English, producing surface-form variation while preserving the underlying content [@sennrich2016; @halterman2025]. We apply back-translation to all rare-bin training examples using Hindi, Urdu, and Bengali as pivot languages, chosen because these languages preserve South Asian place names and group acronyms that European pivot languages tend to mangle.

**T5 paraphrase augmentation.** As an alternative to back-translation, we use Flan-T5-Large to generate paraphrases of rare-bin training examples directly in English [@chung2022; @parolin2022b]. This produces more diverse surface-form variation than back-translation but at the cost of greater risk that the paraphrase alters numerically critical content. The same rare-bin examples targeted by back-translation are augmented here, allowing a direct comparison of the two approaches.

### Results

Table 3 reports the results. The three reweighting strategies (weighted sampling, loss weighting, and targeted oversampling) all increase overall MAE relative to the baseline and none improves bin 3–5 or bin 6+ exact match. All three push the model toward higher count predictions on ambiguous narratives, increasing errors on the common low-count cases that dominate the test set without producing compensating gains in the rare bins. Back-translation improves on the baseline, raising bin 3–5 exact match from 86.7% to 88.0% while holding bin 6+ exact match at the baseline level, but the difference is not statistically significant. T5 paraphrase augmentation is directionally similar but weaker, returning bin 3–5 exact match to the baseline level. No strategy improves bin 6+ MAE, which remains near 4.0 for all interventions against a baseline of 1.97.

--Table 3 about here--

These results suggest that the bin 6+ gap between Flan-T5-Large (71.1% exact match) and T5-XL-QLoRA (84.2%) reflects a model capacity constraint rather than a data imbalance problem. The failure patterns identified in the previous section (multi-group enumeration, implicit totals, claimed casualties) require the model to aggregate counts across clauses and reason about what to include. These are mainly arithmetic reasoning tasks that scale with model capacity rather than training data volume. The T5-XL-QLoRA result, achieved without any rare-bin intervention, suggests the gap is closeable with a larger model, and that reweighting and augmentation cannot substitute for the capacity of a much larger model.

## LLM Prompt Engineering Interventions

The baseline Llama-3.1-8B model miscodes roughly one in ten bin 3–5 events and roughly one in five bin 6+ events. We evaluate four prompt engineering strategies designed to improve performance on these rare bins. One adds a protocol clarification to the system prompt, instructing the model to count claimed attacker casualties even when bodies were not recovered. Three vary the content and composition of few-shot demonstrations, progressively targeting the narrative structures most associated with high-count miscoding. GPT-4o-mini serves as the ceiling reference for the LLM track. The full text of the prompts is available in Appendix B.

### Intervention Strategies

**Protocol clarification.** The baseline prompt does not specify how to handle attacker casualties that are described as claimed or unrecovered. This strategy adds a single instruction to the system prompt directing the model to count all reported deaths on all sides, including claimed attacker casualties even when bodies were not recovered. This directly encodes the maximalist counting protocol established in the data collection design.

**Bin-balanced few-shot.** This strategy replaces the zero-shot baseline with five demonstrations sampled to cover the full range of death counts, including at least one example from each bin. The goal is to make the count range salient to the model before it reads the target narrative, without specifically targeting the failure patterns identified earlier.

**Hard-case few-shot.** Rather than sampling for bin coverage, this strategy selects four demonstrations that exemplify the specific narrative structures driving miscoding in the baseline, including events where attacker casualties are described as claimed or inferred and events where deaths must be summed across multiple armed groups. The examples are drawn from the validation set and excluded from evaluation.

**Combined few-shot.** This strategy combines the bin-balanced and hard-case example sets (nine demonstrations total) with the protocol clarification instruction. The bin-balanced examples provide count-range calibration, the hard-case examples anchor the difficult narrative structures concretely, and the protocol instruction aligns the model on what to count.

### Results

Table 4 reports the results. The protocol clarification strategy is counterproductive. Overall MAE rises from 0.235 to 0.318 and bin 6+ exact match falls from 78.9% to 76.3%. However, the few-shot strategies improve performance at every step. Bin-balanced demonstrations reduce overall MAE to 0.199 while holding bin 6+ exact match at the baseline level. Hard-case demonstrations reduce overall MAE further to 0.113 and raise bin 6+ exact match to 81.6%, the strongest rare-bin result among all Llama strategies. The combined strategy improves on all three metrics, with overall MAE falling to 0.107, bin 6+ exact match rising to 89.5%, and bin 3–5 exact match holding at 88.0%. All three figures nominally exceed GPT-4o-mini, though none of the differences are statistically significant.

-- Table 4 about here --

These results suggest that demonstrations rather than instructions are the active ingredient in few-shot prompting for this task. The protocol clarification instruction specifies what to count but provides no examples of how to handle the ambiguous cases that drive miscoding in the baseline, including claimed casualties, implied totals, and multi-group enumeration. Without anchoring examples, the model applies the instruction too broadly, increasing predicted counts on cases where deaths are genuinely uncertain. The hard-case demonstrations address uncertainty by showing the model how to handle each failure type, while the protocol instruction adds value within the combined prompt because the examples first establish when claimed casualties should be counted and when they should not.
