# Paper revision to-do list

## Introduction (done)
- [x] Rewrite intro to frame three debates: encoder vs decoder, rare categories, domain-specific pretraining
- [x] Foreshadow findings without over-detailing

## Theory / literature review (drafted, needs revision)
- [x] Reduce repetitive citations across intro and theory section (several refs appear in both)
- [x] Add CS literature on how encoders and decoders are actually trained (MLM, CLM, attention mechanisms) to enrich theory beyond social scientists' characterizations

## SATP dataset section (needs expansion)
- [x] Discuss the ontology of our coding scheme and its relationship to the GTD ontology
- [x] Discuss lexical complexity of categories (some lexically distinctive, others diffuse)
- [x] Add 1-2 example incident descriptions to give the reader a flavor of the data
- [x] Add Fetzer's earlier work on automating SATP conflict coding using NLP to the history paragraph

## Methods section (needs expansion and renaming)
- [x] Rename section (something other than "Methods")
- [x] Review modeling code and explain key choices in more detail (hyperparameters, training procedure, evaluation metrics, GPT prompting strategy, imbalance handling implementation)

## Discussion section (new, before conclusion)
- [ ] Synthesize key insights from results
- [ ] Incorporate scratch.md material: bias/validity, privacy and data sovereignty, open vs closed-source models
- [ ] Practical implications: decision tree for researchers (no labeled data → generative models → is .9 F1 good enough?; some labeled data → fine-tuned encoders overtake at ~1k examples; etc.)
- [ ] Field implications: still need funding for annotations, but fewer annotations may suffice; case for smaller targeted annotation efforts rather than massive coding projects
- [ ] Abandoned/defunded projects as motivation for efficient pipelines
- [ ] "Good enough" threshold: when does automated coding justify replacing human coders?

## Abstract and title
- [x] Update to reflect broadened scope

## Rerender with higher reps
- [x] Before sending out be sure to increase reps for bootstrapping in YAML config

## Follow-up
- [ ] Contact Rebecca Cordell about her APSA panel paper (unpublished, potentially relevant)
