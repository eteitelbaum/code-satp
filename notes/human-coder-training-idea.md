# Research Idea: Gold-Standard Training for Human Coders

## The Idea

Political science coding projects measure intercoder reliability as an endpoint —
do two coders agree? — but typically lack a principled process for using expert-coded
gold-standard labels to *improve* coder performance iteratively over time.

The proposal: treat human coders as learnable models. Use expert-coded gold-standard
items as training signal, give coders feedback on their errors, and track a learning
curve across rounds — analogous to how a supervised model is trained on labeled data.
This is distinct from standard intercoder reliability, which measures agreement at a
point in time rather than using disagreement as a training signal.

## Motivation

Emerged from a specific problem in the SATP death-count dataset: the coding manual
gave no instructions on how to handle claimed vs. confirmed attacker deaths (bodies
not recovered), so coders followed an implicit "record as reported" default. This
created an epistemologically ambiguous ground truth that only became visible when
model errors revealed it. Better coder training on edge cases — guided by expert
gold-standard labels — might have surfaced and resolved this ambiguity before the
dataset was built.

## Adjacent Literatures to Check

- **Crowdsourcing/MTurk**: "gold standard" trap items embedded in annotation tasks
  are used for quality control (flagging/removing bad workers) but not iterative
  training. See Northcutt et al. (2021) confident learning for the related problem
  of identifying label errors via model-human disagreement.
- **NLP annotation projects**: Large projects (OntoNotes, SuperGLUE) have training
  phases with pre-labeled practice items and feedback, but these are treated as
  onboarding rather than a principled iterative learning process.
- **Survey methodology**: Calibration studies and interviewer training have some
  structural similarity.
- **Active learning**: Selects informative examples for human annotation but does
  not use the human coder's error rate as a training signal for the coder.

## The Gap (Tentative)

No political science methods paper, to my knowledge, explicitly frames human coders
as learnable models and proposes an iterative gold-standard feedback protocol with
a tracked learning curve. If this gap is real, the contribution would be both
methodological (the training protocol) and empirical (showing that coder error rates
decline across rounds on specific edge-case categories).

## Next Steps

- Do a proper literature search before investing further: political science
  methodology journals, computational social science, and NLP annotation literature.
- Key search terms: "annotator training," "gold standard annotation," "coder
  calibration," "human-in-the-loop annotation," "annotation quality improvement."
- The claim "this hasn't been done" could turn on one paper — verify before committing.
