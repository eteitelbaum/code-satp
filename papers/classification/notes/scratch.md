# Scratch notes for conclusion / future sections

## Bias and validity (cut from intro)

Large language models carry systematic biases that can distort downstream analysis. Fatality estimates vary significantly depending on the language of the query [@steinert2025], and treating model-predicted labels as error-free leads to biased parameter estimates even when classification accuracy exceeds 90 percent [@egami2024]. These concerns are compounded by the nondeterministic nature of generative models, which poses challenges for replicability [@abdurahman2025]. Fine-tuning on human-coded data may mitigate some of these biases, but the extent to which it does so is an empirical question.

## Privacy and data sovereignty (cut from intro)

Researchers working with sensitive conflict data face privacy constraints that commercial APIs cannot satisfy. Only open-source models that can be fine-tuned and deployed locally address concerns about transmitting information on informants, ongoing operations, or vulnerable populations to third-party servers [@weber2024; @grossmann2023].

## Abandoned/defunded projects (cut from intro)

The field is littered with formerly ambitious coding projects that have scaled back or ceased operations as funding expired, including the Cingranelli-Richards Human Rights Dataset (CIRI) and, in its original manually supervised form, GDELT. Hybrid human-machine pipelines like the SPEED project attempted to combine the comparative advantages of each [@nardulli2015], and ensemble classifiers were proposed to pre-filter irrelevant source material before human coding [@croicu2015]. But no approach achieved the combination of accuracy, scalability, and affordability needed to make automated event coding routine.

## "Good enough" threshold (cut from intro)

Researchers in this space rarely confront a fundamental question: what level of automated classification performance justifies replacing human coders? The implicit standard is that higher accuracy is always better, but human inter-coder reliability is itself imperfect. Major datasets disagree at non-trivial rates on event classification for the same conflicts [@eck2012]. The appropriate benchmark for automated systems may not be perfection but parity with trained human coders on equivalent tasks.
