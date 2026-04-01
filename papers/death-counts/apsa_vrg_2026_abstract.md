# APSA Virtual Research Group 2026 Abstract

**Workshop:** Aligning Computational Tools for the Political Science Research Lifecycle
**Date:** April 15-16, 2026
**Submission deadline:** February 25, 2026

From Narratives to Numbers: Evaluating Automated Fatality Count Extraction for Conflict Event Data

Fatality counts are central outcome variables in quantitative conflict research, yet extracting them accurately from narrative event reports is a non-trivial task. A single summary may mention multiple numeric values, conflate deaths with injuries or arrests, and use indirect language. High-fatality events are rare but analytically consequential while aggregate accuracy metrics can mask systematic failures. We examine how well automated models can recover death counts from conflict event narratives using hand-coded incident summaries from the South Asia Terrorism Portal on the Maoist insurgency in India, comparing fine-tuned sequence-to-sequence transformer models against prompted large language models and a supervised regression baseline (ConfliBERT).

We make three contributions. First, the best fine-tuned seq2seq models perform comparably to proprietary LLMs and both substantially outperform the regression baseline, suggesting that fine-tuned open-weight models are viable substitutes in resource-constrained research settings or when data sensitivity prevents the use of external APIs. Second, aggregate error metrics conceal systematic failures on high-fatality events. We propose bin-level reporting as a standard complement and evaluate the ability of targeted remediation strategies, including oversampling of rare bins and data augmentation, to improve tail performance without degrading accuracy on low-fatality events. Third, we identify potential parsing failures as a unique reliability problem distinct from model accuracy, and propose a diagnostics framework covering format compliance, parse success rates, and failure modes as part of a broader set of reporting standards for extraction pipelines.
