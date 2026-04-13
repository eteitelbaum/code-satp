# Models and Evaluation Framework

To evaluate model performance across the full range of the count distribution, I implement a two-track experimental design. The first track fine-tunes a family of seq2seq models directly on the SATP training data. The second evaluates instruction-tuned large language models through prompt engineering, without updating model parameters. Both tracks are evaluated on the same held-out test set and assessed against a shared encoder baseline that combines a conflict-domain pretrained model with a Poisson regression head. The evaluation combines aggregate metrics with a bin-level decomposition across five count bins (0, 1, 2, 3–5, and 6+) to assess performance on rare high-fatality events.

## Seq2seq Models

I evaluate six seq2seq models plus a ceiling reference, all available via the HuggingFace Model Hub. Each incident summary is formatted as a question-answering input with the prompt "How many people were killed? Answer with only a number." prepended to the text.

- *Flan-T5-Large* [@chung2022]: A 770M parameter encoder-decoder model instruction fine-tuned on a mixture of supervised tasks. This is the primary model in the seq2seq track.
- *T5-Base* and *T5-Large* [@raffel2020]: Standard encoder-decoder comparators at 220M and 770M parameters respectively.
- *IndicBART* [@dabre2022]: An encoder-decoder model pretrained on 11 Indic languages and English, evaluated for potential advantages on South Asian named entities and place names.
- *mT5-Base* [@xue2021]: A multilingual T5 variant pretrained on 101 languages at 300M parameters.
- *NT5-Small* [@zhong2023]: A numeracy-augmented T5 variant with a pretraining objective designed to improve numeric reasoning.
- *T5-XL-QLoRA* [@dettmers2023]: A 3B parameter T5 model fine-tuned using quantized low-rank adaptation, included as a ceiling reference.

All models were fine-tuned using the Adafactor optimizer with a learning rate of $3 \times 10^{-5}$, a batch size of 8, and a maximum of 10 epochs with early stopping patience of 2. Input sequences were tokenized to a maximum of 512 tokens. Generated outputs were capped at 16 tokens, sufficient to accommodate any plausible fatality count. The best checkpoint was selected based on validation loss, with a fixed random seed of 42 across all experiments.

## Decoder Models

I evaluate three open-source instruction-tuned decoder models plus a proprietary ceiling reference. All models are evaluated in prompting mode on the same held-out test set, with no parameter updates.

- *Llama-3.1-8B-Instruct* [@dubey2024]: An 8B parameter decoder model instruction-tuned for conversational and task-following use. This is the primary model in the LLM track.
- *Mistral-7B* [@jiang2023mistral] and *Mixtral-8×7B* [@jiang2024mixtral]: Comparator models at 7B and 56B effective parameters respectively.
- *GPT-4o-mini* [@openai2023]: Proprietary API model included as a ceiling reference.

All models receive the same baseline prompt (L0):

> How many people were killed? Answer with only a number. Return JSON exactly as: {"fatalities": \<integer\>}. If no fatalities are mentioned, use 0.

Generation was capped at 48 tokens. Responses were parsed by extracting the integer value from the JSON field; responses that could not be parsed were mapped to 0.^[Prompt non-compliance rates were low across all models; see the supplementary materials for the full parser reliability analysis.]

## Encoder Baseline

To contextualize the generative model results, I also evaluate an encoder-based baseline combining ConfliBERT with a Poisson regression head. ConfliBERT [@hu2022] is a BERT-base model pretrained on a large corpus of political violence and conflict-related text, making it the natural encoder-side reference point for tasks in this domain. A linear regression head is appended to the encoder's sequence-level representation and trained to predict the log-rate of a Poisson distribution. At inference, predictions are exponentiated and rounded to the nearest non-negative integer. Training followed the same procedure as the seq2seq models except that the standard AdamW optimizer was used with a learning rate of $2 \times 10^{-5}$. This baseline anchors all statistical significance tests reported for both tracks.

## Evaluation Framework

Performance is evaluated using mean absolute error (MAE) as the primary metric, supplemented by root mean squared error (RMSE), within-1 accuracy, within-2 accuracy, and nonzero MAE.^[mae] All metrics are reported both overall and separately across five count bins (0, 1, 2, 3–5, and 6+). Bins 0 and 1 together account for the majority of test incidents, so a model that fails on rarer bins can still appear competitive in aggregate. Confidence intervals are obtained via paired bootstrap resampling with $n = 5{,}000$ replications.

^[mae]: An alternative is token-level F1 (as used by @zhong2023) which awards partial credit for string overlap between predicted and gold outputs. For integer count predictions, MAE better captures prediction quality by measuring error magnitude directly.

Because baseline models fail systematically on rare high-fatality bins, I evaluate two classes of targeted intervention. For the seq2seq track, five training-side strategies are applied to Flan-T5-Large: weighted sampling (S1), loss weighting (S2), targeted oversampling (S3), back-translation augmentation via South Asian pivot languages (S4), and T5 paraphrase augmentation (S5). For the prompting track, four strategies are applied to Llama-3.1-8B: protocol clarification (L1), bin-balanced few-shot examples (L2), hard-case few-shot examples (L3), and a combined prompt (L4). Both classes of intervention are described alongside their results in §6.
