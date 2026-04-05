# S4/S5 Augmentation Strategy: Back-Translation and T5 Paraphrase

Notes for implementing S4 (back-translation) and S5 (T5 paraphrase) augmentation for
rare-bin improvement.

## Status

S1–S3 and S6 complete. S4/S5 diagnostic complete (April 2026). Both strategies cleared
the viability threshold for bin 3–5. Integration into
`models/count-models/death-count-extraction-seq2seq-rare-bin.ipynb` is the next step.

See `s4-s5-diagnostic-results.md` for full diagnostic results and pass-rate tables.

---

## S4 — Back-Translation

### Pivot languages

Hindi (`hi`), Urdu (`ur`), and Bengali (`bn`). The rationale: SATP incident summaries
contain Indian place names (district names, forest areas), group names (CPI-Maoist, CoBRA,
STF, CRPF), and narrative conventions from South Asian journalism. South Asian pivot
languages preserve these better than European pivots (French, German) which may mangle
or drop unfamiliar proper nouns.

**Diagnostic finding:** All three pivot languages perform identically (~82% overall,
88.3% for bin 3–5, ~69% for bin 6+). No language is preferable. Use all three.

### Implementation

The `BackTranslationAugmentation` class is in:
`models/classification-models/imbalance-handling/imbalance_handling_strategies.py`

It uses `deep-translator` (free Google Translate API, no key required). Requires internet
access on Colab. Rate-limit delay of 0.5s between calls is sufficient for standard Colab
runtimes; increase to 1.0s if translation errors appear in the output.

**Important:** instantiate one augmenter per pivot language to get deterministic
per-language output. The default `augment_text()` picks a random language from the list:

```python
augmenters_bt = {
    'hi': BackTranslationAugmentation(target_languages=['hi']),
    'ur': BackTranslationAugmentation(target_languages=['ur']),
    'bn': BackTranslationAugmentation(target_languages=['bn']),
}
```

Integration steps:
1. Filter training set to bins 3–5 and 6+ only
2. Run back-translation through all three pivot languages
3. Discard only empty/null augmented texts (see filter below)
4. Concatenate passing examples onto `train_df` before tokenization
5. Assign same bin weights as the originals (they belong to the same bin)
6. Combine with S1 (WeightedRandomSampler) for S1+S4 combined run

### Augmentation filter

Use a minimal null/empty filter only — do not apply a count-preservation check:

```python
def augmentation_valid(augmented_text: str) -> bool:
    return bool(augmented_text and isinstance(augmented_text, str) and augmented_text.strip())
```

**Rationale:** The diagnostic showed that ~18% of examples fail a count-preservation
check, but most failures are structural — multi-group arithmetic cases where the total
is implicit (e.g. "five police and two civilians killed", label=7). The numeral "7"
never appears in the original text either, so no augmentation method can pass the check
for these examples. These are exactly the hard cases the model needs to learn from.
Filtering them out makes the augmented training set *less* representative of difficult
examples. The true rate of genuine augmentation corruption (garbled text, wrong numbers)
is well below 5%. A null filter catches those without discarding good hard examples.

### Dependencies

Add to `models/count-models/requirements.txt` (already done):
```
deep-translator>=1.11.4
```

No additional requirements beyond what is already in count-models/requirements.txt.

---

## S5 — T5 Paraphrase

### Model

`google/flan-t5-large` — same model used in the classification paper experiments.
**Do not use `Vamsi/T5_Paraphrase_Paws`** — that model uses a different prompt format
(`"paraphrase: {text} </s>"`) incompatible with the prompt in `T5ParaphraseAugmentation`.

### Implementation

The `T5ParaphraseAugmentation` class is in:
`models/classification-models/imbalance-handling/imbalance_handling_strategies.py`

Call `paraphrase()` directly rather than `augment_rare_classes()` — the latter uses
`label_cols` designed for multi-label classification, not count bins:

```python
t5_aug = T5ParaphraseAugmentation(model_name="google/flan-t5-large")

paraphrases = t5_aug.paraphrase(
    text,
    num_return_sequences=2,
    seed=RANDOM_SEED + i
)
```

Apply the same `augmentation_valid` null/empty filter as S4 before adding to training data.

### Failure modes (from diagnostic)

| Failure type | Count (of 676) | Share |
|---|---|---|
| number_dropped | 93 | 13.8% |
| wrong_number | 44 | 6.5% |
| numeral_to_word | 0 | 0% |

- `number_dropped`: T5 rewrites at too high an abstraction level, losing numbers entirely.
  Increasing `max_new_tokens` reduces truncation-related cases.
- `wrong_number`: Two sub-causes — (a) multi-group arithmetic (same structural issue as
  S4); (b) incidental numbers in the text (ages, group sizes) replacing the count.
- `numeral_to_word`: Zero cases. Unlike back-translation, T5 does not convert "8" to
  "eight" — it either preserves numerals or drops them.

### Dependencies and version pinning

**Critical:** `T5ParaphraseAugmentation` uses the `text2text-generation` pipeline task,
which is broken in `transformers>=5.0.0`. Pin to `transformers==4.57.1`.

This is already pinned in `models/count-models/requirements.txt`. The classification-models
requirements.txt uses `transformers>=4.46` (unpinned) — do not install from that file for
this task. See `notes/dependency-pinning.md`.

**Runtime restart required:** Colab pre-installs transformers 5.0.0. Installing 4.57.1
via pip does not take effect until the runtime is restarted. The Colab setup cell in
`augmentation-diagnostics.ipynb` handles this automatically with `os.kill(os.getpid(), 9)`
— after the auto-restart, skip the setup cell and run from the imports cell down.

```
Device set to use cuda:0
🔍 Model: google/flan-t5-large
   - Using transformers version: 4.57.1  ← confirm this before running
```

The "pipelines sequentially on GPU" warning is harmless — ignore it.

### Additional dependencies

Add to `models/count-models/requirements.txt` (already done):
```
sentence-transformers>=2.2.2   # for similarity filtering in T5ParaphraseAugmentation
sentencepiece>=0.1.99          # for T5 tokenizer
```

---

## Integration into death-count-extraction-seq2seq-rare-bin.ipynb

### Where augmented data is added

Insert augmentation between data loading and tokenization — the same position as S3's
targeted oversampling. Add passing augmented examples to `train_df`, then proceed to
`prepare_seq2seq_data()` and tokenization as normal.

### Recommended combined strategy: S1 + S4

```python
# 1. Apply S4 back-translation augmentation
augmented_rows = []
for lang, aug in augmenters_bt.items():
    for _, row in rare_df.iterrows():
        aug_texts = aug.augment_text(row['incident_summary'], num_augmentations=1)
        aug_text = aug_texts[0] if aug_texts else None
        if aug_text and aug_text.strip():   # discard only empty/null outputs
            new_row = row.copy()
            new_row['incident_summary'] = aug_text
            augmented_rows.append(new_row)
        time.sleep(0.5)

aug_df = pd.DataFrame(augmented_rows)
train_df_augmented = pd.concat([train_df, aug_df], ignore_index=True)

# 2. Apply S1 WeightedRandomSampler on the augmented training set
# (augmented examples get same bin weights as their source examples)
```

### Reporting augmentation statistics

Log before training:
- n rare-bin examples before augmentation
- n augmented examples generated per language (i.e. non-null outputs)
- n total rare-bin examples after augmentation

This goes in the strategy results table and is useful for the methods section.

---

## Data column names (important)

The main dataset (`satp_clean.csv`) uses:
- `first_action` — not `action_type` — for filtering to Armed Assault / Bombing
- `incident_number` — not `incident_id` — as the row identifier

The val/test CSVs in `models/count-models/data/` also use `incident_number`.
