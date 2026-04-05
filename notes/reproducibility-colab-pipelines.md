# Reproducibility of Colab Training Pipelines

Notes for preparing this work for review or publication. Written April 2026.

## The Core Problem

Colab notebooks install the latest version of packages unless explicitly pinned. A
`requirements.txt` entry like `transformers>=4.35.0` resolves to whatever is current at
install time. When transformers 5.0 released (January 2025), all seq2seq training in the
death-count pipeline silently broke: training loss decreased normally but generation
collapsed to a single constant output (predicting 0 or 1 for every example, regardless
of input). The cause was a breaking change in how `GenerationConfig` interacts with
`Seq2SeqTrainer` and `predict_with_generate=True`, combined with possible Adafactor
optimizer behavior changes.

This was discovered in April 2026 when attempting to re-run the rare-bin experiments.
The fix was to pin `transformers==4.57.1`, recovered from the `transformers_version`
field in `config.json` of a saved fine-tuned model on Google Drive.

## What Was Pinned and Why

`models/count-models/requirements.txt` now pins `transformers==4.57.1` exactly. This is
the version under which all seq2seq death-count models (S0 baseline, and rare-bin
strategies S1–S3) were trained. Results are confirmed reproducible under this version.

The LLM notebooks (`death-count-extraction-llms.ipynb`,
`death-count-extraction-llms-rare-bin.ipynb`) use the OpenAI and Google Gemini APIs and
are not sensitive to the HuggingFace stack in the same way, but they share the same
`requirements.txt`.

## What Is Not Yet Pinned (Do This Before Submission)

The `requirements.txt` pins `transformers` but leaves transitive dependencies at lower
bounds (`datasets>=2.14.0`, `accelerate>=0.33`, etc.). These will resolve to whatever is
current on Colab at install time and could introduce breakage as those libraries evolve.
The most volatile are:

- `tokenizers` — the Rust-based tokenizer library, evolves fast
- `huggingface_hub` — controls model weight fetching and caching
- `accelerate` — deep integration with `Trainer`

**Action required before submission:** After a successful full run (all strategies
complete), add a cell to the notebook that captures the complete resolved environment:

```python
import subprocess
result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
with open(NEW_RESULTS / 'pip_freeze_colab.txt', 'w') as f:
    f.write(result.stdout)
print("Saved pip freeze to Drive")
```

Download `pip_freeze_colab.txt` from Drive and commit it to the repo as
`models/count-models/requirements.lock`. This is the definitive reproducibility artifact:
someone can recreate the exact environment from it regardless of what has changed in the
intervening years. Also record the Colab runtime type (e.g., A100, L4, T4) and Python
version in the notebook's opening cell.

## Versioning Notebooks

The rare-bin notebook (`death-count-extraction-seq2seq-rare-bin.ipynb`) has a version
counter (`NOTEBOOK_VERSION = "1.0"`) and a transformers version check printed at import
time. When re-running, verify the output reads:

```
transformers version: 4.57.1 (expected: 4.57.1)
```

A warning printed here means the environment is wrong and results will differ.

## The Broader Lesson

ML library versioning is semantically meaningful in a way that software libraries
generally are not — the same training code under a different library version can produce
a qualitatively different model. For publication, the version pin in `requirements.txt`
and a frozen `requirements.lock` together constitute the computational environment
description that reviewers and replicators need.

Papers routinely cite exact library versions in appendices. The minimum recommended set
for this pipeline:

- Python version (e.g., 3.11.x)
- `transformers==4.57.1`
- `torch` version (from pip freeze)
- `tokenizers` version (from pip freeze)
- CUDA version (from `torch.version.cuda`)
- GPU type (from `torch.cuda.get_device_name(0)`)

A one-cell block that prints all of these at the top of each notebook is worth adding
before submission.

## Timeline Expectations

- **1–2 years**: Pipeline runs as-is with `transformers==4.57.1` pinned.
- **2–4 years**: Transitive dependencies may require additional pins. The `requirements.lock` file would allow reconstructing the working environment.
- **4+ years**: Colab Python runtime version changes or major PyTorch version bumps could require a containerized environment (Docker) to guarantee exact reproduction. At that point, the `requirements.lock` + recorded Python/CUDA versions are the inputs needed to build such a container.
