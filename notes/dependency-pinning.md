# Dependency Pinning: Notes and TODOs

## The problem

The classification-models `requirements.txt` uses `transformers>=4.46`, an unpinned lower bound. When run on a fresh Colab session in April 2026 this resolved to `transformers==5.0.0`, which broke the `text2text-generation` pipeline task used by `T5ParaphraseAugmentation`. The count-models `requirements.txt` correctly pins `transformers==4.57.1` and is unaffected.

This is a reproducibility risk: anyone running the classification notebooks on a current Colab runtime will get transformers 5.x and may encounter failures.

## Current state

| requirements.txt | transformers pin | Status |
|---|---|---|
| `models/count-models/requirements.txt` | `==4.57.1` | Pinned — safe |
| `models/classification-models/requirements.txt` | `>=4.46` | Unpinned — at risk |

## TODO

1. Check which transformers version was actually used when the classification paper experiments ran (check Colab session history or notebook outputs for printed version numbers).
2. Pin `transformers` in `models/classification-models/requirements.txt` to that version.
3. Consider whether other unpinned packages in the classification requirements pose similar risks (`sentence-transformers>=2.2.2`, `accelerate>=0.30`, `datasets>=2.19`).
4. For the final paper artifact, consider producing a `pip freeze` snapshot of the full environment used for each set of experiments and including it in the repo.

## Broader guidance

For academic publications, pinning exact versions (`==`) is best practice — reproducibility requires that another researcher can reconstruct the exact environment. For industry, a two-layer system is common: loose bounds in a human-readable `requirements.in`, fully locked output in `requirements.txt` via `pip-compile` or `poetry lock`.

The practical tradeoff: pinning prevents surprise breakage but requires deliberate maintenance as package versions age off indexes. For a point-in-time paper artifact this is acceptable; for a long-running service it requires a maintenance plan.
