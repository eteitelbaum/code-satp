# Death Count LLM Notebook Review Notes

Working notes from reviewing `models/count-models/death-count-extraction-llms.ipynb` and related utilities.

## Notebook Scope and Workflow

- `death-count-extraction-llms.ipynb` is a prompt-based inference/evaluation notebook, not a training notebook.
- It runs multiple LLMs (open-source and API models) on SATP incident summaries and evaluates extracted death counts.
- It includes an optional T5/Flan-T5 prompting path (`load_t5` + `run_t5_batch`) inside the LLM notebook, but that is still inference, not fine-tuning.
- `EVAL_SPLIT` toggles which held-out split to evaluate on (`val` vs `test`) for experimentation discipline, not train/test fitting.

## Utilities Import Behavior

- `from utils import ...` pulls names re-exported by `models/count-models/utils/__init__.py`.
- The imported names are functions defined inside module files (mainly `llm_utils.py` and `metrics_utils.py`), not separate files.
- `llm_already_done` is an alias for `already_done` from `llm_utils.py`.
- `get_task_results_dir` is in `utils/file_io.py` and is imported directly (`from utils.file_io import ...`) because it is not re-exported from `utils/__init__.py`.

## llm_utils.py Usage Notes

- Importing selected functions from `utils` still causes `llm_utils.py` top-level code to execute because `utils/__init__.py` imports from it.
- Module-level constants in `llm_utils.py` (e.g., `INSTR`, `USE_T5_FEWSHOT`) are used internally by utility functions, even if not directly imported into the notebook namespace.
- `make_input(...)` builds the prompt string only; it does not get the model answer.
- Model runners (`run_causal_batch`, `run_openai_batch`, `run_gemini_batch`, `run_t5_batch`) call prompt builders internally, send prompts to models, and return raw outputs.

## Prompting Design (Seq2Seq vs LLM)

- `make_input(...)` is a JSON-oriented prompt for decoder/API LLMs.
- `make_input_t5(...)` is a simpler prompt intended for T5-family seq2seq models in prompt mode.
- `prepare_seq2seq_data(...)` in `data_utils.py` is separate from `make_input_t5(...)`; it prepares training/fine-tuning inputs and targets, not LLM notebook inference prompts.
- Prompt wording is duplicated across `llm_utils.py` and `data_utils.py` (acceptable for now, but a future refactor could centralize templates to reduce drift).

## Few-Shot Prompting (T5 in LLM Notebook)

- A T5 few-shot prompting mechanism exists in `llm_utils.py`:
  - `set_t5_fewshot(...)`
  - `make_input_t5_fewshot(...)`
  - `run_t5_batch(...)` uses `USE_T5_FEWSHOT` to switch prompt style
- In `death-count-extraction-llms.ipynb`, the T5/Flan-T5 block (including the few-shot toggle) is commented out.
- The commented template sets `T5_FEWSHOT = False` by default.
- No repo-wide evidence was found of `set_t5_fewshot(True)` being used in `models/count-models`.

## Evidence From Saved Results

- `papers/death-counts/results/death-counts-llms/` contains one saved result artifact set per model label (e.g., `gpt4o_mini`, `llama3_8b`, `mistral_7b`, `mixtral_8x7b`, `flan_t5_xl`, `gemini_flash`).
- Only one `flan_t5_xl` run is present, and filenames/metrics do not encode prompt mode (zero-shot vs few-shot), so the exact prompt configuration cannot be recovered from saved artifacts alone.
- Likely interpretation: the saved `flan_t5_xl` run was a single/zero-shot attempt rather than a preserved few-shot comparison run.

## Presentation Consistency Check (PSSI Extraction Presentation)

- The death-count images in `presentations/extraction-models-pssi/images/death-counts/` are byte-for-byte identical to generated images in `papers/death-counts/data-viz/images/`.
- The corresponding plotting scripts read from `papers/death-counts/results/death-counts-llms`.
- The LLM plots shown in the presentation only include four LLM models (`gpt4o_mini`, `llama3_8b`, `mistral_7b`, `mixtral_8x7b`) and do not include `flan_t5_xl` or `gemini_flash` in those particular plots.

## Parsing Observations (Important)

- LLM notebook uses `parse_fatalities(...)` from `llm_utils.py`.
- Seq2seq notebook uses `extract_number(...)` / `parse_prediction(...)` from `extraction_utils.py`.
- These parsing paths are not the same.
- `extract_number(...)` handles number words (e.g., `"two"`) up to ten.
- `parse_fatalities(...)` does not currently parse number words; it prefers JSON `"fatalities"` or digit extraction.
- This can materially understate LLM performance when models output words instead of digits (observed example in saved `flan_t5_xl.csv`: `"Two"` parsed as `0`).
- `parse_fatalities(...)` also silently maps unparsable outputs to `0`, which can hide parsing failures and inflate apparent zero predictions.

## Practical Implications Discussed

- Few-shot prompting may help formatting consistency and some edge cases for decoder LLMs, but likely only modestly on this narrow extraction task.
- For current measured performance, parser improvements (especially number-word handling and better failure tracking) may matter more than few-shot prompting.
- For fine-tuned seq2seq models, few-shot prompting is less central than for decoder/API LLMs; the main levers are training data, prompt consistency, and parsing/post-processing.
