"""Utilities for running LLM inference for death count extraction."""

import os
import re
import time
from typing import List, Optional, Tuple, Callable, Any, Dict
from pathlib import Path
import importlib

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from huggingface_hub.utils import GatedRepoError


# Configuration constants
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16
USE_4BIT = False

# ── Prompt variants for rare-bin intervention study ───────────────────────────
#
# L0 (baseline): zero-shot, no attacker-death guidance (original experiments)
# L1: adds one sentence clarifying that claimed attacker deaths should be counted
# L2: bin-balanced few-shot (one clear example per bin: 0, 1, 2, 3-5, 6+)
# L3: hard-case few-shot targeting known failure modes (multi-group arithmetic,
#     succumbed to injuries, claimed/unrecovered attacker deaths)
#
# All few-shot examples are drawn from the validation set to avoid test leakage.
# The run_causal_batch / run_openai_batch functions accept an optional prompt_fn
# argument (default: make_input / L0) so variants can be swapped without
# touching calling code.
# ─────────────────────────────────────────────────────────────────────────────

# L0 — baseline instruction (unchanged from original experiments)
INSTR = (
    "How many people were killed? Answer with only a number. "
    "Return JSON exactly as: {\"fatalities\": <integer>}. If no fatalities are mentioned, use 0."
)

# L1 — attacker deaths clarification (same base instruction + one sentence)
INSTR_L1 = (
    "How many people were killed? Answer with only a number. "
    "Return JSON exactly as: {\"fatalities\": <integer>}. If no fatalities are mentioned, use 0. "
    "Count all reported deaths on all sides, including claimed attacker casualties "
    "even if bodies were not recovered."
)

# Optional toggle to enable few-shot prompting for T5 models
USE_T5_FEWSHOT = False

def set_t5_fewshot(enabled: bool) -> None:
    """
    Enable or disable few-shot prompting for T5 models at runtime.

    This allows notebooks to control the prompting style without editing this file.
    """
    global USE_T5_FEWSHOT
    USE_T5_FEWSHOT = bool(enabled)


def make_input(text: str) -> str:
    """L0 (baseline): zero-shot prompt, no attacker-death guidance."""
    return f"{INSTR}\n\nText: {text}\nAnswer:"


def make_input_l1(text: str) -> str:
    """L1: zero-shot with attacker deaths clarification."""
    return f"{INSTR_L1}\n\nText: {text}\nAnswer:"


def make_input_l2(text: str) -> str:
    """
    L2: bin-balanced few-shot prompt.

    One example per bin (0, 1, 2, 3-5, 6+) drawn from the training set
    (incident IDs: 306301001, 801090901, 204120501, 207230601, 804291301).
    Examples are short, unambiguous cases covering the full count range and
    demonstrating the killed/injured distinction.
    """
    examples = [
        # Bin 0 — injuries only, no deaths (training ID 306301001)
        (
            "Seven CRPF personnel were injured in a landmine blast in the Bijapur District.",
            '{"fatalities": 0}'
        ),
        # Bin 1 — single confirmed death (training ID 801090901)
        (
            "CPI-Maoists killed one villager in Garwah district",
            '{"fatalities": 1}'
        ),
        # Bin 2 — two deaths (training ID 204120501)
        (
            "CPI-Maoist cadres kill two farmers in the Rohtas District.",
            '{"fatalities": 2}'
        ),
        # Bin 3-5 — multiple deaths; injured are not counted (training ID 207230601)
        (
            "Three persons were killed and five others injured by the CPI-Maoist "
            "at Khaira village in Lakhisarai District.",
            '{"fatalities": 3}'
        ),
        # Bin 6+ — large confirmed count (training ID 804291301)
        (
            "Seven CPI-Maoist cadres were killed in a gun battle with SFs in Latehar District.",
            '{"fatalities": 7}'
        ),
    ]
    shot_block = "\n\n".join(f"Text: {ex}\nAnswer: {ans}" for ex, ans in examples)
    return f"{INSTR_L1}\n\n{shot_block}\n\nText: {text}\nAnswer:"

def make_input_l3(text: str) -> str:
    """
    L3: hard-case few-shot prompt targeting known failure modes.

    Examples cover: multi-group arithmetic (sum all groups), claimed attacker
    casualties where bodies were carried away, deaths from succumbed injuries,
    and claimed deaths without body recovery (injured bystanders don't count).
    All examples drawn from the training set
    (incident IDs: 312160501, 201130801, 303031602, 312081501).
    """
    examples = [
        # Multi-group arithmetic: sum across all named groups (training ID 312160501)
        (
            "Three Maoists and a civilian were killed during an encounter at "
            "Bhejji locality in the Dantewada District.",
            '{"fatalities": 4}'
        ),
        # Claimed attacker deaths: count even when bodies were carried away (training ID 201130801)
        (
            "Police claimed to have killed six cadres of the CPI-Maoist in an "
            "encounter at Bangudwa Naktaia hills in the Gaya District. The Deputy "
            "Superintendent of Police said that dead bodies of the slain Maoists "
            "could not be recovered from the encounter site as these were taken "
            "away by their colleagues.",
            '{"fatalities": 6}'
        ),
        # Succumbed to injuries: delayed deaths still count (training ID 303031602)
        (
            "Three troopers of CoBRA were killed and at least 15 others were "
            "injured in an encounter with CPI-Maoist cadres in Sukma District. "
            "Officials said while two Commandos had succumbed to bullet injuries "
            "on March 3, their colleague died on March 4. At least 15 others "
            "were injured.",
            '{"fatalities": 3}'
        ),
        # Injuries don't count; claimed deaths without body recovery do (training ID 312081501)
        (
            "Five security personnel, including two STF troopers, were injured "
            "when CPI-Maoist cadres ambushed a team of SFs in Sukma District. "
            "Police also claimed to have gunned down at least 15 Maoists in the "
            "encounter although no bodies were recovered from the spot.",
            '{"fatalities": 15}'
        ),
    ]
    shot_block = "\n\n".join(f"Text: {ex}\nAnswer: {ans}" for ex, ans in examples)
    return f"{INSTR_L1}\n\n{shot_block}\n\nText: {text}\nAnswer:"

def make_input_l4(text: str) -> str:
    """
    L4: combined few-shot prompt (L2 + L3 examples, 9 shots total).

    Merges the bin-balanced coverage of L2 with the hard-case targeting of L3.
    All 9 examples drawn from the training set (IDs: 306301001, 801090901,
    204120501, 207230601, 804291301, 312160501, 201130801, 303031602, 312081501).
    Tests whether combining both example sets outperforms either alone.

    NOTE: requires max_input_tokens=1024 in run_causal_batch to avoid
    truncating long narratives (fixed prompt ~412 tokens).
    """
    l2_examples = [
        # training ID 306301001
        (
            "Seven CRPF personnel were injured in a landmine blast in the Bijapur District.",
            '{"fatalities": 0}'
        ),
        # training ID 801090901
        (
            "CPI-Maoists killed one villager in Garwah district",
            '{"fatalities": 1}'
        ),
        # training ID 204120501
        (
            "CPI-Maoist cadres kill two farmers in the Rohtas District.",
            '{"fatalities": 2}'
        ),
        # training ID 207230601
        (
            "Three persons were killed and five others injured by the CPI-Maoist "
            "at Khaira village in Lakhisarai District.",
            '{"fatalities": 3}'
        ),
        # training ID 804291301
        (
            "Seven CPI-Maoist cadres were killed in a gun battle with SFs in Latehar District.",
            '{"fatalities": 7}'
        ),
    ]
    l3_examples = [
        # training ID 312160501
        (
            "Three Maoists and a civilian were killed during an encounter at "
            "Bhejji locality in the Dantewada District.",
            '{"fatalities": 4}'
        ),
        # training ID 201130801
        (
            "Police claimed to have killed six cadres of the CPI-Maoist in an "
            "encounter at Bangudwa Naktaia hills in the Gaya District. The Deputy "
            "Superintendent of Police said that dead bodies of the slain Maoists "
            "could not be recovered from the encounter site as these were taken "
            "away by their colleagues.",
            '{"fatalities": 6}'
        ),
        # training ID 303031602
        (
            "Three troopers of CoBRA were killed and at least 15 others were "
            "injured in an encounter with CPI-Maoist cadres in Sukma District. "
            "Officials said while two Commandos had succumbed to bullet injuries "
            "on March 3, their colleague died on March 4. At least 15 others "
            "were injured.",
            '{"fatalities": 3}'
        ),
        # training ID 312081501
        (
            "Five security personnel, including two STF troopers, were injured "
            "when CPI-Maoist cadres ambushed a team of SFs in Sukma District. "
            "Police also claimed to have gunned down at least 15 Maoists in the "
            "encounter although no bodies were recovered from the spot.",
            '{"fatalities": 15}'
        ),
    ]
    all_examples = l2_examples + l3_examples
    shot_block = "\n\n".join(f"Text: {ex}\nAnswer: {ans}" for ex, ans in all_examples)
    return f"{INSTR_L1}\n\n{shot_block}\n\nText: {text}\nAnswer:"

def make_input_t5(text: str) -> str:
    """
    Create a T5-friendly prompt (simpler seq2seq style).
    
    Flan-T5 models typically perform best with a plain instruction followed by context,
    without JSON schema or chat-style scaffolding.
    """
    return f"How many people were killed? Answer with only a number.\n\n{text}"

def make_input_t5_fewshot(text: str, shots: Optional[list[tuple[str, str]]] = None) -> str:
    """
    Create a few-shot prompt for T5 models to stabilize zero-shot extraction.
    
    Args:
        text: The input incident summary
        shots: Optional list of (example_text, example_answer) pairs
    """
    if shots is None:
        shots = [
            ("An encounter took place but no casualties were reported.", "0"),
            ("Maoists killed two villagers in the forest.", "2"),
            ("A blast injured five people; no one was killed.", "0"),
        ]
    header = "How many people were killed? Answer with only a number."
    examples = []
    for s, a in shots:
        examples.append(f"Text: {s}\nAnswer: {a}")
    return f"{header}\n\n" + "\n\n".join(examples) + f"\n\nText: {text}\nAnswer:"

def parse_fatalities(s: str, return_metadata: bool = False):
    """
    Parse fatalities count from model output.

    Args:
        s: Model output string (may contain JSON or plain number)
        return_metadata: If True, return a dict with prediction and parse diagnostics
            instead of just the int. Existing callers passing return_metadata=False
            (the default) are unaffected.

    Returns:
        int: Extracted fatalities count (0 if not found), when return_metadata=False.
        dict: When return_metadata=True:
            - prediction (int): extracted count (0 on failure)
            - parse_success (bool): whether a count was confidently extracted
            - parse_method (str): 'json_key', 'digit_regex', or 'failure'
            - defaulted_to_zero_due_to_parse_failure (bool)
    """
    if not s:
        if return_metadata:
            return {"prediction": 0, "parse_success": False,
                    "parse_method": "failure", "defaulted_to_zero_due_to_parse_failure": True}
        return 0
    # Normalize whitespace and strip common code fences/backticks
    s = str(s).strip()
    # If fenced in triple backticks, extract inner content
    try:
        import re as _re
        fenced = _re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=_re.IGNORECASE)
        if fenced:
            s = fenced.group(1).strip()
    except Exception:
        pass
    # Remove stray single backticks
    if "`" in s:
        s = s.replace("`", "")

    # Try to extract from JSON format first
    m = re.search(r'"fatalities"\s*:\s*(-?\d+)', s or "")
    if m:
        result = max(0, int(m.group(1)))
        if return_metadata:
            return {"prediction": result, "parse_success": True,
                    "parse_method": "json_key", "defaulted_to_zero_due_to_parse_failure": False}
        return result

    # Otherwise scan for integers and prefer small, plausible casualty counts
    # This avoids accidentally capturing unrelated large numbers (years, ids, etc.).
    nums = [int(x) for x in re.findall(r'\d+', s or "")]
    if not nums:
        if return_metadata:
            return {"prediction": 0, "parse_success": False,
                    "parse_method": "failure", "defaulted_to_zero_due_to_parse_failure": True}
        return 0
    plausible = [n for n in nums if 0 <= n <= 200]
    result = plausible[0] if plausible else max(0, nums[0])
    if return_metadata:
        return {"prediction": result, "parse_success": True,
                "parse_method": "digit_regex", "defaulted_to_zero_due_to_parse_failure": False}
    return result


def time_inference_call(inference_func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """
    Time an inference function call and return results with timing.
    
    Args:
        inference_func: The inference function to call
        *args, **kwargs: Arguments to pass to the inference function
        
    Returns:
        tuple: (outputs, timing_dict) where timing_dict contains:
            - total_time_seconds: Total inference time
            - time_per_item_seconds: Average time per item
            - throughput_items_per_second: Items processed per second
            - num_items: Number of items processed
    """
    start_time = time.time()
    outputs = inference_func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    
    num_items = len(outputs) if isinstance(outputs, list) else 1
    timing = {
        'total_time_seconds': elapsed_time,
        'time_per_item_seconds': elapsed_time / num_items if num_items > 0 else 0,
        'throughput_items_per_second': num_items / elapsed_time if elapsed_time > 0 else 0,
        'num_items': num_items
    }
    
    return outputs, timing


def _resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Resolve a Hugging Face token from explicit argument or environment.

    Args:
        explicit_token: Token passed to the function

    Returns:
        Optional[str]: Token string if available
    """
    if explicit_token:
        return explicit_token

    for env_var in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token

    return None


def load_causal(model_id: str, token: Optional[str] = None):
    """
    Load a causal language model (for instruction-tuned models like Llama, Mistral).
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        tuple: (tokenizer, model)
    """
    hf_token = _resolve_hf_token(token)

    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # Use NF4 with double quantization for better memory efficiency
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Prefer bfloat16 compute on Ampere+ (safe on others; falls back when unsupported)
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=hf_token
        )
        # For decoder-only models (e.g., Llama, Mistral), left padding is required for batched generation
        # and avoids transformer warnings about right-padding.
        try:
            tok.padding_side = "left"
        except Exception:
            pass
        # Ensure a pad token exists; fall back to eos when undefined
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token_id = tok.eos_token_id
        model_kwargs = {
            "device_map": "auto",
            "token": hf_token,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = DTYPE

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
    except GatedRepoError as exc:
        hint = (
            f"Access to the gated model '{model_id}' requires an approved Hugging Face token. "
            "Visit the model card, request access if needed, then provide your token via "
            "`HUGGINGFACE_TOKEN` or `HF_TOKEN` environment variables, or call "
            "`huggingface_hub.login()` before loading the model."
        )
        if hf_token is None:
            hint += " No token was detected in the current environment."
        raise RuntimeError(hint) from exc

    return tok, model


def load_t5(model_id: str):
    """
    Load a T5 seq2seq model.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        tuple: (tokenizer, model)
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=DTYPE
    )
    return tok, model


@torch.inference_mode()
def run_causal_batch(
    tok,
    model,
    texts: List[str],
    max_new_tokens: int = 48,
    max_input_tokens: int = 512,
    batch_size: int = 16,
    show_progress: bool = True,
    prompt_fn: Optional[Callable[[str], str]] = None,
):
    """
    Batched generation for decoder-only LMs (Llama/Mistral) with left padding.

    Args:
        tok: Tokenizer
        model: Causal language model
        texts: List of input texts
        max_new_tokens: Maximum tokens to generate
        max_input_tokens: Maximum input tokens (truncate for speed)
        batch_size: Number of prompts to process in parallel
        show_progress: Whether to show progress
        prompt_fn: Function mapping raw text to a prompt string. Defaults to
            make_input (L0 baseline). Pass make_input_l1, make_input_l2, or
            make_input_l3 for rare-bin intervention variants.

    Returns:
        list: List of model output strings
    """
    if prompt_fn is None:
        prompt_fn = make_input

    # Deterministic generation
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.do_sample = False
        except AttributeError:
            pass
        try:
            if getattr(model.generation_config, "temperature", None) not in (None, 1.0):
                model.generation_config.temperature = 1.0
        except AttributeError:
            pass

    # Ensure pad token exists for batching
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id

    prompts = [prompt_fn(t) for t in texts]
    outs: List[str] = []
    total = len(prompts)
    num_batches = (total + batch_size - 1) // batch_size

    progress_bar = tqdm(total=total, desc="Generating", leave=False) if (show_progress and tqdm and total > 0) else None
    use_simple_progress = show_progress and (progress_bar is None) and total > 0
    if use_simple_progress:
        print(f"  Processing 0/{total}...", end="\r", flush=True)

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        if use_simple_progress:
            print(f"  Processing {end}/{total}...", end="\r", flush=True)

        inputs = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens
        ).to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id
        )

        # Decode each output in the batch (skip prompt portion)
        for i in range(len(batch_prompts)):
            input_len = inputs["input_ids"][i].shape[0]
            out = tok.decode(gen[i][input_len:], skip_special_tokens=True).strip()
            outs.append(out)
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()
    elif use_simple_progress:
        print(f"  Completed {total}/{total}      ")

    return outs


@torch.inference_mode()
def run_t5_batch(
    tok, 
    model, 
    texts: List[str], 
    max_new_tokens: int = 32,
    max_input_tokens: int = 512,
    batch_size: int = 16,
    show_progress: bool = True
):
    """
    Batched generation for T5 seq2seq models with truncation.
    """
    prompts = [
        make_input_t5_fewshot(t) if USE_T5_FEWSHOT else make_input_t5(t)
        for t in texts
    ]
    outs: List[str] = []
    total = len(prompts)
    num_batches = (total + batch_size - 1) // batch_size
    truncated = 0

    progress_bar = tqdm(total=total, desc="Generating (T5)", leave=False) if (show_progress and tqdm and total > 0) else None
    use_simple_progress = show_progress and (progress_bar is None) and total > 0
    if use_simple_progress:
        print(f"  Processing 0/{total}...", end="\r", flush=True)

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        if use_simple_progress:
            print(f"  Processing {end}/{total}...", end="\r", flush=True)

        encoded = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        # Approximate truncation count: inputs at max length
        try:
            ids = encoded.get("input_ids")
            if ids is not None:
                # ids is a tensor [batch, seq]
                if hasattr(ids, "shape") and ids.shape[1] >= max_input_tokens:
                    # Count all rows that hit the ceiling
                    truncated += int((ids.shape[1] >= max_input_tokens))
        except Exception:
            pass

        tensor_inputs = {
            "input_ids": encoded["input_ids"].to(model.device),
            "attention_mask": encoded["attention_mask"].to(model.device),
        }
        gen = model.generate(
            **tensor_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        for i in range(gen.shape[0]):
            outs.append(tok.decode(gen[i], skip_special_tokens=True).strip())
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()
    elif use_simple_progress:
        print(f"  Completed {total}/{total}      ")

    if truncated > 0:
        print(f"⚠️  Warning: {truncated} input(s) truncated to {max_input_tokens} tokens.")

    return outs


def run_openai_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 50,
    rate_limit_delay: float = 0.05,
    max_concurrency: int = 8,
    show_progress: bool = True,
    prompt_fn: Optional[Callable[[str], str]] = None,
):
    """
    Parallel OpenAI calls with bounded concurrency; preserves order.

    Args:
        prompt_fn: Function mapping raw text to a prompt string. Defaults to
            make_input (L0 baseline). Pass make_input_l1, make_input_l2, or
            make_input_l3 for rare-bin intervention variants.
    """
    # Try to get API key from various sources
    if api_key is None:
        try:
            from google.colab import userdata
            api_key = userdata.get('openai_api_key')
        except ImportError:
            import os
            api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set OPENAI_API_KEY environment variable or add 'openai_api_key' to Colab secrets."
        )
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai>=1.0.0"
        )

    if prompt_fn is None:
        prompt_fn = make_input
    prompts = [prompt_fn(t) for t in texts]
    outs: List[str] = [""] * len(prompts)
    total = len(prompts)
    errors = 0

    progress_bar = tqdm(total=total, desc="Generating (OpenAI)", leave=False) if (show_progress and tqdm and total > 0) else None
    use_simple_progress = show_progress and (progress_bar is None) and total > 0
    if use_simple_progress:
        print(f"  Processing 0/{total}...", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(idx: int, prompt: str) -> tuple[int, str, bool]:
        ok = True
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            out = response.choices[0].message.content.strip()
        except Exception:
            out = ""
            ok = False
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        return idx, out, ok

    if total > 0:
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
            futures = [executor.submit(_one, i, p) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                idx, out, ok = fut.result()
                outs[idx] = out
                if not ok:
                    errors += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                elif use_simple_progress:
                    done = sum(1 for f in futures if f.done())
                    print(f"  Processing {done}/{total}...", flush=True)
    
    if progress_bar is not None:
        progress_bar.close()
    elif use_simple_progress:
        print(f"  Completed {total}/{total}      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
    
    return outs


def run_gemini_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash-latest",
    max_output_tokens: int = 50,
    rate_limit_delay: float = 0.05,
    show_progress: bool = True,
    max_retries: int = 4,
    max_concurrency: int = 8,
):
    """
    Parallel Gemini calls with bounded concurrency, retries, and JSON-only responses.
    """
    # Try to get API key from various sources
    if api_key is None:
        try:
            from google.colab import userdata
            api_key = userdata.get('gemini_api_key')
        except ImportError:
            import os
            api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY environment variable or add 'gemini_api_key' to Colab secrets."
        )
    
    google_api_exceptions = None

    try:
        import google.generativeai as genai
        try:
            google_api_exceptions = importlib.import_module("google.api_core.exceptions")
        except ImportError:  # pragma: no cover - optional dependency
            google_api_exceptions = None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. Install with: pip install google-generativeai"
        )
    except Exception as exc:
        if google_api_exceptions and isinstance(exc, google_api_exceptions.GoogleAPIError):
            raise RuntimeError(
                f"Failed to initialize Gemini model '{model_name}'. "
                "If you are using Gemini 1.5 models, ensure your account has access and "
                "that you are on google-generativeai>=0.7.0."
            ) from exc
        raise RuntimeError(
            f"Failed to initialize Gemini client: {exc}"
        ) from exc
    
    prompts = [make_input(t) for t in texts]
    outs: List[str] = [""] * len(prompts)
    total = len(prompts)
    errors = 0
    diagnostics: List[Dict[str, Any]] = []
    # Progress handling (mirror causal/T5 style)
    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating (Gemini)", leave=False)
        else:
            use_simple_progress = True
            print(f"  Processing 0/{total}...", flush=True)
    
    # Configure generation to prefer raw JSON output (avoid code fences)
    gen_config: Dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
        "response_mime_type": "application/json",
    }
    # Configure permissive safety for violence, if available
    safety_settings = None
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            }
        ]
    except Exception:
        safety_settings = None

    def _process_one(idx: int, prompt: str) -> tuple[int, str, Optional[str]]:
        last_error_local: Optional[str] = None
        out_local = ""
        finish_reason = None
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=gen_config,
                    safety_settings=safety_settings
                )
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason == 2:
                        if hasattr(candidate, 'content') and candidate.content.parts:
                            out_local = candidate.content.parts[0].text.strip()
                        else:
                            out_local = ""
                    elif hasattr(response, 'text') and response.text:
                        out_local = response.text.strip()
                    else:
                        out_local = ""
                else:
                    out_local = response.text.strip() if hasattr(response, 'text') and response.text else ""
                break
            except Exception as exc:
                last_error_local = str(exc)
                base = 0.5
                sleep_s = base * (2 ** attempt)
                try:
                    import random  # local import
                    sleep_s += random.random() * 0.2
                except Exception:
                    pass
                time.sleep(sleep_s)
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        if not out_local and last_error_local:
            diagnostics.append({
                "index": idx,
                "len_chars": 0,
                "finish_reason": finish_reason,
                "error": last_error_local,
            })
        return idx, out_local.strip(), last_error_local if not out_local else None

    if total > 0:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except Exception:
            # Fallback to sequential if futures unavailable
            for i, p in enumerate(prompts):
                idx, out, err = _process_one(i, p)
                outs[idx] = out
                if err:
                    errors += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                elif use_simple_progress:
                    print(f"  Processing {i + 1}/{total}...", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
                futures = [executor.submit(_process_one, i, p) for i, p in enumerate(prompts)]
                for fut in as_completed(futures):
                    try:
                        idx, out, err = fut.result()
                        outs[idx] = out
                        if err:
                            errors += 1
                    except Exception:
                        errors += 1
                    finally:
                        if progress_bar is not None:
                            progress_bar.update(1)
                        elif use_simple_progress:
                            done = sum(1 for f in futures if f.done())
                            print(f"  Processing {done}/{total}...", flush=True)
    
    if progress_bar is not None:
        progress_bar.close()
    elif use_simple_progress:
        print(f"  Completed {total}/{total}      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
        sample = [d for d in diagnostics if d.get("error")][:5]
        if sample:
            print("  Example error diagnostics (up to 5):")
            for d in sample:
                print(f"   - idx={d.get('index')}, error={d.get('error')}")
    
    return outs


def already_done(model_name: str, output_dir: Path) -> bool:
    """
    Check if a model's results already exist.
    
    Args:
        model_name: Model identifier
        output_dir: Output directory to check
        
    Returns:
        bool: True if results file exists
    """
    return (output_dir / f"{model_name}.csv").exists()

