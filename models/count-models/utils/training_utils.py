"""Utilities for model training configuration and cleanup."""

import gc
import math
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments


def create_seq2seq_training_args(
    output_dir,
    batch_size=8,
    learning_rate=3e-5,
    num_epochs=3,
    seed=42,
    generation_max_length=32
):
    """
    Create standard Seq2SeqTrainingArguments for seq2seq models.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        generation_max_length: Maximum length for generated outputs during evaluation (default: 32, suitable for most structured extraction tasks)
        
    Returns:
        Seq2SeqTrainingArguments object
    """
    # Use bf16 on Ampere+ (e.g., A100) for speed+stability, otherwise fp16 on GPU
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    supports_tf32 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        supports_tf32 = major >= 8

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=supports_tf32,
        optim="adafactor",
        report_to="none",
        seed=seed
    )


def create_regression_training_args(
    output_dir,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    seed=42
):
    """
    Create standard TrainingArguments for regression models.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        
    Returns:
        TrainingArguments object
    """
    supports_tf32 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        supports_tf32 = major >= 8

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=False,
        bf16=False,
        tf32=supports_tf32,
        report_to="none",
        seed=seed
    )


def create_qa_training_args(
    output_dir,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    seed=42
):
    """
    Create standard TrainingArguments for QA models.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        
    Returns:
        TrainingArguments object
    """
    # Use bf16 on Ampere+ (e.g., A100) for speed+stability, otherwise fp16 on GPU
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    supports_tf32 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        supports_tf32 = major >= 8

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=supports_tf32,
        report_to="none",
        seed=seed
    )


def compute_bin_weights(labels, cap_ratio=4.0):
    """
    Compute per-example sampling/loss weights inversely proportional to bin frequency.

    Bins: 0 → label==0, 1 → label==1, 2 → label==2, 3 → label in [3,5], 4 → label>=6.
    Weights are frequency-inverse, then capped at cap_ratio × the median weight, then
    sqrt-scaled to avoid destabilising training with very large weights.

    Args:
        labels: list or array of integer total_fatalities values
        cap_ratio: maximum multiple of the median weight before sqrt scaling (default 4.0)

    Returns:
        list of float weights, one per example, in the same order as labels
    """
    def to_bin(n):
        if n == 0:   return 0
        if n == 1:   return 1
        if n == 2:   return 2
        if n <= 5:   return 3
        return 4

    bins = [to_bin(n) for n in labels]
    counts = [0] * 5
    for b in bins:
        counts[b] += 1

    total = len(labels)
    # Inverse frequency: rare bins get higher weight
    raw = [total / c if c > 0 else 0.0 for c in counts]

    # Cap at cap_ratio × median of nonzero weights
    nonzero = [w for w in raw if w > 0]
    nonzero.sort()
    median = nonzero[len(nonzero) // 2]
    cap = cap_ratio * median
    capped = [min(w, cap) for w in raw]

    # Sqrt scaling to soften the distribution
    scaled = [math.sqrt(w) for w in capped]

    return [scaled[b] for b in bins]


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer that oversamples rare-bin examples via WeightedRandomSampler.

    Pass `sample_weights` (one float per training example) as a keyword argument
    to __init__. All other arguments are forwarded to Seq2SeqTrainer unchanged.

    Usage:
        weights = compute_bin_weights(train_labels)
        trainer = WeightedSeq2SeqTrainer(
            sample_weights=weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            ...
        )
    """

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = sample_weights

    def get_train_dataloader(self):
        if self._sample_weights is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        weights = torch.tensor(self._sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class LossWeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer that applies per-example loss weights to the seq2seq cross-entropy.

    The standard T5 loss averages token-level cross-entropy across the sequence.
    This trainer scales each example's mean token loss by a scalar weight before
    averaging across the batch, so rare-bin examples contribute more to gradient updates.

    Weights are read from a `sample_weight` column in the tokenized dataset, which
    DataCollatorForSeq2Seq stacks into a (batch_size,) tensor. Add the column to the
    tokenized dataset before passing it to the trainer:

        weights = compute_bin_weights(train_labels)
        train_tokenized = train_tokenized.add_column("sample_weight", weights)

    Usage:
        weights = compute_bin_weights(train_labels)
        train_tokenized = train_tokenized.add_column("sample_weight", weights)
        trainer = LossWeightedSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            ...
        )
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop sample_weight before forwarding to model (model doesn't accept it)
        w = inputs.pop("sample_weight", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)

        if w is None:
            # No weights provided — fall back to standard mean loss
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits  # (batch, seq_len, vocab)

        # Compute per-example mean token loss manually
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        mask = (shift_labels != -100).float()
        per_example_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        loss = (per_example_loss * w.float().to(per_example_loss.device)).mean()
        return (loss, outputs) if return_outputs else loss


def cleanup_model(*objects):
    """
    Clean up model objects and free GPU memory.
    
    Args:
        *objects: Variable number of objects to delete (model, trainer, tokenizer, etc.)
    """
    # Delete all passed objects
    for obj in objects:
        del obj
    
    # Run garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

