# bootstrap_rare_bin_tables.R
# Paired bootstrap significance tests for rare-bin intervention experiments (T3, T4).
# Tests each strategy vs. the original baseline model (bold) and vs. ceiling (dagger).
# Metrics: overall MAE, nonzero MAE, bin 3-5 MAE, bin 3-5 exact, bin 6+ MAE, bin 6+ exact.
#
# T3 (seq2seq): baseline = original Flan-T5-Large; ceiling = T5-XL-QLoRA
# T4 (LLMs):   baseline = original Llama-3.1-8B;  ceiling = GPT-4o-mini L0
#
# Uses original (not re-run) baseline predictions for consistency with T1/T2.
#
# Run from papers/death-counts/ directory:
#   Rscript data-viz/bootstrap_rare_bin_tables.R
#
# Outputs:
#   results/bootstrap_rare_bin_seq2seq.csv
#   results/bootstrap_rare_bin_llms.csv

library(tidymodels)
library(dplyr)
library(purrr)
library(readr)

N_BOOT <- 5000
set.seed(42)

SEQ2SEQ_DIR  <- "results/death-counts-seq2seq"
LLM_DIR      <- "results/death-counts-llms"
RAREBIN_S    <- "results/rare-bins/seq2seq"
RAREBIN_L    <- "results/rare-bins/llms"

# ── Metric helpers ─────────────────────────────────────────────────────────────

assign_bin <- function(count) {
  dplyr::case_when(
    count == 0              ~ "0",
    count == 1              ~ "1",
    count == 2              ~ "2",
    count >= 3 & count <= 5 ~ "3-5",
    count >= 6              ~ "6+",
    TRUE                    ~ NA_character_
  )
}

compute_mae         <- function(y, p) mean(abs(y - p), na.rm = TRUE)
compute_exact       <- function(y, p) mean(y == p,     na.rm = TRUE)
compute_nonzero_mae <- function(y, p) {
  idx <- y > 0
  if (!any(idx, na.rm = TRUE)) return(NA_real_)
  mean(abs(y[idx] - p[idx]), na.rm = TRUE)
}
compute_pvalue <- function(diffs) {
  diffs <- diffs[!is.na(diffs)]
  if (length(diffs) == 0) return(NA_real_)
  2 * min(mean(diffs <= 0), mean(diffs >= 0))
}

# ── Compute all six metrics for one split ─────────────────────────────────────

metrics_from_split <- function(d, pred_col) {
  y  <- d$true_label
  p  <- d[[pred_col]]
  b  <- d$bin
  y35 <- y[b == "3-5"]; p35 <- p[b == "3-5"]
  y6  <- y[b == "6+"];  p6  <- p[b == "6+"]
  tibble(
    overall_mae  = compute_mae(y, p),
    nonzero_mae  = compute_nonzero_mae(y, p),
    bin35_mae    = if (length(y35) > 0) compute_mae(y35, p35)   else NA_real_,
    bin35_exact  = if (length(y35) > 0) compute_exact(y35, p35) else NA_real_,
    bin6_mae     = if (length(y6)  > 0) compute_mae(y6, p6)     else NA_real_,
    bin6_exact   = if (length(y6)  > 0) compute_exact(y6, p6)   else NA_real_
  )
}

# ── Paired bootstrap for rare-bin metrics ─────────────────────────────────────

run_rare_bin_bootstrap <- function(df, n_boot = N_BOOT) {
  # df columns: true_label, bin, pred_base, pred_model
  bootstraps(df, times = n_boot) |>
    mutate(diffs = map(splits, function(s) {
      d <- analysis(s)
      base  <- metrics_from_split(d, "pred_base")
      model <- metrics_from_split(d, "pred_model")
      model - base
    })) |>
    select(-splits) |>
    unnest(diffs) |>
    summarise(across(everything(), compute_pvalue, .names = "{.col}_p"))
}

# ── Seq2seq rare-bin (T3) ──────────────────────────────────────────────────────

cat("=== Seq2seq rare-bin bootstrap (T3) ===\n")

# Original Flan-T5-Large as baseline (consistent with T1)
t5_large_base <- read_csv(
  file.path(SEQ2SEQ_DIR, "death_counts_flan-t5-large_predictions.csv"),
  show_col_types = FALSE
) |>
  transmute(incident_number,
            true_label,
            bin = assign_bin(true_label),
            pred_base = prediction)

# T5-XL-QLoRA ceiling
t5_xl_ceil <- read_csv(
  file.path(SEQ2SEQ_DIR, "death_counts_flan-t5-xl-lora_predictions.csv"),
  show_col_types = FALSE
) |> select(incident_number, pred_ceiling = prediction)

seq2seq_strategies <- list(
  list(label = "S1", file = "s1_weighted_sampler_test_predictions.csv"),
  list(label = "S2", file = "s2_loss_weighted_test_predictions.csv"),
  list(label = "S3", file = "s3_targeted_examples_test_predictions.csv"),
  list(label = "S4", file = "s4_backtranslation_test_predictions.csv"),
  list(label = "S5", file = "s5_t5paraphrase_test_predictions.csv")
)

seq2seq_pvals <- map_dfr(seq2seq_strategies, function(s) {
  cat("  ", s$label, "...\n")
  strat_preds <- read_csv(
    file.path(RAREBIN_S, s$file), show_col_types = FALSE
  ) |> select(incident_number, pred_model = prediction)

  df <- t5_large_base |>
    inner_join(strat_preds, by = "incident_number") |>
    inner_join(t5_xl_ceil,  by = "incident_number")

  p_base <- df |>
    select(true_label, bin, pred_base, pred_model) |>
    run_rare_bin_bootstrap() |>
    rename_with(~ paste0(.x, "_base"))

  p_ceil <- df |>
    transmute(true_label, bin,
              pred_base  = pred_ceiling,
              pred_model) |>
    run_rare_bin_bootstrap() |>
    rename_with(~ paste0(.x, "_ceil"))

  bind_cols(tibble(strategy = s$label), p_base, p_ceil)
})

# S0 row (T5-Large vs ConfliBERT is in T1; here just test S0 vs ceiling)
cat("  S0 (baseline vs ceiling) ...\n")
df_s0_ceil <- t5_large_base |>
  inner_join(t5_xl_ceil, by = "incident_number") |>
  transmute(true_label, bin,
            pred_base  = pred_ceiling,
            pred_model = pred_base)  # S0 IS pred_base here
p_s0_ceil <- run_rare_bin_bootstrap(df_s0_ceil) |>
  rename_with(~ paste0(.x, "_ceil"))
p_s0_base <- tibble(
  overall_mae_p_base = NA_real_, nonzero_mae_p_base = NA_real_,
  bin35_mae_p_base   = NA_real_, bin35_exact_p_base  = NA_real_,
  bin6_mae_p_base    = NA_real_, bin6_exact_p_base   = NA_real_
)
seq2seq_pvals <- bind_rows(
  tibble(strategy = "S0") |> bind_cols(p_s0_base, p_s0_ceil),
  seq2seq_pvals
)

write_csv(seq2seq_pvals, "results/bootstrap_rare_bin_seq2seq.csv")
cat("Saved results/bootstrap_rare_bin_seq2seq.csv\n\n")

# ── LLM rare-bin (T4) ─────────────────────────────────────────────────────────

cat("=== LLM rare-bin bootstrap (T4) ===\n")

# Original Llama-3.1-8B as baseline (consistent with T2)
llama_base <- read_csv(
  file.path(LLM_DIR, "llama3_8b.csv"),
  show_col_types = FALSE
) |>
  transmute(incident_number,
            true_label,
            bin = assign_bin(true_label),
            pred_base = llama3_8b_prediction)

# GPT-4o-mini L0 ceiling
gpt_l0_ceil <- read_csv(
  file.path(LLM_DIR, "gpt4o_mini.csv"),
  show_col_types = FALSE
) |> select(incident_number, pred_ceiling = gpt4o_mini_prediction)

llm_strategies <- list(
  list(label = "L1", file = "llama3_8b_l1_test.csv"),
  list(label = "L2", file = "llama3_8b_l2_test.csv"),
  list(label = "L3", file = "llama3_8b_l3_test.csv"),
  list(label = "L4", file = "llama3_8b_l4_test.csv")
)

llm_pvals <- map_dfr(llm_strategies, function(s) {
  cat("  ", s$label, "...\n")
  strat_preds <- read_csv(
    file.path(RAREBIN_L, s$file), show_col_types = FALSE
  ) |> select(incident_number, pred_model = prediction)

  df <- llama_base |>
    inner_join(strat_preds, by = "incident_number") |>
    inner_join(gpt_l0_ceil, by = "incident_number")

  p_base <- df |>
    select(true_label, bin, pred_base, pred_model) |>
    run_rare_bin_bootstrap() |>
    rename_with(~ paste0(.x, "_base"))

  p_ceil <- df |>
    transmute(true_label, bin,
              pred_base  = pred_ceiling,
              pred_model) |>
    run_rare_bin_bootstrap() |>
    rename_with(~ paste0(.x, "_ceil"))

  bind_cols(tibble(strategy = s$label), p_base, p_ceil)
})

# L0 row: test vs ceiling only
cat("  L0 (baseline vs ceiling) ...\n")
df_l0_ceil <- llama_base |>
  inner_join(gpt_l0_ceil, by = "incident_number") |>
  transmute(true_label, bin,
            pred_base  = pred_ceiling,
            pred_model = pred_base)
p_l0_ceil <- run_rare_bin_bootstrap(df_l0_ceil) |>
  rename_with(~ paste0(.x, "_ceil"))
p_l0_base <- tibble(
  overall_mae_p_base = NA_real_, nonzero_mae_p_base = NA_real_,
  bin35_mae_p_base   = NA_real_, bin35_exact_p_base  = NA_real_,
  bin6_mae_p_base    = NA_real_, bin6_exact_p_base   = NA_real_
)
llm_pvals <- bind_rows(
  tibble(strategy = "L0") |> bind_cols(p_l0_base, p_l0_ceil),
  llm_pvals
)

write_csv(llm_pvals, "results/bootstrap_rare_bin_llms.csv")
cat("Saved results/bootstrap_rare_bin_llms.csv\n")
cat("\nDone.\n")
