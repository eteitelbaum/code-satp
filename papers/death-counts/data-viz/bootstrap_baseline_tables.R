# bootstrap_baseline_tables.R
# Paired bootstrap significance tests for baseline model comparisons (T1, T2).
# Tests each model vs. ConfliBERT-Poisson (bold) and vs. ceiling model (dagger).
# Display values come from JSON point estimates, NOT bootstrap means.
#
# Run from papers/death-counts/ directory:
#   Rscript data-viz/bootstrap_baseline_tables.R
#
# Outputs:
#   results/bootstrap_baseline_seq2seq.csv
#   results/bootstrap_baseline_llms.csv

library(tidymodels)
library(dplyr)
library(purrr)
library(readr)

N_BOOT <- 5000
set.seed(42)

SEQ2SEQ_DIR <- "results/death-counts-seq2seq"
LLM_DIR     <- "results/death-counts-llms"

# ── Metric helpers ─────────────────────────────────────────────────────────────

compute_mae         <- function(y, p) mean(abs(y - p), na.rm = TRUE)
compute_rmse        <- function(y, p) sqrt(mean((y - p)^2, na.rm = TRUE))
compute_within1     <- function(y, p) mean(abs(y - p) <= 1, na.rm = TRUE)
compute_within2     <- function(y, p) mean(abs(y - p) <= 2, na.rm = TRUE)
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

# ── Paired bootstrap for one model pair ───────────────────────────────────────
# Returns tibble of p-values (one per metric). Uses difference distributions;
# point estimates for display are read separately from JSONs.

run_paired_bootstrap <- function(df, n_boot = N_BOOT) {
  # df columns: true_label, pred_base, pred_model
  bootstraps(df, times = n_boot) |>
    mutate(diffs = map(splits, function(s) {
      d  <- analysis(s)
      y  <- d$true_label
      pb <- d$pred_base
      pm <- d$pred_model
      tibble(
        mae_diff         = compute_mae(y, pm)         - compute_mae(y, pb),
        rmse_diff        = compute_rmse(y, pm)        - compute_rmse(y, pb),
        within1_diff     = compute_within1(y, pm)     - compute_within1(y, pb),
        within2_diff     = compute_within2(y, pm)     - compute_within2(y, pb),
        nonzero_mae_diff = compute_nonzero_mae(y, pm) - compute_nonzero_mae(y, pb)
      )
    })) |>
    select(-splits) |>
    unnest(diffs) |>
    summarise(
      mae_p         = compute_pvalue(mae_diff),
      rmse_p        = compute_pvalue(rmse_diff),
      within_1_p    = compute_pvalue(within1_diff),
      within_2_p    = compute_pvalue(within2_diff),
      nonzero_mae_p = compute_pvalue(nonzero_mae_diff)
    )
}

# ── Seq2seq ────────────────────────────────────────────────────────────────────

cat("=== Seq2seq baseline bootstrap ===\n")

# All seq2seq predictions in one file
combined <- read_csv(
  file.path(SEQ2SEQ_DIR, "death_counts_predictions_combined.csv"),
  show_col_types = FALSE
) |>
  select(incident_number, true_label,
         conflibert  = `conflibert-poisson_pred`,
         t5_base     = `flan-t5-base_pred`,
         t5_large    = `flan-t5-large_pred`,
         t5_xl       = `flan-t5-xl-lora_pred`,
         indicbart   = indicbart_pred,
         mt5         = `mt5-base_pred`,
         nt5         = `nt5-small_pred`)

# Models to test: all except conflibert (baseline) and t5_xl (ceiling)
test_models <- c("t5_base", "t5_large", "indicbart", "mt5", "nt5")

seq2seq_pvals <- map_dfr(test_models, function(model) {
  cat("  vs baseline:", model, "...\n")
  df_base <- combined |>
    transmute(true_label,
              pred_base  = conflibert,
              pred_model = .data[[model]])
  p_base <- run_paired_bootstrap(df_base) |>
    rename_with(~ paste0(.x, "_base"))

  cat("  vs ceiling:", model, "...\n")
  df_ceil <- combined |>
    transmute(true_label,
              pred_base  = t5_xl,
              pred_model = .data[[model]])
  p_ceil <- run_paired_bootstrap(df_ceil) |>
    rename_with(~ paste0(.x, "_ceil"))

  bind_cols(tibble(model = model), p_base, p_ceil)
})

# Ceiling row: t5_xl vs conflibert baseline (for bold test); no ceil test
cat("  ceiling vs baseline ...\n")
df_ceil_base <- combined |>
  transmute(true_label, pred_base = conflibert, pred_model = t5_xl)
p_ceil_base <- run_paired_bootstrap(df_ceil_base) |>
  rename_with(~ paste0(.x, "_base"))
p_ceil_ceil <- tibble(
  mae_p_ceil = NA_real_, rmse_p_ceil = NA_real_, within_1_p_ceil = NA_real_,
  within_2_p_ceil = NA_real_, nonzero_mae_p_ceil = NA_real_
)
seq2seq_pvals <- bind_rows(
  seq2seq_pvals,
  bind_cols(tibble(model = "t5_xl"), p_ceil_base, p_ceil_ceil)
)

write_csv(seq2seq_pvals, "results/bootstrap_baseline_seq2seq.csv")
cat("Saved results/bootstrap_baseline_seq2seq.csv\n\n")

# ── LLMs ───────────────────────────────────────────────────────────────────────

cat("=== LLM baseline bootstrap ===\n")

# Conflibert baseline (LLM dir copy, same test set)
conflibert_llm <- read_csv(
  file.path(LLM_DIR, "death_counts_conflibert-poisson_predictions.csv"),
  show_col_types = FALSE
) |> select(incident_number, true_label, pred_conflibert = prediction)

# GPT ceiling
gpt_ceil <- read_csv(
  file.path(LLM_DIR, "gpt4o_mini.csv"),
  show_col_types = FALSE
) |> select(incident_number, pred_ceiling = gpt4o_mini_prediction)

llm_models <- list(
  list(name = "llama3_8b",    file = "llama3_8b.csv",    col = "llama3_8b_prediction"),
  list(name = "mistral_7b",   file = "mistral_7b.csv",   col = "mistral_7b_prediction"),
  list(name = "mixtral_8x7b", file = "mixtral_8x7b.csv", col = "mixtral_8x7b_prediction")
)

llm_pvals <- map_dfr(llm_models, function(m) {
  cat("  vs baseline:", m$name, "...\n")
  preds <- read_csv(file.path(LLM_DIR, m$file), show_col_types = FALSE) |>
    select(incident_number, pred_model = !!sym(m$col))

  df <- conflibert_llm |>
    inner_join(preds, by = "incident_number") |>
    inner_join(gpt_ceil, by = "incident_number")

  p_base <- df |>
    transmute(true_label, pred_base = pred_conflibert, pred_model) |>
    run_paired_bootstrap() |>
    rename_with(~ paste0(.x, "_base"))

  cat("  vs ceiling:", m$name, "...\n")
  p_ceil <- df |>
    transmute(true_label, pred_base = pred_ceiling, pred_model) |>
    run_paired_bootstrap() |>
    rename_with(~ paste0(.x, "_ceil"))

  bind_cols(tibble(model = m$name), p_base, p_ceil)
})

# GPT ceiling row: vs conflibert baseline; no ceil test
cat("  ceiling (gpt4o_mini) vs baseline ...\n")
df_gpt <- conflibert_llm |>
  inner_join(gpt_ceil, by = "incident_number")
p_gpt_base <- df_gpt |>
  transmute(true_label, pred_base = pred_conflibert, pred_model = pred_ceiling) |>
  run_paired_bootstrap() |>
  rename_with(~ paste0(.x, "_base"))
p_gpt_ceil <- tibble(
  mae_p_ceil = NA_real_, rmse_p_ceil = NA_real_, within_1_p_ceil = NA_real_,
  within_2_p_ceil = NA_real_, nonzero_mae_p_ceil = NA_real_
)
llm_pvals <- bind_rows(
  llm_pvals,
  bind_cols(tibble(model = "gpt4o_mini"), p_gpt_base, p_gpt_ceil)
)

write_csv(llm_pvals, "results/bootstrap_baseline_llms.csv")
cat("Saved results/bootstrap_baseline_llms.csv\n")
cat("\nDone.\n")
