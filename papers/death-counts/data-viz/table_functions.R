# table_functions.R
# Functions to build gt tables for baseline (T1/T2) and rare-bin (T3/T4) results.
# Sourced by satp-counts.qmd. All functions are pure — they take data as arguments.
#
# Display conventions:
#   Bold (cell_text weight = "bold")  — significantly better than base model (p < .05)
#   Dagger (†) appended to value      — not significantly different from ceiling (p > .05)
#
# Table orientation: metrics as rows, models as columns.
#   Leftmost data column:  ConfliBERT-Poisson (reference baseline)
#   Rightmost data column: ceiling model

library(gt)
library(dplyr)
library(tidyr)
library(purrr)
library(jsonlite)

ALPHA <- 0.05

# ── Point estimate loaders ─────────────────────────────────────────────────────

#' Load overall metrics for all seq2seq baseline models from JSON files.
#' Returns a tibble: model, mae, rmse, within_1, within_2, nonzero_mae
load_seq2seq_baseline_metrics <- function(results_dir = "results/death-counts-seq2seq") {
  models <- list(
    list(id = "conflibert",  file = "death_counts_conflibert-poisson_metrics.json"),
    list(id = "t5_base",     file = "death_counts_flan-t5-base_metrics.json"),
    list(id = "t5_large",    file = "death_counts_flan-t5-large_metrics.json"),
    list(id = "indicbart",   file = "death_counts_indicbart_metrics.json"),
    list(id = "mt5",         file = "death_counts_mt5-base_metrics.json"),
    list(id = "nt5",         file = "death_counts_nt5-small_metrics.json"),
    list(id = "t5_xl",       file = "death_counts_flan-t5-xl-lora_metrics.json")
  )
  map_dfr(models, function(m) {
    j <- fromJSON(file.path(results_dir, m$file))$overall
    tibble(
      model       = m$id,
      mae         = j$mae,
      rmse        = j$rmse,
      within_1    = j$within_1,
      within_2    = j$within_2,
      nonzero_mae = j$nonzero_mae
    )
  })
}

#' Load overall metrics for all LLM baseline models from JSON files.
#' Returns a tibble: model, mae, rmse, within_1, within_2, nonzero_mae
load_llm_baseline_metrics <- function(results_dir = "results/death-counts-llms") {
  models <- list(
    list(id = "conflibert",   file = "death_counts_conflibert-poisson_metrics.json"),
    list(id = "llama3_8b",    file = "llama3_8b_metrics.json"),
    list(id = "mistral_7b",   file = "mistral_7b_metrics.json"),
    list(id = "mixtral_8x7b", file = "mixtral_8x7b_metrics.json"),
    list(id = "gpt4o_mini",   file = "gpt4o_mini_metrics.json")
  )
  map_dfr(models, function(m) {
    j <- fromJSON(file.path(results_dir, m$file))$overall
    tibble(
      model       = m$id,
      mae         = j$mae,
      rmse        = j$rmse,
      within_1    = j$within_1,
      within_2    = j$within_2,
      nonzero_mae = j$nonzero_mae
    )
  })
}

#' Load rare-bin metrics for seq2seq strategies.
#' Uses original T5-Large baseline and T5-XL-QLoRA ceiling from JSON;
#' intervention metrics from the results CSV.
#' Returns tibble: strategy, overall_mae, nonzero_mae, bin35_mae, bin35_exact,
#'                 bin6_mae, bin6_exact
load_seq2seq_rare_bin_metrics <- function(
    results_dir   = "results/death-counts-seq2seq",
    rarebin_dir   = "results/rare-bins/seq2seq"
) {
  # Original T5-Large baseline (from JSON, consistent with T1)
  t5l <- fromJSON(file.path(results_dir, "death_counts_flan-t5-large_metrics.json"))
  baseline_row <- tibble(
    strategy    = "S0",
    overall_mae = t5l$overall$mae,
    nonzero_mae = t5l$overall$nonzero_mae,
    bin35_mae   = t5l$bins[["3-5"]]$mae,
    bin35_exact = t5l$bins[["3-5"]]$exact_match,
    bin6_mae    = t5l$bins[["6+"]]$mae,
    bin6_exact  = t5l$bins[["6+"]]$exact_match
  )

  # Ceiling T5-XL-QLoRA (from JSON)
  t5xl <- fromJSON(file.path(results_dir, "death_counts_flan-t5-xl-lora_metrics.json"))
  ceiling_row <- tibble(
    strategy    = "T5-XL",
    overall_mae = t5xl$overall$mae,
    nonzero_mae = t5xl$overall$nonzero_mae,
    bin35_mae   = t5xl$bins[["3-5"]]$mae,
    bin35_exact = t5xl$bins[["3-5"]]$exact_match,
    bin6_mae    = t5xl$bins[["6+"]]$mae,
    bin6_exact  = t5xl$bins[["6+"]]$exact_match
  )

  # ConfliBERT (from JSON, same test set)
  cb <- fromJSON(file.path(results_dir, "death_counts_conflibert-poisson_metrics.json"))
  conflibert_row <- tibble(
    strategy    = "conflibert",
    overall_mae = cb$overall$mae,
    nonzero_mae = cb$overall$nonzero_mae,
    bin35_mae   = cb$bins[["3-5"]]$mae,
    bin35_exact = cb$bins[["3-5"]]$exact_match,
    bin6_mae    = cb$bins[["6+"]]$mae,
    bin6_exact  = cb$bins[["6+"]]$exact_match
  )

  # Intervention strategies S1-S5 from results CSV
  csv <- readr::read_csv(
    file.path(rarebin_dir, "seq2seq_rare_bin_results.csv"),
    show_col_types = FALSE
  )
  strategy_labels <- c(
    "S1 — Weighted sampling",
    "S2 — Loss weighting",
    "S3 — Targeted examples",
    "S4 — Back-translation",
    "S5 — T5 paraphrase"
  )
  interventions <- csv |>
    dplyr::filter(strategy %in% strategy_labels) |>
    dplyr::transmute(
      strategy    = dplyr::case_when(
        strategy == "S1 — Weighted sampling"  ~ "S1",
        strategy == "S2 — Loss weighting"     ~ "S2",
        strategy == "S3 — Targeted examples"  ~ "S3",
        strategy == "S4 — Back-translation"   ~ "S4",
        strategy == "S5 — T5 paraphrase"      ~ "S5"
      ),
      overall_mae = overall_mae,
      nonzero_mae = nonzero_mae,
      bin35_mae   = bin35_mae,
      bin35_exact = bin35_exact,  # already 0-1 proportion
      bin6_mae    = bin6_mae,
      bin6_exact  = bin6_exact
    )

  bind_rows(conflibert_row, baseline_row, interventions, ceiling_row)
}

#' Load rare-bin metrics for LLM strategies.
#' Uses original Llama-3.1-8B baseline from JSON; GPT-4o-mini L0 ceiling from
#' its metrics JSON; intervention metrics from the results CSV.
#' Returns tibble: strategy, overall_mae, nonzero_mae, bin35_mae, bin35_exact,
#'                 bin6_mae, bin6_exact
load_llm_rare_bin_metrics <- function(
    llm_dir     = "results/death-counts-llms",
    rarebin_dir = "results/rare-bins/llms"
) {
  # Original Llama baseline (from JSON)
  ll <- fromJSON(file.path(llm_dir, "llama3_8b_metrics.json"))
  baseline_row <- tibble(
    strategy    = "L0",
    overall_mae = ll$overall$mae,
    nonzero_mae = ll$overall$nonzero_mae,
    bin35_mae   = ll$bins[["3-5"]]$mae,
    bin35_exact = ll$bins[["3-5"]]$exact_match,
    bin6_mae    = ll$bins[["6+"]]$mae,
    bin6_exact  = ll$bins[["6+"]]$exact_match
  )

  # GPT-4o-mini L0 ceiling (from baseline metrics JSON)
  gpt_l0 <- fromJSON(file.path(llm_dir, "gpt4o_mini_metrics.json"))
  ceiling_row <- tibble(
    strategy    = "GPT-L0",
    overall_mae = gpt_l0$overall$mae,
    nonzero_mae = gpt_l0$overall$nonzero_mae,
    bin35_mae   = gpt_l0$bins[["3-5"]]$mae,
    bin35_exact = gpt_l0$bins[["3-5"]]$exact_match,
    bin6_mae    = gpt_l0$bins[["6+"]]$mae,
    bin6_exact  = gpt_l0$bins[["6+"]]$exact_match
  )

  # ConfliBERT (from LLM dir JSON)
  cb <- fromJSON(file.path(llm_dir, "death_counts_conflibert-poisson_metrics.json"))
  conflibert_row <- tibble(
    strategy    = "conflibert",
    overall_mae = cb$overall$mae,
    nonzero_mae = cb$overall$nonzero_mae,
    bin35_mae   = cb$bins[["3-5"]]$mae,
    bin35_exact = cb$bins[["3-5"]]$exact_match,
    bin6_mae    = cb$bins[["6+"]]$mae,
    bin6_exact  = cb$bins[["6+"]]$exact_match
  )

  # Intervention strategies L1-L4 from results CSV
  csv <- readr::read_csv(
    file.path(rarebin_dir, "llm_rare_bin_results.csv"),
    show_col_types = FALSE
  )
  interventions <- csv |>
    dplyr::filter(Model == "Llama-3.1-8B",
                  Strategy %in% c("L1 Attacker deaths clarification",
                                  "L2 Bin-balanced few-shot",
                                  "L3 Hard-case few-shot",
                                  "L4 Combined few-shot (L2+L3)")) |>
    dplyr::transmute(
      strategy    = dplyr::case_when(
        Strategy == "L1 Attacker deaths clarification" ~ "L1",
        Strategy == "L2 Bin-balanced few-shot"         ~ "L2",
        Strategy == "L3 Hard-case few-shot"            ~ "L3",
        Strategy == "L4 Combined few-shot (L2+L3)"     ~ "L4"
      ),
      overall_mae = `Overall MAE`,
      nonzero_mae = `Nonzero MAE`,
      bin35_mae   = `Bin 3-5 MAE`,
      bin35_exact = `Exact 3-5 (%)` / 100,
      bin6_mae    = `Bin 6+ MAE`,
      bin6_exact  = `Exact 6+ (%)` / 100
    )

  bind_rows(conflibert_row, baseline_row, interventions, ceiling_row)
}

# ── Helper: format a cell value with optional dagger ──────────────────────────

fmt_cell <- function(value, decimals, dagger) {
  formatted <- formatC(value, digits = decimals, format = "f")
  if (!is.na(dagger) && dagger) paste0(formatted, "\u2020") else formatted
}

fmt_pct_cell <- function(value, decimals = 1, dagger) {
  formatted <- formatC(value * 100, digits = decimals, format = "f")
  if (!is.na(dagger) && dagger) paste0(formatted, "\u2020") else formatted
}

# ── Build gt table: baseline (T1 / T2) ────────────────────────────────────────

#' @param metrics_df  Output of load_seq2seq_baseline_metrics() or load_llm_baseline_metrics()
#' @param pvals_df    Output of read_csv("results/bootstrap_baseline_seq2seq.csv") or _llms.csv
#' @param model_order Character vector of model IDs in left-to-right column order.
#'                    First = reference baseline (ConfliBERT), last = ceiling.
#' @param col_labels  Named character vector mapping model IDs to display labels.
#' @param title       Table title string.
#' @param n_boot      Number of bootstrap reps (for subtitle).
make_baseline_gt <- function(metrics_df, pvals_df, model_order, col_labels, title = NULL,
                             n_boot = 5000) {
  metric_ids    <- c("mae", "rmse", "within_1", "within_2", "nonzero_mae")
  metric_labels <- c("MAE", "RMSE", "Within-1", "Within-2", "Nonzero MAE")
  lower_better  <- c(TRUE, TRUE, FALSE, FALSE, TRUE)
  is_pct        <- c(FALSE, FALSE, TRUE, TRUE, FALSE)

  # Build character matrix: rows = metrics, cols = model_order
  reference_id <- model_order[1]
  ceiling_id   <- model_order[length(model_order)]

  rows <- map(seq_along(metric_ids), function(mi) {
    mid  <- metric_ids[mi]
    vals <- map_chr(model_order, function(mod) {
      pt  <- metrics_df |> dplyr::filter(model == mod) |> dplyr::pull(!!sym(mid))
      if (mod == reference_id) {
        if (is_pct[mi]) fmt_pct_cell(pt, dagger = FALSE)
        else            fmt_cell(pt, 2, dagger = FALSE)
      } else if (mod == ceiling_id) {
        if (is_pct[mi]) fmt_pct_cell(pt, dagger = FALSE)
        else            fmt_cell(pt, 2, dagger = FALSE)
      } else {
        prow <- pvals_df |> dplyr::filter(model == mod)
        p_base <- prow[[paste0(mid, "_p_base")]]
        p_ceil <- prow[[paste0(mid, "_p_ceil")]]
        dagger <- !is.na(p_ceil) && p_ceil > ALPHA
        if (is_pct[mi]) fmt_pct_cell(pt, dagger = dagger)
        else            fmt_cell(pt, 2, dagger = dagger)
      }
    })
    setNames(as.list(vals), model_order) |>
      as_tibble() |>
      dplyr::mutate(Metric = metric_labels[mi], .before = 1)
  }) |> bind_rows()

  # Build gt table
  tbl <- rows |>
    gt() |>
    tab_source_note(
      source_note = paste0(
        "Bold = significantly better than ConfliBERT (paired bootstrap, n = ",
        n_boot, ", p < .05). \u2020 = not significantly different from ceiling (p > .05). ",
        "MAE and Nonzero MAE: lower is better. Within-1 and Within-2: higher is better."
      )
    ) |>
    tab_style(
      style     = cell_text(weight = "bold"),
      locations = cells_column_labels(columns = dplyr::last_col())
    ) |>
    tab_style(
      style     = cell_text(weight = "bold"),
      locations = cells_column_labels(columns = 2)  # ConfliBERT column label
    ) |>
    cols_label(.list = setNames(as.list(col_labels), model_order)) |>
    tab_options(
      table.font.size        = px(10),
      data_row.padding       = px(2),
      source_notes.font.size = px(8)
    )

  # Bold cells that are significantly better than baseline (includes ceiling)
  for (mod in model_order[seq(2, length(model_order))]) {
    prow <- pvals_df |> dplyr::filter(model == mod)
    for (mi in seq_along(metric_ids)) {
      mid      <- metric_ids[mi]
      p_base   <- prow[[paste0(mid, "_p_base")]]
      pt_model <- metrics_df |> dplyr::filter(model == mod) |> dplyr::pull(!!sym(mid))
      pt_ref   <- metrics_df |> dplyr::filter(model == reference_id) |> dplyr::pull(!!sym(mid))
      better   <- if (lower_better[mi]) pt_model < pt_ref else pt_model > pt_ref
      if (!is.na(p_base) && p_base < ALPHA && better) {
        tbl <- tbl |> tab_style(
          style     = cell_text(weight = "bold"),
          locations = cells_body(columns = !!sym(mod), rows = mi)
        )
      }
    }
  }
  tbl
}

# ── Build gt table: rare-bin interventions (T3 / T4) ─────────────────────────

#' @param metrics_df  Output of load_seq2seq_rare_bin_metrics() or load_llm_rare_bin_metrics()
#' @param pvals_df    Output of read_csv("results/bootstrap_rare_bin_seq2seq.csv") or _llms.csv
#' @param strategy_order Character vector of strategy IDs in column order.
#'                        First = baseline (S0/L0), last = ceiling.
#' @param col_labels  Named character vector mapping IDs to display labels.
#' @param title       Table title string.
#' @param n_boot      Number of bootstrap reps.
make_rare_bin_gt <- function(metrics_df, pvals_df, strategy_order, col_labels, title = NULL,
                             n_boot = 5000) {
  metric_ids    <- c("overall_mae", "nonzero_mae", "bin35_mae",
                     "bin35_exact", "bin6_mae", "bin6_exact")
  metric_labels <- c("MAE", "Nonzero MAE",
                     "Bin 3-5 MAE", "Bin 3-5 Exact (%)",
                     "Bin 6+ MAE",  "Bin 6+ Exact (%)")
  lower_better  <- c(TRUE, TRUE, TRUE, FALSE, TRUE, FALSE)
  is_pct        <- c(FALSE, FALSE, FALSE, TRUE, FALSE, TRUE)

  baseline_id  <- strategy_order[1]   # S0 or L0
  ceiling_id   <- strategy_order[length(strategy_order)]

  rows <- map(seq_along(metric_ids), function(mi) {
    mid  <- metric_ids[mi]
    vals <- map_chr(strategy_order, function(sid) {
      pt <- metrics_df |> dplyr::filter(strategy == sid) |> dplyr::pull(!!sym(mid))
      if (sid == ceiling_id) {
        if (is_pct[mi]) fmt_pct_cell(pt, dagger = FALSE)
        else            fmt_cell(pt, 2, dagger = FALSE)
      } else {
        prow   <- pvals_df |> dplyr::filter(strategy == sid)
        p_base <- prow[[paste0(mid, "_p_base")]]
        p_ceil <- prow[[paste0(mid, "_p_ceil")]]
        dagger <- !is.na(p_ceil) && p_ceil > ALPHA
        if (sid == baseline_id) {
          # Baseline: only apply dagger (no bold vs itself)
          if (is_pct[mi]) fmt_pct_cell(pt, dagger = dagger)
          else            fmt_cell(pt, 2, dagger = dagger)
        } else {
          if (is_pct[mi]) fmt_pct_cell(pt, dagger = dagger)
          else            fmt_cell(pt, 2, dagger = dagger)
        }
      }
    })
    setNames(as.list(vals), strategy_order) |>
      as_tibble() |>
      dplyr::mutate(Metric = metric_labels[mi], .before = 1)
  }) |> bind_rows()

  # Build gt table
  tbl <- rows |>
    gt() |>
    tab_source_note(
      source_note = paste0(
        "Bold = significantly better than baseline model (paired bootstrap, n = ",
        n_boot, ", p < .05). \u2020 = not significantly different from ceiling (p > .05). ",
        "MAE rows: lower is better. Exact (%) rows: higher is better."
      )
    ) |>
    tab_style(
      style     = cell_text(weight = "bold"),
      locations = cells_column_labels(columns = dplyr::last_col())
    ) |>
    cols_label(.list = setNames(as.list(col_labels), strategy_order)) |>
    tab_options(
      table.font.size        = px(10),
      data_row.padding       = px(2),
      source_notes.font.size = px(8)
    )

  # Bold cells significantly better than baseline (S0/L0)
  intervention_ids <- strategy_order[seq(2, length(strategy_order) - 1)]
  for (sid in intervention_ids) {
    prow <- pvals_df |> dplyr::filter(strategy == sid)
    for (mi in seq_along(metric_ids)) {
      mid      <- metric_ids[mi]
      p_base   <- prow[[paste0(mid, "_p_base")]]
      pt_model <- metrics_df |> dplyr::filter(strategy == sid) |> dplyr::pull(!!sym(mid))
      pt_ref   <- metrics_df |> dplyr::filter(strategy == baseline_id) |> dplyr::pull(!!sym(mid))
      better   <- if (lower_better[mi]) pt_model < pt_ref else pt_model > pt_ref
      if (!is.na(p_base) && p_base < ALPHA && better) {
        tbl <- tbl |> tab_style(
          style     = cell_text(weight = "bold"),
          locations = cells_body(columns = !!sym(sid), rows = mi)
        )
      }
    }
  }
  tbl
}
