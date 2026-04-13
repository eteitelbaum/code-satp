# heatmap_functions.R
# Functions to build bin-level performance heatmaps (F1/F2).
# Sourced by satp-counts.qmd.
#
# Displays exact match (%) by model × bin, with models ordered by overall MAE.
# Seq2seq and LLM panels are combined into a single vertically stacked figure.

library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(jsonlite)
library(patchwork)

# ── Data loaders ──────────────────────────────────────────────────────────────

#' Load bin-level exact match for all seq2seq models from JSON files.
#' Returns long tibble: model, model_label, bin, exact_match, overall_mae
load_seq2seq_bin_metrics <- function(results_dir = "results/death-counts-seq2seq") {
  models <- list(
    list(id = "conflibert", label = "ConfliBERT-Poisson",  file = "death_counts_conflibert-poisson_metrics.json"),
    list(id = "t5_base",    label = "Flan-T5-Base",        file = "death_counts_flan-t5-base_metrics.json"),
    list(id = "indicbart",  label = "IndicBART",           file = "death_counts_indicbart_metrics.json"),
    list(id = "mt5",        label = "mT5-Base",            file = "death_counts_mt5-base_metrics.json"),
    list(id = "nt5",        label = "NT5-Small",           file = "death_counts_nt5-small_metrics.json"),
    list(id = "t5_large",   label = "Flan-T5-Large",       file = "death_counts_flan-t5-large_metrics.json"),
    list(id = "t5_xl",      label = "T5-XL-QLoRA",         file = "death_counts_flan-t5-xl-lora_metrics.json")
  )
  map_dfr(models, function(m) {
    j <- fromJSON(file.path(results_dir, m$file))
    overall_mae <- j$overall$mae
    map_dfr(names(j$bins), function(bin) {
      tibble(
        model       = m$id,
        model_label = m$label,
        bin         = bin,
        exact_match = j$bins[[bin]]$exact_match,
        n           = j$bins[[bin]]$n,
        overall_mae = overall_mae
      )
    })
  })
}

#' Load bin-level exact match for all LLM baseline models from JSON files.
#' Returns long tibble: model, model_label, bin, exact_match, overall_mae
load_llm_bin_metrics <- function(results_dir = "results/death-counts-llms") {
  models <- list(
    list(id = "conflibert",   label = "ConfliBERT-Poisson", file = "death_counts_conflibert-poisson_metrics.json"),
    list(id = "mixtral_8x7b", label = "Mixtral-8x7B",      file = "mixtral_8x7b_metrics.json"),
    list(id = "mistral_7b",   label = "Mistral-7B",         file = "mistral_7b_metrics.json"),
    list(id = "llama3_8b",    label = "Llama-3.1-8B",       file = "llama3_8b_metrics.json"),
    list(id = "gemini_flash", label = "Gemini Flash",       file = NULL),  # excluded
    list(id = "gpt4o_mini",   label = "GPT-4o-mini",        file = "gpt4o_mini_metrics.json")
  )
  # Drop excluded models
  models <- Filter(function(m) !is.null(m$file), models)
  map_dfr(models, function(m) {
    j <- fromJSON(file.path(results_dir, m$file))
    overall_mae <- j$overall$mae
    map_dfr(names(j$bins), function(bin) {
      tibble(
        model       = m$id,
        model_label = m$label,
        bin         = bin,
        exact_match = j$bins[[bin]]$exact_match,
        n           = j$bins[[bin]]$n,
        overall_mae = overall_mae
      )
    })
  })
}

# ── Heatmap plot ──────────────────────────────────────────────────────────────

#' Build a bin × model heatmap of exact match rates.
#'
#' @param bin_df     Output of load_seq2seq_bin_metrics() or load_llm_bin_metrics()
#' @param title      Plot title
#' @param bin_order  Character vector specifying bin display order (x-axis).
#'                   Defaults to c("0", "1", "2", "3-5", "6+").
make_bin_heatmap <- function(
    bin_df,
    title,
    bin_order = c("0", "1", "2", "3-5", "6+")
) {
  # Order models by overall MAE ascending (best model at top)
  model_order <- bin_df |>
    distinct(model_label, overall_mae) |>
    arrange(desc(overall_mae)) |>
    pull(model_label)

  plot_df <- bin_df |>
    filter(bin %in% bin_order) |>
    mutate(
      bin         = factor(bin, levels = bin_order),
      model_label = factor(model_label, levels = model_order),
      exact_pct   = exact_match * 100,
      label_text  = sprintf("%.1f", exact_pct)
    )

  ggplot(plot_df, aes(x = bin, y = model_label, fill = exact_pct)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = label_text), size = 3.2, color = "black") +
    scale_fill_gradient2(
      low      = "#d73027",
      mid      = "#fee090",
      high     = "#1a9850",
      midpoint = 90,
      limits   = c(60, 100),
      oob      = scales::squish,
      name     = "Exact\nmatch (%)"
    ) +
    scale_x_discrete(
      labels = c("0" = "0\ndeaths", "1" = "1\ndeath", "2" = "2\ndeaths",
                 "3-5" = "3\u20135\ndeaths", "6+" = "6+\ndeaths")
    ) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid    = element_blank(),
      axis.text.x   = element_text(size = 9),
      axis.text.y   = element_text(size = 9),
      plot.title    = element_text(face = "bold", size = 12),
      legend.position = "right"
    )
}

#' Build a vertically stacked combined heatmap: seq2seq panel on top, LLM panel below.
#' ConfliBERT-Poisson appears in both panels as a reference row.
#' Panels share a color scale; the legend is collected on the right.
#'
#' @param seq2seq_df  Output of load_seq2seq_bin_metrics()
#' @param llm_df      Output of load_llm_bin_metrics()
#' @param bin_order   Bin display order (x-axis)
make_combined_heatmap <- function(
    seq2seq_df,
    llm_df,
    bin_order = c("0", "1", "2", "3-5", "6+")
) {
  fill_scale <- scale_fill_gradient2(
    low      = "#d73027",
    mid      = "#fee090",
    high     = "#1a9850",
    midpoint = 90,
    limits   = c(60, 100),
    oob      = scales::squish,
    name     = "Exact\nmatch (%)"
  )

  p_seq2seq <- make_bin_heatmap(seq2seq_df, title = "(a) Seq2Seq Models", bin_order = bin_order) +
    fill_scale +
    theme(
      axis.text.x     = element_blank(),
      axis.ticks.x    = element_blank(),
      legend.position = "none"
    )

  p_llm <- make_bin_heatmap(llm_df, title = "(b) Large Language Models", bin_order = bin_order) +
    fill_scale +
    theme(legend.position = "right")

  p_seq2seq / p_llm +
    plot_layout(heights = c(7, 5), guides = "collect") &
    theme(legend.position = "right")
}
