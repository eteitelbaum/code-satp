#!/usr/bin/env python3
"""
Lexical diversity analysis for action type labels.

Two complementary analyses:
1. TTR-based measures (within/across-document vocabulary diversity)
2. Top-word coverage: for each label, what % of its documents contain
   the most frequent content word? Tests whether labels have a reliable
   lexical anchor across training examples.

Output: printed tables and figures saved to this directory.
"""

import re
import math
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # non-interactive; view saved output files
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
DATA_CSV = HERE / "../../../data/action_type.csv"
SUMMARY_CSV = HERE / "../../../papers/classification/results/actiontype_test1_summary.csv"

LABELS = ["armed_assault", "arrest", "bombing", "infrastructure",
          "surrender", "seizure", "abduction"]

LABEL_DISPLAY = {
    "armed_assault":  "Armed Assault",
    "arrest":         "Arrest",
    "bombing":        "Bombing",
    "infrastructure": "Infra. Attack",
    "surrender":      "Surrender",
    "seizure":        "Seizure",
    "abduction":      "Abduction",
}

STOPWORDS = {
    # Standard English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "as", "it",
    "its", "this", "that", "which", "who", "what", "when", "where", "how",
    "he", "she", "they", "we", "i", "you", "his", "her", "their", "our",
    "into", "after", "before", "during", "over", "under", "about", "than",
    "also", "one", "two", "three", "up", "out", "if", "so", "then",
    "while", "since", "through", "between", "among", "such", "other", "more",
    "all", "each", "both", "any", "most", "some", "few", "many", "much",
    "being", "there", "here", "them", "him", "those", "these", "against",
    "near", "across", "around", "along", "upon", "within", "without",
    "however", "although", "because", "though", "whom", "whose",
    # Generic location / admin words that appear in almost every SATP report
    "district", "village", "area", "state", "station", "block", "zone",
    "range", "circle", "sector", "division", "tehsil", "taluk", "mandal",
    "region", "locality", "town", "city", "road",
    # Corpus-wide generic actor / report words (near-zero PMI across all labels)
    "cpi", "maoist", "maoists", "naxal", "naxalite", "naxalites",
    "cadres", "cadre", "said", "identified", "local",
}

# ---------------------------------------------------------------------------
# Lexical measures
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens only."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def corpus_ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def guiraud_r(tokens: list[str]) -> float:
    return len(set(tokens)) / math.sqrt(len(tokens)) if tokens else 0.0


def mean_doc_ttr(texts: list[str]) -> float:
    ttrs = [corpus_ttr(tokenize(t)) for t in texts if t.strip()]
    return float(np.mean(ttrs)) if ttrs else 0.0


def doc_frequency(texts: list[str]) -> Counter:
    """Count how many documents contain each content word."""
    counts = Counter()
    for text in texts:
        counts.update(set(content_tokens(text)))
    return counts


def top_words(texts: list[str], n: int = 5) -> list[tuple[str, float]]:
    """Top-n content words by document frequency, with coverage (fraction of docs)."""
    n_docs = len(texts)
    if n_docs == 0:
        return []
    df = doc_frequency(texts)
    return [(word, count / n_docs) for word, count in df.most_common(n)]


def union_coverage(texts: list[str], words: list[str]) -> float:
    """Fraction of documents containing at least one word from the list."""
    n_docs = len(texts)
    if n_docs == 0:
        return 0.0
    hit = sum(1 for t in texts if any(w in set(content_tokens(t)) for w in words))
    return hit / n_docs


class PMIWord(NamedTuple):
    word: str
    pmi: float
    coverage: float   # fraction of label's docs containing the word


def pmi_scores(label_texts: list[str], all_texts: list[str],
               min_doc_freq: int = 10) -> list[PMIWord]:
    """
    Rank content words by PMI with the label.

    PMI(word, label) = log2[ P(word | label) / P(word) ]

    A high PMI means the word appears far more often in this label's
    documents than in the corpus as a whole — it is lexically distinctive.
    Words like 'district' score near zero; 'surrendered' scores high for
    Surrender because it is rare across the corpus but near-universal in
    that label's documents.
    """
    n_all = len(all_texts)
    n_label = len(label_texts)
    if n_all == 0 or n_label == 0:
        return []

    global_df = Counter()
    for t in all_texts:
        global_df.update(set(content_tokens(t)))

    label_df = doc_frequency(label_texts)

    scored = []
    for word, label_count in label_df.items():
        if label_count < min_doc_freq:
            continue
        p_word_given_label = label_count / n_label
        p_word = global_df[word] / n_all
        if p_word > 0:
            pmi = math.log2(p_word_given_label / p_word)
            scored.append(PMIWord(word, pmi, p_word_given_label))

    # Sort by coverage among words that are distinctive (PMI > 0), so we
    # surface the most reliable anchor word, not the most extreme rarity.
    distinctive = [w for w in scored if w.pmi > 0]
    return sorted(distinctive, key=lambda x: -x.coverage)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

action_df = pd.read_csv(DATA_CSV)

summary_df = pd.read_csv(SUMMARY_CSV)
full = summary_df[summary_df["fraction_raw"] == 1.0]

f1_cols = {label: f"test_{label}_f1-score" for label in LABELS}
mean_f1_series = full[[col for col in f1_cols.values()]].mean()
label_f1 = {label: mean_f1_series[col] for label, col in f1_cols.items()}

# ---------------------------------------------------------------------------
# Compute all metrics per label
# ---------------------------------------------------------------------------

all_texts = action_df["incident_summary"].dropna().tolist()

rows = []
for label in LABELS:
    texts = (
        action_df[action_df[label] == 1]["incident_summary"]
        .dropna()
        .tolist()
    )
    all_tokens = [tok for t in texts for tok in tokenize(t)]
    top5 = top_words(texts, n=5)
    top3_words_list = [w for w, _ in top5[:3]]
    pmi = pmi_scores(texts, all_texts)
    top5_pmi = pmi[:5]

    # Distinctive vocab coverage: fraction of docs containing at least one
    # word with PMI >= 1.0 (i.e. at least 2x more common in this label than
    # in the corpus overall). Models the "few synonyms" hypothesis directly —
    # a label with a reliable lexical anchor scores near 100% with one word;
    # a label requiring many synonyms scores lower even summing across all of them.
    PMI_THRESHOLD = 1.0
    distinctive_words = [w.word for w in pmi if w.pmi >= PMI_THRESHOLD]
    distinctive_cov = union_coverage(texts, distinctive_words)

    # Best single anchor: highest-coverage word that clears the PMI threshold
    anchor = next((w for w in pmi if w.pmi >= PMI_THRESHOLD), None)

    rows.append({
        "label":            LABEL_DISPLAY[label],
        "label_raw":        label,
        "n_docs":           len(texts),
        "vocab_size":       len(set(all_tokens)),
        "corpus_ttr":       corpus_ttr(all_tokens),
        "guiraud_r":        guiraud_r(all_tokens),
        "mean_doc_ttr":     mean_doc_ttr(texts),
        "top1_word":        top5[0][0] if top5 else "",
        "top1_coverage":    top5[0][1] if top5 else 0.0,
        "top3_coverage":    union_coverage(texts, top3_words_list),
        "top5":             top5,
        "pmi_word":         top5_pmi[0].word if top5_pmi else "",
        "pmi_score":        top5_pmi[0].pmi if top5_pmi else 0.0,
        "pmi_coverage":     top5_pmi[0].coverage if top5_pmi else 0.0,
        "top5_pmi":         top5_pmi,
        "n_distinctive":    len(distinctive_words),
        "distinctive_cov":  distinctive_cov,
        "anchor_word":      anchor.word     if anchor else "",
        "anchor_coverage":  anchor.coverage if anchor else 0.0,
        "anchor_pmi":       anchor.pmi      if anchor else 0.0,
        "f1_mean":          label_f1[label],
    })

results = pd.DataFrame(rows).sort_values("f1_mean", ascending=False)

# ---------------------------------------------------------------------------
# Table 1: TTR measures
# ---------------------------------------------------------------------------

print("\n── TTR-Based Lexical Diversity ─────────────────────────────────────────────")
print(f"{'Label':<18} {'N':>6} {'Vocab':>7} {'CorpTTR':>8} {'Guiraud':>8} {'DocTTR':>8} {'F1':>7}")
print("-" * 66)
for _, r in results.iterrows():
    print(
        f"{r['label']:<18} {r['n_docs']:>6} {r['vocab_size']:>7} "
        f"{r['corpus_ttr']:>8.3f} {r['guiraud_r']:>8.2f} "
        f"{r['mean_doc_ttr']:>8.3f} {r['f1_mean']:>7.3f}"
    )
print()
for metric, name in [
    ("corpus_ttr",   "Corpus TTR  "),
    ("guiraud_r",    "Guiraud's R "),
    ("mean_doc_ttr", "Mean doc TTR"),
]:
    rho, p = stats.spearmanr(results[metric], results["f1_mean"])
    print(f"  Spearman ρ ({name} vs F1): {rho:+.3f}  (p = {p:.3f})")

# ---------------------------------------------------------------------------
# Table 2: Top-word coverage
# ---------------------------------------------------------------------------

print("\n── Top-Word Coverage ───────────────────────────────────────────────────────")
print(f"{'Label':<18} {'F1':>7}  {'Top word':<14} {'Top-1 cov':>10} {'Top-3 cov':>10}  Top 5 words (doc freq %)")
print("-" * 95)
for _, r in results.iterrows():
    top5_str = "  ".join(f"{w} ({cov:.0%})" for w, cov in r["top5"])
    print(
        f"{r['label']:<18} {r['f1_mean']:>7.3f}  {r['top1_word']:<14} "
        f"{r['top1_coverage']:>9.1%} {r['top3_coverage']:>9.1%}   {top5_str}"
    )

print()
for metric, name in [
    ("top1_coverage", "Top-1 coverage"),
    ("top3_coverage", "Top-3 coverage"),
]:
    rho, p = stats.spearmanr(results[metric], results["f1_mean"])
    print(f"  Spearman ρ ({name} vs F1): {rho:+.3f}  (p = {p:.3f})")

# ---------------------------------------------------------------------------
# Table 3: PMI — most distinctive words per label
# ---------------------------------------------------------------------------

print("\n── PMI: Most Distinctive Words per Label ───────────────────────────────────")
print(f"{'Label':<18} {'F1':>7}  {'Top PMI word':<16} {'PMI':>6} {'Coverage':>10}  Top 5 by PMI")
print("-" * 100)
for _, r in results.iterrows():
    top5_str = "  ".join(f"{w.word} ({w.pmi:.1f}, {w.coverage:.0%})" for w in r["top5_pmi"])
    print(
        f"{r['label']:<18} {r['f1_mean']:>7.3f}  {r['pmi_word']:<16} "
        f"{r['pmi_score']:>6.2f} {r['pmi_coverage']:>9.1%}   {top5_str}"
    )

print()
for metric, name in [
    ("pmi_score",       "Top PMI score      "),
    ("pmi_coverage",    "Top-1 PMI coverage "),
    ("distinctive_cov", "All-PMI≥1 coverage "),
]:
    rho, p = stats.spearmanr(results[metric], results["f1_mean"])
    print(f"  Spearman ρ ({name} vs F1): {rho:+.3f}  (p = {p:.3f})")

print("\n── Distinctive Vocab Coverage (union of all PMI ≥ 1.0 words) ───────────────")
print(f"{'Label':<18} {'F1':>7} {'N distinct':>11} {'Coverage':>10}  Top 5 PMI words")
print("-" * 100)
for _, r in results.iterrows():
    top5_str = "  ".join(f"{w.word} ({w.pmi:.1f}, {w.coverage:.0%})" for w in r["top5_pmi"])
    print(
        f"{r['label']:<18} {r['f1_mean']:>7.3f} {r['n_distinctive']:>11} "
        f"{r['distinctive_cov']:>9.1%}   {top5_str}"
    )

# ---------------------------------------------------------------------------
# Figure 1: TTR scatter plots (existing)
# ---------------------------------------------------------------------------

fig1, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, (metric, xlabel) in zip(axes, [
    ("guiraud_r",    "Guiraud's R  (unique types / √tokens)"),
    ("mean_doc_ttr", "Mean per-document TTR"),
]):
    rho, p = stats.spearmanr(results[metric], results["f1_mean"])
    sizes = (results["n_docs"] / results["n_docs"].max()) * 300 + 40
    ax.scatter(results[metric], results["f1_mean"],
               s=sizes, alpha=0.75, color="steelblue",
               edgecolors="white", linewidths=0.8)
    for _, row in results.iterrows():
        ax.annotate(row["label"], xy=(row[metric], row["f1_mean"]),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    coeffs = np.polyfit(results[metric], results["f1_mean"], 1)
    x_range = np.linspace(results[metric].min(), results[metric].max(), 200)
    ax.plot(x_range, np.polyval(coeffs, x_range), "r--", alpha=0.45, linewidth=1.2)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(f"ρ = {rho:+.2f},  p = {p:.3f}", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
axes[0].set_ylabel("Mean F1 (full training, all encoder models)", fontsize=10)
fig1.text(0.5, -0.02, "Point size ∝ number of labeled examples.",
          ha="center", fontsize=8, color="gray")
fig1.suptitle("TTR-Based Lexical Diversity vs. Model Performance — Action Type Labels",
              fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
fig1.savefig(HERE / "lexical_diversity_vs_f1.png", dpi=150, bbox_inches="tight")
fig1.savefig(HERE / "lexical_diversity_vs_f1.pdf", bbox_inches="tight")
print(f"\nFig 1 saved → lexical_diversity_vs_f1.png")

# ---------------------------------------------------------------------------
# Figure 2: PMI coverage — bar chart + scatter
# ---------------------------------------------------------------------------

plot_df = results.sort_values("f1_mean", ascending=True)  # ascending for horizontal bars

fig2, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Bar chart: distinctive vocab coverage per label ---
y = np.arange(len(plot_df))
ax_bar.barh(y, plot_df["distinctive_cov"], color="steelblue", alpha=0.85)

for i, (_, row) in enumerate(plot_df.iterrows()):
    top_word = row["top5_pmi"][0].word if row["top5_pmi"] else ""
    top_pmi  = row["top5_pmi"][0].pmi  if row["top5_pmi"] else 0
    ax_bar.text(row["distinctive_cov"] + 0.005, i,
                f'top: "{top_word}" (PMI {top_pmi:.1f}),  {row["n_distinctive"]} distinct words',
                va="center", fontsize=7.5, color="steelblue")

ax_bar.set_yticks(y)
ax_bar.set_yticklabels(plot_df["label"], fontsize=9)
ax_bar.set_xlabel("Fraction of docs containing ≥1 word with PMI ≥ 1.0", fontsize=9)
ax_bar.set_title("Distinctive vocab coverage per label\n(ordered by F1, low → high)",
                 fontsize=9)
ax_bar.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_bar.grid(axis="x", linestyle=":", alpha=0.4)
ax_bar.set_xlim(0, 1.35)

# --- Scatter: distinctive vocab coverage vs F1 ---
rho_cov, p_cov = stats.spearmanr(results["distinctive_cov"], results["f1_mean"])

sizes = (results["n_docs"] / results["n_docs"].max()) * 300 + 40
ax_scatter.scatter(results["distinctive_cov"], results["f1_mean"],
                   s=sizes, alpha=0.8, color="steelblue",
                   edgecolors="white", linewidths=0.8)
for _, row in results.iterrows():
    top_word = row["top5_pmi"][0].word if row["top5_pmi"] else ""
    ax_scatter.annotate(
        f"{row['label']}\n(\"{top_word}\")",
        xy=(row["distinctive_cov"], row["f1_mean"]),
        xytext=(0, 7), textcoords="offset points",
        ha="center", va="bottom", fontsize=7.5,
    )

coeffs = np.polyfit(results["distinctive_cov"], results["f1_mean"], 1)
x_range = np.linspace(results["distinctive_cov"].min(), results["distinctive_cov"].max(), 200)
ax_scatter.plot(x_range, np.polyval(coeffs, x_range), "r--", alpha=0.45, linewidth=1.2)

ax_scatter.set_xlabel("Distinctive vocab coverage (union of all PMI ≥ 1.0 words)", fontsize=10)
ax_scatter.set_ylabel("Mean F1 (full training, all encoder models)", fontsize=10)
ax_scatter.set_title(f"ρ = {rho_cov:+.2f},  p = {p_cov:.3f}", fontsize=9)
ax_scatter.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_scatter.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax_scatter.grid(axis="y", linestyle=":", alpha=0.4)

fig2.suptitle("Distinctive Vocabulary Coverage vs. Model Performance — Action Type Labels",
              fontsize=11, fontweight="bold")
fig2.text(0.5, -0.02, "Point size ∝ number of labeled examples.",
          ha="center", fontsize=8, color="gray")
plt.tight_layout()
fig2.savefig(HERE / "pmi_coverage_vs_f1.png", dpi=150, bbox_inches="tight")
fig2.savefig(HERE / "pmi_coverage_vs_f1.pdf", bbox_inches="tight")
print(f"Fig 2 saved → pmi_coverage_vs_f1.png")

# ---------------------------------------------------------------------------
# Figure 3: Two-factor summary — distinctiveness × volume → F1
# ---------------------------------------------------------------------------

fig3, ax = plt.subplots(figsize=(7, 5.5))

cmap = plt.cm.RdYlGn
f1_vals = results["f1_mean"].values
norm = plt.Normalize(vmin=f1_vals.min() - 0.02, vmax=f1_vals.max() + 0.01)

sc = ax.scatter(
    results["anchor_coverage"],
    results["n_docs"],
    c=results["f1_mean"],
    cmap=cmap, norm=norm,
    s=220, zorder=3,
    edgecolors="white", linewidths=1.2,
)

# Custom label offsets to avoid overlap
offsets = {
    "Arrest":        ( 0.01,  80,  "left"),
    "Surrender":     ( 0.01, -120, "left"),
    "Armed Assault": (-0.02,  80,  "right"),
    "Bombing":       (-0.02, -120, "right"),
    "Seizure":       (-0.02,  80,  "right"),
    "Infra. Attack": ( 0.01, -120, "left"),
    "Abduction":     ( 0.01,  60,  "left"),
}

for _, row in results.iterrows():
    dx, dy, ha = offsets.get(row["label"], (0.01, 60, "left"))
    ax.annotate(
        f"{row['label']}\n(\"{row['anchor_word']}\", {row['anchor_coverage']:.0%})",
        xy=(row["anchor_coverage"], row["n_docs"]),
        xytext=(row["anchor_coverage"] + dx, row["n_docs"] + dy),
        ha=ha, va="center", fontsize=8,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
    )

cbar = fig3.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Mean F1 (full training, all encoder models)", fontsize=9)

ax.set_xlabel("Anchor-word coverage  (top PMI ≥ 1.0 word, fraction of label's docs)",
              fontsize=9)
ax.set_ylabel("Training set size (number of labeled examples)", fontsize=9)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.grid(linestyle=":", alpha=0.35)

# Light shading to suggest quadrants
ax.axvline(0.75, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(1000, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.77, 2800, "distinctive\n+ high volume", fontsize=7.5,
        color="gray", va="center")
ax.text(0.30, 2800, "diffuse\n+ high volume", fontsize=7.5,
        color="gray", va="center", ha="center")
ax.text(0.77, 550, "distinctive\n+ low volume", fontsize=7.5,
        color="gray", va="center")
ax.text(0.30, 550, "diffuse\n+ low volume", fontsize=7.5,
        color="gray", va="center", ha="center")

fig3.suptitle(
    "Lexical Distinctiveness × Training Volume → Model Performance\nAction Type Labels",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
fig3.savefig(HERE / "distinctiveness_volume_f1.png", dpi=150, bbox_inches="tight")
fig3.savefig(HERE / "distinctiveness_volume_f1.pdf", bbox_inches="tight")
print(f"Fig 3 saved → distinctiveness_volume_f1.png")
