# Framing Relative to Prior Work

## The Core Concern

Simon et al. (UCDP-AEC, KONVENS 2025) pre-empts some of what might seem like the paper's main findings:

- They show generative models are competitive on death count extraction: Flan-T5-Large
  reaches 81.0% exact match on Deaths. NOTE: Text2Event scores 53.4%, but DEGREE
  (another "extractive" baseline, BART-based but input-anchored) reaches 81.3%, matching
  Flan-T5-Large. The extractive-vs-generative comparison is therefore not a clean win for
  seq2seq. Do not use Simon et al. as empirical support for the claim that extractive
  models fail on death counts.
- They show fine-tuned T5-scale models can hit ~81% exact match on death fields
- Their "abstractive inference is necessary" framing covers the same ground as the argument
  for seq2seq over encoder classification

If the paper's main claim were "seq2seq works for conflict death count extraction," Simon et al.
have essentially published that already, even if it's not their headline finding.

---

## What This Analysis Adds That Simon et al. Don't Touch

### 1. The distribution problem — the real gap

UCDP by definition only includes events with at least one confirmed death. Their value space
is small integers clustered near 1–5, highly repetitive across train and test. Our dataset is
71% zeros with a long tail out to 100+. The problem of *when does the model know to predict
zero* and *how does it handle high-fatality outliers* simply does not exist in their data.
Their "deaths is easy" finding is partly an artifact of data construction, not a general result.

### 2. The rare-bin analysis

No intervention study, no bin-level breakdown, no analysis of systematic failure on high-count
cases in Simon et al. The insight that overall MAE flatters models that systematically fail on
the policy-relevant cases is entirely absent from their paper. High-fatality events drive
conflict severity indices, escalation assessments, and policy responses — getting them wrong
is not a marginal problem.

### 3. Prompting vs. fine-tuning comparison

Simon et al. evaluate only fine-tuned models — no zero-shot or few-shot LLM baselines at all.
The parallel tracks (prompt engineering vs. training interventions) and the finding that
few-shot Llama matches GPT-4o-mini on bin 6+ is entirely new.

### 4. Instruction-following asymmetry

L1 protocol clarification is helpful for GPT-4o-mini, harmful for Llama (0.235 → 0.318 MAE).
This is a concrete, practically useful finding for researchers choosing between models that
nothing in Simon et al. touches.

### 5. Coding protocol analysis

The maximalist vs. conservative coding debate and the "epistemic discounting" interpretation
of model errors — this is a conflict studies contribution. Simon et al. inherit UCDP's
conservative methodology without examining it. The finding that models' lower predictions on
"claimed/no bodies" cases reflect a defensible epistemic choice is specific to this domain
and not present in any prior NLP work on count extraction.

### 6. Benchmark interpretability — the diagnostic value of a focused dataset

A focused single-conflict dataset is not just a limitation to apologize for; it is a
methodological asset for diagnosing model behavior. In multi-conflict datasets like
UCDP-AEC or LEMONADE, a model error on a death count field could stem from several
confounded sources: failure to identify which actors are involved, wrong conflict framing,
unfamiliar reporting conventions, cross-lingual confusion, or actual count extraction
failure. Because SATP covers a single conflict with a fixed cast of actors and consistent
narrative conventions, actor identification is already solved. Every event is CRPF/police
vs. CPI-Maoist vs. civilians, drawn from the same portal. When a model fails, the failure
is specifically about count extraction.

This matters most for the per-side decomposition (see todo-per-side-decomposition.md). In
a multi-conflict dataset you could not cleanly attribute asymmetric errors across sides to
reporting conventions, because actor misidentification could explain the same pattern. The
focused dataset rules that out, enabling the move from "the model fails" to "the model
fails specifically when counts are claimed rather than confirmed."

More broadly, there is a methodological argument here about what makes a good benchmark
for extraction tasks. Homogeneity is a feature, not a bug, when the goal is diagnostic
precision. The 10,000 SATP events give statistical power on a well-defined problem, whereas
a globally diverse dataset conflates many sources of difficulty simultaneously. This could
be framed as a secondary contribution: not just "here is a benchmark" but "here is an
argument for what properties make a benchmark useful for isolating extraction failures."

---

## Recommended Reframing

The paper should not be primarily framed as "here is a system for extracting death counts"
— Simon et al. have done enough of that. The stronger framing is:

> **Existing evaluations of automated count extraction mask systematic failure on rare
> high-fatality events, which are precisely the cases that matter most for conflict
> measurement.**

This makes the primary contribution the *evaluation design* and the *rare-bin intervention
study*, not the basic seq2seq extraction result. Simon et al. become a useful reference point
establishing baseline capability — and the paper pushes past it on the dimension they did
not examine.

The paper's structure of argument then becomes:

1. Prior work shows generative models can extract death counts (Simon et al., Zhong et al.)
2. But existing evaluations use aggregate MAE, which masks bin-level failure
3. In conflict data with skewed distributions, this matters: high-fatality cases are rare
   but policy-critical
4. We run a systematic study of interventions on both prompting and training sides
5. Findings: few-shot prompting is highly effective for LLMs; training interventions help
   seq2seq on bin 3-5 but cannot close the bin 6+ gap (a capacity problem, not a data problem)
6. Practical upshot: for conflict researchers without GPU resources, few-shot Llama is now
   competitive with fine-tuned T5-XL on the hardest cases

This framing is honest about what is new and positions the paper as doing for count
extraction what Simon et al. did for event schemas — but on the dimension they missed.
