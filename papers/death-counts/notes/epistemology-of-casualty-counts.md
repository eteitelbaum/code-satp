# Epistemology of Casualty Counts

## The Core Problem

Death counts in conflict are not simple facts waiting to be extracted — they are products
of politically structured reporting environments. Three structural sources of distortion:

1. **Perpetrator incentives:** Governments and armed groups downplay deaths attributable
   to them to manage domestic opposition, avoid international sanctions, and maintain
   legitimacy.
2. **Victim group incentives:** Targeted groups have incentives to emphasize and sometimes
   inflate casualties to attract international solidarity and invoke humanitarian law.
3. **Access constraints:** Journalists are killed, access is blocked, infrastructure
   disrupted, and rural violence is systematically under-reported (Price & Ball 2015).

This means that any "ground truth" death count is itself a product of politically
structured reporting, and different sources will produce systematically different figures
for the same event.

---

## Three Methodological Responses in the Literature

### 1. Conservative lower-bound (UCDP)

Only count deaths that can be confirmed from at least one independent source. Treat
single-source security force claims of enemy casualties as unverifiable and exclude them
from the best estimate (or assign to the low estimate only). This minimizes false
positives at the cost of systematic undercount. UCDP GED embeds this approach in its
Deaths Low / Deaths High / Deaths Best distinction.

### 2. Reporting-based / maximalist (ACLED, SATP)

Record what was reported. ACLED codes the figure reported in the source and records the
source separately, making the evidentiary basis transparent. SATP uses a maximalist
protocol: count all reported deaths on all sides, including claimed attacker casualties
without body recovery. The SATP coding in this paper adheres to this protocol at 99.9%
(component sum match across dataset). This approach prioritizes completeness and
transparency over conservative verification.

### 3. Statistical estimation (HRDAG)

Use multiple systems estimation (MSE) and other statistical methods to estimate the true
death toll from partial, overlapping witness lists and databases. Treats the "true" count
as a latent quantity to be estimated, explicitly modeling the reporting process as biased
and incomplete (Ball et al.). This is the most epistemologically sophisticated approach
but requires multiple independent data sources and is not applicable to event-level
extraction from single documents.

---

## What Steinert & Kazenwadel (2025) Add

Steinert & Kazenwadel audit GPT-3.5's fatality estimates for Israeli and Turkish airstrikes
by querying in the attacker's language (Hebrew, Turkish) vs. the targeted group's language
(Arabic, Kurdish). Key findings:

- GPT gives ~27% lower estimates in attacker language vs. targeted-group language
- GPT frequently evades entirely in attacker language (denying the event occurred):
  Hebrew 29%, Turkish 38% evasion rate vs. Arabic 5%, Kurdish 0%
- Word frequency analysis shows Arabic/Kurdish responses emphasize civilians, children,
  UN condemnations; Hebrew/Turkish emphasize security operations, "terrorist"
- GPT substitutes highly salient training-data events for the queried event (Turkish
  responses frequently describe PKK cave executions rather than the queried airstrike)

**Crucially, they have no ground truth.** UCDP GED is used only as an event registry to
identify which airstrikes to ask about. Their research question is purely comparative:
does the estimate vary by language? They bracket the accuracy question entirely, justifying
this by noting that airstrike death counts are "often disputed and difficult to verify."

Their theoretical claim: GPT produces the modal response from whichever language-specific
training corpus is activated by the query language. The model faithfully mirrors the
information politics of each language community. The evasion mechanism is novel and
arguably worse than search engine bias — it presents as authoritative and neutral while
actually reflecting the attacker-language information environment.

---

## Implications for This Paper

### 1. Our approach sidesteps the Steinert problem

The Steinert finding applies to zero-shot LLM queries where the model draws on its
entire training corpus. Our approach anchors the model to a specific source document and
asks it to extract from that text only. This sidesteps training-data language politics:
the model is not recalling from memory, it is reading a specific SATP narrative. The
prompt design ("How many people were killed? Return JSON exactly as: {fatalities: <int>}")
enforces this grounding.

### 2. Epistemic discounting vs. political bias — two different mechanisms

Steinert shows models undercount deaths attributed to the attacker in the attacker's
language — a reflection of training corpus politics. Our finding is that models undercount
claimed Maoist deaths specifically — because the model applies an evidentiary discount to
unconfirmed claims ("bodies taken away," "claimed by security forces"). These produce
similar outcomes (undercounting) but via different mechanisms:

- Steinert: bias from training data language distribution
- Our paper: evidentiary reasoning applied to explicit textual uncertainty markers

Our mechanism is more defensible: the model is responding to genuine epistemic signals
in the text, not to language-stratified training data. A model that discounts "claimed"
casualties is doing something reasonable; a model that gives different numbers in Hebrew
vs. Arabic is doing something politically structured.

### 3. Our ground truth is documented and defensible

Unlike Steinert, we have an explicit, internally consistent coding protocol (maximalist)
with 99.9% adherence verified against component sums. The epistemological challenge of
contested death counts does not undermine our evaluation — it shapes the choice of
protocol and requires disclosure. Readers who prefer conservative UCDP-style coding can
treat our results as an upper bound; the model's tendency to discount claimed casualties
actually moves predictions toward the conservative end.

### 4. Worth a paragraph in the paper

The Steinert paper is worth citing in the discussion to make this contrast explicit:
prior work has shown that naive LLM querying for conflict fatalities is unreliable
because it mirrors training-data language biases (Steinert & Kazenwadel 2025). Our
document-grounded approach is not subject to this problem, but introduces a different
epistemological question: whose counting standard should the model apply? The answer
requires explicit protocol specification — which is exactly what the maximalist/
conservative discussion provides.

---

## Key References

- Steinert, C.V. and Kazenwadel, D. (2025). "How User Language Affects Conflict Fatality
  Estimates in ChatGPT." *Journal of Peace Research* 62(4): 1128–1143. [steinert2025]
- Eck, K. (2012). "In Data We Trust? A Comparison of UCDP GED and ACLED Conflict Events
  Datasets." *Cooperation and Conflict* 47(1): 124–141. [eck2012]
- Price, M. and Ball, P. (2015). The Limits of Observation for Understanding Mass Violence.
  *Canadian Journal of Law and Society* 30(2): 237–257. [not in bib — consider adding]
