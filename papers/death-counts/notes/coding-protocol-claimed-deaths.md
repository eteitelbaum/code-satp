# Coding Protocol: Claimed Attacker Deaths and the Upper/Lower Bound Question

Working note on a methodological issue that affects interpretation of model performance
on high-count cases. Relevant for the paper text or a footnote.

## The Issue

`total_fatalities` in `deaths.csv` is confirmed as a maximalist (upper-bound) count:
it sums deaths on all sides, including attacker/Maoist casualties reported by security
forces even when bodies were not recovered. This is established by component-column
analysis (see `high-count-diagnostics.md`): `total_fatalities` = sum of five component
columns in 99.9% of cases, and `maoist_fatalities > 0` in 48% of 6+ bin cases.

The question is whether this is the right coding choice — and whether models that
return lower counts on "claimed, no bodies recovered" cases are wrong or merely
applying a stricter evidentiary standard.

## How Major Conflict Datasets Handle This

**UCDP (Uppsala Conflict Data Program)** uses a conservative, lower-bound methodology.
A death is coded only when there is sufficient source confirmation. Single-source
government claims of rebel/attacker deaths — especially without body recovery — are
typically discounted or excluded. UCDP explicitly distinguishes "best estimate,"
"low estimate," and "high estimate" and defaults to the low estimate when sources
conflict or are unreliable. The UCDP GED (Georeferenced Event Dataset) documentation
notes that security-force-reported combatant deaths are among the most unreliable
figures in conflict reporting.

**ACLED (Armed Conflict Location & Event Data Project)** takes a different approach:
it codes the event and records fatality figures as reported, but it codes the *source*
of the claim separately, allowing downstream users to filter by source reliability.
ACLED does not adjudicate whether a claimed figure is accurate; it records what was
reported. This is closer to the SATP approach, but ACLED at least preserves
provenance metadata.

**SATP / this dataset** appears to follow a text-extraction protocol that does not
distinguish confirmed from claimed deaths. The students coded what was in the narrative,
and narratives frequently present police claims as statements of fact ("Police claimed
to have killed six cadres") without additional verification. The result is a dataset
that is internally consistent but epistemically heterogeneous: some high counts are
confirmed by recovered bodies; others rest entirely on security force claims.

## Why This Is Especially Problematic in the Indian Maoist Context

The Indian Maoist (Naxalite) conflict has documented cases of "fake encounters" —
incidents where security forces killed civilians or unarmed individuals and subsequently
classified them as Maoists. Several such cases have been adjudicated in Indian courts
and by the National Human Rights Commission. This means a non-trivial share of
"Maoist fatalities" in the dataset may reflect not just unconfirmed claims but
potentially fabricated ones. The "bodies carried away by colleagues" narrative is a
recurring feature of fake encounter cases precisely because the absence of bodies
eliminates forensic accountability.

This does not invalidate the dataset — SATP records what was officially reported, and
that is itself a meaningful data source — but it does mean that `maoist_fatalities`
figures in high-count cases should be treated with more skepticism than, say,
`security_fatalities` figures, which are typically confirmed by official records and
next-of-kin accounts.

## Implications for Model Evaluation

When models return lower counts on "claimed, no bodies recovered" cases, they are not
simply making errors — they are applying an implicit evidentiary discount that is
defensible under a UCDP-style methodology. The model's count is more epistemically
conservative; the ground truth is more epistemically permissive. Neither is
categorically wrong.

This matters for how we present the L1/L3 prompt interventions in the paper. The
L1 instruction ("count claimed attacker casualties even if bodies were not recovered")
is not telling models to be more accurate — it is telling them to match the specific
coding protocol that the human coders applied. That is the right framing. Alternatives:

- We could describe the gap on these cases as a "protocol alignment gap" rather than
  a model error.
- A footnote could note that under a UCDP lower-bound protocol, the models' default
  behavior on "claimed/no bodies" cases would be considered correct.
- If reviewers push back on the maximalist protocol, the response is that the
  protocol is internally consistent (99.9% match to component sum), that it aligns
  with how SATP presents the data, and that we instruct models to match it explicitly.

## Suggested Footnote Language (Draft)

> Our measure of `total_fatalities` follows a maximalist coding protocol that includes
> all reported deaths on all sides, including attacker casualties claimed by security
> forces even where bodies were not independently confirmed. This is consistent with
> ACLED-style reporting-based coding but differs from the UCDP lower-bound approach,
> which discounts single-source security-force claims of combatant deaths. Models
> prompted without this clarification tend to apply an implicit evidentiary discount
> on claimed casualties, producing counts that are lower than our ground truth but
> consistent with stricter evidentiary standards. We instruct models explicitly to
> follow the maximalist protocol to align extraction behavior with the coding protocol
> the human annotators applied.

## Sources on Fake Encounters and Body Disposal (researched 2026-04-12)

A search for sources specifically documenting body removal/disappearance as forensic obstruction in fake encounter cases found the following. Worth developing for a future draft or a separate paper on the epistemology of conflict casualty counts.

**Strongest source for the Maoist/Chhattisgarh context:**
- Sarkeguda 2012 Judicial Commission (Justice V.K. Agrawal) — Government of Chhattisgarh inquiry found post-mortem reports "written without the post-mortem actually being ever conducted" and "clear manipulation of the investigation." 17 villagers killed and falsely classified as Maoists, including 7 minors. Reported in *Scroll.in* (Dec 4, 2019) https://scroll.in/latest/945423 and *Al Jazeera* (Dec 4, 2019) https://www.aljazeera.com/news/2019/12/4/indian-govt-report-17-adivasi-falsely-dubbed-maoists-shot-dead

**Body disposal as forensic obstruction (Chhattisgarh, recent):**
- Operation Black Forest (May 2025): 8 bodies cremated in defiance of AP High Court order, families physically detained during cremation. *The Wire* https://m.thewire.in/article/rights/narayanpur-basavaraju-maoists-cremated-families; *Scroll.in* https://scroll.in/article/1082902

**Most theoretically precise statement of the mechanism (Punjab analogy):**
- Human Rights Watch, *Protecting the Killers: A Policy of Impunity in Punjab, India* (October 2007): "To hide the evidence of their crimes, security forces secretly disposed of the bodies, usually by cremating them." Thousands cremated as "unclaimed." Punjab context, not Maoist, but explicit connection between body disposal and forensic obstruction. https://www.hrw.org/reports/2007/india1007/india1007.htm

**Body burial as concealment (State Dept., Chhattisgarh):**
- U.S. Department of State, *2008 Human Rights Report: India*: documents a Chhattisgarh case where police buried bodies in a forest to conceal deaths; under media pressure bodies were exhumed and autopsies confirmed gunshot deaths. https://2009-2017.state.gov/j/drl/rls/hrrpt/2008/sca/119134.htm

**Supreme Court guidelines violated by body removal:**
- *PUCL v. State of Maharashtra* (2014) 10 SCC 635: 16 mandatory guidelines for encounter investigations including videographed post-mortems in district hospitals; cremation without family consent violates these guidelines. https://www.livelaw.in/breaking-killings-police-encounters-affect-credibility-rule-law-administration-criminal-justice-system-supreme-court-issues-16-guidelines-investigation-police-encounters

**Academic work (Kashmir, not Maoist):**
- "Reproducing Regimes of Impunity: Fake Encounters and the Informalization of Everyday Violence in Kashmir Valley," *Cultural Studies* 24(1), 2010. DOI: 10.1080/09502380903221117. Paywalled; most theoretically precise academic framing but wrong conflict.

No peer-reviewed academic journal article making this argument specifically for the Naxalite context was found. The documentation comes from investigative journalism and judicial commissions.

## Action Item

Decide whether to address this in the main text (framing section) or as a footnote.
If a footnote, attach it at the point where the L1 intervention is introduced and
its motivation explained. The UCDP/ACLED contrast is useful context for reviewers
familiar with conflict data; it frames the issue as a known methodological debate
rather than a flaw specific to this dataset.
