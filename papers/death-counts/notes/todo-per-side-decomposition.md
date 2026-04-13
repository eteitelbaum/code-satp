# TO DO: Per-Side Death Count Decomposition Analysis

## The Opportunity

Human coders already coded deaths broken down by side (security forces, Maoists, civilians).
This is not a re-coding burden — it is an analysis extension that could strengthen the paper
considerably, particularly relative to Simon et al. (UCDP-AEC) who use a similar decomposition
but cannot connect it to evidentiary structure the way SATP data allows.

## What This Would Add

The per-side breakdown makes the epistemic discounting finding *testable* rather than
illustrative. Currently the argument that models apply an evidentiary discount to claimed
Maoist casualties is inferred from specific hard cases in the error analysis. With per-side
ground truth you could test it systematically: run models on "how many security force
personnel were killed?" vs. "how many Maoists were killed?" and show that model error is
asymmetric across sides. This would be a substantively novel finding — not just about NLP
model behavior but about how conflict reporting conventions shape automated extraction.

## Questions to Resolve Before Proceeding

1. **Does the total equal the sum of components?** The overall coding adheres to the
   maximalist protocol at 99.9%. Verify that security + Maoist + civilian deaths sum to
   the total at the same rate before building an analysis on the components.

2. **Is the distribution problem equally severe for each side?** Security force deaths
   are more precisely reported and confirmed — the rare-bin problem may be less acute there.
   Maoist deaths will have more "claimed/no bodies" ambiguity. Civilian deaths may be rarest
   of all. If distributions differ by side, the rare-bin analysis becomes richer.

3. **Do the SATP narratives typically give per-side counts explicitly?** If narratives
   usually say "3 CRPF and 5 Maoists killed" rather than just "8 killed," the per-side
   extraction task may actually be *easier* than the total — no arithmetic needed, just
   reading off separate numbers. That contrast with the total-count task would itself be
   an interesting finding.

## Where It Fits in the Paper

If the above checks out, this belongs in the paper proper rather than future work —
probably as a section after the baseline results, using the decomposition to diagnose
*why* errors occur where they do. Suggested framing: connect extraction accuracy to the
evidentiary structure of conflict reporting (confirmed vs. claimed casualties), which is
a contribution that Simon et al. cannot make with UCDP data.

If the analysis is messy or the components don't sum cleanly, this can be reframed as
future work with the epistemic discounting angle as the motivation.

## Status note (2026-04-12)

The 99.9% component-sum check is already confirmed (9,908 of 9,919 cases; 11 exceptions are data entry gaps). Question 1 above is therefore resolved. Questions 2 and 3 remain open.

During §5 drafting, the epistemic discounting finding was illustrated with specific hard cases (Category 2 errors) but not tested systematically. The per-side decomposition would make this testable — model error on Maoist deaths vs. security force deaths — and could be a standalone contribution. Deferred pending decision on whether to include in this paper or a follow-up.
