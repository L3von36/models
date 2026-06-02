# MARST manuscript — journal-reviewer report

**Manuscript:** `ARTICLE.tex` (25 pages, 7,497 words, single-column NeurIPS/arXiv style)
**Source notebook:** `marst-learnable-results.ipynb`
**Date of review:** 2026-06-02
**Reviewer role:** senior co-author preparing the paper for submission to
**Transportation Research Part C** / **IEEE T-ITS**
**Rubric:** novelty / technical correctness / experimental rigor /
reproducibility / statistical validity / writing quality / publication
readiness

---

## 1. Overall recommendation

**Major revision needed before submission.**  The science is sound,
the leak audit and statistical machinery are unusually strong for this
venue, and the negative finding on per-sensor learnable decay is
genuinely creditable.  But three structural issues — the parameter-count
argument, the absence of a load-balancing ablation, and a sub-optimal
KNN baseline — will all be flagged by a competent reviewer.  None
requires re-running the main experiments; all can be addressed in 1–2
days of focused work plus prose edits.

---

## 2. Major strengths

| # | Strength |
|---|---|
| **S1** | **Strict streaming protocol with an empirical leak audit.** The per-dataset held-out-truth corruption test plus future-perturbation test (§3.5, §5.8) is rare in this literature. The 14-baseline `NO LEAK / CAUSAL` verdict is a hard claim that survives scrutiny and gives the paper a deployability story that BRITS-forward, iTransformer, etc. share but few discuss explicitly. |
| **S2** | **Statistical rigor is appropriate for the venue.** 3 training seeds × 5 eval-mask seeds = 15 paired evaluations, Wilcoxon signed-rank, Holm-Bonferroni across the four dataset comparisons, all four reach *p* < 0.001. Table 3 cleanly separates the headline claim from noise. |
| **S3** | **Honest negative finding.** The per-sensor learnable decay parameter is reported as a near-null result (std ≈ 0.003 around mean ≈ 0.95) and the implication for the contribution claim is acknowledged in §6. Exactly the kind of self-critical discussion reviewers respect. |
| **S4** | **JamMAE consistency check.** The pipeline asserts that the MAE leader and JamMAE leader are the same model on every dataset. Domain-appropriate operational measure rather than a synthetic add-on metric. |
| **S5** | **Reproducibility is strong.** Concrete hyperparameters, RNG discipline (per-thread NumPy and CUDA generators), 500-step train/eval gap, exact dataset slicing, runtime, and a named notebook — Appendix B is at the level a third party could reproduce the tables. |

---

## 3. Major weaknesses

### W1 — No ablation on the load-balancing auxiliary loss
The aux loss at λ = 0.02 is presented as a design move (§3.4) and as
one of four contributions (§1), but the manuscript shows no λ-sweep, no
λ = 0 comparison, and no quantitative evidence that the load-balanced
variant beats the unbalanced variant on the headline metric.  The
qualitative "the gate collapses without it, look at the ablation rows"
argument in §5.7 is not a substitute for a single comparison row
λ = 0 vs λ = 0.02 on MAE.  **A reviewer will demand this.**
*Effort to fix:* 4 hours on the same 2× T4 setup.

### W2 — KNN baseline is sub-optimal — and we have the data to prove it
The Q12 KNN *k*-sweep in cell 15 (output: `k=10 → MAE=19.60` on PEMS08,
vs `k=5 → MAE=20.18`) shows *k* = 10 is better on every dataset, but
the main results table reports *k* = 5.  A reviewer will ask why we
picked the worse setting.  Either re-run with *k* = 10 or report the
sweep explicitly and acknowledge the choice.  The current paper makes
neither move.
*Effort:* 30 minutes.

### W3 — Parameter-count gap acknowledged but not addressed
MARST is 17× iTransformer.  The Discussion promises a compute-matched
ablation as "future work."  A competent reviewer (especially at T-ITS)
will not accept this as sufficient — they will say "run one
200K-parameter MARST seed and report the row."  Without that row the
win is vulnerable to the "you bought it with parameters" rebuttal even
if the architectural prior really is doing work.
*Effort:* 1 hour training + 15 min prose.

### W4 — ImputeFormer is cited but never compared, and the framing is now awkward
After the recent edit, the paper says "We do not run ImputeFormer as
a baseline ... it is not strictly causal."  This is honest but creates
a problem: ImputeFormer is offline, so it is the strongest published
method on PEMS-BAY/METR-LA *in the offline setting*.  Without an
offline-vs-streaming comparison, a reviewer cannot tell whether MARST
is competitive with the offline state-of-the-art.  Two clean options:
- (a) add ImputeFormer's published numbers as a non-comparable
  reference row in Table 2 with a footnote ("offline; not directly
  comparable");
- (b) remove the ImputeFormer mention entirely except as related work.

### W5 — Missing-pattern table has only 3 of 4 datasets
Table 7 reports PEMS-BAY, METR-LA, PEMS08; PEMS04 is absent.  Either
add it (the notebook prints it — `MARST 19.99 / 21.X / 27.X` should be
extractable) or explain the omission.  Reviewers spot this gap
immediately.

### W6 — Per-sensor learnable decay is sold as a contribution and as a null result simultaneously
This is awkward.  Recommend re-positioning: drop it from the
contributions list in §1 (contribution #2), describe it instead as a
*design probe* in §3.2 whose outcome — a negative finding — is reported
in §5.5.  The Discussion already handles it well; the Introduction
should not promote it as a contribution.

### W7 — Only three sparsity levels (50, 80, 95 %)
Standard in this benchmark is six (50, 60, 70, 80, 90, 95).
Defensible to keep three, but the omission of 70 and 90 should be
flagged in Limitations explicitly.  Currently it is mentioned in
passing.

---

## 4. Required revisions (must fix before submission)

| # | Action | Effort |
|---|---|---|
| **R1** | Add a single row to the results: `MARST (λ = 0)` vs `MARST (λ = 0.02)` MAE on all four datasets, same 3-seed protocol.  Without this row the load-balancing-loss contribution claim is unfounded. | 4 h training |
| **R2** | Either re-run the KNN imputer baseline with *k* = 10 (the Q12-sweep winner) or add a sentence in §4 ("Baselines") explaining the *k* = 5 choice and reporting the *k*-sweep result in a footnote.  Currently the paper uses a sub-optimal setting silently. | 30 min |
| **R3** | Add PEMS04 to the missing-pattern table (Table 7).  The data is in the notebook output. | 10 min |
| **R4** | Move per-sensor learnable decay out of the contributions list in §1.  Re-cast contribution #2 as the load-balancing aux loss (more defensible), and report per-sensor learnable decay as a design probe with a negative-finding outcome in §3.2 / §5.5. | 30 min prose |
| **R5** | Add a row to Table 9 (parameter count) or a paragraph in §6 with a compute-matched mini-MARST result (hidden = 64, layers = 3, ~200 K parameters).  Even a single training seed is enough for the parameter-fairness rebuttal. | 1 h train + 15 min |

**Total required-revision effort:** ≈ 1 day of training compute on
Kaggle 2× T4 + 1.5 h of prose / table edits.

---

## 5. Suggested revisions (would strengthen but not blocking)

| # | Action |
|---|---|
| **SR1** | Report JamMAE for every baseline in an extended-metrics table per dataset, not just LOCF.  The pipeline already computes it (cell 15 extended-metrics block). |
| **SR2** | Add ImputeFormer's published numbers as a non-comparable reference row in Table 2 with a clear "(offline)" footnote so reviewers can see the offline state-of-the-art context.  Quoting published numbers is acceptable. |
| **SR3** | Add one speed-dataset example to Figure 11 (currently only PEMS08 flow).  The qualitative panel is more informative with both regimes represented. |
| **SR4** | Ablate the 2-hop neighbour-mean input feature (*n*₂ₕₒₚ).  Currently described as "extra spatial context" with no quantitative evidence.  One row is enough. |
| **SR5** | Replace the present-tense `\paragraph{Limitations.}` block with a numbered subsection so it is easier for reviewers to reference (L1, L2, ...). |
| **SR6** | Add 2025–2026 streaming-imputation references if any have appeared post-July 2025; current Related Work has BayOTIDE and STAMImputer but not much beyond. |
| **SR7** | Reframe the "soft-LOCF decay sweep" in Table 6 as an *initialisation sensitivity* test, not a hyperparameter sweep, to be precise about what is being measured. |
| **SR8** | Section 6 has seven paragraphs; merge "The headline win is real" and "Streaming deployment" (both essentially recap §5.1 framing), and split the over-long Limitations paragraph into its constituent points. |

---

## 6. Minor / cosmetic issues

- **Figure 6** caption (`mae vs jammae`) shows PEMS08 only.  Consider a footnote that the same pattern holds on the other three datasets.
- **Table 1** (positioning matrix).  The "Per-pos. mixture" column has `yes (neural experts)` for STAMImputer; consider adding `(classical anchors)` for MARST's row to make the comparison crisp at a glance.
- **`references.bib`** does not include Pi-Transformer; required only if SR6 is adopted.
- **Hyphenation.** "JamMAE" is one word in the prose but rendered as "Jam-MAE" or "JamMAE" inconsistently in some captions.  Pick one.

---

## 7. Simulated specialist reviews

### Friendly reviewer (PhD supervisor tone)
> "Strong, deployable, honest paper.  The leak audit and JamMAE
> consistency check are unusual and good.  Two things before submission:
> (i) the load-balancing ablation row, (ii) the compute-matched
> mini-MARST.  With those, ready to send.  Keep the negative finding on
> per-sensor learnable decay — that is a sign of a mature researcher,
> not a weakness."

### Harsh reviewer (top conference tone)
> "The contribution surface is thin.  (1) MARST is a mixture of three
> textbook predictors with a transformer trunk and a load-balancing
> loss adopted from Switch Transformer; nothing here is architecturally
> new.  (2) The KNN baseline is run at *k* = 5 while the authors' own
> Q12 sweep shows *k* = 10 is better — careless or selective.  (3) The
> parameter count is 17× the strongest baseline; the comparison is not
> fair.  (4) ImputeFormer is mentioned as state-of-the-art but never
> compared.  (5) The load-balancing claim is qualitative; show me the
> λ = 0 row.  Reject without these fixes.  With them, weak accept."

### TRC reviewer
> "The streaming constraint and leak audit address a real deployment
> concern that the imputation community has under-discussed.  The
> four-dataset benchmark and the statistical machinery are appropriate.
> Three required revisions: (i) PEMS04 in the missing-pattern table,
> (ii) the load-balancing aux-loss ablation, (iii) clarification of
> the KNN-*k* choice.  The cross-domain validation gap is acceptable for
> this venue; transportation-only is in scope."

### IEEE T-ITS reviewer
> "Methodology is sound, statistics are correct, reproducibility is
> good.  The parameter-efficiency comparison is the weakest link — I
> want a compute-matched row.  The deployment claim ('strictly
> streaming, leak-audited') is the differentiator and should be
> foregrounded in the Introduction more prominently.  Also: report
> JamMAE for every baseline in the extended-metrics table, not just
> LOCF."

---

## 8. Revision checklist (printable)

### Before submission — required
- [ ] **R1.** Add λ = 0 vs λ = 0.02 ablation row to Results.
- [ ] **R2.** Switch KNN baseline to *k* = 10 (or report sweep).
- [ ] **R3.** Add PEMS04 row to missing-pattern table.
- [ ] **R4.** Move per-sensor learnable decay out of Contributions list.
- [ ] **R5.** Add compute-matched mini-MARST row (hidden 64, 3 layers).

### Suggested — would strengthen
- [ ] **SR1.** Report JamMAE for every baseline.
- [ ] **SR2.** ImputeFormer reference row in Table 2.
- [ ] **SR3.** Speed-dataset example in qualitative figure.
- [ ] **SR4.** 2-hop neighbour-mean feature ablation.
- [ ] **SR5.** Numbered limitations subsection.
- [ ] **SR6.** Newer streaming-imputation citations.
- [ ] **SR7.** Reframe decay-sweep table as initialisation sensitivity.
- [ ] **SR8.** Tighten Discussion section structure.

### Cosmetic
- [ ] Figure 6 caption — note cross-dataset generalisation.
- [ ] Table 1 — add `(classical anchors)` to MARST's row.
- [ ] Hyphenation pass on `JamMAE`.

---

## 9. Bottom line

This is a **publishable paper after R1–R5 are addressed.**  None of the
required revisions invalidate the existing results; they fill gaps
that any reviewer will spot.  The science as it stands is honest, the
negative finding on per-sensor learnable decay is genuinely creditable,
and the streaming + leak-audit story is the strongest single asset.

R3 and R4 can be executed in the current session without new
experiments.  R1, R2, R5 require notebook re-runs on Kaggle.

---

*Generated 2026-06-02 from `ARTICLE.tex` and `marst-learnable-results.ipynb`.*
