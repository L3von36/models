# MARST: Contributions, Framing, and Honest Defence

A reference document for the thesis article on MARST. It catalogues the four legitimate contributions of the work, distinguishes them from things that are *not* contributions, and supplies the verbs and phrasings that keep each claim defensible without overselling.

This document is meant to be read alongside `THESIS.md`. Anywhere `THESIS.md` makes a claim, this document explains *why* the claim is defensible and *how* to articulate it.

---

## 0. Executive summary

MARST is a careful application of well-known techniques to a well-defined problem with rigorous evaluation. It is **not** a paradigm-shifting architectural invention. The four real contributions are:

1. A **specific architectural combination** — three causal classical anchors fed as input features to a strictly causal spatiotemporal transformer, blended by a learned per-position softmax mixture and refined by a residual correction. The exact combination has not been published before.
2. **State-of-the-art empirical results** on the four standard traffic benchmarks (PEMS-BAY, METR-LA, PEMS04, PEMS08), statistically significant under Wilcoxon Holm-corrected *p* < 0.001, against 18 baselines including the 2024 published state of the art (ImputeFormer, iTransformer).
3. **Diagnostic and engineering improvement**: identification of natural expert collapse in the anchor gate, followed by application of a Switch-Transformer-style load-balancing auxiliary loss that improves MAE on three of four traffic datasets.
4. **Cross-domain validation**: the same architecture wins on UCI Electricity Load Diagrams without retuning, with interpretable per-domain adaptation of the learned anchor mixture π.

These four are real, modest, defensible. They are exactly what one expects of a careful master-thesis-level paper in transportation. They are *not* enough to claim a NeurIPS-grade architectural novelty, and the article should not pretend otherwise.

---

## 1. The four contributions, with evidence and framing

### Contribution 1 — The MARST architecture (specific combination)

**Claim.** A strictly causal spatiotemporal transformer that imputes missing traffic readings by learning a per-position softmax mixture over three interpretable classical anchors (LOCF, HA, KNN), refined by a small residual correction. This specific combination of (a) classical anchors as input features, (b) per-position softmax gating, (c) residual correction, and (d) strict causal protocol has not appeared together in the spatiotemporal imputation literature.

**Supporting evidence.**
- A literature search across four angles (traffic imputation, multi-anchor mixture experts, classical-prior + neural, mixture-of-experts for time series) and a 2024 benchmarking survey of 11+ state-of-the-art methods (arXiv 2412.04733) confirmed that no published method feeds LOCF/HA/KNN as input features to a learned per-position softmax mixture under a strict streaming protocol.
- The closest cousin is **STAMImputer** (IJCAI 2025), which uses a softmax-gated mixture of *neural* experts (multi-head attention, low-rank graph attention, feed-forward) under bidirectional attention — not classical anchors under strict causality.
- The hybrid statistical-neural paradigm itself has strong precedent (ES-RNN won M4 2018; Wide & Deep at Google 2016; N-BEATS at ICLR 2020) but has not been instantiated for streaming traffic imputation.

**Where it lives in the code.** Cell 4 of `marst-multigpu.ipynb` defines `MaskedSTTransformerMARST`. The forward pass in §5 of `CODE_EXPLANATION.md` maps directly to the architectural description in §3.3 of the thesis article.

**Defensible framing (use these verbs).**
- "We **propose** MARST, a strictly causal spatiotemporal transformer that…"
- "MARST **applies** the hybrid statistical-neural paradigm of ES-RNN [Smyl 2020], Wide & Deep [Cheng et al. 2016], and N-BEATS [Oreshkin et al. 2020] to streaming traffic imputation."
- "To our knowledge, this **specific combination** has not been published in the imputation literature."

**Indefensible framing (avoid).**
- "We **invent** the multi-anchor mixture-of-experts paradigm." → no, that's Mixtral / Switch Transformer / STAMImputer.
- "We are the **first** to use LOCF / HA / KNN for imputation." → no, those are decades old.
- "Our architecture is **fundamentally novel**." → no, every component is standard; only the combination is.

---

### Contribution 2 — State-of-the-art empirical results

**Claim.** MARST attains the lowest Mean Absolute Error on every one of the four standard traffic benchmarks (PEMS-BAY, METR-LA, PEMS04, PEMS08) against 18 strong baselines including the 2024 published state-of-the-art ImputeFormer and iTransformer. Wins are statistically significant under a Wilcoxon signed-rank test with Holm correction at *p* < 0.001 on every dataset. The same model also wins on JamMAE — the regime-restricted MAE on congested timestamps — on every dataset.

**Supporting evidence.**
- §5.1 and §5.2 of `THESIS.md` reports the headline 3-seed numbers and Wilcoxon test outputs.
- The pipeline automatically prints `>> JamMAE leader matches MAE leader: 'MARST (ours, multi-anchor)'` on every dataset (see §6.3 of `THESIS.md`).
- The full extended-metrics table (MAE, JamMAE, RMSE, MedAE, MAPE %, R², Pearson, MBE, Hit@τ at two tolerances) is in `REPORT.md`.

**Why this is a contribution.** Empirical superiority over 18 baselines including the most recent published methods is the standard bar for transportation papers. The Wilcoxon test plus Holm correction is a rigorous statistical defence; the JamMAE consistency check is a domain-appropriate additional measure. The wins are not single-seed lucky runs — they are 3-training-seed × 5-eval-mask = 15-evaluation averages with tight standard deviations.

**Defensible framing.**
- "MARST **attains** the lowest MAE on all four standard traffic benchmarks."
- "Wins are **statistically significant** under the Wilcoxon signed-rank test (Holm-adjusted *p* < 0.001)."
- "MARST is the **JamMAE leader** on every benchmark."

**Indefensible framing.**
- "MARST **dominates** all prior work." → tone too strong; specify the four datasets.
- "MARST is **provably optimal**." → no, you have no convexity / regret bound.

---

### Contribution 3 — Expert-collapse diagnosis and load-balancing fix

**Claim.** We observe that the learned anchor gate π naturally collapses onto a dominant anchor per dataset (e.g. HA carries 78 % of weight on PEMS04) — a classical mixture-of-experts pathology documented in the Switch Transformer literature. We **adopt** the load-balancing auxiliary loss from Switch Transformer [Fedus, Zoph, Shazeer 2021] and apply it for the first time, to our knowledge, to a classical-anchor mixture in spatiotemporal imputation. With λ = 0.02 the loss preserves soft per-dataset specialisation while breaking the expert-collapse pattern and improves MAE on three of four datasets (METR-LA, PEMS04, PEMS08; PEMS-BAY shows a ~1 % regression).

**Supporting evidence.**
- The natural collapse pattern is observable in the per-dataset π distribution: PEMS-BAY → LOCF 0.55–0.64, PEMS04 → HA 0.76–0.85, PEMS08 → HA 0.70–0.79.
- After adding the load-balance loss at λ = 0.02 (3-seed protocol):
  - PEMS-BAY MAE: 1.4488 → 1.4620 (regression of +0.013, 1 %)
  - METR-LA MAE: 3.3492 → 3.3287 (improvement of −0.021)
  - PEMS04 MAE: 20.123 → 20.019 (improvement of −0.103)
  - PEMS08 MAE: 16.380 → 15.976 (improvement of −0.404, 2.5 %)
- A λ-sensitivity sweep (0, 0.02, 0.05) demonstrates the win is robust to the weight choice in this range.
- Reference: Fedus, Zoph, Shazeer (2021). "Switch Transformers: Scaling to Trillion Parameter Models." arXiv:2101.03961.

**Why this is a contribution.** The diagnostic is observable, reproducible, and aligns with a well-documented MoE failure mode. The fix is a direct application of a known technique, but the application *to a spatiotemporal anchor-mixture imputation gate* has not been published. The empirical impact is measurable: 2.5 % MAE improvement on PEMS08 is well outside the noise envelope.

**Defensible framing.**
- "We **identify** that the anchor gate exhibits the natural mixture-of-experts expert-collapse pattern [Fedus et al. 2021]."
- "We **adopt** the Switch-Transformer-style load-balancing auxiliary loss and apply it to the anchor gate. To our knowledge this is the **first application** of MoE load balancing to a classical-anchor mixture in spatiotemporal imputation."
- "At λ = 0.02 the load-balanced variant improves MAE on three of four datasets, with a 2.5 % MAE reduction on PEMS08."

**Indefensible framing.**
- "We **invented** the load-balancing aux loss." → no, Fedus et al. 2021.
- "We **discovered** mixture-of-experts collapse." → no, well-documented since 2017.

---

### Contribution 4 — Cross-domain validation and per-domain interpretability

**Claim.** The same MARST architecture with identical hyperparameters is evaluated on UCI Electricity Load Diagrams 2011–2014. MARST attains 29.47 ± 0.10 kWh MAE, reducing error by 25.4 % over Historical Average. The learned anchor mixture π adapts to the new domain — HA dominates at 0.78–0.89 (vs ~0.78 on traffic flow datasets), KNN drops to 0.00 — without any per-domain tuning. The interpretability pattern is consistent: temporal-persistence-dominant datasets (slow-changing freeway speeds on PEMS-BAY) yield high LOCF weight; seasonality-dominant datasets (flow, electricity load) yield high HA weight; KNN is consistently the smallest contributor.

**Supporting evidence.**
- §5.6 and §5.5 of `THESIS.md` reports the cross-domain MAE and the cross-domain π table.
- The 3-seed standard deviation on electricity is 0.10 kWh — very tight, indicating the result is reproducible and robust.
- The π distribution is consistent across all five seeds and across all five datasets.

**Why this is a contribution.** Most traffic imputation methods either (a) don't validate on non-traffic data or (b) require per-domain retuning to work. MARST does neither — same architecture, same hyperparameters, automatic per-domain π adaptation. This strengthens the architectural claim from "a method for traffic" to "a general method demonstrated on transportation data."

**Defensible framing.**
- "We **demonstrate** that the same architecture and hyperparameters generalise to UCI Electricity Load Diagrams without retuning."
- "The learned anchor mixture π **adapts** automatically to the dominant regularity of each domain."
- "This is a **strengthening** of the architectural claim, not a separate methodological contribution."

**Indefensible framing.**
- "MARST is **universal**." → no, it's been validated on 5 datasets across 2 domains.
- "We **solve** electricity load imputation." → no, you applied your traffic model and got measurable improvement.

---

## 2. What is *not* a contribution (honest transparency)

These are things that might *look* like contributions if framed creatively, but are not. Avoid claiming them:

| Item | Why it isn't a contribution |
|---|---|
| The spatiotemporal transformer backbone | Standard architecture. Self-attention with causal masking has been in the literature since 2017. |
| The three classical anchors (LOCF, HA, KNN) | All decades old. Anyone could think of using them. |
| Mixing classical predictions with a neural network | ES-RNN won M4 in 2018 doing exactly this. Wide & Deep from 2016. The *strategy* is well-established. |
| Mixture-of-experts with softmax gating | Originally from Shazeer et al. 2017. Switch Transformer (2021), Mixtral (2023), STAMImputer (2025). |
| The load-balancing aux loss | Fedus et al., Switch Transformer 2021. You applied it; you didn't invent it. |
| Residual learning | He et al., ResNet 2016. |
| Causal masking in self-attention | Vaswani et al., original Transformer paper 2017. |
| Jam-weighted loss | A small recipe choice. Don't claim it as a "novel domain-specific loss." |
| Sparsity curriculum | A small training recipe. Don't claim it as a novel curriculum-learning insight. |
| The leak audit protocol | This is scientific hygiene, not a contribution. Mention it as a methodological discipline, not as a "new evaluation framework." |
| JamMAE as a metric | Defining a regime-restricted MAE is trivial. Mention as a domain-appropriate evaluation choice, not as a "new metric." |

In the thesis article, frame these as **engineering choices** or **standard techniques**, not as contributions. Reviewers will respect the discipline.

---

## 3. Positioning matrix

The single most useful figure for the thesis introduction is this positioning matrix. It immediately tells a reviewer what gap MARST fills.

| Method | Year | Causal | Classical priors as inputs | Per-position softmax mixture | PEMS-BAY / METR-LA |
|---|---|---|---|---|---|
| BRITS [Cao 2018] | 2018 | bidirectional | no | no | yes |
| GRIN [Cini 2022] | 2022 | bidirectional | no | no | yes |
| SAITS [Du 2022] | 2022 | no | no | no | – |
| PriSTI [Liu 2023] | 2023 | no (diffusion) | no | no | yes |
| ImputeFormer [Nie 2024] | 2024 | no | no | no | yes |
| STAMImputer [IJCAI 2025] | 2025 | bidirectional | no | yes (neural experts) | no |
| **MARST (ours)** | – | **yes** | **yes (LOCF + HA + KNN)** | **yes (per sensor, time)** | **yes (all four)** |

The last row identifies the gap. The "yes / no" entries are individually verifiable from the cited papers; the matrix is defensible if challenged.

---

## 4. Recommended verbs by contribution type

| Verb | When to use it | Example |
|---|---|---|
| propose | When you are presenting a thing you built | "We **propose** MARST, an…" |
| apply | When you are using a known technique in a new context | "We **apply** the hybrid statistical-neural paradigm to streaming traffic imputation." |
| adopt | When you are using a technique from a different field, citing the source | "We **adopt** the Switch-Transformer load-balancing loss [Fedus et al. 2021]." |
| identify | When you are pointing out a problem or phenomenon | "We **identify** that the anchor gate exhibits expert collapse." |
| demonstrate | When you are providing empirical evidence of a property | "We **demonstrate** cross-domain generalisation on UCI Electricity." |
| attain | When you are reporting a result | "MARST **attains** the lowest MAE on…" |
| introduce | Reserved for things that are genuinely yours | (Use sparingly. The MARST decomposition is the only candidate.) |

Avoid: **invent**, **discover**, **revolutionise**, **fundamentally novel**, **first ever**, **paradigm shift**. These overstate the contribution and undermine reviewer trust.

---

## 5. The one-paragraph defence brief

If your advisor or a reviewer asks "what is actually new about MARST?", reply with the following paragraph verbatim:

> "MARST proposes a specific architectural combination that has not been published before: classical causal anchors (LOCF, HA, KNN) fed as input features to a strictly causal spatiotemporal transformer, blended by a learned per-position softmax mixture and refined by a residual correction. The hybrid statistical-neural paradigm itself has strong precedent in ES-RNN, Wide & Deep, and N-BEATS, but its instantiation for streaming spatiotemporal traffic imputation is, to our knowledge, novel. Beyond the architecture, the paper additionally identifies natural expert collapse in the anchor gate, adopts a Switch-Transformer-style load-balancing auxiliary loss to address it, and reports a measurable improvement on three of four traffic datasets. Finally, MARST validates cross-domain generality on UCI Electricity Load Diagrams without hyperparameter retuning, providing inspectable interpretability through the per-domain anchor mixture distribution. The architectural components are standard; the empirical wins are real and statistically significant; the load-balancing diagnostic and the cross-domain validation are honest contributions at a master-thesis-paper scale, not paradigm-shifting findings."

This is calibrated. It claims exactly what is defensible and concedes what is not. It will hold up under thoughtful questioning.

---

## 6. Things to do before submitting

Before sending the thesis or article anywhere, verify these items against the document:

- [ ] The thesis Introduction lists exactly the four contributions in §1 of this document. Not three (overselling makes you look careless). Not five (claiming JamMAE or the leak audit would look amateur).
- [ ] Every "we propose / introduce" verb is checked against §4 above.
- [ ] The positioning matrix in §3 above appears in or near the Related Work section.
- [ ] The cross-domain electricity result is positioned as "a generality validation," not as a separate contribution.
- [ ] The Discussion section explicitly concedes the limitations: parameter count vs iTransformer, mixed anchor ablation on speed datasets, the soft-LOCF wiring detail you discovered.
- [ ] The Conclusion does not use the word "novel" more than once. The word "propose" can appear several times. The word "demonstrate" can appear as often as needed.

---

## 7. Bottom line

You have four real contributions for a transportation thesis article. They are calibrated, defensible, and emerged from careful engineering and evaluation. None of them would alone carry a top-tier ML conference paper, but together they form a complete master-thesis-level contribution that will hold up under reviewer scrutiny at a transportation venue (TRC, IEEE T-ITS, Transportmetrica).

Frame the work with calibrated verbs ("propose," "apply," "adopt," "identify," "demonstrate"), concede the limitations honestly, and resist the temptation to label things as "novel" that are merely "applied for the first time in this specific way." Reviewers respect the discipline; they punish the overreach.

The result is an article that says exactly what it should: *we built a careful application of well-known techniques to a real problem and showed measurable, statistically significant improvement under rigorous evaluation, with interpretable behaviour and cross-domain generality*. That is a thesis. That is enough.
