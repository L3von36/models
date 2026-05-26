# MARST — Novelty Positioning

Draft positioning / related-work text for the thesis. Citation keys match
`docs/references.bib` (`saits`, `imputeformer`, `grin`, `anchormoe`, `past`).

> **Status / caveats**
> - Novelty claims are hedged with "to our knowledge" — confirm with a Google
>   Scholar / Semantic Scholar / arXiv / institutional-thesis search and a
>   Turnitin similarity check before submitting.
> - The SOTA/quantitative claim is narrow (PEMS04, 80% sparsity, 3 seeds).
>   Re-run on PEMS08 before generalising.
> - For a graph-imputation audience, add at least one *quantitative* comparison
>   against PAST or Anchor-MoE, not just prose.

---

## Novelty and positioning

Transformer- and graph-based imputers have become the mainstream for sparse
spatiotemporal traffic data. Pure-attention models such as SAITS \cite{saits}
and the low-rank-induced ImputeFormer \cite{imputeformer}, and graph-recurrent
models such as GRIN \cite{grin}, learn the imputation map end-to-end from the
observed signal alone, with no explicit structural prior on the output. A
second, recent line of work introduces *gating* over multiple internal
representations — Anchor-MoE \cite{anchormoe} anchors a mixture of experts
around a data-driven mean, and PAST \cite{past} gates between learned *primary*
and *auxiliary* spatiotemporal pathways. Our model, the **Multi-Anchor Residual
Spatiotemporal Transformer (MARST)**, differs from both lines in *what* it
mixes. Rather than gating opaque learned experts, MARST forms, at every
(sensor, time) cell, a softmax mixture $\pi \in \Delta^2$ over **three
classical, interpretable, causal imputation estimators** —
last-observation-carried-forward (recency), the per-sensor time-of-day
historical average (seasonality), and the mean of *currently observed* graph
neighbours (spatial context). The blended anchor
$a = \pi_{\text{LOCF}}a_{\text{LOCF}} + \pi_{\text{HA}}a_{\text{HA}} + \pi_{\text{KNN}}a_{\text{KNN}}$
serves as a structured prior, and the spatiotemporal transformer trunk —
alternating causal-masked temporal attention and mask-aware spatial attention —
is tasked only with predicting a *residual* correction $r$, gated onto the
anchor by a meta-gate $\alpha$: $\hat{y} = a + \alpha r$. To our knowledge, no
prior work mixes these specific human-interpretable causal baselines as the
residual base of a masked spatiotemporal transformer.

This design yields two properties absent from the comparators above. First, the
gate $\pi$ is **directly interpretable**: because each anchor is a named
estimator, the learned mixture reveals *which prior the model trusts where*
(e.g., shifting weight from recency toward seasonality as a sensor's last
observation grows stale), rather than producing the uninterpretable weights of a
learned-expert MoE. Second, anchoring the prediction in strong classical
estimators and learning only the correction is what stabilises the model under
**extreme sparsity** (up to 95% blind), where a from-scratch attention map has
too little signal to fit. We further enforce strict **anti-leakage invariants**
throughout — a zero-diagonal adjacency so the spatial anchor never reads a
sensor's own masked value, fully causal anchors, and train-only statistics —
which much of the imputation literature does not make explicit.

We position MARST against its own single-anchor predecessor,
**MaskedSTTransformerV5**, as a controlled internal ablation: V5 shares MARST's
trunk, residual head, meta-gate, training budget, and seeds, but uses a *single*
`soft-LOCF` anchor. On PEMS04 (80% sparsity, three training seeds), the
multi-anchor mixture reduces MAE from $21.12$ to $20.13$ veh/5 min, a $4.7\%$
improvement that is significant under a paired test ($p = 5.6\times10^{-6}$) and
consistent across every secondary metric (RMSE, MedAE, MAPE, $R^2$, Pearson,
bias, and hit-rate). This isolates the multi-anchor design — rather than model
capacity or training procedure — as the source of the gain.
