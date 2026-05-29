# Related Work: Streaming/Causal Spatiotemporal Imputation

**Compiled for MARST positioning.** Last updated: 2026-05-29.

This document catalogs published work on the streaming/causal/online imputation
protocol that MARST adopts. It evaluates each candidate for: (a) whether it is
genuinely streaming-causal (no future peeking), (b) whether it is spatiotemporal,
(c) whether it can serve as a direct comparison baseline for MARST.

---

## TL;DR

| Result | Status |
|---|---|
| Direct precedent for "streaming imputation" | **BayOTIDE (ICML 2024)** — found |
| Direct precedent that is *also* neural + spatiotemporal | **None found.** This is MARST's gap. |
| Closest spatiotemporal-aware causal model | Forecasting architectures retrofitted to imputation (STID, DLinear, PatchTST, iTransformer, DCRNN, GWN) — already in MARST's benchmark |

**MARST's contribution:** the first **neural, spatiotemporal-aware, multi-anchor** streaming imputation method. BayOTIDE established the streaming protocol but uses a Bayesian/GP formulation with independent channels and no graph; MARST extends the protocol to deep spatiotemporal models that exploit sensor-graph structure.

---

## Tier 1 — Direct Precedent

### BayOTIDE (ICML 2024)

- **Full title:** "BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition"
- **Authors:** Shikai Fang, Qingsong Wen, Yingtao Luo, Shandian Zhe, Liang Sun
- **Venue:** Proceedings of the 41st International Conference on Machine Learning, Vienna, 2024 (PMLR 235)
- **Code:** https://github.com/xuangu-fang/BayOTIDE
- **arXiv:** https://arxiv.org/abs/2308.14906

#### Key quote (from page 2, second column of the paper)
> *"To the best of our knowledge, BayOTIDE is the first online probabilistic imputation method of multivariate time series that could fit streaming data well."*

#### Protocol
- **Strictly online inference.** At time t_{n+1}, the posterior uses only D_{tn} = {y_1, …, y_n} ∪ y_{n+1}. See their Equation 10 and Algorithm 1.
- **Closed-form sequential Bayesian updates.** Kalman filter + RTS smoother; no future timesteps consumed during inference.
- **Their Table 1 explicitly categorizes inference manner.** BayOTIDE = "online"; every comparison method (TIDER, statistical baselines, DNN-based, diffusion-based including CSDI and CSBI) = "offline". The streaming-vs-offline distinction is theirs, not invented here.

#### Architecture (high level)
- Functional decomposition: time series X(t) = U V(t), where V(t) = [trend factors, seasonality factors]
- Each factor is a Gaussian Process with Matérn kernel (trend) or periodic kernel (seasonality)
- GPs converted to LTI-SDE → state-space model → online inference via Kalman filter
- D_r trend factors + D_s seasonality factors (small numbers, e.g., D_r=1, D_s=3 in synthetic experiments)
- Bayesian: uncertainty quantification via posterior variance

#### Datasets used by BayOTIDE
- Traffic-Guangzhou (214 channels, 500 timestamps)
- Solar-Power (137 channels, 52,560 timestamps)
- Uber-Move (7,489 channels, 744 timestamps)
- **None of these overlap with PEMS-BAY / METR-LA / PEMS04 / PEMS08**, so direct numerical comparison to their reported tables is not possible without re-running their model on the PEMS benchmarks.

#### Baselines compared against in BayOTIDE
Deterministic:
1. SimpleMean — column-wise mean imputation
2. BRITS (Cao et al. 2018) — RNN with time decay (the original is bidirectional)
3. NAOMI (Liu et al. 2019) — bidirectional RNN with adversarial training
4. SAITS (Du et al. 2023) — transformer with self-attention
5. TIDER (Liu et al. 2023) — disentangled temporal representations

Probabilistic:
1. Multi-Task GP (Bonilla et al. 2008) — multi-output Gaussian process
2. GP-VAE (Fortuin et al. 2020) — Gaussian-process VAE
3. CSDI (Tashiro et al. 2021) — conditional diffusion
4. CSBI (Chen et al. 2023) — Schrödinger Bridge diffusion

Plus ablations of their own model.

#### What BayOTIDE is *not*
- Not neural (it's Bayesian/GP)
- Not spatiotemporal — treats channels as independent, no graph adjacency
- Not multi-anchor — has trend + seasonality decomposition but no LOCF/HA/KNN mixture
- Not evaluated on the standard PEMS-BAY / METR-LA traffic benchmarks

#### Citation framing for MARST's paper
> *"BayOTIDE (Fang et al., ICML 2024) established the streaming imputation protocol but uses Bayesian Gaussian-process decomposition over independent channels, leaving open the question of how a **neural, spatiotemporal-aware** model can achieve streaming imputation while exploiting sensor-graph structure and learned multi-anchor priors. MARST addresses this gap."*

---

## Tier 2 — Adjacent but Not a Direct Match

### ON-Traffic (arXiv 2025-03)

- **Full title:** "ON-Traffic: An Operator Learning Framework for Online Traffic Flow Estimation and Uncertainty Quantification from Lagrangian Sensors"
- **Authors:** Jake Rap, Amritam Das (Eindhoven University of Technology)
- **arXiv:** https://arxiv.org/abs/2503.14053
- **Code:** https://github.com/STC-Lab/ON-Traffic

#### Why not a direct match
- Uses **Lagrangian (probe vehicle) sensors**, not static loop detectors. Different sensor model entirely.
- Uses **operator learning** (DeepONet variants: VIDON-based branch + trunk + nonlinear decoder) — fundamentally different architecture family from MARST.
- **Receding-horizon scheme** has both past horizon Δ_past *and* prediction horizon Δ_pred. So it's part-imputation, part-forecasting, not pure imputation.
- Includes PINN-style physics-based losses, requires fundamental-diagram-aware modeling.

#### What is shared
- "Online" framing — continuous processing as data arrives.
- Uncertainty quantification.
- Causal in the sense that predictions at time t_c only use measurements from t ∈ [t_c − Δ_past, t_c].

#### Citation framing
Cite as an adjacent online traffic-state estimation work, but note that MARST addresses the static-sensor imputation problem (loop detectors, PEMS-style), not the moving-probe Lagrangian setting.

### SANNI (Lobachevskii J. Math. 2024)

- **Full title:** "SANNI: Online Imputation of Missing Values in Multivariate Time Series Based on Deep Learning and Behavioral Patterns"
- **Venue:** Lobachevskii Journal of Mathematics (Springer), 2024
- **Link:** https://link.springer.com/article/10.1134/S1995080224606854

#### Why not a direct match
- Designed for **behavioral pattern** imputation, not spatiotemporal traffic.
- Two-stage architecture (Recognizer + Reconstructor) operates over "snippets" of typical activity patterns.
- Not evaluated on traffic benchmarks; tested on activity/behavior datasets.
- Lower-tier venue than ICLR/NeurIPS/KDD/AAAI.

#### Citation framing
Brief acknowledgment as an alternative online imputation approach in different problem domains. Not a direct competitor.

---

## Tier 3 — Excluded by Protocol

These published methods are well-known imputation SOTA but use **bidirectional / offline** context, which is incompatible with MARST's streaming protocol. They should be cited in the paper with a one-paragraph justification for non-comparison.

| Method | Year/Venue | Architecture | Why excluded |
|---|---|---|---|
| **BRITS** | NeurIPS 2018 | Bidirectional RNN with time decay | Bidirectional |
| **NAOMI** | NeurIPS 2019 | Bidirectional RNN + adversarial training | Bidirectional |
| **GP-VAE** | AISTATS 2020 | Gaussian-process VAE | VAE encoder sees full window |
| **mTAN** | ICLR 2021 | Multi-time attention | Bidirectional attention |
| **CSDI** | NeurIPS 2021 | Conditional score-based diffusion | Both-sides self-attention |
| **GRIN** | ICLR 2022 | Bidirectional graph imputation network | Bidirectional |
| **SAITS (full)** | ESWA 2023 | Self-attention masked imputation | Bidirectional attention (you have a *causal* variant in your benchmark) |
| **TimesNet** | ICLR 2023 | Temporal 2D variation (Fourier blocks) | Uses full-window FFT |
| **TIDER** | ICLR 2023 | Disentangled temporal representations | Encoder sees full window |
| **PriSTI** | ICDE 2023 | Conditional diffusion + graph | Both-sides context |
| **CSBI** | NeurIPS 2023 | Schrödinger Bridge | Bidirectional |
| **ImputeFormer** | KDD 2024 | Low-rankness Transformer | Bidirectional attention (partial causal variant possible) |
| **GSLI** | AAAI 2025 | Graph structure learning + cross-time attention | Bidirectional |
| **Casper** | **CIKM 2024** | Causality-aware (SCM-style) GNN with Prompt-Based Decoder + Spatiotemporal Causal Attention | "Causal" refers to causal *inference* (SCM, backdoor paths), not temporal causality; uses bidirectional context |
| **PSW-I** | **ICLR 2025** | Optimal-transport alignment using Pairwise Spectrum Distance (frequency-domain DFT) + Selective Matching Regularization | Iterative batch-based refinement of imputed values, not streaming; operates on full sequences |
| **KAI** | 2026 (Springer) | Kalman-Attention Imputation — Kalman **smoothing** + self-attention with mask recognition | Kalman *smoother* (not just filter) is bidirectional; uses both forward and backward Kalman passes |
| **MIRACLE** | 2021 | Causally-aware imputation via learning missing data mechanisms | "Causal" in causal-inference sense (missing-mechanism modeling); not temporal causality |
| **Causal View of TSI** | IJCAI 2025 | Variational inference with normalizing-flow architecture under nonlinear ICA framework | Identifiability of missing mechanisms (MCAR/MAR/MNAR); not temporal-causal protocol |

#### Suggested paper paragraph (for "Related Work / Why these are excluded")
> *"A substantial body of recent imputation work — including BRITS, NAOMI, GP-VAE, mTAN, CSDI, GRIN, SAITS, TimesNet, TIDER, PriSTI, CSBI, ImputeFormer, GSLI, Casper, KAI, and PSW-I — operates under the offline protocol, using bidirectional context or iterative batch refinement to fill each missing value. Following BayOTIDE (Fang et al., 2024), we treat the streaming (causal, past-only) protocol as a distinct and harder task, motivated by online deployment scenarios where future observations are unavailable. We therefore do not compare against these offline methods directly, but acknowledge their strong reported performance in the offline setting."*

#### Note on naming confusion with "causal"
The word "causal" appears in three distinct senses across this literature; the MARST paper should be explicit about which sense it intends to avoid reviewer confusion:
1. **Temporal causality** — predictions at time t use only observations at times ≤ t. This is MARST's sense.
2. **Causal inference** (Pearl-style) — distinguishing cause from correlation via structural causal models, backdoor paths, etc. Used by Casper, MIRACLE, "Causal View of TSI."
3. **Missing-mechanism causality** — modeling the cause of missingness (MCAR/MAR/MNAR). Used by MIRACLE.
Senses 2 and 3 are about *what causes data* and *what causes missingness*; sense 1 is about *what timesteps are available at inference*. Different problems, same word.

---

## What MARST's Benchmark Currently Contains

Of MARST's 17 baselines + MARST itself, classified against this landscape:

| Tier | Models | Count |
|---|---|---|
| Imputation-native, causal | LOCF, KNN Imputer, SAITS (causal variant), MARST itself | 4 |
| Classical / task-agnostic (used as imputation baseline) | HA, Global Mean (= BayOTIDE's SimpleMean), Ridge, MLP, LSTM, 2L-LSTM, GRU, TCN | 8 |
| Spatiotemporal forecasting architectures (causal by design) | DCRNN, GWN, STID, DLinear, PatchTST, iTransformer | 6 |
| **Total** | | **18** |

After the strategy reframe to "streaming/causal spatiotemporal imputation," the forecasting architectures are no longer awkward outsiders — they are **causal-by-construction baselines** that fit the protocol naturally. This reframes a perceived weakness as a defensible design choice.

---

## Recommended Additions to MARST's Benchmark

Ranked by effort vs. value.

### 1. BRITS-forward (high value, low effort)
- BayOTIDE used full BRITS as a baseline. The forward-only variant is the causal analog and fits MARST's protocol.
- You had this as "BRITS-lite" earlier; removed during the lite-cleanup. **Re-adding it as "BRITS (forward, causal)" is the right move now** that streaming/causal is the explicit protocol.
- Effort: ~30 lines of code, ~5 min training per dataset.

### 2. BayOTIDE itself, as a black-box baseline (high value, medium-high effort)
- Run their public code on PEMS-BAY / METR-LA / PEMS04 / PEMS08 using their default hyperparameters.
- Export predictions as `.npz`, load into MARST's evaluation pipeline for MAE/RMSE comparison.
- Skip CRPS/NLLK (their probabilistic metrics) — just use the posterior mean as the point prediction.
- Effort: 1–2 days. Mostly glue work (data loaders, hyperparameter tuning).
- Trade-off: yields a real number for the headline table, validates the streaming protocol comparison.

### 3. TIDER-causal (low-medium value, medium effort)
- TIDER (ICLR 2023) is a state-of-the-art deterministic offline imputation method.
- Causalizing it requires restructuring the seasonal/trend decomposition encoder to be left-padded only.
- Effort: ~150 lines + 1 training run per dataset.
- Optional — TIDER itself is not streaming, and the causalized version would be a custom variant rather than a faithful reproduction.

### 4. Multi-Task GP (online) — low value, high effort
- BayOTIDE compares against MultiTaskGP. A naïve forward-only Kalman-filter implementation is possible but redundant once BayOTIDE itself is added (BayOTIDE generalizes MTGP).
- Skip unless explicitly requested.

---

## Implications for the MARST Paper

### Positioning
1. **Cite BayOTIDE prominently** — in the abstract, introduction, and related work. This single citation does enormous work: it validates the protocol, prevents the "this isn't standard imputation" pushback, and gives MARST a clear differentiation axis.

2. **State the triple novelty over BayOTIDE explicitly:**
   - Neural (BayOTIDE is Bayesian/GP)
   - Spatiotemporal graph-aware (BayOTIDE treats channels independently)
   - Multi-anchor mixture (BayOTIDE uses single functional decomposition)

3. **Reframe the forecasting-derived baselines as "causal spatiotemporal baselines"** rather than apologetically labeling them "adapted from forecasting." They were always causal — that fits the streaming protocol perfectly.

### What not to do
- Don't add CSDI / GRIN / PriSTI / GP-VAE as baselines. They're explicitly offline.
- Don't apologize for the protocol difference — it's a deliberate, citable design choice (BayOTIDE precedent).
- Don't oversell as "first streaming imputation" — BayOTIDE is. Be precise: "first neural / spatiotemporal-aware / multi-anchor streaming imputation."

---

## Notes for Continued Research

The literature search underlying this document covered (searches conducted 2026-05-29):
- BayOTIDE (ICML 2024) — confirmed direct precedent
- ON-Traffic (March 2025) — different sensor model (Lagrangian probe vehicles), adjacent
- SANNI (2024, Lobachevskii J.) — different problem domain, brief mention
- Casper (CIKM 2024) — SCM causality, not temporal causality
- PSW-I (ICLR 2025, Optimal Transport for TSI) — alignment-based, batch-iterative; not streaming
- KAI (2026, Springer) — Kalman smoothing + attention; bidirectional smoother, not streaming
- MIRACLE (2021) — missing-mechanism causal inference; not temporal-causal
- Causal View of TSI (IJCAI 2025) — identifiability of missing mechanisms; not temporal-causal
- Awesome_Imputation curated list (GitHub) — comprehensive index, no other streaming candidates found
- Deep Learning for Multivariate Time Series Imputation Survey (IJCAI 2025) — provides offline/online taxonomy

**Verdict:** Across NeurIPS 2024, ICLR 2025, KDD 2024, AAAI 2025, CIKM 2024, IJCAI 2025, ICML 2024, and arXiv preprints through May 2026, **the only published method matching MARST's strict streaming-imputation protocol is BayOTIDE.** Several methods use the word "causal" but mean causal inference (MIRACLE, Casper, "Causal View of TSI") or refer to causal masks in attention (most forecasting models) rather than the streaming-inference protocol. This confirms MARST's positioning gap and reinforces the recommendation to cite BayOTIDE as the precedent and position MARST as its neural-spatiotemporal-multi-anchor extension.

**Possible follow-up searches** if positioning needs further validation:
- BayOTIDE follow-up work / citations of BayOTIDE
- Online recursive matrix factorization for traffic data (Yu et al. TRMF and successors)
- Continual learning for time series imputation
- Memory-augmented streaming neural networks for missing data

---

## Sources

- [BayOTIDE: Bayesian Online Multivariate Time Series Imputation (ICML 2024)](https://arxiv.org/pdf/2308.14906)
- [ON-Traffic: Operator Learning for Online Traffic Flow Estimation (2025)](https://arxiv.org/pdf/2503.14053)
- [Casper: Causality-Aware Spatiotemporal GNN for Imputation (CIKM 2024)](https://arxiv.org/pdf/2403.11960)
- [PSW-I: Optimal Transport for Time-Series Imputation (ICLR 2025)](https://haoxuanli-pku.github.io/papers/ICLR%2025%20-%20Optimal%20Transport%20for%20Time%20Series%20Imputation.pdf)
- [KAI: Scalable Kalman-Attention Imputation Method (Springer 2026)](https://link.springer.com/chapter/10.1007/978-3-032-18455-9_14)
- [MIRACLE: Causally-Aware Imputation via Learning Missing Data Mechanisms (2021)](https://arxiv.org/pdf/2111.03187)
- [Causal View of Time Series Imputation (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0532.pdf)
- [SANNI: Online Imputation via Behavioral Patterns (2024)](https://link.springer.com/article/10.1134/S1995080224606854)
- [Awesome Deep Learning for Time-Series Imputation (curated list)](https://github.com/WenjieDu/Awesome_Imputation)
- [Deep Learning for Multivariate Time Series Imputation: A Survey (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/1187.pdf)
- [Graph Structure Learning for Spatial-Temporal Imputation (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/download/32081/34236)
- [FastSTI: Fast Conditional Pseudo-Numerical Diffusion (2024)](https://arxiv.org/pdf/2410.15248)
- [SAITS: Self-Attention-based Imputation for Time Series (ESWA 2023)](https://arxiv.org/pdf/2202.08516)
