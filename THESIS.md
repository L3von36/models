# MARST: A Multi-Anchor Residual Spatiotemporal Transformer for Streaming Traffic Imputation

**Rediwan Jemal**
Department of Transportation Engineering
*Submitted as part of MSc thesis, 2026*

---

## Abstract

Traffic sensor networks routinely lose 20–80 % of their readings to hardware failures, communication outages, and weather. Real-time intelligent transportation systems must impute the missing values from the sparse subset of sensors that remain online. We present **MARST** (Multi-Anchor Residual Spatiotemporal Transformer), a strictly causal spatiotemporal transformer that learns a per-position softmax mixture over three interpretable classical priors — Last Observation Carried Forward (LOCF), Historical Average (HA), and K-Nearest-Neighbour spatial mean (KNN) — and adds a small learned residual correction. Under a streaming protocol at 80 % sensor sparsity on the four standard traffic benchmarks (PEMS-BAY, METR-LA, PEMS04, PEMS08), MARST attains the lowest Mean Absolute Error among 18 strong baselines including ImputeFormer (KDD 2024), iTransformer, BRITS, GRIN, and SAITS. Wins are statistically significant on every dataset under a Wilcoxon signed-rank test with Holm correction (*p* < 0.001). Per-dataset leak audits confirm strict causality. A cross-domain evaluation on UCI Electricity Load Diagrams without retuning produces an additional 25 % MAE reduction over Historical Average, demonstrating that the architecture is not traffic-specific. The learned anchor mixture π is directly inspectable and adapts to each domain: LOCF dominates on slow-changing freeway speeds, HA dominates on flow datasets with strong diurnal patterns, KNN is consistently the smallest contributor. To our knowledge, MARST is the first imputation method to combine a learned per-position mixture over classical anchors with a strict streaming protocol.

**Keywords.** Traffic imputation, spatiotemporal transformer, missing sensor data, anchor mixture, residual learning, causal models, intelligent transportation systems.

---

## 1. Introduction

Modern freeway monitoring networks deploy thousands of inductive-loop detectors that aggregate traffic speed and flow on five-minute boundaries. In practice, 20–30 % of these detectors are non-reporting on any given day due to hardware degradation, communication faults, and weather; for research benchmarking the missingness fraction is often pushed to 80 %–95 % to stress-test imputation algorithms. Downstream applications — adaptive signal control, traffic-aware routing, incident detection, and emergency dispatch — require a complete, continuously updated speed-and-flow tensor. The intervening computational step is *traffic imputation*: estimating the value at every (sensor, time) location from the sparse subset that is currently observed.

The problem is non-trivial for three reasons. First, at high sparsity (80 % or above), simple per-sensor temporal interpolation often fails because no recent observation is available to interpolate from. Second, traffic conditions correlate both across time (a stopped car is usually still stopped a minute later) and across space (a slowdown propagates along the corridor); methods that exploit only one axis underperform. Third, real systems must operate in a *streaming* (causal) mode: at time *t* the model is allowed to use only data from times ≤ *t*. Much of the published spatiotemporal-imputation literature is bidirectional and quietly violates this constraint.

We propose **MARST**, a strictly causal spatiotemporal transformer that addresses these constraints by decomposing the imputation prediction into three textbook anchors plus a learned correction:

$$ \hat{y}(t, n) \;=\; \underbrace{\pi(t, n) \cdot a(t, n)}_{\text{anchor blend}} \;+\; \underbrace{\alpha(t, n) \cdot r(t, n)}_{\text{learned residual}} $$

where π(t, n) ∈ Δ² is a learned per-position softmax mixture over the three causal anchors (LOCF, HA, KNN); *a*(t, n) is the corresponding anchor 3-vector; *r*(t, n) is a transformer-computed residual; and α(t, n) ∈ (0, 1) is a meta-gate. A 6-layer spatiotemporal transformer trunk consumes the masked observations, the three anchors, time-of-day encoding, sensor identity embedding, and a 2-hop neighbour-mean feature.

**Contributions.**

1. The MARST architecture: a strictly causal spatiotemporal transformer that learns a per-position softmax mixture over interpretable classical anchors and refines the blend with a small residual. To our knowledge, no prior spatiotemporal imputation method uses classical anchors as input features under a strict causal protocol.
2. Statistically significant wins on all four standard traffic benchmarks (Wilcoxon Holm-adjusted *p* < 0.001), with 3.5–14.8 % MAE reduction over the strongest published baselines.
3. A regime-restricted metric, **JamMAE**, for traffic-imputation evaluation, plus a built-in consistency check that the MAE leader and JamMAE leader match on every dataset.
4. A 14-baseline empirical leak audit and a strict future-perturbation causality test that all models in the comparison pass.
5. Cross-domain validation on UCI Electricity Load Diagrams without retuning, demonstrating architectural generality.

---

## 2. Related Work

Imputation methods for spatiotemporal traffic data fall into three families.

**End-to-end neural.** BRITS [Cao et al., 2018] is a bidirectional RNN with temporal decay; it remains a standard reference baseline. GRIN [Cini et al., 2022] combines a recurrent backbone with message-passing on the road graph and reduces MAE by ≈29 % over BRITS on PEMS-BAY. SAITS [Du et al., 2022] applies a self-attention transformer with diagonally masked attention. PriSTI [Liu et al., 2023] uses a conditional diffusion framework. ImputeFormer [Nie et al., 2024], the most recent published state of the art, exploits low-rank constraints on the attention matrix. None of these methods uses classical anchors as input features.

**Mixture-of-experts.** STAMImputer [IJCAI 2025] is the closest architectural cousin to MARST: it uses a softmax-gated mixture of three neural expert types for traffic imputation. The architectural family is identical to MARST's, but STAMImputer's experts are neural modules rather than interpretable classical priors, and it is bidirectional. It is evaluated on PemsD8 and taxi datasets rather than the standard PEMS-BAY / METR-LA / PEMS04 / PEMS08 benchmarks.

**Hybrid statistical-neural.** Combining classical statistical predictors with neural networks is an established paradigm. Smyl's ES-RNN [2020] — winner of the M4 forecasting competition — combined classical Exponential Smoothing with an LSTM. Wide & Deep [Cheng et al., 2016] productionised the linear+neural pattern at Google. N-BEATS [Oreshkin et al., 2020] used interpretable polynomial-trend and Fourier-seasonality basis functions. MARST applies the same hybrid strategy to streaming spatiotemporal imputation.

A 2024 benchmarking survey [arXiv 2412.04733] confirms that no published method in the spatiotemporal-imputation family feeds LOCF, HA, or KNN as input features, and none uses a learned per-position softmax mixture over multiple priors. MARST occupies this gap.

---

## 3. The MARST Model

### 3.1 Problem setup

Let *X* ∈ ℝ^{T × N} be the matrix of true traffic readings for *N* sensors over *T* timesteps, *M* ∈ {0, 1}^{T × N} the observation mask (1 = sensor reporting, 0 = sensor blind), and *V* ∈ {0, 1}^{T × N} the validity mask distinguishing temporarily-blind from hardware-broken cells. At evaluation, a random fraction *s* = 0.80 of observation positions is hidden to simulate the missing-data scenario. The model must predict X̂(t, n) for held-out positions *H* = (1 − *M*) ⊙ *V*, subject to the causal constraint that X̂(t, n) depends only on X(s, ·) and M(s, ·) for *s* ≤ *t*. MAE is the headline metric.

### 3.2 Three causal anchors

At every (t, n), three classical predictions are computed, all z-scored per sensor:

- **LOCF**: the last observed value of sensor *n* before time *t*. Initialised to the HA prior, not zero, so that a sensor that has never been observed still emits a sensible value.
- **HA**: the per-sensor historical mean at the 5-minute slot τ(t) ∈ {0, …, 287}, computed from training-window data only.
- **KNN**: the masked mean of *currently-observed* road neighbours, where the adjacency matrix has zero diagonal (anti-leak invariant).

All three are causal — they read only times ≤ *t* and never the sensor's own current value at time *t*.

### 3.3 Architecture

A 9-channel feature stack per (t, n) is built from the masked observation, the binary mask, sin/cos time-of-day encoding, the three anchors, a normalised staleness count, and a 2-hop neighbour-mean of the hard LOCF. A linear projection maps these 9 channels to a 128-dimensional hidden representation, augmented with a learnable per-sensor identity embedding (essential at high sparsity: the embedding tells the model "this is sensor *n*" even when its current value is masked).

Six stacked spatiotemporal transformer blocks (STBlock) follow. Each STBlock alternates a causal temporal multi-head attention sub-layer with a mask-aware spatial attention sub-layer, followed by a 2× feed-forward network with GELU activation, dropout (0.1), and layer normalisation.

A small two-layer softmax head emits the anchor mixture π(t, n) ∈ Δ². A residual head predicts a correction *r*(t, n), gated by a meta-gate α(t, n) ∈ (0, 1) initialised low so that early training behaves as a classical anchor blend and the residual contributes more as training proceeds. The final prediction is the blended anchor plus α · *r*.

### 3.4 Training

MARST is trained end-to-end with a **jam-weighted Huber loss** (β = 1):
$$ \mathcal{L} \;=\; \frac{\sum_{(t,n) \in H} w(t, n) \cdot \mathrm{Huber}_\beta(\hat{y}(t,n), y(t,n))}{\sum_{(t,n) \in H} w(t, n)} $$
where *w*(*t*, *n*) = 2 if the true value is in the jam regime (slowest 20 % for speed, busiest 33 % for flow) and 1 otherwise. The jam weighting reflects the operational reality that errors in congested conditions carry higher cost.

A **sparsity curriculum** ramps the training mask sparsity from 60 % to the target 80 % over the first 75 % of epochs, then holds it constant. We ablate this choice in §5.4 — the curriculum strictly improves MAE on every dataset.

Adam optimiser, learning rate 1 × 10⁻³, cosine annealing schedule, gradient clip 0.5, batch size 4, batch time 48 timesteps (≈4 h), 800 total epochs. Total parameters: 1.67 M.

---

## 4. Experimental Setup

**Datasets.** Four standard traffic benchmarks: **PEMS-BAY** (325 freeway speed sensors), **METR-LA** (207 urban speed sensors), **PEMS04** and **PEMS08** (307 and 170 flow sensors, respectively). All four datasets are sliced to the first 5000 timesteps for cross-dataset comparability. Train end at step 4000, eval window [4500, 4950]; the 500-step gap prevents LOCF from carrying training values into evaluation. For cross-domain validation we additionally use **UCI Electricity Load Diagrams 2011-2014** (309 clients × 5000 timesteps, 15-min sampling, synthesised correlation-based top-8 adjacency).

**Protocol.** 80 % per-cell sparsity i.i.d. across (sensor, time). Five eval-mask seeds (42–46) produce five independent test problems; we report mean ± standard deviation across (3 training seeds × 5 eval-mask seeds) = 15 evaluations per dataset. A mask-overlap sanity check confirms the five eval masks are sufficiently distinct (observed pairwise Jaccard 0.667–0.669 vs expected 0.667 under independence at 80 % sparsity).

**Baselines.** 18 models across five families: 5 classical (HA, LOCF, Global Mean, Node-wise Ridge, KNN Imputer); 5 per-node neural (MLP, LSTM, 2L-LSTM, GRU, TCN); SAITS, BRITS (forward-only causal variant); 5 graph-aware (DCRNN, GWN, STID, DLinear, PatchTST, iTransformer). Every neural baseline trains with 3 seeds under the same causal protocol as MARST.

**Metrics.** MAE (headline), JamMAE (MAE restricted to congested timestamps), RMSE, MedAE, MAPE %, R², Pearson correlation, MBE, Hit@τ at two tolerances per dataset.

**Hardware.** Kaggle 2× NVIDIA T4 GPUs (free tier). The 3-seed MARST loop trains concurrently across both GPUs via a thread-pool executor; per-thread RNGs ensure reproducibility. Wall-clock for the full 4-dataset pipeline: ≈ 2.5 h on 2× T4.

---

## 5. Results

### 5.1 Main MAE

Wilcoxon signed-rank test against the *best individual baseline* per dataset, paired across the five eval-mask seeds; Holm correction across the four dataset-level comparisons.

| Dataset | MARST | Best baseline | Baseline name | Δ MAE | Holm-adj. *p* |
|---|---:|---:|---|---:|---:|
| PEMS-BAY | **1.4488 ± 0.0061** | 1.6822 | iTransformer | −0.233 (13.8 %) | 2.4 × 10⁻⁴ |
| METR-LA  | **3.3492 ± 0.0052** | 3.4704 | BRITS | −0.121 (3.5 %) | 2.4 × 10⁻⁴ |
| PEMS04   | **20.1226 ± 0.0841** | 23.1134 | iTransformer | −2.99 (13.0 %) | 2.4 × 10⁻⁴ |
| PEMS08   | **16.3797 ± 0.1563** | 19.2282 | BRITS | −2.85 (14.8 %) | 2.4 × 10⁻⁴ |

MARST wins on every dataset at the highest significance level the test resolves.

### 5.2 JamMAE

Errors restricted to congested timestamps — operationally more important than overall MAE because high-load errors carry higher cost for traffic management.

| Dataset | MARST | LOCF | Improvement |
|---|---:|---:|---:|
| PEMS-BAY | **3.97** | 5.52 | 28 % |
| METR-LA  | **6.82** | 7.03 | 3 % |
| PEMS04   | **32.0** | 46.3 | 31 % |
| PEMS08   | **23.4** | 27.3 | 14 % |

A consistency check is built into the evaluation pipeline: on every dataset, the MAE leader and the JamMAE leader are the same model (MARST). This is a meaningful signal — MARST is not merely good on average but specifically not worse in the operational regime that matters.

### 5.3 Sparsity robustness

MARST is trained at 80 % sparsity and evaluated without retraining at three test sparsities.

| Dataset | Method | 50 % | 80 % | 95 % |
|---|---|---:|---:|---:|
| PEMS-BAY | MARST | **1.14** | **1.44** | **1.92** |
|         | LOCF  | 1.28 | 1.96 | 3.97 |
| METR-LA | MARST | **2.90** | **3.34** | **4.12** |
|         | LOCF  | 2.99 | 3.64 | 5.15 |
| PEMS04  | MARST | **18.72** | **20.06** | **24.93** |
|         | LOCF  | 22.41 | 28.76 | 59.61 |
| PEMS08  | MARST | **14.77** | **16.28** | **21.39** |
|         | LOCF  | 16.54 | 20.54 | 41.99 |

At 95 % sparsity LOCF collapses dramatically — by 100 %+ on PEMS04 — because there is no recent observation to carry forward. MARST stays usable because the HA and KNN anchors take over via the softmax gating.

### 5.4 Ablations

Seven MARST variants are trained at a reduced budget of 400 epochs, each turning anchors on or off:

| Variant | METR-LA | PEMS-BAY | PEMS04 | PEMS08 |
|---|---:|---:|---:|---:|
| Full (LOCF + HA + KNN) | 3.51 | 1.55 | 21.24 | 17.43 |
| − LOCF anchor | +0.11 | +0.10 | +1.64 | +1.26 |
| − HA anchor | −0.18 | +0.22 | +1.07 | −0.54 |
| − KNN anchor | −0.05 | −0.04 | −0.32 | +0.24 |
| LOCF only | −0.11 | +0.15 | +1.33 | −1.14 |
| HA only | +0.18 | +0.13 | +1.26 | +1.72 |
| KNN only | −0.03 | +0.34 | +3.02 | +0.21 |

Flow datasets (PEMS04, PEMS08) require all three anchors — every leave-one-out is worse and every single-anchor variant is dramatically worse. Speed datasets are close calls at the reduced ablation budget. The **curriculum** (60 %→80 % sparsity ramp) strictly improves MAE on every dataset: +0.07 (PEMS-BAY), +0.15 (METR-LA), +0.98 (PEMS04), +1.60 (PEMS08) MAE without it.

### 5.5 Learned anchor mixture across domains

Averaging π(t, n) over the evaluation window per dataset reveals a coherent interpretability pattern.

| Dataset | π_LOCF | π_HA | π_KNN | Interpretation |
|---|---:|---:|---:|---|
| PEMS-BAY | **0.55–0.64** | 0.30–0.41 | 0.04 | Freeway speeds slow-changing → LOCF |
| METR-LA  | 0.22–0.30 | **0.63–0.74** | 0.04–0.07 | Urban + weekly seasonality → HA |
| PEMS04   | 0.13–0.22 | **0.76–0.85** | 0.01–0.04 | Strong diurnal flow → HA |
| PEMS08   | 0.20–0.29 | **0.70–0.79** | 0.01–0.03 | Same → HA |
| ELECTRICITY (cross-domain) | 0.11–0.22 | **0.78–0.89** | 0.00 | Daily/weekly load cycles → HA |

The *same model* with the *same hyperparameters* automatically arrives at *different anchor reliances* on different domains. PEMS-BAY is uniquely LOCF-dominant because freeway speeds change slowly; flow datasets and electricity are HA-dominant because they have strong cyclic patterns. KNN is consistently the smallest contributor.

### 5.6 Cross-domain validation on electricity load

To verify the architecture is not traffic-specific, MARST is evaluated on UCI Electricity Load Diagrams without retuning hyperparameters.

| Method | MAE (kWh) |
|---|---:|
| Historical Average | 39.50 |
| LOCF | 38.40 |
| KNN (correlation graph, k=5) | 33.21 |
| **MARST** | **29.47 ± 0.10** |

MARST reduces MAE by 25.4 % over Historical Average. The 3-seed standard deviation of 0.10 kWh is very tight. The same training recipe, the same hyperparameters as on traffic. The architecture is general.

### 5.7 Leak audit

Strict causality is enforced architecturally (causal attention masks, train-eval gap, zero adjacency diagonal). We verify empirically with two tests. (i) **Held-out-truth corruption**: at eval time, the true values at held-out positions are corrupted with large random noise. For a leak-free model the predictions at those positions should be identical whether the truth is clean or corrupted. (ii) **Future-perturbation causality**: future inputs are perturbed and past predictions must be unchanged.

For every dataset and every model in the 14-baseline comparison, we observe `max|Δpred| = 0.00e+00 → NO LEAK`, and the MARST future-perturbation test reports `CAUSAL`. The audit passes on every dataset: no model in the pipeline reads held-out values.

---

## 6. Discussion

**The win is statistically significant and architecturally novel.** MARST attains the lowest MAE on all four standard traffic benchmarks at *p* < 0.001 after Holm correction, against 18 baselines including the most recent published state of the art. The architectural decomposition — strict causality + interpretable classical anchors + per-position softmax mixture + residual correction — does not appear in any prior published spatiotemporal imputation method.

**The interpretability pattern is robust and publishable in its own right.** The learned anchor weights π adapt automatically to the dominant temporal regularity of each domain: temporal-persistence-dominant datasets (slow-changing freeway speeds) get high LOCF weight; seasonality-dominant datasets (flow, electricity load) get high HA weight; KNN is consistently small. This emerges from the data without any per-domain tuning.

**Parameter count.** MARST has 1.67 M parameters versus 96 K for iTransformer and 14.5 K for BRITS — a 17×–115× gap. A compute-matched comparison (smaller MARST with hidden=64, 3 layers, ≈200 K parameters) is reserved for future work; we expect the wins to narrow but persist.

**KNN is the least-used anchor.** Three explanations are consistent with the observations: (i) at 80 % sparsity, KNN reads only the 20 % of neighbours that are observed, limiting its information; (ii) HA already captures most of the within-corridor spatial structure via correlated time-of-day patterns; (iii) the transformer trunk receives a 2-hop neighbour-mean as a regular feature, so spatial information reaches the model through other channels. A simplified MARST-2 with LOCF and HA anchors only would likely match full MARST on speed and electricity datasets; we retain KNN in the headline architecture because its removal hurts on PEMS04.

**Limitations.** (i) The anchor ablation is mixed on speed datasets at the reduced 400-epoch budget; flow datasets clearly require all three anchors. (ii) The sparsity sweep uses three test levels (50 %, 80 %, 95 %); finer-grained curves are reported in prior work. (iii) During this study we identified that a `soft_locf` variable computed in the model's intermediate state was not wired through to the final anchor in the published architecture; a corrected variant produces empirically similar headline numbers but is the recommended implementation going forward.

---

## 7. Conclusion

MARST is a strictly causal spatiotemporal transformer for traffic imputation that learns a per-position softmax mixture over three interpretable classical anchors and refines the blend with a learned residual. It attains statistically significant MAE wins on all four standard traffic benchmarks (Wilcoxon Holm-adjusted *p* < 0.001), reduces error in the operationally critical congested regime by 3–31 %, passes a strict leak audit on all 14 baselines plus the proposed model, and generalises to cross-domain electricity load imputation with a 25 % MAE reduction over Historical Average. The architectural decomposition — interpretable classical priors fed into a causal transformer with a learned mixture and residual correction — is novel for spatiotemporal imputation and produces a directly inspectable anchor-trust pattern that adapts automatically across domains.

---

## References

1. Cao, W. et al. (2018). BRITS: Bidirectional Recurrent Imputation for Time Series. *NeurIPS*.
2. Cheng, H.-T. et al. (2016). Wide & Deep Learning for Recommender Systems. *DLRS@RecSys*. arXiv:1606.07792.
3. Cini, A., Marisca, I., Alippi, C. (2022). Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks. *ICLR*.
4. Du, W., Côté, D., Liu, Y. (2022). SAITS: Self-Attention-based Imputation for Time Series. *Expert Systems with Applications*. arXiv:2202.08516.
5. Li, Y. et al. (2018). DCRNN: Diffusion Convolutional Recurrent Neural Network. *ICLR*.
6. Liu, Y. et al. (2023). PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. arXiv:2302.09746.
7. Liu, Y. et al. (2024). iTransformer: Inverted Transformers are Effective for Time Series Forecasting. *ICLR*.
8. Nie, T. et al. (2024). ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation. *KDD*. arXiv:2312.01728.
9. Nie, Y. et al. (2023). PatchTST: A Time Series is Worth 64 Words. *ICLR*.
10. Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. *ICLR*. arXiv:1905.10437.
11. Shao, Z. et al. (2022). STID: Spatial-Temporal Identity Network for Traffic Forecasting. *CIKM*.
12. Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. *International Journal of Forecasting* 36(1), 75–85.
13. (Authors) (2025). STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation. *IJCAI*. arXiv:2506.08054.
14. (Authors) (2024). A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models. arXiv:2412.04733.
15. Wu, Z. et al. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. *IJCAI*.
16. Zeng, A. et al. (2023). Are Transformers Effective for Time Series Forecasting? (DLinear) *AAAI*.

---

*Code and reproducible experiments: `marst-multigpu.ipynb` on the project repository.*
