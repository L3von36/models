# Related Work and Novelty Defence — MARST

A literature review situating MARST against the published landscape, structured around the question a reviewer is most likely to ask: **"LOCF, HA, and KNN are textbook methods from decades ago — what's actually new here?"**

The short answer: the *anchors* aren't new, but **using classical methods as inputs to a neural network and learning a per-position gate over them** is a well-established paradigm with at least a decade of precedent. MARST is the first application of that paradigm to streaming spatiotemporal traffic imputation. This document collects the citations and quantitative positioning needed to make that case to a reviewer.

---

## 1. The strategy MARST uses — published precedent

Combining classical statistical methods with neural networks is **a recognised paradigm with a decade of citations**. Three Tier-1 papers establish it:

### 1.1 ES-RNN — Smyl (2020), M4 competition winner

- **What it did.** Hand-coded **Exponential Smoothing** (a 1960s textbook method) combined with an **LSTM**. ES handles level, trend, and seasonality; the LSTM learns the deviations.
- **Result.** Won the M4 forecasting competition by a wide margin over 60+ submissions, including pure-neural and pure-statistical entries.
- **Quote.** "Pure machine learning and neural network methods performed worse than standard algorithms like ARIMA or Exponential Smoothing… the winner of the competition, with a solid margin, was Slawek's hybrid Exponential Smoothing-Recurrent Neural Networks (ES-RNN) method." (Uber Engineering blog summarising Smyl 2020)
- **Reference.** Smyl, S. (2020). "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting." *International Journal of Forecasting* 36(1):75–85.
- **How it maps to MARST.** ES = LOCF/HA-style anchor; LSTM = the transformer. The exact same logic ("classical structure + neural correction") applied to imputation instead of forecasting.

### 1.2 Wide & Deep — Cheng et al. (2016), Google

- **What it did.** Jointly trained a **wide linear model** with hand-crafted cross-product features (memorisation) and a **deep neural network** (generalisation). Both branches contribute to the final output.
- **Result.** Productionised at Google Play with over 1 billion active users. Demonstrated that the joint model significantly outperforms either wide-only or deep-only.
- **Reference.** Cheng et al. (2016). "Wide & Deep Learning for Recommender Systems." *DLRS@RecSys*. arXiv:1606.07792.
- **How it maps to MARST.** Wide = anchor branch (LOCF/HA/KNN, interpretable, hand-engineered); Deep = the transformer. Same complementarity argument.

### 1.3 N-BEATS — Oreshkin et al. (2020), ICLR

- **What it did.** Stacked MLP blocks projected onto **interpretable basis functions** — polynomials for trend, Fourier series for seasonality. Each block has a "backcast" (what it explains) and a "forecast" (what it contributes); later blocks see the residual.
- **Result.** Improved forecast accuracy by 11% over statistical benchmarks and 3% over the previous M4 winner (ES-RNN).
- **Reference.** Oreshkin, Carpov, Chapados, Bengio (2020). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." *ICLR*. arXiv:1905.10437.
- **How it maps to MARST.** N-BEATS uses **fixed mathematical bases** (polynomial, Fourier) as the structured components; MARST uses **operational priors** (LOCF, HA, KNN) appropriate to traffic imputation. The decomposition philosophy is the same.

### 1.4 Tier-2 precedents — hybrid statistical-neural forecasting

A broad class of published methods follows the same recipe:

- **LSTNet** (Lai et al. 2018, SIGIR) — augments a CNN/RNN with a classical autoregressive component; the AR branch handles scale variations the neural branch misses.
- **NBEATSx** (Olivares et al. 2022) — extends N-BEATS to accept exogenous statistical features.
- **DeepAR** (Salinas et al. 2020, IJF) — combines an LSTM with a probabilistic output head using classical distribution priors.
- **ARIMA-LSTM hybrids** — many published instances in COVID forecasting, electricity demand, financial volatility. Classical model fits trend/seasonality; neural network fits residuals.
- **Wang et al. (2021)**, "A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Mortality Modeling," arXiv:2112.08618 — explicit fusion of classical statistics with deep nets.

The general pattern is so well-established that recent surveys treat "hybrid statistical-neural" as a named family of methods.

---

## 2. The closest cousins in spatiotemporal imputation

These are the papers a reviewer will compare MARST against directly.

### 2.1 STAMImputer — Zhao et al. (2025), IJCAI — *closest in architectural spirit*

- **Architecture.** Mixture-of-Experts for traffic imputation. Three expert types: Temporal Experts (multi-head attention), Spatial Experts (low-rank graph attention), Observation Experts (feed-forward). Softmax gating combines their outputs.
- **Where it differs from MARST.**
  1. Experts are *neural modules*, not classical anchors. **No LOCF, HA, or KNN as inputs.**
  2. **Bidirectional**, not strictly causal. Not deployable as a live streaming system.
  3. Tested on PemsD8, SZ-Taxi, DiDi-SZ, NYC-Taxi. **Not PEMS-BAY or METR-LA.**
- **Reference.** "STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation." IJCAI 2025. arXiv:2506.08054.
- **Take-away.** Closest design family (softmax gate over multiple experts on traffic imputation), but the experts are neural and the protocol is offline. MARST's contribution of using **classical, interpretable anchors as the experts** is not present here.

### 2.2 ImputeFormer — Nie et al. (2024), KDD — *current SOTA on PEMS-BAY/METR-LA*

- **Architecture.** Transformer with low-rank constraints on the attention matrix; generalises across traffic, energy, and air quality.
- **Where it differs from MARST.** No anchors; no gating; robustness comes from low-rank inductive bias.
- **Reference.** Nie et al. (2024). "ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation." *KDD*. arXiv:2312.01728.
- **Take-away.** The strongest direct competitor on your headline numbers. Your iTransformer baseline already beats published ImputeFormer levels on most metrics, and MARST beats iTransformer significantly — so the cross-paper comparison is favourable.

### 2.3 GRIN — Cini et al. (2022), ICLR — *standard graph-imputation baseline*

- **Architecture.** Recurrent + message-passing graph neural network.
- **Result.** Reduces MAE by ~29% vs BRITS on PEMS-BAY (point-missing setting).
- **Reference.** Cini, Marisca, Alippi (2022). "Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks." *ICLR*. [GitHub repo](https://github.com/Graph-Machine-Learning-Group/grin).
- **Take-away.** The graph-imputation reference point everyone cites.

### 2.4 SAITS, BRITS — Du et al. (2022), Cao et al. (2018) — *classical neural baselines*

- **SAITS:** Self-attention transformer with diagonally-masked attention. Du, Côté, Liu (2022). arXiv:2202.08516.
- **BRITS:** Bidirectional RNN with temporal decay. Cao et al. (2018), *NeurIPS*.
- **Status.** Both already in your benchmark (#11 SAITS, #12 BRITS). MARST beats them significantly on every dataset (Holm-corrected p < 0.001).

### 2.5 Bridge-TS — (2025) — *closest in the "use a cheap prediction as a prior" idea*

- **Architecture.** Generative time-series imputation that takes the output of a **pretrained transformer** as a prior, then runs a data-to-data generation process on top. Explicitly explores "compositional priors" combining multiple pretrained estimators.
- **Where it differs from MARST.** Priors are heavy neural models, not interpretable classical methods. No softmax-gated mixture — uses diffusion-style refinement.
- **Reference.** "Exploiting the Prior of Generative Time Series Imputation." arXiv:2512.23832.
- **Take-away.** Same broad strategy of "feed the network a cheap guess and let it correct it," but at a much heavier compute scale and without interpretability.

### 2.6 SNI — Statistical-Neural Interaction (2026) — *closest in spirit on the tabular side*

- **Architecture.** Tabular imputation. Computes correlation-derived priors per feature, then **regularises** transformer attention toward the prior via a learned head-wise coefficient.
- **Where it differs from MARST.** Gating is *per-head*, not per-position. Priors are correlation matrices, not LOCF/HA/KNN. Task is tabular (MIMIC-IV, NHANES), not spatiotemporal traffic.
- **Reference.** "Statistical-Neural Interaction Networks for Interpretable Mixed-Type Data Imputation." arXiv:2601.12380.
- **Take-away.** Same general philosophy ("couple statistical priors with neural attention") but the priors and gating are different.

### 2.7 The 2024 imputation survey — confirmation that no one does *exactly* what MARST does

- **Reference.** "A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models." arXiv:2412.04733 (2024).
- **Coverage.** 11+ state-of-the-art methods including BRITS, GRIN, SAITS, PriSTI, ImputeFormer, STD-PLM, GCASTN, LATC, MagiNet, etc.
- **Key finding for our purposes.** None of the surveyed methods feed classical anchors (LOCF, HA, KNN) as input features. None use a learned per-position softmax mixture over multiple imputation priors. **MARST's design occupies a gap in the published literature.**

---

## 3. Positioning matrix

| Method | Year | Causal / streaming? | Classical priors as inputs? | Learned per-position mixture? | PEMS-BAY / METR-LA tested? |
|---|---|---|---|---|---|
| BRITS | 2018 | bidirectional | no | no | yes |
| GRIN | 2022 | bidirectional | no | no | yes |
| SAITS | 2022 | no | no | no | – |
| PriSTI | 2023 | no (diffusion) | no | no | yes |
| ImputeFormer | 2024 | no | no | no | yes |
| Bridge-TS | 2025 | no | neural priors | no (diffusion refine) | – |
| STAMImputer | 2025 | bidirectional | no | yes (over neural experts) | no (PemsD8, taxi) |
| SNI | 2026 | n/a (tabular) | yes (correlation) | per-head, not per-position | no (tabular) |
| **MARST (ours)** | – | **yes** | **yes (LOCF + HA + KNN)** | **yes (per (sensor, time))** | **yes (all 4)** |

The last row identifies the gap MARST fills.

---

## 4. The defence brief — a three-paragraph response to the textbook objection

If a reviewer writes "LOCF, HA, and KNN are decades old — what's actually new?", reply with:

> **The combination, not the components, is the contribution.** Combining classical statistical methods with neural networks is a recognised paradigm with a decade of high-impact precedent. The M4 forecasting competition (Makridakis et al. 2020) was won by Smyl's ES-RNN (Smyl 2020), an explicit hybrid of Exponential Smoothing and LSTM that beat 60+ submissions including pure-neural and pure-statistical methods. Wide & Deep (Cheng et al. 2016) productionised the same strategy at Google. N-BEATS (Oreshkin et al. 2020) extended the idea to interpretable basis decomposition. Our contribution is not the existence of classical anchors — it is their **specific application to streaming spatiotemporal imputation, with a per-position softmax gate over three causal anchors and a small residual correction**.

> **No prior imputation work uses this combination.** A 2024 benchmarking survey of 11+ state-of-the-art spatiotemporal imputation methods (BRITS, GRIN, SAITS, PriSTI, ImputeFormer, STD-PLM, GCASTN, …) confirms that none feed classical anchors (LOCF, HA, KNN) as input features and none use a learned per-position softmax mixture. The closest cousin is STAMImputer (IJCAI 2025), which uses softmax-gated mixture-of-experts for traffic imputation, but its experts are themselves neural modules (multi-head attention, graph attention, feed-forward), not interpretable classical priors, and it is offline/bidirectional rather than strictly causal.

> **The paradigm is established; the design and the empirical evidence are ours.** MARST is novel in (i) using interpretable classical anchors as features inside a spatiotemporal transformer, (ii) gating them **per (sensor, time)** rather than per-head or per-batch, (iii) preserving strict causality so the model is deployable as a live system (verified by a per-dataset leak audit that all 14 baselines and MARST pass), and (iv) demonstrating Wilcoxon-significant wins (Holm-corrected p < 0.001) across four standard traffic benchmarks (PEMS-BAY, METR-LA, PEMS04, PEMS08) against 18 strong baselines including the published SOTA.

This concedes the reviewer's narrow point (the anchors are textbook), establishes that the *strategy* is well-precedented (three citations from top venues), shows the *combination* is gap-filling (a recent survey confirms no one has done it), and closes with what is *uniquely* MARST's contribution (the four enumerated novelties + the empirical evidence).

---

## 5. Suggested paper introduction framing

> "Recent imputation work falls into three families: **(i) end-to-end neural methods** that learn from raw values (BRITS [Cao 2018], SAITS [Du 2022], GRIN [Cini 2022], ImputeFormer [Nie 2024]), **(ii) prior-informed methods** that regularise attention toward statistical patterns (SNI [2026], Pi-Transformer), and **(iii) mixture-of-experts methods** that blend multiple neural experts via learned gates (STAMImputer [IJCAI 2025]). MARST occupies a fourth, unfilled position: a strictly causal spatiotemporal transformer that learns a **per-position softmax mixture over interpretable classical priors** (LOCF, HA, KNN) and refines the blend with a small residual correction. This decomposition strategy follows the hybrid statistical-neural paradigm established by ES-RNN [Smyl 2020], Wide & Deep [Cheng 2016], and N-BEATS [Oreshkin 2020] but, to our knowledge, is the first instantiation of that paradigm in spatiotemporal traffic imputation. The interpretability of the learned gate π allows direct inspection of which classical regime — local time, daily seasonality, or spatial neighbourhood — the model trusts at each (sensor, time) pair."

---

## 6. Where MARST's claim is weakest — be honest

Two reviewer objections to anticipate and concede partially:

**Objection A: "You're just adding domain-specific feature engineering."**
Partly true — providing LOCF/HA/KNN as inputs *is* feature engineering. The defence: the contribution is teaching the network *when* to trust each anchor (the learned gate π changes per (sensor, time)), and the empirical alternative — giving the network only raw values and hoping it discovers these regularities — is what the 18 published baselines do. They lose by Wilcoxon-significant margins.

**Objection B: "Your anchor ablation is mixed on speed datasets."**
Also partly true. On METR-LA, the "− HA anchor" variant beats full MARST by 0.18 mph; on PEMS-BAY, "− KNN anchor" wins by 0.04 mph. The honest framing: the multi-anchor gate is **clearly load-bearing on flow datasets** (PEMS04 and PEMS08 — every leave-one-out is +1 to +3 worse) and **roughly neutral on speed** at the reduced 400-epoch ablation budget. Don't oversell the unconditional claim that all three anchors are always essential.

---

## 7. Bibliography (BibTeX-ready)

```bibtex
@article{smyl2020hybrid,
  title={A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting},
  author={Smyl, Slawek},
  journal={International Journal of Forecasting},
  volume={36}, number={1}, pages={75--85}, year={2020},
}

@inproceedings{cheng2016wide,
  title={Wide \& Deep Learning for Recommender Systems},
  author={Cheng, Heng-Tze and others},
  booktitle={Workshop on Deep Learning for Recommender Systems},
  year={2016}, eprint={1606.07792},
}

@inproceedings{oreshkin2020nbeats,
  title={N-BEATS: Neural basis expansion analysis for interpretable time series forecasting},
  author={Oreshkin, Boris N. and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua},
  booktitle={ICLR}, year={2020}, eprint={1905.10437},
}

@inproceedings{cao2018brits,
  title={BRITS: Bidirectional Recurrent Imputation for Time Series},
  author={Cao, Wei and Wang, Dong and Li, Jian and Zhou, Hao and Li, Lei and Li, Yitan},
  booktitle={NeurIPS}, year={2018},
}

@article{du2022saits,
  title={SAITS: Self-Attention-based Imputation for Time Series},
  author={Du, Wenjie and C{\^o}t{\'e}, David and Liu, Yan},
  journal={Expert Systems with Applications}, year={2023}, eprint={2202.08516},
}

@inproceedings{cini2022grin,
  title={Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks},
  author={Cini, Andrea and Marisca, Ivan and Alippi, Cesare},
  booktitle={ICLR}, year={2022},
}

@inproceedings{nie2024imputeformer,
  title={ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation},
  author={Nie, Tongyi and others},
  booktitle={KDD}, year={2024}, eprint={2312.01728},
}

@inproceedings{stamimputer2025,
  title={STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation},
  booktitle={IJCAI}, year={2025}, eprint={2506.08054},
}

@misc{stti2024survey,
  title={A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models},
  year={2024}, eprint={2412.04733},
}

@inproceedings{lai2018lstnet,
  title={Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks},
  author={Lai, Guokun and Chang, Wei-Cheng and Yang, Yiming and Liu, Hanxiao},
  booktitle={SIGIR}, year={2018},
}
```

---

## 8. URLs and source list

- ES-RNN (Smyl): https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153
- Fast ES-RNN (GPU implementation, arXiv 2019): https://arxiv.org/pdf/1907.03329
- Uber blog summary of M4 win: https://www.uber.com/us/en/blog/m4-forecasting-competition/
- Wide & Deep (Cheng et al. 2016): https://arxiv.org/pdf/1606.07792
- N-BEATS (Oreshkin et al. 2020): https://arxiv.org/pdf/1905.10437
- NBEATSx (Olivares et al. 2022): https://www.sciencedirect.com/science/article/pii/S0169207022000413
- Statistics-Deep Learning Hybrid (Wang et al. 2021): https://arxiv.org/abs/2112.08618
- ImputeFormer (Nie et al. 2024): https://arxiv.org/abs/2312.01728
- GRIN (Cini et al. 2022, code): https://github.com/Graph-Machine-Learning-Group/grin
- SAITS (Du et al. 2022): https://arxiv.org/pdf/2202.08516
- STAMImputer (IJCAI 2025): https://arxiv.org/html/2506.08054v1
- 2024 Imputation Survey: https://arxiv.org/html/2412.04733v2
- SNI (Statistical-Neural Interaction Networks): https://arxiv.org/html/2601.12380v1
- Bridge-TS (Generative TS imputation prior): https://arxiv.org/pdf/2512.23832
- Pi-Transformer (prior-informed dual-attention): https://arxiv.org/pdf/2509.19985
- Casper (causality-aware spatiotemporal GNN imputation): https://arxiv.org/html/2403.11960v1
- A Hybrid Framework Integrating Traditional Models and Deep Learning (2025): https://www.mdpi.com/1099-4300/27/7/695
