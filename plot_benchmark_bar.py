"""
Standalone script: regenerate fig01_benchmark_bar.png with final 5-seed results.
No GPU / dataset required — all numbers are baked in from the evaluation run.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Final 5-seed results (seeds 42-46, 80% sparsity, PEMS-BAY) ──────────────
# Format: name -> (mean_MAE, std_MAE)
RESULTS = {
    "1. Historical Average (HA)":            (3.1198, 0.0051),
    "2. LOCF (Last-Obs Carried Forward)":    (1.9592, 0.0123),
    "3. Global Mean (per-node train mean)":  (5.5168, 0.0072),
    "4. Node-wise Ridge Regression":         (3.1204, 0.0051),
    "5. KNN Imputer (k=5, masked-dist)":     (2.1841, 0.0143),
    "6.  MLP (per-node + node-emb)":         (3.5799, 0.0054),
    "7.  LSTM (per-node)":                   (2.2654, 0.0194),
    "8.  BiLSTM->2L-LSTM (causal, fixed)":   (2.2226, 0.0172),
    "9.  GRU (per-node)":                    (2.3004, 0.0274),
    "10. TCN (causal dilated, per-node)":    (1.9849, 0.0201),
    "11. SAITS-lite (causal Attn + node-emb, fixed)": (3.4992, 0.0290),
    "12. BRITS-lite (forward GRU only, fixed)": (1.9662, 0.0220),
    "13. DCRNN-lite (DiffGCN + GRU)":        (3.1279, 0.0169),
    "14. ASTGCN-lite (GCN + Causal Attn, fixed)": (3.6502, 0.0215),
    "15. GWN-lite (Adaptive GCN + Gated TCN)": (2.7291, 0.0071),
    "16. MaskedSTTransformerV3-FIXED":       (1.5776, 0.0117),
    "17. MaskedSTTransformerV4":             (1.5298, 0.0089),
    "18. MaskedSTTransformerV5 (ours)":      (1.3977, 0.0051),
}

FAMILY_COLOR = {
    "stat":     "#9E9E9E",
    "linear":   "#78909C",
    "nonparam": "#90A4AE",
    "nn":       "#42A5F5",
    "rnn":      "#29B6F6",
    "conv":     "#26C6DA",
    "attn":     "#66BB6A",
    "graph":    "#FFA726",
    "ours":     "#B71C1C",
}
FAMILY_LABEL = {
    "stat": "Statistical", "linear": "Linear", "nonparam": "Non-parametric",
    "nn": "Neural (per-node)", "rnn": "RNN", "conv": "TCN",
    "attn": "Attention", "graph": "Graph", "ours": "Ours",
}
MODEL_FAMILY = {
    "1. Historical Average (HA)":           "stat",
    "2. LOCF (Last-Obs Carried Forward)":   "stat",
    "3. Global Mean (per-node train mean)":  "stat",
    "4. Node-wise Ridge Regression":         "linear",
    "5. KNN Imputer (k=5, masked-dist)":     "nonparam",
    "6.  MLP (per-node + node-emb)":         "nn",
    "7.  LSTM (per-node)":                   "rnn",
    "8.  BiLSTM->2L-LSTM (causal, fixed)":   "rnn",
    "9.  GRU (per-node)":                    "rnn",
    "10. TCN (causal dilated, per-node)":    "conv",
    "11. SAITS-lite (causal Attn + node-emb, fixed)": "attn",
    "12. BRITS-lite (forward GRU only, fixed)": "rnn",
    "13. DCRNN-lite (DiffGCN + GRU)":        "graph",
    "14. ASTGCN-lite (GCN + Causal Attn, fixed)": "graph",
    "15. GWN-lite (Adaptive GCN + Gated TCN)": "graph",
    "16. MaskedSTTransformerV3-FIXED":       "ours",
    "17. MaskedSTTransformerV4":             "ours",
    "18. MaskedSTTransformerV5 (ours)":      "ours",
}
SHORT = {
    "1. Historical Average (HA)":           "Hist. Avg.",
    "2. LOCF (Last-Obs Carried Forward)":   "LOCF",
    "3. Global Mean (per-node train mean)":  "Global Mean",
    "4. Node-wise Ridge Regression":         "Ridge",
    "5. KNN Imputer (k=5, masked-dist)":     "KNN (k=5)",
    "6.  MLP (per-node + node-emb)":         "MLP",
    "7.  LSTM (per-node)":                   "LSTM",
    "8.  BiLSTM->2L-LSTM (causal, fixed)":   "BiLSTM→LSTM",
    "9.  GRU (per-node)":                    "GRU",
    "10. TCN (causal dilated, per-node)":    "TCN",
    "11. SAITS-lite (causal Attn + node-emb, fixed)": "SAITS-lite",
    "12. BRITS-lite (forward GRU only, fixed)": "BRITS-lite",
    "13. DCRNN-lite (DiffGCN + GRU)":        "DCRNN-lite",
    "14. ASTGCN-lite (GCN + Causal Attn, fixed)": "ASTGCN-lite",
    "15. GWN-lite (Adaptive GCN + Gated TCN)": "GWN-lite",
    "16. MaskedSTTransformerV3-FIXED":       "MST-V3",
    "17. MaskedSTTransformerV4":             "MST-V4",
    "18. MaskedSTTransformerV5 (ours)":      "MST-V5 ★",
}

ha_mae   = RESULTS["1. Historical Average (HA)"][0]
locf_mae = RESULTS["2. LOCF (Last-Obs Carried Forward)"][0]

# Sort worst → best so best appears at the top of the horizontal bar chart
items  = sorted(RESULTS.items(), key=lambda kv: kv[1][0], reverse=True)
labels = [SHORT.get(k, k) for k, _ in items]
maes   = np.array([v[0] for _, v in items])
stds   = np.array([v[1] for _, v in items])
colors = [FAMILY_COLOR[MODEL_FAMILY.get(k, "stat")] for k, _ in items]

fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(labels))
ax.barh(y, maes, xerr=stds, color=colors, height=0.65,
        error_kw=dict(ecolor="#333", capsize=3, elinewidth=1.2))
ax.axvline(ha_mae,   color="#FF7043", ls="--", lw=1.5, alpha=0.8,
           label=f"HA ({ha_mae:.2f})")
ax.axvline(locf_mae, color="#78909C", ls=":",  lw=1.5, alpha=0.8,
           label=f"LOCF ({locf_mae:.2f})")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("MAE (km/h)")
ax.set_title("PEMS-BAY Imputation Benchmark  |  80% Sparsity, 5 Seeds",
             fontsize=12, fontweight="bold")
ax.invert_yaxis()

# Family legend
seen_fams = dict.fromkeys(MODEL_FAMILY.get(k, "stat") for k, _ in items)
handles = [mpatches.Patch(color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
           for f in seen_fams]
leg1 = ax.legend(handles=handles, loc="lower right", fontsize=8,
                 title="Model family", title_fontsize=8, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, max(maes) * 1.13)

# Annotate our best model
best_k = "18. MaskedSTTransformerV5 (ours)"
bm, _ = RESULTS[best_k]
bi = labels.index(SHORT[best_k])
ax.annotate(f"+{100*(ha_mae-bm)/ha_mae:.1f}% vs HA",
            xy=(bm, bi), xytext=(bm + 0.08, bi - 0.6),
            fontsize=8.5, color="#B71C1C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.3))

plt.tight_layout()
plt.savefig("fig01_benchmark_bar.png", dpi=150, bbox_inches="tight")
print("Saved fig01_benchmark_bar.png")
print(f"V5 MAE: {bm:.4f}  |  +{100*(ha_mae-bm)/ha_mae:.1f}% vs HA")
