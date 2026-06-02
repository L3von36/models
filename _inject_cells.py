"""Inject R1/R2/R5 cells from KAGGLE_CELLS_R1_R2_R5.md into the notebook.

Reads marst-learnable-results.ipynb, inserts a new section after cell 15
(the run_dataset driver), and writes marst-learnable-results-R1R2R5.ipynb.
Original notebook is not modified.
"""
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = "marst-learnable-results.ipynb"
DST = "marst-learnable-results-R1R2R5.ipynb"

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)


import uuid


def _cell_id() -> str:
    """Generate an 8-char cell id (nbformat 4.5+ recommends one per cell)."""
    return uuid.uuid4().hex[:8]


def code_cell(source: str) -> dict:
    """Return a notebook cell dict for a code cell, source as a single string
    (matching the original notebook's convention; nbformat 4.5 spec allows
    either string or list of strings)."""
    src = source.rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def md_cell(source: str) -> dict:
    """Return a notebook cell dict for a markdown cell, source as a single string."""
    src = source.rstrip("\n")
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": src,
    }


# ---- new cells to inject --------------------------------------------------

HEADER_MD = """\
## R1 / R2 / R5 — review-required revisions

Three extra experiments queued by the journal-reviewer report.  Cells A–E
below close the gaps:

| Cell | Closes | Runtime on 2× T4 |
|---|---|---|
| A | shared dataset-setup helper used by B/C/D | — |
| B | **R1** — load-balancing aux loss ablation (λ = 0 vs λ = 0.02) | ≈ 25 min |
| C | **R2** — KNN imputer at k = 10 (Q12-sweep winner) vs k = 5 headline | ≈ 5 min |
| D | **R5** — compute-matched mini-MARST (≈ 200 K parameters) | ≈ 12 min |
| E | aggregation + paper-ready ASCII tables | < 1 min |

Run in order; B/C/D depend on A.  E depends on all three.  Sidecar JSON
files (`R1_load_balance_ablation.json`, `R2_knn_k10.json`,
`R5_compute_matched.json`) are written without touching the headline
`results_<DATASET>.json` files.
"""

CELL_A = """\
# =============================================================================
# Cell A — shared dataset-setup helper for R1 / R2 / R5
# =============================================================================
import numpy as np
import torch

def _r_setup_dataset(name):
    \"\"\"Re-create the dataset-scoped globals that run_dataset(name) sets up.

    Sets (as globals so the existing helpers see them):
      DATASET, CFG, STEPS_PER_DAY, NUM_NODES,
      value_raw, valid_raw, node_means, node_stds, speed_norm, ha_prior_np,
      A_t, speed_gpu, valid_gpu, ha_prior, node_means_t, node_stds_t,
      m_TN  (per-EVAL_SEED eval mask)
    Does NOT train anything or compute baselines.
    \"\"\"
    g = globals()
    g['DATASET'] = name
    g['CFG']     = DATASETS[name]
    g['STEPS_PER_DAY'] = int(CFG.get('steps_per_day', 288))

    # ---- raw values + validity mask -----------------------------------------
    value_raw = load_raw_array()[:WINDOW]
    if CFG['kind'] == 'speed':
        valid_raw = (value_raw > 0).astype(np.float32)
    else:
        valid_raw = np.ones_like(value_raw, dtype=np.float32)
    g['value_raw'] = value_raw
    g['valid_raw'] = valid_raw
    g['NUM_NODES'] = int(value_raw.shape[1])

    # ---- per-sensor normalisation (training window only) --------------------
    vt   = value_raw[:TRAIN_END]
    vtv  = valid_raw[:TRAIN_END]
    nm   = (vt * vtv).sum(0) / (vtv.sum(0) + 1e-8)
    sq   = ((vt - nm) ** 2 * vtv).sum(0) / (vtv.sum(0) + 1e-8)
    ns   = np.sqrt(sq + 1e-6).astype(np.float32)
    nm   = nm.astype(np.float32)
    speed_norm = ((value_raw - nm) / ns).astype(np.float32)
    g['node_means'] = nm
    g['node_stds']  = ns
    g['speed_norm'] = speed_norm

    # ---- HA prior on training window only -----------------------------------
    slot_idx = (np.arange(WINDOW) % STEPS_PER_DAY).astype(np.int64)
    ha = np.zeros_like(speed_norm)
    for s in range(STEPS_PER_DAY):
        sel  = slot_idx[:TRAIN_END] == s
        sub  = speed_norm[:TRAIN_END][sel]
        subv = valid_raw [:TRAIN_END][sel]
        sums = (sub * subv).sum(0)
        cnts = subv.sum(0) + 1e-8
        ha[slot_idx == s] = sums / cnts
    g['ha_prior_np'] = ha

    # ---- adjacency (diagonal zeroed = spatial anti-leak invariant) ----------
    adj = load_adjacency(NUM_NODES).astype(np.float32)
    np.fill_diagonal(adj, 0)

    # ---- ship to GPU --------------------------------------------------------
    g['A_t']          = torch.tensor(adj,        device=device)
    g['speed_gpu']    = torch.tensor(speed_norm, device=device)
    g['valid_gpu']    = torch.tensor(valid_raw,  device=device)
    g['ha_prior']     = torch.tensor(ha,         device=device)
    g['node_means_t'] = torch.tensor(nm,         device=device)
    g['node_stds_t']  = torch.tensor(ns,         device=device)

    # ---- 5 eval-mask seeds --------------------------------------------------
    m_TN = {}
    for sd in EVAL_SEEDS:
        rng = np.random.default_rng(sd)
        m_TN[sd] = (rng.random((EVAL_LEN, NUM_NODES)) > SPARSITY).astype(np.float32)
    g['m_TN'] = m_TN

    print(f'  [setup] {name}: N={NUM_NODES}, T={WINDOW}, '
          f'eval=[{EVAL_START}:{EVAL_START+EVAL_LEN}], '
          f'sparsity={SPARSITY}, kind={CFG["kind"]}')

print('_r_setup_dataset() ready.')
"""

CELL_B = """\
# =============================================================================
# Cell B — R1: Load-balancing aux loss ablation (lambda=0 vs lambda=0.02)
# Re-trains MARST with LOAD_BALANCE_WEIGHT=0.0 on every dataset and
# compares against the headline lambda=0.02 result loaded from
# results_<DATASET>.json. Saves R1_load_balance_ablation.json.
# Runtime: ~25 min on 2x T4.
# =============================================================================
import json, time

R1_OUT_JSON   = 'R1_load_balance_ablation.json'
R1_RESULTS    = {}
_R1_SAVED_LB  = LOAD_BALANCE_WEIGHT  # restore at end of cell

for _ds in DATASETS_TO_RUN:
    print(f'\\n{"="*70}\\n  R1: MARST lambda=0 on {_ds}\\n{"="*70}')
    _r_setup_dataset(_ds)

    # Train MARST with LB disabled (3 seeds, in parallel across GPUs)
    globals()['LOAD_BALANCE_WEIGHT'] = 0.0
    ctor = lambda: MaskedSTTransformerMARST(
        NUM_NODES, A_t, node_means_t, node_stds_t,
        hidden=128, n_heads=4, n_layers=6, dropout=0.1).to(device)
    jobs = [dict(ctor=ctor, epochs=TRAIN_EPOCHS,
                 batch_size=BATCH_SIZE_MARST, seed=ts,
                 label=f'MARST_lb0(seed={ts})') for ts in TRAIN_SEEDS]
    t0 = time.time()
    nets = _train_parallel(jobs)
    elapsed = time.time() - t0

    per_seed_per_eval = []
    for net in nets:
        per_seed_per_eval.append(_eval_st_mae(net).tolist())

    # Restore aux weight immediately so any later cell behaves normally
    globals()['LOAD_BALANCE_WEIGHT'] = _R1_SAVED_LB

    # Load headline lambda=0.02 result for paired comparison
    with open(f'results_{_ds}.json') as f:
        hd = json.load(f)
    marst_key = next(k for k in hd['results'] if 'MARST' in k)
    lambda_002_per_eval = list(hd['results'][marst_key])

    mean_lb0  = float(np.mean(per_seed_per_eval))
    mean_lb02 = float(np.mean(lambda_002_per_eval))
    R1_RESULTS[_ds] = dict(
        lambda_0_per_seed_per_eval = per_seed_per_eval,
        lambda_002_per_eval        = lambda_002_per_eval,
        lambda_0_mae               = mean_lb0,
        lambda_0_std               = float(np.std(np.array(per_seed_per_eval).mean(1))),
        lambda_002_mae             = mean_lb02,
        delta                      = mean_lb0 - mean_lb02,
        pct_change                 = 100.0 * (mean_lb0 - mean_lb02) / mean_lb02,
        wall_time_seconds          = elapsed,
    )
    print(f'\\n  [{_ds}]')
    print(f'    lambda=0    MAE: {mean_lb0:.4f}  '
          f'(std across 3 seeds: {R1_RESULTS[_ds]["lambda_0_std"]:.4f})')
    print(f'    lambda=0.02 MAE: {mean_lb02:.4f}  (headline)')
    print(f'    delta:           {R1_RESULTS[_ds]["delta"]:+.4f}  '
          f'({R1_RESULTS[_ds]["pct_change"]:+.2f}%)')
    print(f'    wall time:       {elapsed/60:.1f} min')

with open(R1_OUT_JSON, 'w') as f:
    json.dump(R1_RESULTS, f, indent=2)
print(f'\\nSaved {R1_OUT_JSON}')
"""

CELL_C = """\
# =============================================================================
# Cell C — R2: KNN imputer at k=10 (paired against k=5 headline)
# Uses masked-Euclidean distance (only observed coords contribute to
# pairwise distance, column re-normalisation prevents self-leakage).
# Saves R2_knn_k10.json. Runtime: ~5 min (CPU only).
# =============================================================================
import json
import numpy as np

R2_OUT_JSON = 'R2_knn_k10.json'
R2_RESULTS  = {}

def _knn_mae_per_seed(value_raw, valid_raw, mask, k):
    \"\"\"Masked-distance KNN imputer evaluated on held-out cells.\"\"\"
    EW = value_raw[EVAL_START:EVAL_START + EVAL_LEN]      # [EL, N]
    EV = valid_raw[EVAL_START:EVAL_START + EVAL_LEN]      # [EL, N]
    held = (mask == 0) & (EV > 0.5)                        # [EL, N]
    M   = mask * EV                                         # observed cells
    Xm  = EW * M                                            # zero where unobserved

    EL, N = Xm.shape
    out = np.zeros_like(EW)
    for t in range(EL):
        m_t = M[t]
        if m_t.sum() < 2:
            out[t] = EW[t]
            continue
        m_pair = M * m_t[None, :]
        cnt    = m_pair.sum(1)
        diff2  = ((Xm - Xm[t][None, :]) ** 2 * m_pair).sum(1)
        dist   = np.sqrt(diff2 / np.maximum(cnt, 1))
        dist[t] = np.inf
        dist[cnt < 1] = np.inf
        nbr_idx = np.argpartition(dist, min(k, EL - 1))[:k]
        nbr_X = EW[nbr_idx]
        nbr_M = M [nbr_idx]
        num   = (nbr_X * nbr_M).sum(0)
        den   = nbr_M.sum(0) + 1e-8
        out[t] = num / den

    err = np.abs(out[held] - EW[held])
    return float(err.mean())

for _ds in DATASETS_TO_RUN:
    print(f'\\n=== R2 on {_ds} ===')
    _r_setup_dataset(_ds)

    per_eval_k10 = []
    for sd in EVAL_SEEDS:
        mae = _knn_mae_per_seed(value_raw, valid_raw, m_TN[sd], k=10)
        per_eval_k10.append(mae)

    with open(f'results_{_ds}.json') as f:
        hd = json.load(f)
    knn_key = next((k for k in hd['results'] if 'KNN' in k), None)
    k5_per_eval = list(hd['results'][knn_key]) if knn_key else None

    R2_RESULTS[_ds] = dict(
        k10_per_eval = per_eval_k10,
        k10_mae      = float(np.mean(per_eval_k10)),
        k10_std      = float(np.std (per_eval_k10)),
        k5_per_eval  = k5_per_eval,
        k5_mae       = float(np.mean(k5_per_eval)) if k5_per_eval else None,
        delta        = (float(np.mean(per_eval_k10)) - float(np.mean(k5_per_eval)))
                       if k5_per_eval else None,
    )
    print(f'  k=10 MAE: {R2_RESULTS[_ds]["k10_mae"]:.4f}  '
          f'+/- {R2_RESULTS[_ds]["k10_std"]:.4f}  (5 eval seeds)')
    if k5_per_eval:
        print(f'  k=5  MAE: {R2_RESULTS[_ds]["k5_mae"]:.4f}  (headline)')
        print(f'  delta:    {R2_RESULTS[_ds]["delta"]:+.4f}  '
              f'(negative = k=10 better)')

with open(R2_OUT_JSON, 'w') as f:
    json.dump(R2_RESULTS, f, indent=2)
print(f'\\nSaved {R2_OUT_JSON}')
"""

CELL_D = """\
# =============================================================================
# Cell D — R5: Compute-matched mini-MARST (hidden=64, layers=3, ~200K params)
# Trains a smaller MARST and reports MAE alongside the headline 1.67M-parameter
# config. Single training seed (budget). Saves R5_compute_matched.json.
# Runtime: ~12 min on 2x T4 (4 datasets x 1 seed).
# =============================================================================
import json, time

R5_OUT_JSON = 'R5_compute_matched.json'
R5_RESULTS  = {}
_seeds      = [0]   # extend to [0, 1, 2] for the full 3-seed protocol

for _ds in DATASETS_TO_RUN:
    print(f'\\n{"="*70}\\n  R5: Mini-MARST on {_ds}\\n{"="*70}')
    _r_setup_dataset(_ds)

    ctor_mini = lambda: MaskedSTTransformerMARST(
        NUM_NODES, A_t, node_means_t, node_stds_t,
        hidden=64, n_heads=4, n_layers=3, dropout=0.1).to(device)

    _tmp   = ctor_mini()
    params = sum(p.numel() for p in _tmp.parameters())
    del _tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    jobs = [dict(ctor=ctor_mini, epochs=TRAIN_EPOCHS,
                 batch_size=BATCH_SIZE_MARST, seed=s,
                 label=f'Mini-MARST(s={s})') for s in _seeds]
    t0   = time.time()
    nets = _train_parallel(jobs)
    elapsed = time.time() - t0

    per_seed_per_eval = [_eval_st_mae(net).tolist() for net in nets]
    mini_mae = float(np.mean(per_seed_per_eval))

    with open(f'results_{_ds}.json') as f:
        hd = json.load(f)
    full_marst_key = next(k for k in hd['results'] if 'MARST' in k)
    full_per_eval  = list(hd['results'][full_marst_key])
    full_mae       = float(np.mean(full_per_eval))

    full_params = int(hd.get('model_stats', {}).get(full_marst_key, {})
                       .get('params', 1_666_831))

    R5_RESULTS[_ds] = dict(
        mini_params        = int(params),
        mini_per_seed_eval = per_seed_per_eval,
        mini_mae           = mini_mae,
        full_mae           = full_mae,
        full_params        = full_params,
        delta_mae          = mini_mae - full_mae,
        pct_change         = 100.0 * (mini_mae - full_mae) / full_mae,
        param_ratio        = full_params / max(1, params),
        wall_time_seconds  = elapsed,
        n_seeds            = len(_seeds),
    )
    print(f'\\n  [{_ds}]')
    print(f'    Mini-MARST: {params:>9,} params  MAE {mini_mae:.4f}')
    print(f'    Full MARST: {full_params:>9,} params  MAE {full_mae:.4f}')
    print(f'    Delta MAE : {R5_RESULTS[_ds]["delta_mae"]:+.4f}  '
          f'({R5_RESULTS[_ds]["pct_change"]:+.2f}%)')
    print(f'    Param / : {R5_RESULTS[_ds]["param_ratio"]:.1f}x')
    print(f'    wall time : {elapsed/60:.1f} min')

with open(R5_OUT_JSON, 'w') as f:
    json.dump(R5_RESULTS, f, indent=2)
print(f'\\nSaved {R5_OUT_JSON}')
"""

CELL_E = """\
# =============================================================================
# Cell E — aggregation & paper-ready tables for R1, R2, R5
# Run AFTER B, C, D. Loads the three sidecar JSON files and prints ASCII
# tables in the same format as the existing notebook output blocks.
# =============================================================================
import json
import numpy as np

def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

R1 = _load('R1_load_balance_ablation.json')
R2 = _load('R2_knn_k10.json')
R5 = _load('R5_compute_matched.json')

ds_order = ['METR-LA', 'PEMS-BAY', 'PEMS04', 'PEMS08']

# ---- R1 -----------------------------------------------------------
if R1:
    print('\\n' + '=' * 76)
    print('  R1 - Load-balancing aux loss ablation (lambda=0 vs lambda=0.02)')
    print('=' * 76)
    print(f'{"Dataset":<10} {"lambda=0":>12} {"lambda=0.02":>14} {"delta":>10} {"% change":>10}')
    print('-' * 76)
    for ds in ds_order:
        if ds not in R1: continue
        r = R1[ds]
        print(f'{ds:<10} {r["lambda_0_mae"]:>12.4f} {r["lambda_002_mae"]:>14.4f} '
              f'{r["delta"]:>+10.4f} {r["pct_change"]:>+9.2f}%')
    print('=' * 76)
    print('Interpretation:')
    print('  delta > 0   -> load balancing helps MAE on this dataset')
    print('  delta ~ 0   -> load balancing is interpretability-only (neutral MAE)')
    print('  delta < 0   -> load balancing hurts MAE (rare; flags lambda too high)')
else:
    print('[R1 results not found - run Cell B first]')

# ---- R2 -----------------------------------------------------------
if R2:
    print('\\n' + '=' * 60)
    print('  R2 - KNN imputer: k=10 vs k=5 headline')
    print('=' * 60)
    print(f'{"Dataset":<10} {"k=5":>10} {"k=10":>10} {"delta":>10}')
    print('-' * 60)
    for ds in ds_order:
        if ds not in R2: continue
        r = R2[ds]
        print(f'{ds:<10} {r["k5_mae"]:>10.4f} {r["k10_mae"]:>10.4f} '
              f'{r["delta"]:>+10.4f}')
    print('=' * 60)
else:
    print('[R2 results not found - run Cell C first]')

# ---- R5 -----------------------------------------------------------
if R5:
    print('\\n' + '=' * 80)
    print('  R5 - Compute-matched mini-MARST (hidden=64, layers=3)')
    print('=' * 80)
    print(f'{"Dataset":<10} {"Full MAE":>10} {"Mini MAE":>10} {"delta":>9} '
          f'{"% chg":>8} {"param /":>9}')
    print('-' * 80)
    for ds in ds_order:
        if ds not in R5: continue
        r = R5[ds]
        print(f'{ds:<10} {r["full_mae"]:>10.4f} {r["mini_mae"]:>10.4f} '
              f'{r["delta_mae"]:>+9.4f} {r["pct_change"]:>+7.2f}% '
              f'{r["param_ratio"]:>8.1f}x')
    print('-' * 80)
    any_ds = next((d for d in ds_order if d in R5), None)
    if any_ds:
        print(f'  Mini-MARST: {R5[any_ds]["mini_params"]:,} params')
        print(f'  Full MARST: {R5[any_ds]["full_params"]:,} params')
        print(f'  Seeds used per dataset: {R5[any_ds]["n_seeds"]}')
    print('=' * 80)
else:
    print('[R5 results not found - run Cell D first]')
"""

# ---- find insertion point (after cell with run_dataset driver loop) -------
# Cell 15 contains `for _ds in DATASETS_TO_RUN:` and `run_dataset(_ds)`
insert_after = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "DATASETS_TO_RUN" in src and "run_dataset(_ds)" in src:
        insert_after = i
        break

if insert_after is None:
    raise RuntimeError(
        "Could not find the run_dataset driver cell to insert after."
    )

print(f"Found driver loop at cell index {insert_after}; inserting 6 new cells "
      f"(1 markdown header + 5 code) immediately after.")

new_cells = [
    md_cell(HEADER_MD),
    code_cell(CELL_A),
    code_cell(CELL_B),
    code_cell(CELL_C),
    code_cell(CELL_D),
    code_cell(CELL_E),
]

# Inject in order so they appear in the right position
nb["cells"] = (
    nb["cells"][: insert_after + 1] + new_cells + nb["cells"][insert_after + 1 :]
)

# Save with conservative JSON formatting that matches jupyter's output
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {DST}: {len(nb['cells'])} cells total "
      f"({len(nb['cells']) - 6} original + 6 new).")
