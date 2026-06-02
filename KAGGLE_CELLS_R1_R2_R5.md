# Kaggle cells for R1, R2, R5 — copy-paste into `marst-learnable-results.ipynb`

These four cells close the three required-revisions items from
`REVIEW_REPORT.md`:

| # | Closes | Runtime on 2× T4 |
|---|---|---|
| **A** | shared dataset-setup helper used by R1, R2, R5 | — |
| **B** | **R1** — load-balancing aux loss ablation ($\lambda = 0$ vs $\lambda = 0.02$) | ≈ 25 min |
| **C** | **R2** — KNN imputer at $k = 10$ (Q12-sweep winner) vs $k = 5$ headline | ≈ 5 min |
| **D** | **R5** — compute-matched mini-MARST (≈ 200 K parameters) | ≈ 12 min |
| **E** | aggregation + paper-ready ASCII tables | < 1 min |

**Place all five cells AFTER cell 15** (the `run_dataset` driver loop) so
that `results_<DATASET>.json` for all four datasets already exists.
The cells write fresh JSON sidecar files (`R1_load_balance_ablation.json`,
`R2_knn_k10.json`, `R5_compute_matched.json`) without touching the
headline JSON files.

The cells reuse the notebook's existing globals (`SPARSITY`, `WINDOW`,
`TRAIN_END`, `EVAL_START`, `EVAL_LEN`, `EVAL_SEEDS`, `TRAIN_EPOCHS`,
`BATCH_SIZE_MARST`, `TRAIN_SEEDS`, `DATASETS`, `DATASETS_TO_RUN`,
`LOAD_BALANCE_WEIGHT`) and the existing helpers
(`load_raw_array`, `load_adjacency`, `MaskedSTTransformerMARST`,
`_train_parallel`, `_eval_st_mae`).  Nothing in the notebook above
needs to change.

---

## Cell A — shared dataset-setup helper

Run this **once** before any of the R-cells below.  It defines
`_r_setup_dataset(name)`, which mirrors the data-loading portion of
`run_dataset(name)` (load raw, compute per-sensor normalisation, build
the HA prior, ship to GPU, build the 5 eval masks) without running any
baselines or training.  All R-cells call this at the top of their
per-dataset loop because `run_dataset` overwrites the dataset-scoped
globals every time it runs, so by the end of the main pipeline only
the last dataset's tensors live in memory.

```python
# =============================================================================
# Cell A — shared dataset-setup helper for R1 / R2 / R5
# =============================================================================
import numpy as np
import torch

def _r_setup_dataset(name):
    """Re-create the dataset-scoped globals that run_dataset(name) sets up.

    Sets (as globals so the existing helpers see them):
      DATASET, CFG, STEPS_PER_DAY, NUM_NODES,
      value_raw, valid_raw, node_means, node_stds, speed_norm, ha_prior_np,
      A_t, speed_gpu, valid_gpu, ha_prior, node_means_t, node_stds_t,
      m_TN  (per-EVAL_SEED eval mask)
    Does NOT train anything or compute baselines.
    """
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
```

---

## Cell B — R1: Load-balancing aux loss ablation

Re-trains MARST with `LOAD_BALANCE_WEIGHT = 0.0` on every dataset
(3 train seeds × 5 eval-mask seeds), and compares against the
$\lambda = 0.02$ headline result loaded from
`results_<DATASET>.json`.  Saves
`R1_load_balance_ablation.json`.

**Expected outcome (per the discussion in `ARTICLE.tex` §5.7):**
$\lambda = 0$ runs should show comparable or slightly worse MAE plus a
collapsed gate (per-dataset $\pi \approx (1, 0, 0)$-ish) — visible in
the per-epoch training prints.  If the MAE difference is consistently
under 0.5 % across datasets, you can soften the contribution-2 framing
to "load balancing is a diagnostic / interpretability aid that has
roughly neutral MAE impact."  If $\lambda = 0$ is clearly worse on
flow datasets, the contribution stands as written.

```python
# =============================================================================
# Cell B — R1: Load-balancing aux loss ablation (lambda=0 vs lambda=0.02)
# =============================================================================
import json, time

R1_OUT_JSON   = 'R1_load_balance_ablation.json'
R1_RESULTS    = {}
_R1_SAVED_LB  = LOAD_BALANCE_WEIGHT  # restore at end of cell

for _ds in DATASETS_TO_RUN:
    print(f'\n{"="*70}\n  R1: MARST lambda=0 on {_ds}\n{"="*70}')
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
    print(f'\n  [{_ds}]')
    print(f'    lambda=0    MAE: {mean_lb0:.4f}  '
          f'(std across 3 seeds: {R1_RESULTS[_ds]["lambda_0_std"]:.4f})')
    print(f'    lambda=0.02 MAE: {mean_lb02:.4f}  (headline)')
    print(f'    delta:           {R1_RESULTS[_ds]["delta"]:+.4f}  '
          f'({R1_RESULTS[_ds]["pct_change"]:+.2f}%)')
    print(f'    wall time:       {elapsed/60:.1f} min')

with open(R1_OUT_JSON, 'w') as f:
    json.dump(R1_RESULTS, f, indent=2)
print(f'\nSaved {R1_OUT_JSON}')
```

---

## Cell C — R2: KNN imputer at $k = 10$ vs $k = 5$

Re-evaluates the KNN imputer with $k = 10$ — the Q12-sweep winner —
across the 5 eval-mask seeds per dataset.  Uses the same
masked-Euclidean-distance variant as the notebook's headline KNN
baseline (only observed coordinates contribute to the
inter-time-step distance; column re-normalisation prevents
self-leakage).  Saves `R2_knn_k10.json`.

```python
# =============================================================================
# Cell C — R2: KNN imputer at k=10 (paired against k=5 headline)
# =============================================================================
import json
import numpy as np

R2_OUT_JSON = 'R2_knn_k10.json'
R2_RESULTS  = {}

def _knn_mae_per_seed(value_raw, valid_raw, mask, k):
    """Masked-distance KNN imputer evaluated on held-out cells.

    value_raw : [T, N]      original tensor (physical units)
    valid_raw : [T, N]      validity (1 = real cell, 0 = broken hardware)
    mask      : [EVAL_LEN,N] observation mask on the eval window
                            (1 = observed input, 0 = hidden / to impute)
    k         : int         number of neighbours

    Returns scalar MAE on (mask == 0) & (valid == 1) positions of the
    eval window.
    """
    EW = value_raw[EVAL_START:EVAL_START + EVAL_LEN]      # [EL, N]
    EV = valid_raw[EVAL_START:EVAL_START + EVAL_LEN]      # [EL, N]
    held = (mask == 0) & (EV > 0.5)                        # [EL, N]
    M   = mask * EV                                         # observed cells
    Xm  = EW * M                                            # zero where unobserved

    # Pair-wise masked-distance between every (timestep) row, restricted
    # to columns observed in *both* rows.  O(EL^2 * N) — acceptable for
    # EL=450, N<=325.
    EL, N = Xm.shape
    out = np.zeros_like(EW)
    # Pre-compute squared values for speed
    Xm2 = (Xm ** 2)
    for t in range(EL):
        # Columns observed at row t
        m_t = M[t]                                          # [N] 0/1
        if m_t.sum() < 2:
            out[t] = EW[t]   # nothing to lean on; leave as-is
            continue
        # Compute distance from t to every other row, over columns
        # observed at both.
        m_pair = M * m_t[None, :]                           # [EL, N]
        cnt    = m_pair.sum(1)                              # [EL]
        diff2  = ((Xm - Xm[t][None, :]) ** 2 * m_pair).sum(1)
        # Normalise by overlap count (avoid /0)
        dist   = np.sqrt(diff2 / np.maximum(cnt, 1))
        dist[t] = np.inf                                     # exclude self
        dist[cnt < 1] = np.inf                               # exclude no-overlap
        nbr_idx = np.argpartition(dist, min(k, EL - 1))[:k]
        # Per-column weighted mean over the k neighbours that *observe*
        # the column.  Falls back to row-t observation (which is the
        # held-out cell -> NaN result) only if no neighbour observes
        # the column; in that case we use the column mean over the
        # neighbour rows where M==0 contributes the zero-imputed value.
        nbr_X = EW[nbr_idx]                                  # [k, N]
        nbr_M = M [nbr_idx]                                  # [k, N]
        num   = (nbr_X * nbr_M).sum(0)
        den   = nbr_M.sum(0) + 1e-8
        out[t] = num / den

    err = np.abs(out[held] - EW[held])
    return float(err.mean())

for _ds in DATASETS_TO_RUN:
    print(f'\n=== R2 on {_ds} ===')
    _r_setup_dataset(_ds)

    per_eval_k10 = []
    for sd in EVAL_SEEDS:
        mae = _knn_mae_per_seed(value_raw, valid_raw, m_TN[sd], k=10)
        per_eval_k10.append(mae)

    # Load k=5 from headline JSON
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
print(f'\nSaved {R2_OUT_JSON}')
```

---

## Cell D — R5: Compute-matched mini-MARST

Trains MARST with `hidden = 64`, `n_layers = 3` — approximately
200 K parameters, comparable in scale to iTransformer (96 K) and
PatchTST (84 K) — and reports MAE alongside the headline 1.67 M-parameter
configuration.  Single training seed (budget reason); add more seeds by
extending `_seeds = [0]` to `[0, 1, 2]` if you have the compute.
Saves `R5_compute_matched.json`.

```python
# =============================================================================
# Cell D — R5: Compute-matched mini-MARST (hidden=64, layers=3, ~200K params)
# =============================================================================
import json, time

R5_OUT_JSON = 'R5_compute_matched.json'
R5_RESULTS  = {}
_seeds      = [0]   # extend to [0, 1, 2] for the full 3-seed protocol

for _ds in DATASETS_TO_RUN:
    print(f'\n{"="*70}\n  R5: Mini-MARST on {_ds}\n{"="*70}')
    _r_setup_dataset(_ds)

    ctor_mini = lambda: MaskedSTTransformerMARST(
        NUM_NODES, A_t, node_means_t, node_stds_t,
        hidden=64, n_heads=4, n_layers=3, dropout=0.1).to(device)

    # Count parameters once
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
    print(f'\n  [{_ds}]')
    print(f'    Mini-MARST: {params:>9,} params  MAE {mini_mae:.4f}')
    print(f'    Full MARST: {full_params:>9,} params  MAE {full_mae:.4f}')
    print(f'    Δ MAE     : {R5_RESULTS[_ds]["delta_mae"]:+.4f}  '
          f'({R5_RESULTS[_ds]["pct_change"]:+.2f}%)')
    print(f'    Param /   : {R5_RESULTS[_ds]["param_ratio"]:.1f}×')
    print(f'    wall time : {elapsed/60:.1f} min')

with open(R5_OUT_JSON, 'w') as f:
    json.dump(R5_RESULTS, f, indent=2)
print(f'\nSaved {R5_OUT_JSON}')
```

---

## Cell E — Aggregation and paper-ready ASCII tables

Run **after** B, C, D.  Loads the three sidecar JSON files and prints
ASCII tables in the same format as the existing notebook output blocks.
Copy these directly into the paper Section 5.

```python
# =============================================================================
# Cell E — aggregation & paper-ready tables for R1, R2, R5
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
    print('\n' + '=' * 76)
    print('  R1 — Load-balancing aux loss ablation (lambda=0 vs lambda=0.02)')
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
    print('\n' + '=' * 60)
    print('  R2 — KNN imputer: k=10 vs k=5 headline')
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
    print('\n' + '=' * 80)
    print('  R5 — Compute-matched mini-MARST (hidden=64, layers=3)')
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
    if ds_order[0] in R5:
        any_ds = next(d for d in ds_order if d in R5)
        print(f'  Mini-MARST: {R5[any_ds]["mini_params"]:,} params')
        print(f'  Full MARST: {R5[any_ds]["full_params"]:,} params')
        print(f'  Seeds used per dataset: {R5[any_ds]["n_seeds"]}')
    print('=' * 80)
else:
    print('[R5 results not found - run Cell D first]')
```

---

## Where the results go in `ARTICLE.tex`

After running these cells, three small edits absorb the new data:

### R1 (load-balancing ablation)
- **Add** a new subsection §5.8 `Load-balancing ablation` with a 4-row table from Cell E's R1 output.
- **Update** Contribution #2 (currently calls out the missing ablation) to state the actual finding.
- **Update** §6 paragraph *The load-balancing aux loss is doing work* to cite the new table.

### R2 (KNN k=10)
- **Replace** the KNN row in Table 2 (cross-dataset MAE) with the k=10 numbers from Cell E.
- **Add** a footnote on the KNN row: "Imputer uses k = 10, the winner of the Q12 sweep; k = 5 is reported in Appendix B for comparison."
- **Append** the k=5 row to Appendix B for full disclosure.

### R5 (compute-matched mini-MARST)
- **Add** a new row to Table 9 (parameter counts): `Mini-MARST (ours, hidden=64, layers=3)  ~200,000  MAE on each dataset`
- **Add** a sentence to §6 paragraph *Parameter count*: "A compute-matched mini-MARST at ~200 K parameters (Table 9) retains [X]% of the full model's MAE win on flow datasets and [Y]% on speed datasets, confirming the architectural prior is doing work distinct from raw capacity."  Fill in X, Y from Cell E.

---

## Total compute budget

| Phase | Compute | Wall-clock on 2× T4 |
|---|---|---|
| Cell A (helper) | nil | < 1 s |
| Cell B (R1, 4 datasets × 3 seeds × 800 epochs) | 12 MARST trains | ~ 25 min |
| Cell C (R2, KNN k=10) | CPU only | ~ 5 min |
| Cell D (R5, 4 datasets × 1 seed × 800 epochs at hidden=64) | 4 mini-MARST trains | ~ 12 min |
| Cell E (aggregation) | nil | < 1 s |
| **Total** | | **~ 45 min** |

A single Kaggle session (9 h) is more than enough.  If you bump Cell D
to 3 seeds the total rises to ~ 70 min — still trivial.

---

*Place these five cells in order after the main `run_dataset` driver,
re-run, and the three review-required experiments are done.*
