import time
import numpy as np
import pandas as pd
import torch

from data.camels_hourly_loader import (
    make_persistent_window_arrays,
    SequentialNonOverlappingSampler,
    bucket_dictionary,
    date_splits,
    output_vars,
    seq_length,
    windows_per_batch,
)
from utils.scalers import denorm_from_scaler


def compute_nse(pred, obs):
    pred = np.asarray(pred).ravel()
    obs = np.asarray(obs).ravel()
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return -999.0
    return 1.0 - np.sum((obs - pred) ** 2) / denom


def predict_block_persistent_allpoints(model, df, base_idx, end_idx, device):
    """
    Persistent predictions (NON-overlapping windows), bounded to [base_idx, end_idx].
    Returns (absolute_obs_indices, denorm_preds) for ALL time steps (dense coverage).
    """
    model.eval()
    X_np, _ = make_persistent_window_arrays(df, base_idx, end_idx, seq_length)
    Xfull = torch.tensor(X_np, dtype=torch.float32, device=device)  # [Nw, L, F]

    preds_idx, preds_val = [], []
    sampler = SequentialNonOverlappingSampler(
        n_samples=Xfull.size(0), windows_per_batch=windows_per_batch
    )
    saved_states = None

    with torch.no_grad():
        for batch_indices in sampler:
            data = Xfull[batch_indices]  # [windows_per_batch, L, F]
            current_states = saved_states
            for window_pos, window_idx in enumerate(batch_indices):
                x = data[window_pos : window_pos + 1]  # [1, L, F]
                out, new_hidden = model(x, hidden_states=current_states)  # [1, L, C]

                # Map ALL steps to absolute indices, store denormed predictions
                seq_pred = out[0, :, 0].cpu().numpy()  # [L]
                for step in range(seq_length):
                    obs_index = int(base_idx) + (window_idx * seq_length) + step
                    preds_idx.append(obs_index)
                    preds_val.append(denorm_from_scaler(float(seq_pred[step])))

                if new_hidden is not None:
                    current_states = (new_hidden[0].detach(), new_hidden[1].detach())
            saved_states = current_states

    return preds_idx, preds_val


def nse_sparse(idx_list, preds_list, df, s, e, var):
    """Compute NSE using only indices inside the segment [s+(L-1), e]."""
    keep = [i for i in idx_list if (s + (seq_length - 1)) <= i <= e]
    if not keep:
        return np.nan
    m = {i: v for i, v in zip(idx_list, preds_list)}
    obs = [df.loc[i, var] for i in sorted(set(keep))]
    prd = [m[i] for i in sorted(set(keep))]
    return compute_nse(prd, obs)


def build_dense_series(idx_list, preds_list, df, s, e, var):
    """
    Build aligned (dates, obs, preds) arrays for plotting, restricted to valid indices.
    """
    keep = [i for i in idx_list if (s + (seq_length - 1)) <= i <= e]
    if not keep:
        return np.array([]), np.array([]), np.array([])
    m = {i: v for i, v in zip(idx_list, preds_list)}
    idx_sorted = sorted(set(keep))
    obs = np.array([df.loc[i, var] for i in idx_sorted])
    prd = np.array([m[i] for i in idx_sorted])
    dates = pd.to_datetime(df.loc[idx_sorted, "date"].values)
    return dates, obs, prd


def eval_persistent_all_basins(model_persistent, device):
    per_basin = {}
    timings = {
        "persistent": {"train": 0.0, "val": 0.0, "test": 0.0},
    }

    print("\n" + "=" * 50)
    print("PERSISTENT LSTM - NASH-SUTCLIFFE EFFICIENCY (NSE)")
    print("=" * 50)

    for ibuc in bucket_dictionary:
        df = bucket_dictionary[ibuc]
        s_tr, e_tr = date_splits[ibuc]["train"]
        s_va, e_va = date_splits[ibuc]["val"]
        s_te, e_te = date_splits[ibuc]["test"]

        # Persistent timing & NSE (ALL points)
        t0 = time.perf_counter()
        idx_tr, p_tr = predict_block_persistent_allpoints(
            model_persistent, df, base_idx=s_tr, end_idx=e_tr, device=device
        )
        timings["persistent"]["train"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        idx_va, p_va = predict_block_persistent_allpoints(
            model_persistent, df, base_idx=s_va, end_idx=e_va, device=device
        )
        timings["persistent"]["val"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        idx_te, p_te = predict_block_persistent_allpoints(
            model_persistent, df, base_idx=s_te, end_idx=e_te, device=device
        )
        timings["persistent"]["test"] += time.perf_counter() - t0

        nse_p_tr = nse_sparse(idx_tr, p_tr, df, s_tr, e_tr, output_vars[0])
        nse_p_va = nse_sparse(idx_va, p_va, df, s_va, e_va, output_vars[0])
        nse_p_te = nse_sparse(idx_te, p_te, df, s_te, e_te, output_vars[0])

        per_basin[ibuc] = dict(pers=(nse_p_tr, nse_p_va, nse_p_te))

        print(f"\nBasin {ibuc}:")
        print(
            f"  Persistent - Train: {nse_p_tr:.4f}, Val: {nse_p_va:.4f}, Test: {nse_p_te:.4f}"
        )

    arr_p = np.array([per_basin[i]["pers"] for i in per_basin], dtype=float)
    avg_pers = np.nanmean(arr_p, axis=0)

    print(f"\n{'=' * 50}")
    print("MACRO-AVERAGE RESULTS (PERSISTENT ONLY):")
    print(f"{'=' * 50}")
    print(
        f"Persistent - Train: {avg_pers[0]:.4f}, Val: {avg_pers[1]:.4f}, Test: {avg_pers[2]:.4f}"
    )

    return per_basin, avg_pers, timings

