import numpy as np
import matplotlib.pyplot as plt

from data.camels_hourly_loader import bucket_dictionary, date_splits, target_col
from utils.metrics import (
    predict_block_persistent_allpoints,
    build_dense_series,
    compute_nse,
)


def plot_persistent_train_val_test_for(ibuc, model_persistent, device, var=target_col):
    """Plot for a single basin id: actual vs persistent predictions for Train/Val/Test."""
    df = bucket_dictionary[ibuc]
    s_tr, e_tr = date_splits[ibuc]["train"]
    s_va, e_va = date_splits[ibuc]["val"]
    s_te, e_te = date_splits[ibuc]["test"]

    # Persistent predictions (ALL points, dense)
    idx_p_tr, pers_train = predict_block_persistent_allpoints(
        model_persistent, df, base_idx=s_tr, end_idx=e_tr, device=device
    )
    idx_p_va, pers_val = predict_block_persistent_allpoints(
        model_persistent, df, base_idx=s_va, end_idx=e_va, device=device
    )
    idx_p_te, pers_test = predict_block_persistent_allpoints(
        model_persistent, df, base_idx=s_te, end_idx=e_te, device=device
    )

    # Build aligned dense series for plotting
    dates_tr, obs_tr, prd_tr = build_dense_series(
        idx_p_tr, pers_train, df, s_tr, e_tr, var
    )
    dates_va, obs_va, prd_va = build_dense_series(
        idx_p_va, pers_val, df, s_va, e_va, var
    )
    dates_te, obs_te, prd_te = build_dense_series(
        idx_p_te, pers_test, df, s_te, e_te, var
    )

    nse_p_tr = compute_nse(prd_tr, obs_tr) if len(obs_tr) > 0 else np.nan
    nse_p_va = compute_nse(prd_va, obs_va) if len(obs_va) > 0 else np.nan
    nse_p_te = compute_nse(prd_te, obs_te) if len(obs_te) > 0 else np.nan

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    panels = [
        ("Train", dates_tr, obs_tr, prd_tr, nse_p_tr),
        ("Val", dates_va, obs_va, prd_va, nse_p_va),
        ("Test", dates_te, obs_te, prd_te, nse_p_te),
    ]
    for ax, (name, dates, obs, preds, nse_p) in zip(axes, panels):
        if len(obs) == 0:
            ax.set_title(f"ibuc={ibuc} — {name} Segment (no data)")
            ax.grid(True, alpha=0.3)
            continue
        ax.plot(dates, obs, "-", linewidth=1.2, color="k", label=f"Actual ({name})")
        ax.plot(
            dates,
            preds,
            ":",
            linewidth=1.4,
            label=f"Persistent {name} NSE={nse_p:.3f}",
        )
        ax.set_ylabel(var)
        ax.set_title(f"ibuc={ibuc} — {name} Segment")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{var} — Last 20 Years | Train(14y)/Val(4y)/Test(2y)", y=0.98)
    plt.tight_layout()
    plt.show()

