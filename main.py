#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from data.camels_hourly_loader import (
    bucket_dictionary,
    n_input,
    n_output,
)
from model.persistent_lstm import LSTMPersistent
from training.persistent_trainer import train_persistent
from utils.metrics import eval_persistent_all_basins
from utils.plotting import plot_persistent_train_val_test_for


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA:", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def main():
    torch.manual_seed(1)
    device = get_device()

    # Model hyper-parameters
    hidden_state_size = 64
    num_layers = 2
    DROPOUT_P = 0.3

    model_persistent = LSTMPersistent(
        num_classes=n_output,
        input_size=n_input,
        hidden_size=hidden_state_size,
        num_layers=num_layers,
        dropout_p=DROPOUT_P,
    ).to(device)

    print("\n" + "=" * 60)
    print(
        "TRAINING PERSISTENT (STATEFUL) LSTM — NON-OVERLAPPING WINDOWS, SEQUENTIAL WITH STATE"
    )
    print("=" * 60)

    model_persistent, _, t_train_persistent = train_persistent(
        model_persistent, device=device
    )

    # -------- EVALUATION: per-basin + macro-average + TIMINGS --------
    per_basin, avg_pers, timings = eval_persistent_all_basins(
        model_persistent, device=device
    )

    print("\n" + "=" * 60)
    print("TIMINGS SUMMARY (seconds) — PERSISTENT ONLY")
    print("=" * 60)
    print(f"TRAIN — Persistent: {t_train_persistent:.3f} s")
    print(
        f"PRED  — Persistent: train={timings['persistent']['train']:.3f}, "
        f"val={timings['persistent']['val']:.3f}, test={timings['persistent']['test']:.3f}"
    )

    # -------- PLOT (all basins) --------
    print("\nPlotting all basins (Persistent vs Actual) …")
    for ibuc in bucket_dictionary.keys():
        print(f"Plotting ibuc={ibuc} …")
        plot_persistent_train_val_test_for(
            ibuc=ibuc,
            model_persistent=model_persistent,
            device=device,
        )


if __name__ == "__main__":
    main()

