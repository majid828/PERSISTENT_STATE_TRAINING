import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data.camels_hourly_loader import train_loader_pers, buckets_for_training, windows_per_batch

# ----------------------------
# Training hyper-parameters
# ----------------------------
num_epochs = 21
learning_rate = np.array([1e-3] * 7 + [5e-4] * 7 + [1e-4] * 7)


def train_persistent(model, device):
    """
    Persistent/stateful: Non-overlapping windows, sequential inside batch,
    carry state across windows/batches. Loss on ALL points in each window.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate[0]), weight_decay=1e-5)
    criterion = nn.MSELoss()

    epoch_losses = []
    t0_train = time.perf_counter()

    for epoch in range(num_epochs):
        for g in optimizer.param_groups:
            g["lr"] = float(learning_rate[epoch])

        epoch_loss = 0.0
        total_windows = 0

        for ibuc in buckets_for_training:
            saved_states = None  # reset between basins

            for (data, targets) in train_loader_pers[ibuc]:
                # data: [windows_per_batch, L, F]
                data, targets = data.to(device), targets.to(device)
                current_states = saved_states

                # Process each window sequentially, carrying state
                for window_pos in range(windows_per_batch):
                    window_x = data[window_pos : window_pos + 1]  # [1, L, F]
                    window_y = targets[window_pos : window_pos + 1]  # [1, L, C]

                    optimizer.zero_grad()
                    out, new_hidden = model(window_x, hidden_states=current_states)
                    loss = criterion(out, window_y)  # ALL steps loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_windows += 1

                    # Detach & keep values for next window
                    if new_hidden is not None:
                        current_states = (
                            new_hidden[0].detach(),
                            new_hidden[1].detach(),
                        )

                # Save for next batch (state continues across batches)
                saved_states = current_states

        avg_epoch_loss = epoch_loss / total_windows if total_windows > 0 else 0.0
        avg_epoch_rmse = float(np.sqrt(avg_epoch_loss))
        epoch_losses.append(avg_epoch_loss)
        print(
            f"[Persistent] Epoch {epoch:02d} | lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"Loss: {avg_epoch_loss:.4f} | RMSE: {avg_epoch_rmse:.4f}"
        )

    t1_train = time.perf_counter()
    print(f"[TIME] Persistent TRAIN total: {t1_train - t0_train:.3f} s")

    results = {
        ibuc: {"loss": epoch_losses, "RMSE": [float(np.sqrt(l)) for l in epoch_losses]}
        for ibuc in buckets_for_training
    }
    return model, results, (t1_train - t0_train)

