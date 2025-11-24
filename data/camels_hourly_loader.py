import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, Sampler

# ----------------------------
# Columns
# ----------------------------
camels_input_cols = [
    "convective_fraction",
    "longwave_radiation",
    "potential_energy",
    "potential_evaporation",
    "pressure",
    "shortwave_radiation",
    "specific_humidity",
    "temperature",
    "total_precipitation",
    "wind_u",
    "wind_v",
]
target_col = "QObs_CAMELS(mm/h)"

input_vars = camels_input_cols
output_vars = [target_col]

# ----------------------------
# Hyper-parameters related to data
# ----------------------------
seq_length = 48
n_input = len(input_vars)
n_output = len(output_vars)

# use last 20 years; 14/4/2 split
YEARS_BACK = 20
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = 14, 4, 2

# Persistent "batch size" = number of windows per batch
windows_per_batch = seq_length


def load_camels_hourly(input_csv, output_csv):
    df_x = pd.read_csv(input_csv)
    df_y = pd.read_csv(output_csv)
    df_x["date"] = pd.to_datetime(df_x["date"])
    df_y["date"] = pd.to_datetime(df_y["date"])
    df_x = df_x[["date"] + camels_input_cols]
    df_y = df_y["date"].to_frame().join(df_y[target_col])
    df = (
        pd.merge(df_x, df_y, on="date", how="inner")
        .dropna(subset=camels_input_cols + [target_col])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df.index = np.arange(len(df))
    return df


# ----------------------------
# Basin definitions
# ----------------------------
BASINS = [
    dict(
        ibuc=0,
        X="CAMELS_data_sample/hourly/nldas_hourly/01333000_hourly_nldas.csv",
        Y="CAMELS_data_sample/hourly/usgs-streamflow/01333000-usgs-hourly.csv",
    ),
    dict(
        ibuc=1,
        X="CAMELS_data_sample/hourly/nldas_hourly/01423000_hourly_nldas.csv",
        Y="CAMELS_data_sample/hourly/usgs-streamflow/01423000-usgs-hourly.csv",
    ),
    dict(
        ibuc=2,
        X="CAMELS_data_sample/hourly/nldas_hourly/02046000_hourly_nldas.csv",
        Y="CAMELS_data_sample/hourly/usgs-streamflow/02046000-usgs-hourly.csv",
    ),
]

# Load raw data for each basin
bucket_dictionary = {b["ibuc"]: load_camels_hourly(b["X"], b["Y"]) for b in BASINS}


# ----------------------------
# Restrict to last N years
# ----------------------------
def restrict_to_last_years(df, years_back=5):
    end_date = df["date"].max()
    start_date = end_date - pd.DateOffset(years=years_back)
    df2 = df[df["date"] >= start_date].copy().reset_index(drop=True)
    df2.index = np.arange(len(df2))
    if len(df2) < seq_length * 4:
        raise ValueError(f"Not enough rows after {years_back}y restriction: {len(df2)}")
    return df2


for ibuc in bucket_dictionary:
    bucket_dictionary[ibuc] = restrict_to_last_years(
        bucket_dictionary[ibuc], years_back=YEARS_BACK
    )


# ----------------------------
# Date-based train/val/test splits
# ----------------------------
def compute_date_splits(df, train_years, val_years, test_years, seq_length):
    """
    Compute train/val/test index boundaries based strictly on years.
    Ensures each split contains at least one full window of length seq_length.
    """
    end_date = df["date"].max()

    # Compute date boundaries
    test_start_date = end_date - pd.DateOffset(years=test_years)
    val_start_date = test_start_date - pd.DateOffset(years=val_years)
    train_start_date = val_start_date - pd.DateOffset(years=train_years)

    # In case dataset is shorter than requested years
    train_start_date = max(train_start_date, df["date"].min())

    # Convert dates to indices
    i_train_start = int(df["date"].searchsorted(train_start_date, side="left"))
    i_val_start = int(df["date"].searchsorted(val_start_date, side="left"))
    i_test_start = int(df["date"].searchsorted(test_start_date, side="left"))
    i_end = len(df) - 1

    # Safety checks: each segment must contain >= seq_length rows
    assert (i_val_start - i_train_start) >= seq_length, "Train slice too short for seq_length"
    assert (i_test_start - i_val_start) >= seq_length, "Val slice too short for seq_length"
    assert (i_end + 1 - i_test_start) >= seq_length, "Test slice too short for seq_length"

    return {
        "train": (i_train_start, i_val_start - 1),
        "val": (i_val_start, i_test_start - 1),
        "test": (i_test_start, i_end),
    }


date_splits = {
    ibuc: compute_date_splits(
        bucket_dictionary[ibuc],
        train_years=TRAIN_YEARS,
        val_years=VAL_YEARS,
        test_years=TEST_YEARS,
        seq_length=seq_length,
    )
    for ibuc in bucket_dictionary
}

buckets_for_training = list(bucket_dictionary.keys())
print("Basins:", buckets_for_training)
for ibuc in buckets_for_training:
    tr = date_splits[ibuc]["train"]
    va = date_splits[ibuc]["val"]
    te = date_splits[ibuc]["test"]
    print(
        f"ibuc={ibuc} Train[{tr[0]},{tr[1]}]  Val[{va[0]},{va[1]}]  Test[{te[0]},{te[1]}]"
    )

# ----------------------------
# Scalers (fitted over all training data)
# ----------------------------
def fit_scaler_multi():
    frames_in, frames_out = [], []
    for ibuc in buckets_for_training:
        s, e = date_splits[ibuc]["train"]
        df = bucket_dictionary[ibuc]
        frames_in.append(df.loc[s:e, input_vars])
        frames_out.append(df.loc[s:e, output_vars])
    df_in = pd.concat(frames_in, axis=0)
    df_out = pd.concat(frames_out, axis=0)
    scaler_in = StandardScaler().fit(df_in)
    scaler_out = StandardScaler().fit(df_out)
    return scaler_in, scaler_out


scaler_in, scaler_out = fit_scaler_multi()


# ============================================================
# Persistent windowed arrays
# ============================================================
def make_persistent_window_arrays(df, start_idx, end_idx, seq_length):
    """Create NON-OVERLAPPING windows for persistent LSTM (step=seq_length)."""
    end_idx = min(end_idx, len(df) - 1)
    start_idx = min(start_idx, end_idx)
    Xin = scaler_in.transform(df.loc[start_idx:end_idx, input_vars])
    Yin = scaler_out.transform(df.loc[start_idx:end_idx, output_vars])

    n_total = Xin.shape[0]
    if n_total < seq_length:
        raise ValueError(f"Slice too short: n_total={n_total}, seq_length={seq_length}")

    n_samples = n_total // seq_length
    X = np.zeros((n_samples, seq_length, n_input), dtype=np.float32)
    Y = np.zeros((n_samples, seq_length, n_output), dtype=np.float32)
    for i in range(n_samples):
        s = i * seq_length
        e = s + seq_length
        X[i] = Xin[s:e]
        Y[i] = Yin[s:e]
    return X, Y


class SequentialNonOverlappingSampler(Sampler):
    """
    Sampler for sequential processing of NON-OVERLAPPING windows within batches.
    Batch size (for persistent training) = windows_per_batch.
    """

    def __init__(self, n_samples: int, windows_per_batch: int):
        self.n_samples = n_samples
        self.windows_per_batch = windows_per_batch
        self.num_batches = n_samples // windows_per_batch

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.windows_per_batch
            yield list(range(start, start + self.windows_per_batch))

    def __len__(self):
        return self.num_batches


def make_persistent_loader(start, end, ibuc_list, windows_per_batch):
    """Make persistent loader with NON-OVERLAPPING windows processed sequentially."""
    loader = {}
    arraysX, arraysY = {}, {}
    for ibuc in ibuc_list:
        df = bucket_dictionary[ibuc]
        X, Y = make_persistent_window_arrays(df, start, end, seq_length)
        arraysX[ibuc], arraysY[ibuc] = X, Y
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        sampler = SequentialNonOverlappingSampler(
            n_samples=X.shape[0], windows_per_batch=windows_per_batch
        )
        loader[ibuc] = DataLoader(ds, batch_sampler=sampler)
    return loader, arraysX, arraysY


def build_persistent_loaders():
    train_loader_pers, val_loader_pers, test_loader_pers = {}, {}, {}

    for ibuc in bucket_dictionary:
        s_tr, e_tr = date_splits[ibuc]["train"]
        s_va, e_va = date_splits[ibuc]["val"]
        s_te, e_te = date_splits[ibuc]["test"]

        # Train
        ld_tr, _, _ = make_persistent_loader(
            s_tr, e_tr, [ibuc], windows_per_batch=windows_per_batch
        )
        train_loader_pers[ibuc] = ld_tr[ibuc]

        # Val
        ld_va, _, _ = make_persistent_loader(
            s_va, e_va, [ibuc], windows_per_batch=windows_per_batch
        )
        val_loader_pers[ibuc] = ld_va[ibuc]

        # Test
        ld_te, _, _ = make_persistent_loader(
            s_te, e_te, [ibuc], windows_per_batch=windows_per_batch
        )
        test_loader_pers[ibuc] = ld_te[ibuc]

    return train_loader_pers, val_loader_pers, test_loader_pers


# Build loaders (persistent only)
train_loader_pers, val_loader_pers, test_loader_pers = build_persistent_loaders()

