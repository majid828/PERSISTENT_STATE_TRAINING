import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import sklearn
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import trange, tqdm

from pyflo import system
from pyflo.nrcs import hydrology
uh484 = system.array_from_csv('distributions/scs484.csv')

g = 1.271e8
time_step = 1
rain_probability_range = {"None": [0.3, 0.4],
                          "Light": [0.4, 0.5],
                          "Heavy": [0.1, 0.3]}

threshold_precip = 0.01 # precip value boundary between "light" and "heavy"
max_precip = 0.25 # max amount of precip possible

# distribution parameters
rain_depth_range = {"Light": [0.0008108, 0.0009759], "Heavy": [0.2341, 0.0101, 0.009250]}
bucket_attributes_range = {"A_bucket": [5e2, 2e3],
                           "H_bucket": [0.1, 0.3],
                           "rA_spigot": [0.1, 0.2], # calculations to be a function of H_bucket
                           "rH_spigot": [0.05, 0.15], # calculations to be a function of H_bucket
                           ### The following two parameters come from standard distributions based on real data.
                           # Do not change these:
                           "K_infiltration": [-13.8857, 1.1835], # location and scale of normal distribution
                           "ET_parameter": [2.2447, 9.9807e-5, 0.0016], # shape, loc, and scale of Weibull min dist

                           "soil_depth": [0.3, 0.8]
                          }
bucket_attributes_list = list(bucket_attributes_range.keys())
bucket_attributes_list.append('A_spigot')
bucket_attributes_list.append('H_spigot')
bucket_attributes_lstm_inputs = ['H_bucket', 'rA_spigot', 'rH_spigot', 'soil_depth']
print("LSTM model input attributes", bucket_attributes_lstm_inputs)
input_vars = ['precip', 'et', 'h_bucket']
input_vars.extend(bucket_attributes_lstm_inputs)
output_vars = ['q_total', 'q_overflow', 'q_spigot']
n_input = len(input_vars)
n_output = len(output_vars)

noise = {"pet": 0.1, "et": 0.1, "q": 0.1, "head": 0.1}
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M3/M2/M1 (Metal) device")
else:
    device = 'cpu'
    print("Using CPU")
hidden_state_size = 128
num_layers = 1
num_epochs = 25
batch_size = 128
seq_length = 32
learning_rate = np.linspace(start=0.001, stop=0.0001, num=num_epochs)

n_buckets_split = {"train": 20, "val": 10,"test": 1}
time_splits = {"warmup":256, "train": 1032, "val": 1032,"test": 1032}


num_records = time_splits["warmup"] + time_splits["train"] + time_splits["val"] + time_splits["test"] + seq_length * 3
n_buckets = n_buckets_split["train"] + n_buckets_split["val"] + n_buckets_split["test"]

def split_parameters():
    # create lists of bucket indices for each set based on the given bucket splits
    buckets_for_training = list(range(0, n_buckets_split['train'] + 1))
    buckets_for_val = list(range(n_buckets_split['train'] + 1,
                                 n_buckets_split['train'] + n_buckets_split['val'] + 1))
    buckets_for_test = list(range(n_buckets - n_buckets_split['test'], n_buckets))

    # determine the time range for each set based on the given time splits
    train_start = time_splits["warmup"] + seq_length
    train_end   = time_splits["warmup"] + time_splits["train"]
    val_start   = train_end + seq_length
    val_end     = val_start + time_splits["val"]
    test_start  = val_end + seq_length
    test_end    = test_start + time_splits["test"]

    # organize the split parameters into separate lists for each set
    train_split_parameters = [buckets_for_training, train_start, train_end]
    val_split_parameters = [buckets_for_val, val_start, val_end]
    test_split_parameters = [buckets_for_test, test_start, test_end]

    return [train_split_parameters, val_split_parameters, test_split_parameters]

[[buckets_for_training, train_start, train_end],
[buckets_for_val, val_start, val_end],
[buckets_for_test, test_start, test_end]]= split_parameters()

def setup_buckets(n_buckets):
    # Boundary conditions
    buckets = {bucket_attribute:[] for bucket_attribute in bucket_attributes_list}
    buckets['A_spigot'] = []
    buckets['H_spigot'] = []
    for i in range(n_buckets):
        for attribute in bucket_attributes_list:
            if attribute == 'A_bucket' or attribute == 'H_bucket' or attribute == 'rA_spigot' or attribute == 'rH_spigot' or attribute == 'soil_depth':
                buckets[attribute].append(np.random.uniform(bucket_attributes_range[attribute][0],
                                                        bucket_attributes_range[attribute][1]))
            if attribute == 'K_infiltration':
                buckets[attribute].append(np.random.normal(bucket_attributes_range[attribute][0],
                                                        bucket_attributes_range[attribute][1]))

            if attribute == "ET_parameter":
                buckets[attribute].append(stats.weibull_min.rvs(bucket_attributes_range[attribute][0],
                                                                bucket_attributes_range[attribute][1],
                                                                bucket_attributes_range[attribute][2]))

        buckets['A_spigot'].append(np.pi * (0.5 * buckets['H_bucket'][i] * buckets['rA_spigot'][i]) ** 2)
        buckets['H_spigot'].append(buckets['H_bucket'][i] * buckets['rH_spigot'][i])

    # Initial conditions
    h_water_level = [np.random.uniform(0, buckets["H_bucket"][i]) for i in range(n_buckets)]
    mass_overflow = [0]*n_buckets

    return buckets, h_water_level, mass_overflow

buckets, h_water_level, mass_overflow = setup_buckets(n_buckets)

def pick_rain_params():
    buck_rain_params = [rain_depth_range,
                        np.random.uniform(rain_probability_range["None"][0],
                                            rain_probability_range["None"][1]),
                        np.random.uniform(rain_probability_range["Heavy"][0],
                                            rain_probability_range["Heavy"][1]),
                        np.random.uniform(rain_probability_range["Light"][0],
                                            rain_probability_range["Light"][1])
                 ]
    return buck_rain_params

def random_rain(preceding_rain, bucket_rain_params):
    depth_range, no_rain_probability, light_rain_probability, heavy_rain_probability = bucket_rain_params
    # some percent of time we have no rain at all
    if np.random.uniform(0.01, 0.99) < no_rain_probability:
        rain = 0

    # When we do have rain, the probability of heavy or light rain depends on the previous hour's rainfall
    else:
        rain = np.inf
        # If last hour was a light rainy hour, or no rain, then we are likely to have light rain this hour
        if preceding_rain < threshold_precip:
            if np.random.uniform(0.0, 1.0) < light_rain_probability:
                while rain < 0 or rain > threshold_precip:
                    rain = stats.gumbel_r.rvs(depth_range["Light"][0], depth_range["Light"][1])
            else:
                # But if we do have heavy rain, then it could be very heavy
                while rain < threshold_precip or rain > max_precip:
                    rain = stats.genpareto.rvs(depth_range["Heavy"][0], depth_range["Heavy"][1], depth_range["Heavy"][2])

        # If it was heavy rain last hour, then we might have heavy rain again this hour
        else:
            if np.random.uniform(0.0, 1.0) < heavy_rain_probability:
                while rain < threshold_precip or rain > max_precip:
                    rain = stats.genpareto.rvs(depth_range["Heavy"][0], depth_range["Heavy"][1], depth_range["Heavy"][2])
            else:
                while rain < 0 or rain > threshold_precip:
                    rain = stats.gumbel_r.rvs(depth_range["Light"][0], depth_range["Light"][1])
    return rain

in_list = {}
for ibuc in range(n_buckets):
    bucket_rain_params = pick_rain_params()
    in_list[ibuc] = [0]
    for i in range(1, num_records):
        in_list[ibuc].append(random_rain(in_list[ibuc][i-1], bucket_rain_params))

def apply_unit_hydrograph(df, ibuc):
    """Given a bucket‐simulation DataFrame with 'q_overflow' and 'q_spigot' (both in m/s normalized by area),
    compute and append a 'q_total' column by routing combined runoff through a unit hydrograph.
    """
    # build the Basin object
    area_acres = buckets["A_bucket"][ibuc] / 4047
    basin = hydrology.Basin(
        area       = area_acres,
        cn         = 83.0,
        tc         = 2.3,
        runoff_dist= uh484,
        peak_factor= 1
    )

    # prepare cumulative‐inch input
    n = len(df)
    q_in = np.zeros((n, 2))
    cum_inches = 0.0
    for i in range(n):
        cum_inches += (df.loc[i,'q_overflow'] + df.loc[i,'q_spigot']) * 39.3701
        q_in[i] = (i, cum_inches)

    # run UH
    full = basin.flood_hydrograph(q_in, interval=1)[:,1]

    # trim or pad to match df length
    if len(full) >= n:
        out = full[:n]
    else:
        out = np.pad(full, (0, n-len(full)), 'constant')

    # convert back to m per time step, normalized by area
    df['q_total'] = out / 35.315 / buckets["A_bucket"][ibuc] * 3600

    return df

def run_bucket_simulation(ibuc):
    columns = ['precip', 'et', 'infiltration', 'h_bucket', 'q_overflow', 'q_spigot', 'q_total']
    columns.extend(bucket_attributes_list)
    # Memory to store model results
    df = pd.DataFrame(index=list(range(len(in_list[ibuc]))), columns=columns)

    # Main loop through time
    for t, precip_in in enumerate(in_list[ibuc]):

        # Add the input mass to the bucket
        h_water_level[ibuc] = h_water_level[ibuc] + precip_in

        # Lose mass out of the bucket. Some periodic type loss, evaporation, and some infiltration...

        # ET (m/s) is the value at each time step taking diurnal fluctuations into account. The definite integral of the following function
        # (excluding noise) from 0 to 24 is equal to ET_parameter, which is measured in m/day.
        et = np.max([0, ((1/7.6394)* buckets["ET_parameter"][ibuc]) * np.sin((np.pi / 12)*t) * np.random.normal(1, noise['pet'])])

        k = 10 ** buckets['K_infiltration'][ibuc]
        L = buckets['soil_depth'][ibuc]

        # Calculate infiltration using Darcy’s Law: Q = (k * ρ * g * A * Δh) / (μ * L) → infiltration = Q / A
        # Final form: infiltration = (k * ρ * g * Δh) / (μ * L), with Δh = soil depth + water level height
        delta_h = h_water_level[ibuc] + L
        infiltration = k * delta_h / L

        h_water_level[ibuc] = np.max([0 , (h_water_level[ibuc] - et)])
        h_water_level[ibuc] = np.max([0 , (h_water_level[ibuc] - infiltration)])
        h_water_level[ibuc] = h_water_level[ibuc] * np.random.normal(1, noise['et'])

        # Overflow if the bucket is too full
        if h_water_level[ibuc] > buckets["H_bucket"][ibuc]:
            mass_overflow[ibuc] = h_water_level[ibuc] - buckets["H_bucket"][ibuc]
            h_water_level[ibuc] = buckets["H_bucket"][ibuc]
            h_water_level[ibuc] = h_water_level[ibuc] - np.random.uniform(0, noise['q'])

        # Calculate head on the spigot
        h_head_over_spigot = (h_water_level[ibuc] - buckets["H_spigot"][ibuc] )
        h_head_over_spigot = h_head_over_spigot * np.random.normal(1, noise['head'])

        # Calculate water leaving bucket through spigot
        if h_head_over_spigot > 0:
            velocity_out = np.sqrt(2 * g * h_head_over_spigot)
            spigot_out_volume = velocity_out *  buckets["A_spigot"][ibuc] * time_step

            # prevents spigot from draining water below H_spigot
            spigot_out = np.min([spigot_out_volume / buckets["A_bucket"][ibuc], h_head_over_spigot])
            h_water_level[ibuc] -= spigot_out
        else:
            spigot_out = 0

        # Save the data in time series
        df.loc[t,'precip'] = precip_in
        df.loc[t,'et'] = et
        df.loc[t,'infiltration'] = infiltration
        df.loc[t,'h_bucket'] = h_water_level[ibuc]
        df.loc[t,'q_overflow'] = mass_overflow[ibuc]
        df.loc[t,'q_spigot'] = spigot_out
        for attribute in bucket_attributes_list:
            df.loc[t, attribute] = buckets[attribute][ibuc]

        mass_overflow[ibuc] = 0

    # --- route through unit hydrograph ---
    df = apply_unit_hydrograph(df, ibuc)

    # ---- mass tracking columns ----
    # all in meters of water per time step, per unit area
    df['cum_precip']   = df['precip'].cumsum()
    df['cum_et']       = df['et'].cumsum()
    df['cum_inf']      = df['infiltration'].cumsum()
    df['cum_runoff']   = df['q_overflow'].cumsum() + df['q_spigot'].cumsum()
    df['storage']      = df['h_bucket']
    df['mass_out_tot'] = df['cum_et'] + df['cum_inf'] + df['cum_runoff'] + df['storage']
    df['residual_frac']= (df['cum_precip'] - df['mass_out_tot']) / df['cum_precip']
    # --------------------------------

    return df

bucket_dictionary = {}

# Define the progress milestones
milestones = [0.2, 0.4, 0.6, 0.8, 1.0]
n_buckets_completed = 0  # Counter for completed buckets

for ibuc in range(n_buckets):
    bucket_dictionary[ibuc] = run_bucket_simulation(ibuc)

    # Increment the completed bucket counter
    n_buckets_completed += 1

    # Calculate the current progress as a fraction
    progress = n_buckets_completed / n_buckets

    # Check if we have reached any of the milestones
    for milestone in milestones:
        if progress >= milestone:
            print(f"Progress: {int(milestone * 100)}% complete.")
            milestones.remove(milestone)  # Remove the milestone once it is reached
            break  # To avoid printing multiple milestones at once

def viz_simulation(ibuc):
    """
    Print bucket characteristics, mass‐balance summary, and plot inputs, outputs, and storage time series.
    """
    df = bucket_dictionary[ibuc]

    # --- Static bucket characteristics ---
    attrs = {
        "Bucket ID":              ibuc,
        "Bucket area (m²)":       df.A_bucket.iloc[0],
        "Spigot area (m²)":       df.A_spigot.iloc[0],
        "Bucket height (m)":      df.H_bucket.iloc[0],
        "Spigot height (m)":      df.H_spigot.iloc[0],
        "Permeability (m²)":      df.K_infiltration.iloc[0],
        "ET parameter (m/day)":   df.ET_parameter.iloc[0],
        "Soil depth (m)":         df.soil_depth.iloc[0],
        "Overflow mean (m/h)":    df.q_overflow.mean(),
        "Overflow max (m/h)":     df.q_overflow.max(),
    }
    for name, val in attrs.items():
        print(f"{name:<25}: {val:.2f}" if name != "Bucket ID" else f"{name:<25}: {val}")

    # --- Final mass‐balance values ---
    final = df.iloc[-1]
    mb = {
        "Total precip (m)":       final['cum_precip'],
        "Total ET (m)":           final['cum_et'],
        "Total infiltration (m)":  final['cum_inf'],
        "Total runoff (m)":       final['cum_runoff'],
        "Final storage (m)":      final['storage'],
        "Mass out + stored (m)":  final['mass_out_tot'],
        "Residual fraction":      final['residual_frac'],
    }
    print("\nMass‐balance summary:")
    for name, val in mb.items():
        if name == "Residual fraction":
            print(f"{name:<25}: {val:.2%}")
        else:
            print(f"{name:<25}: {val:.2f}")

    # --- Time slice for plotting ---
    start = time_splits["warmup"]
    end   = start + 256
    slice_df = df.loc[start:end]

    # --- Create subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True)

    slice_df[['precip', 'et', 'infiltration']].plot(ax=axes[0])
    axes[0].set(title='Model inputs', xlabel='Time (h)', ylabel='Depth (m)')

    slice_df[output_vars].plot(ax=axes[1])
    axes[1].set(title='Model outputs', xlabel='Time (h)', ylabel='Yield (m)')

    slice_df[['h_bucket']].plot(ax=axes[2])
    axes[2].set(title='Bucket water level', xlabel='Time (h)', ylabel='Height (m)')

    plt.tight_layout()
    plt.show()
    plt.close()

for i, ibuc in enumerate(buckets_for_val):
    viz_simulation(ibuc)
    if i > 2:
        break

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, batch_size, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_size, num_classes) # fully connected layer

    def forward(self, x, init_states=None):
        if init_states is None:
            h_t = Variable(torch.zeros(1, x.size(0), self.hidden_size, device=x.device)) # hidden state
            c_t = Variable(torch.zeros(1, x.size(0), self.hidden_size, device=x.device)) # internal state
            init_states = (h_t, c_t)

        out, _ = self.lstm(x, init_states)
        out = self.relu(out)  # Apply ReLU after LSTM to ensure positive values
        prediction = self.fc_1(out) # Dense, fully connected layer
        prediction = self.relu(prediction)  # Apply ReLU after the fully connected layer to ensure non-negative outputs

        return prediction

def check_validation_period(lstm, np_val_seq_X, ibuc, n_plot=100):

    def __make_prediction():
        lstm_output_val = lstm(torch.Tensor(np_val_seq_X[ibuc]).to(device=device))
        val_predictions = {var: [] for var in output_vars}

        for i in range(lstm_output_val.shape[0]):
            for j, var in enumerate(output_vars):
                val_predictions[var].append((lstm_output_val[i, -1, j].cpu().detach().numpy() * \
                                             np.std(df.loc[train_start:train_end, var])) + \
                                            np.mean(df.loc[train_start:train_end, var]))
        return val_predictions

    def __compute_nse(val_predictions):
        nse_values = {}
        for var in output_vars:
            actual_values = df.loc[val_start:val_end, var]
            mean_actual = np.mean(actual_values)
            pred_variance = 0
            obs_variance = 0

            for i, pred in enumerate(val_predictions[var]):
                t = i + seq_length - 1
                pred_variance += np.power((pred - actual_values.values[t]), 2)
                obs_variance += np.power((mean_actual - actual_values.values[t]), 2)

            nse_values[var] = np.round(1 - (pred_variance / obs_variance), 4)

        return nse_values

    def __compute_mass_balance():
        mass_in = df.sum()['precip']
        mass_out = df.sum()['et'] + \
                   df.sum()['q_spigot'] + \
                   df.sum()['q_overflow'] + \
                   df.sum()['infiltration'] + \
                   df.loc[num_records - 1, 'h_bucket']
        return mass_in, mass_out

    df = bucket_dictionary[ibuc]
    val_predictions = __make_prediction()
    nse_values = __compute_nse(val_predictions)
    mass_in, mass_out = __compute_mass_balance()

    for var in output_vars:
        print(f"{var} NSE: {nse_values[var]}")

    residual = (mass_in - mass_out) / mass_in
    print(f"Mass into the system:    {mass_in:.2f}")
    print(f"Mass out of the system:   {mass_out:.2f}")
    print(f"Percent mass residual:    {residual:.0%}")

    fig, axes = plt.subplots(1, len(output_vars), figsize=(20, 3))
    if len(output_vars) == 1:
        axes = [axes]

    for ax, var in zip(axes, output_vars):
        obs_start = val_start + seq_length
        obs_end = obs_start + n_plot - 1
        ax.plot(df.loc[obs_start:obs_end, var].values, label=f"{var} actual")
        ax.plot(val_predictions[var][:n_plot], label=f"LSTM {var} predicted")
        ax.legend()

    plt.show()
    plt.close()

torch.manual_seed(1)
lstm = LSTM1(num_classes=n_output,
             input_size=n_input,
             hidden_size=hidden_state_size,
             num_layers=num_layers,
             batch_size=batch_size,
             seq_length=seq_length).to(device=device)

def fit_scaler():
    frames = [bucket_dictionary[ibuc].loc[train_start:train_end, input_vars] for ibuc in buckets_for_training]
    df_in = pd.concat(frames)
    scaler_in = StandardScaler()
    _ = scaler_in.fit_transform(df_in)

    frames = [bucket_dictionary[ibuc].loc[train_start:train_end, output_vars] for ibuc in buckets_for_training]
    df_out = pd.concat(frames)
    scaler_out = StandardScaler()
    _ = scaler_out.fit_transform(df_out)
    return scaler_in, scaler_out

scaler_in, scaler_out = fit_scaler()

# Compare last k_preds timesteps of output vs target (seq-to-k); unlike seq-to-1, this provides richer temporal feedback
# Use seq-to-seq generally for multiple future steps, seq-to-1 for single-step ahead prediction
k_preds = 1

def make_data_loader(start, end, bucket_list):
    loader = {}
    np_seq_X = {}
    np_seq_y = {}

    for ibuc in bucket_list:
        df = bucket_dictionary[ibuc]
        # scale inputs and outputs
        Xin = scaler_in.transform(df.loc[start:end, input_vars])
        Yin = scaler_out.transform(df.loc[start:end, output_vars])

        # number of samples: full window length = seq_length + k_preds
        n_total = Xin.shape[0]
        n_samples = n_total - seq_length + 1

        # allocate arrays: inputs always seq_length, outputs now seq_length as before
        X = np.zeros((n_samples, seq_length, n_input))
        Y = np.zeros((n_samples, seq_length, n_output))

        for i in range(n_samples):
            t0 = i + seq_length
            X[i] = Xin[i:t0]
            Y[i] = Yin[i:t0]

        np_seq_X[ibuc] = X
        np_seq_y[ibuc] = Y

        ds = torch.utils.data.TensorDataset(
            torch.Tensor(X),
            torch.Tensor(Y)
        )
        loader[ibuc] = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    return loader, np_seq_X, np_seq_y

train_loader, np_train_seq_X, np_train_seq_y = make_data_loader(train_start, train_end, buckets_for_training)
val_loader, np_val_seq_X, np_val_seq_y = make_data_loader(val_start, val_end, buckets_for_val)
test_loader, np_test_seq_X, np_test_seq_y = make_data_loader(test_start, test_end, buckets_for_test)

def train_model(lstm, train_loader, buckets_for_training):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=learning_rate[0])
    epoch_bar = tqdm(range(num_epochs), desc="Training", position=0, total=num_epochs)
    results = {ibuc: {"loss": [], "RMSE": []} for ibuc in buckets_for_training}

    for epoch in epoch_bar:
        for ibuc in buckets_for_training:
            batch_bar = tqdm(
                train_loader[ibuc],
                desc=f"Bucket: {ibuc}, Epoch: {epoch}",
                position=1, leave=False, disable=True
            )

            # --- Training Loop ---
            for data, targets in batch_bar:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()

                out = lstm(data)
                preds = out[:, -k_preds:, :]        # last k_preds timesteps
                true = targets[:, -k_preds:, :]
                loss = criterion(preds, true)
                loss.backward()
                optimizer.step()

                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    RMSE=f"{loss.sqrt().item():.2f}"
                )

            # --- Validation on train set for RMSE ---
            with torch.no_grad():
                rmses = []
                for data_, targets_ in train_loader[ibuc]:
                    data_, targets_ = data_.to(device), targets_.to(device)
                    out_ = lstm(data_)
                    preds_ = out_[:, -k_preds:, :]
                    mse_ = criterion(preds_, targets_[:, -k_preds:, :])
                    rmses.append(mse_.sqrt().item())
                mean_rmse = np.mean(rmses)

            # record metrics
            results[ibuc]["loss"].append(loss.item())
            results[ibuc]["RMSE"].append(mean_rmse)

            epoch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                RMSE=f"{mean_rmse:.2f}"
            )
        print(f"Epoch {epoch} mean RMSE: {mean_rmse:.2f}")

    return lstm, results


lstm, results = train_model(lstm, train_loader, buckets_for_training)
def viz_learning_curve(results, buckets_for_training):
    """
    Plot the mean learning curve with 5th–95th percentile intervals for loss and RMSE.
    """
    # Stack metrics across buckets
    losses = np.stack([results[b]['loss'] for b in buckets_for_training])
    rmses  = np.stack([results[b]['RMSE'] for b in buckets_for_training])
    epochs = np.arange(losses.shape[1])

    # Compute mean and percentile bounds
    mean_loss = losses.mean(axis=0)
    low_loss  = np.percentile(losses, 5, axis=0)
    high_loss = np.percentile(losses, 95, axis=0)

    mean_rmse = rmses.mean(axis=0)
    low_rmse  = np.percentile(rmses, 5, axis=0)
    high_rmse = np.percentile(rmses, 95, axis=0)

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharex=True)

    # Loss plot

    ax1.plot(epochs, mean_loss, label="Mean", c="k")
    ax1.plot(epochs, low_loss,  linestyle="--", label="5/95th pct", c="grey")
    ax1.plot(epochs, high_loss, linestyle="--", c="grey")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend()

    # RMSE plot
    ax2.plot(epochs, mean_rmse, label="Mean", c="k")
    ax2.plot(epochs, low_rmse,  linestyle="--", label="5/95th pct", c="grey")
    ax2.plot(epochs, high_rmse, linestyle="--", c="grey")
    ax2.set(title="RMSE", xlabel="Epoch", ylabel="RMSE")
    ax2.legend()

    plt.suptitle("Learning Curve Summary")
    plt.tight_layout()
    plt.show()


viz_learning_curve(results, buckets_for_training)
for ibuc in buckets_for_val:
    check_validation_period(lstm, np_val_seq_X, ibuc)
