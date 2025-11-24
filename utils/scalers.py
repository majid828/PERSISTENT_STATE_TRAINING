from data.camels_hourly_loader import scaler_out


def denorm_from_scaler(x_std: float) -> float:
    """Denormalize a single standardized discharge value using global scaler_out."""
    mu = float(scaler_out.mean_[0])
    sd = float(scaler_out.scale_[0])
    return max(x_std * sd + mu, 0.0)

