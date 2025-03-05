import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MSE

from call_datasets import gas_demand


if __name__ == "__main__":
    X, y, test_size = gas_demand()
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size].to_frame()
    y_test = y.iloc[-test_size:].to_frame()
    train_size = len(y) - test_size

    model = DeepAR(
        h=test_size,
        input_size=train_size,
        lstm_n_layers=2,
        trajectory_samples=100,
        loss=DistributionLoss(distribution="StudentT", level=[80, 90], return_params=True),
        valid_loss=MQLoss(level=[80, 90]),
        learning_rate=0.005,
        stat_exog_list=list(X_train.columns[-11:]),
        futr_exog_list=list(X_train.columns[:-11]),
        max_steps=100,
        val_check_steps=10,
        early_stop_patience_steps=-1,
        scaler_type="standard",
        enable_progress_bar=True,
    )

    nf = NeuralForecast(models=[model], freq="ME")

    nf.fit(df=y_train, val_size=test_size, random_seed=24)

    print("control")
