import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss,  MQLoss

from call_datasets import gas_demand
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

if __name__ == "__main__":
    df, test_size = gas_demand()
    df = df.rename(columns={'ABONE': 'y'})
    train_size = len(df) - test_size
    df = df.loc[:,["unique_id","ds","y"]]
    deepar = DeepAR(
        h=test_size,
        input_size=test_size-5,
        lstm_n_layers=2,
        trajectory_samples=100,
        loss=DistributionLoss(distribution="StudentT",  return_params=True),
        valid_loss=MQLoss(),
        learning_rate=0.005,
        #futr_exog_list=list(df.columns[1:-2]),
        # hist_exog_list=df.columns[1:-2],
        max_steps=100,
        val_check_steps=10,
        early_stop_patience_steps=-1,
        scaler_type="standard",
        enable_progress_bar=True,
    )

    nf = NeuralForecast(models=[deepar], freq="D")
    nf.fit(df=df[:train_size], val_size=test_size)
    Y_hat_df = nf.predict(futr_df=df[-test_size:])
    print("control")
