import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MSE

from call_datasets import gas_demand




if __name__ == '__main__':
    X, y, test_size = gas_demand()
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size].to_frame()
    y_test = y.iloc[-test_size:].to_frame()


    model = DeepAR(h=12,
                   input_size=len(y_train),
                   lstm_n_layers=1,
                   trajectory_samples=100,
                   loss=MSE(),
                   valid_loss=MSE(),
                   learning_rate=0.005,
                   max_steps=100,
                   val_check_steps=10,
                   early_stop_patience_steps=-1,
                   scaler_type='standard',
                   enable_progress_bar=True,
                   )

    nf = NeuralForecast(
        models=[
            model
        ],
        freq='ME'
    )

    nf.fit(df=y_train, val_size=test_size)

    print("control")