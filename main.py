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
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]


    model = DeepAR(h=12,
                   input_size=24,
                   lstm_n_layers=1,
                   trajectory_samples=100,
                   loss=MSE(outputsize_multiplier=1),
                   valid_loss=MSE,
                   learning_rate=0.005,
                   stat_exog_list=['airline1'],
                   futr_exog_list=['trend'],
                   max_steps=100,
                   val_check_steps=10,
                   early_stop_patience_steps=-1,
                   scaler_type='standard',
                   enable_progress_bar=True,
                   )

    nf = NeuralForecast(
        models=[
            model
        ]
    )

    nf.fit(df=y_train, val_size=test_size)

    print("control")