import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer
from neuralforecast.losses.pytorch import MSE
from neuralforecast.losses.numpy import mse

# Load data
df = pd.read_csv("data/turkey_gas.csv")
df.rename(columns={"Unnamed: 0": "ds", "ABONE": "y"}, inplace=True)
df["ds"] = pd.to_datetime(df["ds"])
df["unique_id"] = "turkey_gas"
df["y"] /= 10**7
df = df.sort_values(by="ds")

# Split into train (80%) and test (20%)
train_size = len(df) - int(len(df) * 0.2)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
forecast_horizon = len(test_df)

# Define hyperparameter search space
param_grid = {
    "input_size": [30, 60, 90],
    "hidden_size": [128, 256, 512],
    "n_heads": [4, 8, 16],
    "dropout": [0.05, 0.1, 0.2],
    "learning_rate": [1e-4, 1e-3, 1e-2],
}

# Generate random search samples
num_trials = 100
random_params = [{key: np.random.choice(values) for key, values in param_grid.items()} for _ in range(num_trials)]

best_mse = float("inf")
best_params = None

# Iterate over random hyperparameter configurations
for params in random_params:
    model = iTransformer(
        h=forecast_horizon,
        input_size=params["input_size"],
        n_series=1,
        hidden_size=params["hidden_size"],
        n_heads=params["n_heads"],
        e_layers=2,
        d_layers=1,
        d_ff=1024,
        factor=1,
        dropout=params["dropout"],
        use_norm=True,
        loss=MSE(),
        max_steps=1000,
        learning_rate=params["learning_rate"],
    )

    # Train the model
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(train_df)

    # Make predictions
    forecast_df = nf.predict(futr_df=test_df)
    fores = forecast_df["iTransformer"] * (10**7)
    tests = test_df["y"] * (10**7)

    # Compute MSE
    current_mse = mse(tests.to_numpy(), fores.to_numpy())

    # Update best model if performance improved
    if current_mse < best_mse:
        best_mse = current_mse
        best_params = params

print(f"Best Hyperparameters: {best_params}")
print(best_mse)
print("Best MSE: {:e}".format(best_mse))
