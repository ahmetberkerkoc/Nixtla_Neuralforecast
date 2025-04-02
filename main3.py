import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    DeepAR,
    Informer,
    iTransformer,
    NBEATS,
    NHITS,
    TimeMixer,
    TFT,
    Autoformer,
    FEDformer,
    VanillaTransformer,
)
from neuralforecast.losses.numpy import mse, mae
from neuralforecast.losses.pytorch import MSE, MAE

# Load data
df = pd.read_csv("data/turkey_gas.csv")

# Rename columns
df.rename(columns={"Unnamed: 0": "ds", "ABONE": "y"}, inplace=True)
df["ds"] = pd.to_datetime(df["ds"])  # Convert to datetime
df["unique_id"] = "turkey_gas"  # Single time-series
df["y"] /= 10**7

# Sort by date
df = df.sort_values(by="ds")

# Split into train (80%) and test (20%)
train_size = len(df) - int(len(df) * 0.2)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
val_size = int(len(df) * 0.1)

# Forecast horizon equals the length of the test set
forecast_horizon = len(test_df)
one_piece_len = 12
num_parts = int(forecast_horizon / one_piece_len)
rest_of_data = forecast_horizon % one_piece_len
# Define and train the DeepAR model
models = [
    # TimeMixer(
    #     h=one_piece_len,
    #     input_size=24,
    #     n_series=2,
    #     scaler_type="standard",
    #     max_steps=500,
    #     early_stop_patience_steps=-1,
    #     val_check_steps=5,
    #     learning_rate=1e-3,
    #     loss=MSE(),
    #     valid_loss=MSE(),
    #     batch_size=32,
    # )
    # iTransformer(
    #     h=one_piece_len,
    #     input_size=60,
    #     n_series=2,
    #     hidden_size=128,
    #     n_heads=2,
    #     e_layers=2,
    #     d_layers=1,
    #     d_ff=4,
    #     factor=1,
    #     dropout=0.1,
    #     use_norm=True,
    #     loss=MSE(),
    #     batch_size=32,
    #     max_steps=1000,
    # )
    # VanillaTransformer(
    #     h=one_piece_len,
    #     input_size=60,
    #     hidden_size=16,
    #     conv_hidden_size=32,
    #     n_head=2,
    #     loss=MSE(),
    #     scaler_type="robust",
    #     learning_rate=1e-3,
    #     max_steps=1000,
    #     #val_check_steps=50,
    #     #early_stop_patience_steps=2,
    # )
    iTransformer(
        h=one_piece_len,
        input_size=24,
        n_series=2,
        hidden_size=128,
        n_heads=2,
        e_layers=2,
        d_layers=1,
        d_ff=4,
        factor=1,
        dropout=0.1,
        use_norm=True,
        loss=MSE(),
        batch_size=32,
        max_steps=500,
    )
]

model_name = "iTransformer"
for i in range(num_parts):
    print(f"STEP {i}")
    train_df = df.iloc[: (i * one_piece_len) + train_size]
    test_df = df.iloc[(i * one_piece_len) + train_size : ((i + 1) * one_piece_len) + train_size]

    nf = NeuralForecast(models=models, freq="D")
    nf.fit(train_df)

    # Make predictions for the test period
    forecast_df = nf.predict(futr_df=test_df)
    forecast_df = forecast_df.iloc[:one_piece_len]
    if i == 0:
        final_forecast_df = forecast_df
    else:
        final_forecast_df = pd.concat([final_forecast_df, forecast_df], axis=0)

if rest_of_data:

    models = [
        # TimeMixer(
        #     h=rest_of_data,
        #     input_size=60,
        #     n_series=2,
        #     scaler_type="standard",
        #     max_steps=1000,
        #     early_stop_patience_steps=-1,
        #     val_check_steps=5,
        #     learning_rate=1e-3,
        #     loss=MAE(),
        #     valid_loss=MAE(),
        #     batch_size=32,
        # )
        iTransformer(
            h=rest_of_data,
            input_size=100,
            n_series=2,
            hidden_size=128,
            n_heads=2,
            e_layers=2,
            d_layers=1,
            d_ff=4,
            factor=1,
            dropout=0.1,
            use_norm=True,
            loss=MSE(),
            batch_size=32,
            max_steps=1000,
        )
    ]

    train_df = df.iloc[:-rest_of_data]
    test_df = df.iloc[-rest_of_data:]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(train_df)
    forecast_df = nf.predict(futr_df=test_df)
    forecast_df = forecast_df.iloc[:rest_of_data]
    final_forecast_df = pd.concat([final_forecast_df, forecast_df], axis=0)

final_forecast_df.reset_index(drop=True, inplace=True)
fores = final_forecast_df[model_name] * (10**7)
fores = fores.to_numpy()

test_df = df.iloc[train_size:]
tests = test_df["y"] * (10**7)
tests = tests.to_numpy()

print("MSE: {:e}".format(mse(tests, fores)))
print("MAE: {:e}".format(mae(tests, fores)))


# Plot train, test, and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train_df["ds"], train_df["y"] * (10**7), label="Train Data", color="blue", linewidth=2)
plt.plot(test_df["ds"], test_df["y"] * (10**7), label="Test Data", color="green", linewidth=2)
plt.plot(
    final_forecast_df["ds"],
    final_forecast_df[model_name] * (10**7),
    label="Forecast",
    color="red",
    linestyle="dashed",
    linewidth=2,
)
plt.axvline(test_df["ds"].iloc[0], color="gray", linestyle="--", label="Test Start")
plt.xlabel("Date")
plt.ylabel("Gas Consumption (ABONE)")
plt.title("{} Forecast on Test Set MSE {:e}".format(model_name, mse(tests, fores)))
plt.legend()
plt.savefig(f"figures/turkey_gas_{model_name}.png")

print("control")
