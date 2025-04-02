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
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("pumadyn-32nm/puma32H.data", sep=",", header=None)
y_orig = df.iloc[:, -1]
df = df.iloc[:, :-1]
df["y"] = y_orig
col = {i: f"{i}" for i in range(32)}
df = df.rename(columns=col)


scaler = MinMaxScaler()
df["y"] = pd.Series(scaler.fit_transform(df["y"].to_numpy()[:, None]).ravel(), df.index)


start_date = "2000-01-01"
df["ds"] = range(len(df))  # Convert to datetime
df["unique_id"] = "elevator"  # Single time-series


# Split into train (80%) and test (20%)
train_size = len(df) - 5  # int(len(df) * 0.2)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Forecast horizon equals the length of the test set
forecast_horizon = len(test_df)

# Define and train the DeepAR model
models = [
    # DeepAR(
    #     h=forecast_horizon,
    #     input_size=30,
    #     max_steps=3000,
    #     scaler_type="standard",
    #     learning_rate=0.00001,
    #     # futr_exog_list=["ABONE_SICAKLIK_HI", "ABONE_SICAKLIK_MIN", "ABONE_RUZGAR_HI", "ABONE_RUZGAR_AVG"]
    #     futr_exog_list=list(train_df.columns[:32]),
    #     batch_size=1,
    # ),
    # Informer(
    #     h=forecast_horizon,
    #     input_size=forecast_horizon,
    #     hidden_size=32,
    #     conv_hidden_size=64,
    #     n_head=2,
    #     loss=MSE(),
    #     futr_exog_list=list(train_df.columns[:32]),
    #     scaler_type="standard",
    #     learning_rate=3e-4,
    #     max_steps=2000,
    # ),
    iTransformer(
        h=forecast_horizon,
        input_size=60,
        n_series=1,
        hidden_size=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=1024,
        factor=1,
        dropout=0.1,
        use_norm=True,
        loss=MSE(),
        max_steps=1000,
        learning_rate=1e-3,
    ),
    # TFT(h=forecast_horizon, input_size=forecast_horizon * 2, loss=MSE(), learning_rate=3e-3),
    # Autoformer(
    #     h=forecast_horizon,
    #     input_size=forecast_horizon * 2,
    #     loss=MSE(),
    # ),
    # FEDformer(
    #     h=forecast_horizon,
    #     input_size=forecast_horizon * 2,
    #     modes=64,
    #     hidden_size=64,
    #     conv_hidden_size=128,
    #     n_head=8,
    #     loss=MSE(),
    #     futr_exog_list=list(train_df.columns[:32]),
    #     learning_rate=1e-3,
    #     max_steps=150,
    # ),
    # TimeMixer(
    #     h=forecast_horizon,
    #     input_size=30,
    #     n_series=2,
    #     scaler_type="standard",
    #     max_steps=1000,
    #     early_stop_patience_steps=-1,
    #     val_check_steps=5,
    #     learning_rate=3e-2,
    #     loss=MSE(),
    #     valid_loss=MSE(),
    #     batch_size=32,
    # ),
    # VanillaTransformer(
    #     h=forecast_horizon,
    #     input_size=forecast_horizon * 2,
    #     hidden_size=32,
    #     conv_hidden_size=64,
    #     n_head=2,
    #     loss=MSE(),
    #     scaler_type="robust",
    #     learning_rate=1e-3,
    #     max_steps=300,
    # ),
]


nf = NeuralForecast(models=models, freq=1)
nf.fit(train_df)

# Make predictions for the test period
forecast_df = nf.predict(futr_df=test_df)
forecast_df = forecast_df.iloc[:forecast_horizon]

for model_name in nf.models:
    model_name = str(model_name)
    fores = forecast_df[model_name]
    fores = fores.to_numpy()
    fores = scaler.inverse_transform(fores[:, None]).ravel()

    tests = y_orig[-forecast_horizon:]

    print("MSE: {:e}".format(mse(tests[-forecast_horizon:].to_numpy(), fores[-forecast_horizon:])))
    print("MAE: {:e}".format(mae(tests[-forecast_horizon:].to_numpy(), fores[-forecast_horizon:])))

    # Plot train, test, and forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["ds"], y_orig[:-forecast_horizon], label="Train Data", color="blue", linewidth=2)
    plt.plot(test_df["ds"], tests, label="Test Data", color="green", linewidth=2)
    plt.plot(
        forecast_df["ds"],
        fores,
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
    plt.savefig(f"figures2/puma_{model_name}.png")

    print("control")
