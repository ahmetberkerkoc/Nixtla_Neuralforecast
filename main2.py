import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.numpy import mse, mae

# Load data
df = pd.read_csv("data/turkey_gas.csv")

# Rename columns
df.rename(columns={'Unnamed: 0': 'ds', 'ABONE': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])  # Convert to datetime
df['unique_id'] = 'turkey_gas'  # Single time-series
df['y'] /= 10**7

# Sort by date
df = df.sort_values(by='ds')

# Split into train (80%) and test (20%)
train_size = len(df) - int(len(df) * 0.2)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
val_size = int(len(df) * 0.1)
# Forecast horizon equals the length of the test set
forecast_horizon = len(test_df)

# Define and train the DeepAR model
models = [DeepAR(h=forecast_horizon,
                    input_size=int(1.5)*forecast_horizon,
                    max_steps=3000,
                    #scaler_type="standard",
                    learning_rate=0.00003,
                    # futr_exog_list=["ABONE_SICAKLIK_HI", "ABONE_SICAKLIK_MIN", "ABONE_RUZGAR_HI", "ABONE_RUZGAR_AVG"]
                    futr_exog_list= list(train_df.columns[3:-1]),
                    batch_size=1
                    # val_size = val_size
                )]
nf = NeuralForecast(models=models, freq='D')
nf.fit(train_df)

# Make predictions for the test period
forecast_df = nf.predict(futr_df=test_df)

fores = forecast_df['DeepAR']*(10**7)
fores = fores.to_numpy()

tests = test_df['y']*(10**7)
tests = tests.to_numpy()

print("MSE: {:e}".format(mse(tests, fores)))
print("MAE: {:e}".format(mae(tests, fores)))


# Plot train, test, and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train_df['ds'], train_df['y']*(10**7), label="Train Data", color='blue', linewidth=2)
plt.plot(test_df['ds'], test_df['y']*(10**7), label="Test Data", color='green', linewidth=2)
plt.plot(forecast_df['ds'], forecast_df['DeepAR']*(10**7), label="Forecast", color='red', linestyle='dashed', linewidth=2)
plt.axvline(test_df['ds'].iloc[0], color='gray', linestyle='--', label='Test Start')
plt.xlabel("Date")
plt.ylabel("Gas Consumption (ABONE)")
plt.title("DeepAR Forecast on Test Set")
plt.legend()
plt.savefig("figures/turkey_gas.png")

print("control")