"""
AEP Energy Demand Forecasting Notebook (Python Script)
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading data...")
df = pd.read_csv('/home/workdir/daily_AEP.csv', parse_dates=['Datetime'], index_col='Datetime')
print(df.head())

# Plot historical
plt.figure(figsize=(12, 6))
plt.plot(df['AEP_MW'])
plt.title('AEP Daily Energy Demand (2004-2018)')
plt.ylabel('MW')
plt.savefig('/home/workdir/historical_plot.png')
print("Historical plot saved.")

# Train/test split
train = df.iloc[:-365]
test = df.iloc[-365:]

# Fit ETS model
print("Fitting ETS model...")
model = ExponentialSmoothing(train['AEP_MW'], trend='add', seasonal='add', seasonal_periods=365)
fit = model.fit()

# Forecast
forecast = fit.forecast(steps=365)

# Evaluation
mae = np.mean(np.abs(test['AEP_MW'] - forecast))
mape = np.mean(np.abs((test['AEP_MW'] - forecast) / test['AEP_MW'])) * 100
print(f"MAE: {mae:.2f} MW")
print(f"MAPE: {mape:.2f}%")

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index[-365:], train['AEP_MW'][-365:], label='Train')
plt.plot(test.index, test['AEP_MW'], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.title('AEP Daily Demand Forecast')
plt.savefig('/home/workdir/forecast_plot.png')
print("Forecast plot saved.")

# Save future forecast
future = fit.forecast(365)
future.to_csv('/home/workdir/future_forecast.csv')
print("Future forecast saved to future_forecast.csv")

print("\nRun this script to reproduce the full analysis.")
