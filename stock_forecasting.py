# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Step 1: Project Setup
# Objective: Forecast future stock prices for Apple Inc. (AAPL)

# Step 2: Data Collection
# Obtain historical stock price data from Yahoo Finance
stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')

# Display the first few rows of the dataset
print(stock_data.head())

# Step 3: Data Preprocessing
# Handling Missing Values (if any)
stock_data = stock_data.ffill()  # Forward fill to handle any missing data

# Focus on the 'Close' price for forecasting
# Data Visualization: Plot the historical stock prices
stock_data['Close'].plot(figsize=(12, 6))
plt.title('Historical Stock Prices for Apple Inc.')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Step 4: Stationarity Check
# Use the Augmented Dickey-Fuller (ADF) test to check stationarity
result = adfuller(stock_data['Close'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If the series is non-stationary, apply differencing
if result[1] > 0.05:
    stock_data['Close_diff'] = stock_data['Close'].diff().dropna()

    # Recheck stationarity after differencing
    result_diff = adfuller(stock_data['Close_diff'].dropna())
    print(f'ADF Statistic (Differenced): {result_diff[0]}')
    print(f'p-value (Differenced): {result_diff[1]}')
else:
    stock_data['Close_diff'] = stock_data['Close']

# Step 5: Modeling
# Use ARIMA model for forecasting
# Hyperparameter tuning is skipped for simplicity (order=(5, 1, 0) is used)
model = ARIMA(stock_data['Close_diff'], order=(5, 1, 0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Step 6: Forecasting
# Define periods for 1 week, 1 month, and 1 year
week_days = 7
month_days = 30
year_days = 252  # Assuming 252 trading days in a year

# Forecast for 1 week, 1 month, and 1 year
forecast_week = model_fit.forecast(steps=week_days)
forecast_month = model_fit.forecast(steps=month_days)
forecast_year = model_fit.forecast(steps=year_days)

# Print the predicted values
print(f"Predicted stock price 1 week from now: {forecast_week.iloc[-1]:.2f}")
print(f"Predicted stock price 1 month from now: {forecast_month.iloc[-1]:.2f}")
print(f"Predicted stock price 1 year from now: {forecast_year.iloc[-1]:.2f}")

# Step 7: Visualization of Forecasts
# Create a new dataframe for visualization
forecast_dates_week = pd.date_range(start=stock_data.index[-1], periods=week_days + 1, closed='right')
forecast_dates_month = pd.date_range(start=stock_data.index[-1], periods=month_days + 1, closed='right')
forecast_dates_year = pd.date_range(start=stock_data.index[-1], periods=year_days + 1, closed='right')

forecast_df_week = pd.DataFrame(forecast_week, index=forecast_dates_week, columns=['Forecast'])
forecast_df_month = pd.DataFrame(forecast_month, index=forecast_dates_month, columns=['Forecast'])
forecast_df_year = pd.DataFrame(forecast_year, index=forecast_dates_year, columns=['Forecast'])

# Plot the forecasts
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Historical')
plt.plot(forecast_df_week, label='1 Week Forecast', color='orange')
plt.plot(forecast_df_month, label='1 Month Forecast', color='green')
plt.plot(forecast_df_year, label='1 Year Forecast', color='red')
plt.title('Apple Inc. Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Step 8: Evaluation
# Evaluate the model using RMSE (for the last 30 days of available data)
actual = stock_data['Close'].iloc[-30:]
predicted = forecast_year[:30]

mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Optional: Save the model to disk
with open('arima_model.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)

# Optional: Deployment
# Consider deploying this model using a dashboard tool like Streamlit or Dash