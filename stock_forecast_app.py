import yfinance as yf
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Function to check stationarity
def check_stationarity(data):
    data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    result = adfuller(data)
    return result[1] <= 0.05  # If p-value is less than 0.05, data is stationary

# Function to forecast stock prices using the SARIMA model
def forecast_stock(stock_symbol, forecast_steps):
    stock_data = yf.download(stock_symbol)
    stock_data = stock_data.ffill()
    stock_data = stock_data.asfreq('B')  # 'B' for business day frequency

    progress_bar = st.progress(0)  # Initialize progress bar
    total_steps = 3  # Total steps in the process

    if not check_stationarity(stock_data['Close']):
        stock_data['Close_diff'] = stock_data['Close'].diff().dropna()
    else:
        stock_data['Close_diff'] = stock_data['Close']
    
    progress_bar.progress(1 / total_steps)  # Update progress

    # Use the SARIMA model with predefined orders
    model = SARIMAX(stock_data['Close_diff'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    
    progress_bar.progress(2 / total_steps)  # Update progress

    st.write("Forecasting...")

    # Forecast for the custom time period
    forecast = model_fit.forecast(steps=forecast_steps)
    progress_bar.progress(3 / total_steps)  # Update progress

    last_close_price = stock_data['Close'].iloc[-1]

    # Convert differenced forecast back to actual price levels
    forecast_prediction = last_close_price + forecast.cumsum().iloc[-1]

    return forecast_prediction

# Function to convert the user's time input to the equivalent number of business days
def convert_time_to_days(time_value, time_unit):
    if time_unit == "Days":
        return time_value
    elif time_unit == "Weeks":
        return time_value * 5  # Approximate number of business days in a week
    elif time_unit == "Months":
        return time_value * 21  # Approximate number of business days in a month
    elif time_unit == "Years":
        return time_value * 252  # Approximate number of business days in a year

# Streamlit UI
st.title('Stock Price Prediction App')

# User inputs
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple):", "AAPL")
time_value = st.number_input("Enter the time period (whole number):", min_value=1, value=1)
time_unit = st.selectbox("Select time unit:", ["Days", "Weeks", "Months", "Years"])

# When the user clicks the button, perform the forecast
if st.button('Predict'):
    forecast_steps = convert_time_to_days(time_value, time_unit)
    prediction = forecast_stock(stock_symbol, forecast_steps)

    st.subheader(f"Prediction for {stock_symbol.upper()} {time_value} {time_unit} from now:")
    st.write(f"Predicted stock price: ${prediction:.2f}")

    st.write("Note: These predictions are based on historical trends and are not financial advice.")
