import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# -----------------------------
# 1. Generate Synthetic Past Weather Data (last 30 days)
# -----------------------------
np.random.seed(42)
today = datetime.today()
dates = pd.date_range(end=today, periods=30, freq="D")

# Simulate seasonal temperature pattern with noise
temperature = 25 + np.sin(np.linspace(0, 6, 30)) * 8 + np.random.normal(0, 2, 30)
df = pd.DataFrame({"Date": dates, "Temperature": temperature})
df.set_index("Date", inplace=True)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("🌤 Weather Forecasting (ARIMA)")
st.write(f"Today is **{today.strftime('%d %B %Y')}**")
st.write("This app uses the past 30 days of synthetic weather data to forecast the next 7 days.")

# Show dataset
st.subheader("Past 30 Days Weather Data")
st.line_chart(df["Temperature"])

# -----------------------------
# 3. Train-Test Split
# -----------------------------
train = df.iloc[:-7]
test = df.iloc[-7:]

# -----------------------------
# 4. Auto ARIMA Model
# -----------------------------
model = auto_arima(train["Temperature"], seasonal=False, trace=False)
forecast = model.predict(n_periods=7)

# -----------------------------
# 5. Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(test["Temperature"], forecast))
st.subheader("Model Evaluation")
st.write(f"**RMSE on last 7 days:** {rmse:.2f}")

# -----------------------------
# 6. Visualization
# -----------------------------
st.subheader("Forecast vs Actual (Last 7 Days)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(test.index, test["Temperature"], label="Actual")
ax.plot(test.index, forecast, label="Forecast", linestyle="--")
ax.legend()
st.pyplot(fig)

# -----------------------------
# 7. Future Forecast (Next 7 Days)
# -----------------------------
future_forecast = model.predict(n_periods=7)
future_dates = pd.date_range(start=today + timedelta(days=1), periods=7, freq="D")
future_df = pd.DataFrame({"Date": future_dates, "Forecasted_Temperature": future_forecast})
future_df.set_index("Date", inplace=True)

st.subheader("Future Forecast (Next 7 Days)")
st.line_chart(future_df["Forecasted_Temperature"])
