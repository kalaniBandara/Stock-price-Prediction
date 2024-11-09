import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st

# Streamlit App
st.title("Spark NZ Stock Price Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing date and close price columns", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    st.write("## Uploaded Data Preview:")
    st.write(data.head())

    # Check if the necessary column is present
    if "Close" not in data.columns:
        st.error("The dataset must have a 'Close' column for the closing prices.")
    else:
        # Handle missing values
        data = data.dropna(subset=["Close"])

        # Data Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

        # Create training data structure
        prediction_days = 60
        x_train = []
        y_train = []

        for x in range(prediction_days, len(scaled_data) - 90):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, epochs=20, batch_size=32)

        # Prepare test data for predictions
        test_data = scaled_data[-(prediction_days + 90):]
        x_test = []
        for x in range(prediction_days, len(test_data)):
            x_test.append(test_data[x - prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Plot results
        actual_prices = data["Close"][-90:].values
        prediction_dates = pd.date_range(start=data.index[-90], periods=90, freq='B')

        plt.figure(figsize=(14, 7))
        plt.plot(data.index[-180:], data["Close"].iloc[-180:], color='blue', label='Train')
        plt.plot(prediction_dates, actual_prices, color='orange', label='Actual Price')
        plt.plot(prediction_dates, predictions, color='green', label='Predicted Price')
        plt.title('Spark NZ Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price (NZD)')
        plt.legend()
        plt.grid()

        st.pyplot(plt.gcf())
        plt.clf()  # Clear figure for next plot to avoid overlap

        # Evaluation of the model
        mse = mean_squared_error(actual_prices, predictions)
        rmse = np.sqrt(mse)
        st.write("## Model Evaluation:")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        # Explanation
        st.write("## Explanation of Results:")
        st.write("The chart above shows the historical closing prices (blue) for training, "
                 "the actual closing prices (orange) over the past 3 months, and the predicted closing prices (green) "
                 "for the same period. The LSTM model was trained using past price data and was able to capture trends "
                 "and fluctuations in the stock price to predict the closing price over the 3-month period.")
