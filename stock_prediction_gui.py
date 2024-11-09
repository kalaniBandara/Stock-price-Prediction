import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import yfinance as yf

# Initialize GUI window
root = tk.Tk()
root.title("Spark NZ Stock Prediction")
root.geometry("400x300")

# Function to load data
def load_data():
    global data, scaler, look_back, X, y
    try:
        data = yf.download('SPK.NZ', start='2018-01-01', end='2023-12-31')
        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        look_back = 30

        # Create sequences
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        messagebox.showinfo("Data Load", "Data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")

# Function to build and train the model
def train_model():
    global model
    try:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32)
        model.save("spark_nz_lstm_model.h5")
        messagebox.showinfo("Model Training", "Model trained and saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")

# Function to predict and plot results
def predict_and_plot():
    try:
        model = load_model("spark_nz_lstm_model.h5")
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        data['Predictions'] = np.nan
        data['Predictions'].iloc[look_back:] = predictions[:, 0]

        plt.figure(figsize=(14,7))
        plt.plot(data['Close'], label="Actual Price")
        plt.plot(data['Predictions'], label="Predicted Price")
        plt.title("Spark NZ Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price (NZD)")
        plt.legend()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {e}")

# GUI Elements
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack(pady=10)

predict_button = tk.Button(root, text="Predict and Plot", command=predict_and_plot)
predict_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=10)

# Run the GUI loop
root.mainloop()
