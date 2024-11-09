import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load and Visualize Data
data = yf.download('SPK.NZ', start='2018-01-01', end='2023-12-31')
data = data[['Close']]

# Plot the closing prices
plt.figure(figsize=(12,6))
plt.plot(data, label='Close Price History')
plt.title('Spark NZ Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price (NZD)')
plt.legend()
plt.show()

# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Define the look-back period (number of past days used to predict the future)
look_back = 30

# Create sequences and labels
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape data for LSTM input [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Build and Compile LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Make Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
mae = mean_absolute_error(y_test_unscaled, predictions)
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Step 7: Plot the Results
train = data[:train_size + look_back]
valid = data[train_size + look_back:]
valid['Predictions'] = predictions

plt.figure(figsize=(14,7))
plt.plot(train['Close'], label='Training Data')
plt.plot(valid[['Close', 'Predictions']])
plt.title('Spark NZ Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (NZD)')
plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')
plt.show()

# from sklearn.metrics import r2_score

# # Calculate additional metrics
# rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
# mae = mean_absolute_error(y_test_unscaled, predictions)
# r2 = r2_score(y_test_unscaled, predictions)
# mape = np.mean(np.abs((y_test_unscaled - predictions) / y_test_unscaled)) * 100

# # Print metrics
# print(f'Root Mean Squared Error (RMSE): {rmse}')
# print(f'Mean Absolute Error (MAE): {mae}')
# print(f'R-squared (R^2) Score: {r2}')
# print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# # Plot the results with metrics
# plt.figure(figsize=(14, 7))
# plt.plot(train['Close'], label='Training Data')
# plt.plot(valid[['Close', 'Predictions']])
# plt.title('Spark NZ Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Close Price (NZD)')
# plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')

# # Display metrics on the plot
# plt.text(valid.index[0], max(valid['Close']) * 1.05,
#          f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}\nMAPE: {mape:.2f}%',
#          bbox=dict(facecolor='white', alpha=0.5),
#          fontsize=10, color='black')

# plt.show()


# from sklearn.metrics import r2_score

# # Calculate RMSE (already computed)
# rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
# print(f'Root Mean Squared Error (RMSE): {rmse}')

# # Mean Absolute Error (MAE) (already computed)
# mae = mean_absolute_error(y_test_unscaled, predictions)
# print(f'Mean Absolute Error (MAE): {mae}')

# # R-squared (R^2) score
# r2 = r2_score(y_test_unscaled, predictions)
# print(f'R-squared (R^2) Score: {r2}')

# # Mean Absolute Percentage Error (MAPE)
# mape = np.mean(np.abs((y_test_unscaled - predictions) / y_test_unscaled)) * 100
# print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

from sklearn.metrics import r2_score

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
mae = mean_absolute_error(y_test_unscaled, predictions)
r2 = r2_score(y_test_unscaled, predictions)
mape = np.mean(np.abs((y_test_unscaled - predictions) / y_test_unscaled)) * 100

# Plot the stock price and predictions
plt.figure(figsize=(14, 7))
plt.plot(train['Close'], label='Training Data')
plt.plot(valid[['Close', 'Predictions']])
plt.title('Spark NZ Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (NZD)')
plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')
plt.show()

# Display the metrics below the plot
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R^2) Score: {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
