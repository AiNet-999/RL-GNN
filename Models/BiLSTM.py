from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

df = read_csv('SP500_Closing_Prices.csv',
              header=0, parse_dates=['Date'], index_col='Date')
df.index = df.index.to_period('D')


input_sequence = 3   # past 3 days as input like MACRO-DRIVEN GNN-LSTM 
forecast_horizon = 1 # 1-day ahead prediction
test_ratio = 0.10
top_n = 5
epochs = 20
batch_size = 16


volatility = df.std().sort_values(ascending=False)
top_stocks = volatility.head(top_n).index.tolist()
print(f"Top {top_n} most volatile stocks: {top_stocks}")

all_predictions = {}
all_tests = {}
rmse_list, mae_list = [], []


def create_supervised_data(series, n_input=3, n_output=1):
    X, y = [], []
    for i in range(len(series) - n_input - n_output + 1):
        X.append(series[i:i+n_input])
        y.append(series[i+n_input:i+n_input+n_output])
    return np.array(X), np.array(y).ravel()


for stock in top_stocks:
    print(f"\nTraining LSTM for stock: {stock}")
    series = df[stock].dropna().values.reshape(-1, 1)
    if len(series) < 10:
        print(f"Skipping {stock} (too few data points)")
        continue
    size = int(len(series) * (1 - test_ratio))
    train, test = series[:size], series[size - input_sequence:]  


    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train).flatten()
    test_scaled = scaler.transform(test).flatten()

    X_train, y_train = create_supervised_data(train_scaled, input_sequence, forecast_horizon)
    X_test, y_test = create_supervised_data(test_scaled, input_sequence, forecast_horizon)

  
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping {stock} (not enough samples after sequence creation)")
        continue


    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
   
    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,        
    restore_best_weights=True  
)

 
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(input_sequence, 1)),   # similar to hidden units which we used in our proposed model to demonstrate the effect
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

 
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,validation_split=0.1,
              callbacks=[early_stop])


    y_pred_scaled = model.predict(X_test, verbose=0).flatten()


    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    rmse_list.append(rmse)
    mae_list.append(mae)

    print(f"RMSE (orig scale): {rmse:.4f}, MAE (orig scale): {mae:.4f}")

    test_index = df.index[-len(y_true):].to_timestamp()
    all_tests[stock] = y_true
    all_predictions[stock] = (test_index, y_pred)

avg_rmse = np.mean(rmse_list)
avg_mae = np.mean(mae_list)

print("\n=== Overall Average Metrics (Bi-LSTM + MinMaxScaler, 1-step Ahead) ===")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")

for stock in top_stocks:
    plt.figure(figsize=(10, 5))
    actual = all_tests[stock]
    dates, preds = all_predictions[stock]
    plt.plot(dates, actual, label='Actual', color='black')
    plt.plot(dates, preds, label='Predicted (BiLSTM, H=1)', linestyle='--', color='blue')
    plt.title(f"{stock} - BiLSTM Forecast (1-Step Ahead, Original Scale)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
