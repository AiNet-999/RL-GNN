from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


df = read_csv('SP500_Closing_Prices-NEW.csv',
              header=0, parse_dates=['Date'], index_col='Date')
df.index = df.index.to_period('D')
df = df.ffill().bfill()

# === Parameters ===
input_sequence = 3   # past 3 days as input
forecast_horizon = 1 # 1-day ahead prediction
test_ratio = 0.10
top_n = 5

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
    print(f"\nTraining XGBoost Regressor for stock: {stock}")
    series = df[stock].dropna().values.reshape(-1, 1)
    if len(series) < 10:
        print(f"Skipping {stock} (too few data points)")
        continue

    # === Split train/test FIRST ===
    size = int(len(series) * (1 - test_ratio))
    train, test = series[:size], series[size - input_sequence:] 


    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train).flatten()
    test_scaled = scaler.transform(test).flatten()


    X_train, y_train = create_supervised_data(train_scaled, input_sequence, forecast_horizon)
    X_test, y_test = create_supervised_data(test_scaled, input_sequence, forecast_horizon)


    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)


    y_pred_scaled = model.predict(X_test)


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

print("\n=== Overall Average Metrics (XGBoost + MinMaxScaler, 1-step Ahead) ===")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MAE: {avg_mae:.4f}")


for stock in top_stocks:
    plt.figure(figsize=(10, 5))
    actual = all_tests[stock]
    dates, preds = all_predictions[stock]
    plt.plot(dates, actual, label='Actual', color='black')
    plt.plot(dates, preds, label='Predicted (XGBoost, H=1)', linestyle='--', color='blue')
    plt.title(f"{stock} - XGBoost Forecast (1-Step Ahead, Original Scale)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
