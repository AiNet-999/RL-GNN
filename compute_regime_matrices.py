!pip install hmmlearn

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

df = pd.read_csv("/content/SP500_Closing_Prices-NEW - Copy.csv", header=None)
print("Original Shape:", df.shape)

df = df.ffill().bfill().fillna(df.mean())
print("Missing values after cleaning:", df.isnull().sum().sum())

num_rows = int(0.70 * len(df))
df_70 = df.iloc[:num_rows]
print("70% Data Shape:", df_70.shape)

returns = np.log(df_70 / df_70.shift(1)).dropna()
print("Return Matrix Shape:", returns.shape)

market_return = returns.mean(axis=1)
market_volatility = returns.std(axis=1)
features = np.column_stack([market_return, market_volatility])

hmm = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=1000,
    random_state=42
)
hmm.fit(features)
hidden_states = hmm.predict(features)

returns = returns.copy()
returns["Regime"] = hidden_states

print("\nRegime Distribution:")
print(pd.Series(hidden_states).value_counts())

state_means = {i: market_return[hidden_states == i].mean() for i in range(3)}
sorted_states = sorted(state_means, key=state_means.get)

bear_state = sorted_states[0]
sideways_state = sorted_states[1]
bull_state = sorted_states[2]

print("\nRegime Mapping:")
print("Bear State:", bear_state)
print("Sideways State:", sideways_state)
print("Bull State:", bull_state)

R_bull = returns[returns["Regime"] == bull_state].drop(columns=["Regime"])
R_bear = returns[returns["Regime"] == bear_state].drop(columns=["Regime"])
R_sideways = returns[returns["Regime"] == sideways_state].drop(columns=["Regime"])

print("\nRegime Shapes:")
print("Bull Shape:", R_bull.shape)
print("Bear Shape:", R_bear.shape)
print("Sideways Shape:", R_sideways.shape)

C_bull = R_bull.corr()
C_bear = R_bear.corr()
C_sideways = R_sideways.corr()

threshold = 0.5

def count_edges(matrix, threshold):
    A = matrix.abs().values
    edges = np.sum(A > threshold) - matrix.shape[0]
    edges = edges // 2
    return edges

bull_edges = count_edges(C_bull, threshold)
bear_edges = count_edges(C_bear, threshold)
side_edges = count_edges(C_sideways, threshold)

print("\n===== Threshold Statistics (τ = 0.5) =====")
print("Bull edges >", threshold, ":", bull_edges)
print("Bear edges >", threshold, ":", bear_edges)
print("Sideways edges >", threshold, ":", side_edges)

C_bull.to_csv("correlation_matrix_bull.csv", index=False)
C_bear.to_csv("correlation_matrix_bear.csv", index=False)
C_sideways.to_csv("correlation_matrix_sideways.csv", index=False)

print("\nSaved Files:")
print("correlation_matrix_bull.csv")
print("correlation_matrix_bear.csv")
print("correlation_matrix_sideways.csv")