# RegimeGNN for Stock Price Forecasting using Market Regimes

This repository contains an implementation of our **RL-GNN** for time-series forecasting of **S&P 500 stock prices**, **FTSE-100**, **SZSE-500 stock prices**.
The model leverages **regime-specific correlation matrices** to capture structural co-movements between stocks, while preserving temporal dynamics through LSTM layers

---

## Overview

The proposed architecture combines:

- **HMM-Based Market Regime Detection** :Detects market regimes (Bull, Bear, Sideways) using historical returns and volatility via Gaussian HMM..
- **LSTM (Long Short-Term Memory):** Captures temporal dependencies in stock price time series.
- **Graph Convolutional Networks (GCN):** Models structural relationships between stocks using regime-specific correlation matrices.
- **Learnable graph adjacency: **Learnable matrices enabling regime-aware forecasting with dynamic structure learning

Unlike traditional correlation-based graphs, the workflow first identifies market regimes, constructs **regime-specific graphs**, and then applies learnable graph convolutions, enabling regime-aware forecasting with dynamic structure learning
---

## Data

### Market Data Sources

- **Historical stock price data**  
  Fetched from Yahoo Finance  
  https://finance.yahoo.com/
---

## Regime-Specific Correlation Matrices
Generated from the HMM-based regime detection pipeline.

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- NetworkX

Install dependencies:

pip install -r requirements.txt

---
## Citations
Please cite the following paper if you use this work or code:
- **Regime-Aware Static and Dynamic Graph Structure Learning for Stock Price Forecasting**, 2026.
© 2026 — All rights reserved.
