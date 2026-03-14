from contextlib import closing
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

DEBUG_RUN_EAGER = True
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
if DEBUG_RUN_EAGER:
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

gpus = tf.config.experimental.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

CLOSING_PRICES_PATH = 'SZSE_Closing_Prices - Copy.csv'
CORRELATION_MATRIX_PATH = 'adjacency_matrix.csv'

closing_prices = pd.read_csv(CLOSING_PRICES_PATH, header=None).fillna(method='ffill').fillna(method='bfill')
adjacency_matrix_raw = pd.read_csv(CORRELATION_MATRIX_PATH, header=None).to_numpy()

num_stocks = 50
closing_prices = closing_prices.iloc[:, :num_stocks].to_numpy()
adjacency_matrix_raw = adjacency_matrix_raw[:num_stocks, :num_stocks]
print("Raw Closing Prices Shape:", closing_prices.shape)

def split_data(data, train_size=0.8):
    num_time_steps = data.shape[0]
    num_train = int(num_time_steps * train_size)
    train_data = data[:num_train]
    test_data = data[num_train:]
    return train_data, test_data

train_raw, test_raw = split_data(closing_prices, 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)
train_scaled = scaler.transform(train_raw)
test_scaled = scaler.transform(test_raw)
train_array = np.expand_dims(train_scaled, axis=-1)
test_array = np.expand_dims(test_scaled, axis=-1)

print(f"Train set size: {train_array.shape}")
print(f"Test set size: {test_array.shape}")

def create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=True, multi_horizon=False):
    inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[:-forecast_horizon], None, sequence_length=input_sequence_length, shuffle=False, batch_size=batch_size
    )
    target_offset = input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[target_offset:, :, 0], None, sequence_length=target_seq_length, shuffle=False, batch_size=batch_size
    )
    dataset = tf.data.Dataset.zip((inputs, targets))
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(100)
    return dataset.prefetch(tf.data.AUTOTUNE)

in_feat = 1
input_sequence_length = 3
forecast_horizon = 1
batch_size = 32

train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False)

def build_transformer_model(num_nodes, input_seq_len, forecast_horizon, d_model=64, num_heads=4, ff_dim=128):
    inputs = layers.Input(shape=(input_seq_len, num_nodes, 1))
    x = layers.Reshape((input_seq_len, num_nodes))(inputs)
    x = layers.Dense(d_model)(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_nodes * forecast_horizon)(x)
    outputs = layers.Reshape((forecast_horizon, num_nodes))(x)
    model = keras.Model(inputs, outputs, name="Transformer_Model")
    return model

model = build_transformer_model(
    num_nodes=num_stocks,
    input_seq_len=input_sequence_length,
    forecast_horizon=forecast_horizon
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = model.fit(train_dataset,  epochs=100,callbacks=[early_stop])

print("\nFinal Training History:")
print(f"Final Training Loss: {history.history['loss'][-1]:.6f}")