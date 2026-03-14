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
if DEBUG_RUN_EAGER:
    tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

import random
import os

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# ==========================================
# Configuration and TensorFlow setup
# ==========================================
DEBUG_RUN_EAGER = True
RUN_QUICK_CHECK = False
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
if DEBUG_RUN_EAGER:
    tf.config.run_functions_eagerly(True)
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
volatility = closing_prices.std(axis=0).to_numpy()
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
    print(f"Input Shape: {data_array[:-forecast_horizon].shape}")
    print(f"Target Shape: {data_array[target_offset:, :, 0].shape}")
    return dataset.prefetch(tf.data.AUTOTUNE)

in_feat = 1
hidden_feat = 32
input_sequence_length = 3
forecast_horizon = 1
batch_size = 32

train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)
#val_dataset = create_tf_dataset(val_array, input_sequence_length, forecast_horizon, batch_size)
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False)

class GraphConv1(layers.Layer):
    def __init__(self, in_feat, out_feat, graph_info, aggregation_type="mean", combination_type="concat", activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = self.add_weight(shape=(in_feat, out_feat), initializer='glorot_uniform', trainable=True)
        self.activation_fn = tf.keras.activations.get(activation)
    def aggregate(self, neighbour_representations):
        func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)
        if func is None:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")
        return func(neighbour_representations, self.graph_info.edges[0], num_segments=self.graph_info.num_nodes)
    def compute_nodes_representation(self, features):
        return tf.matmul(features, self.weight)
    def compute_aggregated_messages(self, features):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)
    def update(self, nodes_representation, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")
        if self.activation_fn is not None:
            return self.activation_fn(h)
        return h
    def call(self, features):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

class GraphAttention(tf.keras.layers.Layer):
    def __init__(self, in_feat, out_feat, graph_info, attn_heads=1, concat_heads=True, activation="elu", **kwargs):
        super().__init__(**kwargs)
        self.graph_info = graph_info
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.activation_fn = tf.keras.activations.get(activation)
        self.kernels = []
        self.attn_kernels = []
        for h in range(attn_heads):
            self.kernels.append(self.add_weight(shape=(in_feat, out_feat), initializer='glorot_uniform', name=f'W_head{h}'))
            self.attn_kernels.append(self.add_weight(shape=(2 * out_feat, 1), initializer='glorot_uniform', name=f'a_head{h}'))
    def call(self, features):
        edge_src, edge_dst = self.graph_info.edges
        outputs = []
        for head in range(self.attn_heads):
            W = self.kernels[head]
            a = self.attn_kernels[head]
            h = tf.tensordot(features, W, axes=[[2], [0]])
            h_src = tf.gather(h, edge_src)
            h_dst = tf.gather(h, edge_dst)
            concat = tf.concat([h_src, h_dst], axis=-1)
            e = tf.squeeze(tf.tensordot(concat, a, axes=[2, 0]), axis=-1)
            e = tf.nn.leaky_relu(e)
            exp_e = tf.exp(e)
            denom = tf.math.unsorted_segment_sum(exp_e, edge_dst, self.graph_info.num_nodes)
            alpha = exp_e / tf.gather(denom, edge_dst)
            weighted_src = h_src * tf.expand_dims(alpha, axis=-1)
            node_repr = tf.math.unsorted_segment_sum(weighted_src, edge_dst, self.graph_info.num_nodes)
            outputs.append(node_repr)
        if self.concat_heads:
            output = tf.concat(outputs, axis=-1)
        else:
            output = tf.add_n(outputs) / self.attn_heads
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

class GraphSAGE_GAT(tf.keras.layers.Layer):
    def __init__(self, in_feat, hidden, out_seq_len, graph_info, input_seq_len=1, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.graph_info = graph_info
        self.input_seq_len = int(input_seq_len)
        self.conv1 = GraphConv1(in_feat, hidden, graph_info, aggregation_type="mean", combination_type="add", activation="relu")
        self.conv2 = GraphConv1(hidden, hidden, graph_info, aggregation_type="mean", combination_type="add", activation=None)
        self.gat = GraphAttention(in_feat=hidden, out_feat=hidden, graph_info=graph_info, attn_heads=1, concat_heads=False, activation="elu")
        self.gate_dense = layers.Dense(hidden, activation=None)
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.readout = layers.Dense(out_seq_len)
    def call(self, inputs, training=False):
        if self.input_seq_len == 1:
            x = tf.squeeze(inputs, axis=1)
        else:
            x = inputs[:, -1, :, :]
        x = tf.transpose(x, [1, 0, 2])
        h1 = self.conv1(x)
        g = self.gat(h1)
        s = self.conv2(h1)
        alpha = tf.sigmoid(self.gate_dense(h1))
        h = alpha * g + (1.0 - alpha) * s
        h = self.dropout(h, training=training)
        h = self.norm(h)
        h = tf.nn.relu(h)
        h = tf.transpose(h, [1, 0, 2])
        out = self.readout(h)
        return tf.transpose(out, [0, 2, 1])

class GraphInfo:
    def __init__(self, edges, num_nodes, adj=None):
        self.edges = edges
        self.num_nodes = num_nodes
        self.adj = adj

def compute_adjacency_matrix(corr, epsilon=0.8):
    adjacency_matrix = np.where(corr> epsilon, 1, 0)
    np.fill_diagonal(adjacency_matrix, 1)
    return adjacency_matrix

epsilon = 0.5
adjacency_matrix = compute_adjacency_matrix(adjacency_matrix_raw, epsilon=epsilon)
node_idx, nbr_idx = np.where(adjacency_matrix == 1)
graph = GraphInfo(edges=(node_idx.tolist(), nbr_idx.tolist()), num_nodes=adjacency_matrix.shape[0], adj=adjacency_matrix)
print(f"Nodes: {graph.num_nodes}, edges: {len(node_idx)}")

st_gcn = GraphSAGE_GAT(
    in_feat=in_feat,
    hidden=hidden_feat,
    out_seq_len=forecast_horizon,
    graph_info=graph,
    input_seq_len=input_sequence_length,
    dropout_rate=0.1
)

inputs = layers.Input((int(input_sequence_length), graph.num_nodes, int(in_feat)))
outputs = st_gcn(inputs)
model = keras.models.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
history = model.fit(train_dataset,  epochs=100, callbacks=[early_stop])
print("\nFinal Training History:")
print(f"Final Training Loss: {history.history['loss'][-1]}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]}")