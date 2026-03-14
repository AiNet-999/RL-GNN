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

class GraphConv1(layers.Layer):
    def __init__(self, in_feat, out_feat, graph_info, aggregation_type="mean", combination_type="concat", activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.activation = layers.Activation(activation)
        src_idx = tf.convert_to_tensor(graph_info.edges[0], dtype=tf.int32)
        tgt_idx = tf.convert_to_tensor(graph_info.edges[1], dtype=tf.int32)
        self.edge_source = src_idx
        self.edge_target = tgt_idx
        self.num_nodes = int(graph_info.num_nodes)
        w_init = keras.initializers.glorot_uniform()
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feat, out_feat), dtype="float32"), trainable=True)

    def aggregate(self, neighbour_representations):
        if self.aggregation_type == "sum":
            agg = tf.math.unsorted_segment_sum(neighbour_representations, self.edge_source, num_segments=self.num_nodes)
        elif self.aggregation_type == "mean":
            agg = tf.math.unsorted_segment_mean(neighbour_representations, self.edge_source, num_segments=self.num_nodes)
        elif self.aggregation_type == "max":
            agg = tf.math.unsorted_segment_max(neighbour_representations, self.edge_source, num_segments=self.num_nodes)
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")
        return agg

    def compute_nodes_representation(self, features):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features):
        neighbour_representations = tf.gather(features, self.edge_target)
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")
        return self.activation(h)

    def call(self, features):
        nodes_repr = self.compute_nodes_representation(features)
        agg_msgs = self.compute_aggregated_messages(features)
        out = self.update(nodes_repr, agg_msgs)
        return out

class GraphInfo:
    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes

def compute_adjacency_matrix(route_distances, epsilon):
    adjacency_matrix = np.where(route_distances > epsilon, 1, 0)
    return adjacency_matrix

adjacency_matrix_raw = compute_adjacency_matrix(adjacency_matrix_raw, epsilon=0.5)
node_indices, neighbor_indices = np.where(adjacency_matrix_raw == 1)
graph1 = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix_raw.shape[0])
print(f"Total number of edges: {len(node_indices)}")

def build_tgcn_model(in_feat, gcn_hidden, forecast_horizon, graph_info, input_seq_len):
    num_nodes = int(graph_info.num_nodes)
    inputs = layers.Input(shape=(input_seq_len, num_nodes, in_feat))

    def graph_conv_step(x, gcn_layer):
        x_t = tf.transpose(x, [1, 0, 2])
        y_t = gcn_layer(x_t)
        y = tf.transpose(y_t, [1, 0, 2])
        return y

    gcn = GraphConv1(
        in_feat,
        gcn_hidden,
        graph_info,
        aggregation_type="mean",
        combination_type="concat",
        activation="relu"
    )

    gcn_outputs = []
    for t in range(input_seq_len):
        xt = layers.Lambda(lambda z: z[:, t])(inputs)
        gt = layers.Lambda(lambda z: graph_conv_step(z, gcn))(xt)
        gcn_outputs.append(gt)

    gcn_sequence = layers.Lambda(lambda z: tf.stack(z, axis=1))(gcn_outputs)
    reshaped = layers.Reshape((input_seq_len, num_nodes * gcn_hidden * 2))(gcn_sequence)
    gru_out = layers.GRU(64, return_sequences=False)(reshaped)
    dense = layers.Dense(num_nodes * forecast_horizon)(gru_out)
    outputs = layers.Reshape((forecast_horizon, num_nodes))(dense)
    model = keras.Model(inputs, outputs, name="TGCN_Model")
    return model

in_feat = 1
gcn_hidden = 64
input_sequence_length = 3
forecast_horizon = 1
batch_size = 32

train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False)

model = build_tgcn_model(
    in_feat,
    gcn_hidden,
    forecast_horizon,
    graph1,
    input_sequence_length
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_dataset, epochs=100)

print("\nFinal Training History:")
print(f"Final Training Loss: {history.history['loss'][-1]:.6f}")