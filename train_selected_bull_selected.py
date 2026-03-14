from contextlib import closing
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import random
import os

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

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


closing_prices = (
    pd.read_csv('SP500_Closing_Prices-NEW - Copy.csv', header=None)
    .ffill()
    .bfill()
)

# KEEP ONLY BULL + STANDARD MATRICES
adjacency_matrix_bull = pd.read_csv('correlation_matrix_bull.csv', header=None).to_numpy()
adjacency_matrix_std = pd.read_csv('adjacency_matrix.csv', header=None).to_numpy()

num_stocks = 50
volatile_idx = [62, 45, 13, 53, 73]

selected_idx = []
i = 0
while len(selected_idx) < num_stocks - len(volatile_idx):
    if i not in volatile_idx:
        selected_idx.append(i)
    i += 1

selected_idx += volatile_idx

print("Selected indices:", selected_idx)
print("Total stocks:", len(selected_idx))

closing_prices = closing_prices.iloc[:, selected_idx]

adjacency_matrix_bull = adjacency_matrix_bull[selected_idx][:, selected_idx]
adjacency_matrix_std = adjacency_matrix_std[selected_idx][:, selected_idx]

closing_prices = closing_prices.to_numpy()

num_time_steps = closing_prices.shape[0]
train_size = int(0.8 * num_time_steps)

train_raw = closing_prices[:train_size]
test_raw = closing_prices[train_size:]

print(f"Train raw shape: {train_raw.shape}, Test raw shape: {test_raw.shape}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)

train_scaled = scaler.transform(train_raw)
test_scaled = scaler.transform(test_raw)

train_array = np.expand_dims(train_scaled, axis=-1)
test_array = np.expand_dims(test_scaled, axis=-1)

print(f"Train array shape: {train_array.shape}")
print(f"Test array shape: {test_array.shape}")


def create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=True, multi_horizon=False):

    inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size
    )

    target_offset = input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    target_seq_length = forecast_horizon if multi_horizon else 1

    targets = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[target_offset:, :, 0],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size
    )

    dataset = tf.data.Dataset.zip((inputs, targets))

    if shuffle:
        dataset = dataset.shuffle(100)

    print(f"Input Shape: {data_array[:-forecast_horizon].shape}")
    print(f"Target Shape: {data_array[target_offset:, :, 0].shape}")

    return dataset.prefetch(16).cache()


in_feat = 1
out_feat = 10
lstm_units = 64
input_sequence_length = 3
forecast_horizon = 1
batch_size = 64

train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)

test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False
)


class GraphConv(layers.Layer):

    def __init__(self, in_feat, out_feat, graph_info, sta, weig,
                 static_adjacency_matrix,
                 aggregation_type="mean",
                 combination_type="concat",
                 activation=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.static_adjacency_matrix = static_adjacency_matrix
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.sta = sta
        self.weig = weig

        self.adjacency_matrix = self.add_weight(
            shape=(graph_info.num_nodes, graph_info.num_nodes),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="learnable_adjacency_matrix"
        )

        self.sigmoid = layers.Activation('sigmoid')

        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"),
            trainable=True
        )

        self.activation = layers.Activation(activation) if activation else None

    def aggregate(self, neighbour_representations):

        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        return aggregation_func(
            neighbour_representations,
            self.graph_info.edges[0],
            num_segments=self.graph_info.num_nodes,
        )

    def compute_nodes_representation(self, features):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features):

        adjacency_matrix = self.sigmoid(self.adjacency_matrix)

        combined_adjacency_matrix = self.sta * self.static_adjacency_matrix + self.weig * adjacency_matrix

        neighbour_representations = tf.gather(features, self.graph_info.edges[1])

        aggregated_messages = self.aggregate(neighbour_representations)

        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation, aggregated_messages):

        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)

        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages

        return self.activation(h) if self.activation else h

    def call(self, features):

        nodes_representation = self.compute_nodes_representation(features)

        aggregated_messages = self.compute_aggregated_messages(features)

        return self.update(nodes_representation, aggregated_messages)


class LSTMGC(layers.Layer):

    def __init__(self, in_feat, out_feat, lstm_units,
                 input_seq_len, output_seq_len,
                 graph_info3, graph_info4,
                 static_adjacency_matrix,
                 graph_conv_params=None, **kwargs):

        super().__init__(**kwargs)

        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None
            }

        self.graph_conv3 = GraphConv(in_feat, out_feat, graph_info3, 1, 0,
                                     static_adjacency_matrix, **graph_conv_params)

        self.graph_conv4 = GraphConv(in_feat, out_feat, graph_info4, 0, 0.5,
                                     static_adjacency_matrix, **graph_conv_params)

        self.lstm3 = layers.LSTM(lstm_units, return_sequences=False)
        self.dense3 = layers.Dense(output_seq_len)

        self.lstm4 = layers.LSTM(lstm_units, return_sequences=False)
        self.dense4 = layers.Dense(output_seq_len)

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def call(self, inputs):

        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out3 = self.graph_conv3(inputs)
        shape = tf.shape(gcn_out3)
        num_nodes, batch_size, input_seq_len, out_feat = shape[0], shape[1], shape[2], shape[3]

        gcn_out3 = tf.reshape(gcn_out3, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out3 = self.lstm3(gcn_out3)
        dense_output3 = self.dense3(lstm_out3)

        gcn_out4 = self.graph_conv4(inputs)
        shape = tf.shape(gcn_out4)
        num_nodes, batch_size, input_seq_len, out_feat = shape[0], shape[1], shape[2], shape[3]

        gcn_out4 = tf.reshape(gcn_out4, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out4 = self.lstm4(gcn_out4)
        dense_output4 = self.dense4(lstm_out4)

        dense_output = dense_output3 + dense_output4

        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))

        return tf.transpose(output, [1, 2, 0])


class GraphInfo:

    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes


def compute_adjacency_matrix(route_distances, epsilon):
    adjacency_matrix = np.where(route_distances > epsilon, 1, 0)
    return adjacency_matrix


adjacency_matrix_bull = compute_adjacency_matrix(adjacency_matrix_bull, epsilon=0.5)
node_indices, neighbor_indices = np.where(adjacency_matrix_bull == 1)

graph3 = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix_bull.shape[0]
)

adjacency_matrix_std = compute_adjacency_matrix(adjacency_matrix_std, epsilon=0.5)
node_indices, neighbor_indices = np.where(adjacency_matrix_std == 1)

graph4 = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix_std.shape[0]
)

st_gcn = LSTMGC(
    in_feat=in_feat,
    out_feat=out_feat,
    lstm_units=lstm_units,
    input_seq_len=input_sequence_length,
    output_seq_len=forecast_horizon,
    graph_info3=graph3,
    graph_info4=graph4,
    static_adjacency_matrix=adjacency_matrix_std
)

inputs = layers.Input((int(input_sequence_length), graph3.num_nodes, int(in_feat)))

lstm_outputs = st_gcn(inputs)

model = keras.models.Model(inputs, lstm_outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

history = model.fit(train_dataset, epochs=100)

print("\nFinal Training History:")
print(f"Final Training Loss: {history.history['loss'][-1]}")
print("Fitness History (each epoch):")