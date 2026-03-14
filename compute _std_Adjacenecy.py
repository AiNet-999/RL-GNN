import pandas as pd
import numpy as np

df = pd.read_csv("/content/SZSE_Closing_Prices - Copy.csv", header=None)
print("Original Shape:", df.shape)

df = df.fillna(method='ffill')
df = df.fillna(method='bfill')
df = df.fillna(df.mean())
print("Missing values after cleaning:", df.isnull().sum().sum())

num_rows = int(0.70 * len(df))
df_70 = df.iloc[:num_rows]
print("70% Data Shape:", df_70.shape)

corr_matrix = df_70.corr()
print("Correlation Matrix Shape:", corr_matrix.shape)

threshold = 0.5
adj_matrix = (corr_matrix.abs() > threshold).astype(int)
np.fill_diagonal(adj_matrix.values, 0)

A = adj_matrix.values
num_nodes = A.shape[0]

ones_count = np.sum(A == 1)
zeros_count = np.sum(A == 0)
unique_edges = ones_count // 2
max_possible_edges = num_nodes * (num_nodes - 1) / 2
density = unique_edges / max_possible_edges
sparsity = 1 - density
degrees = np.sum(A, axis=1)
avg_degree = np.mean(degrees)
max_degree = np.max(degrees)
min_degree = np.min(degrees)

print("\n===== Graph Statistics =====")
print("Number of Nodes:", num_nodes)
print("Total 1s in Matrix:", ones_count)
print("Total 0s in Matrix:", zeros_count)
print("Unique Undirected Edges:", unique_edges)
print("Graph Density:", round(density, 4))
print("Graph Sparsity:", round(sparsity, 4))
print("Average Degree:", round(avg_degree, 2))
print("Maximum Degree:", max_degree)
print("Minimum Degree:", min_degree)

adj_matrix.to_csv("adjacency_matrix.csv", index=False)
print("\nAdjacency matrix saved as adjacency_matrix.csv")