# Load and preprocess the Cora dataset for GCN

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os

# ================================================================
# One-hot Encoding Function
# Converts paper categories into one-hot encoded vectors
# ================================================================
def encode_onehot(labels):
    """
    Convert paper categories (string labels) into one-hot encoded vectors.
    Example:
        ['Neural_Networks', 'Genetic_Algorithms', 'Neural_Networks'] 
        → [[1,0,0,...], [0,1,0,...], [1,0,0,...]]
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# ================================================================
# Normalization Function
# Row-normalizes a sparse matrix (each row sums to 1)
# ================================================================
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# ================================================================
# Convert Scipy Sparse Matrix → PyTorch Sparse Tensor
# ================================================================
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# ================================================================
# Load Cora Dataset
# ================================================================
def load_data(path="./data/", dataset="cora"):
    """Load citation network dataset (Cora only for now)."""
    print(f"Loading {dataset} dataset...")

    # Load cora.content
    idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # Load cora.cites
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # Make the graph symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize features and adjacency
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # Split indices
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # Convert to tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Return tensors
    print("-> Dataset loaded successfully!")
    print("Adjacency shape:", adj.shape)
    print("Feature shape:", features.shape)
    print("Number of classes:", labels.max().item() + 1)
    print("Train/Val/Test split:", len(idx_train), len(idx_val), len(idx_test))

    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":
    load_data()
