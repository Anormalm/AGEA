"""Utility functions for AGEA."""

import yaml
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_prob, y_pred=None, k=100):
    """Compute prediction metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_pred is None:
        y_pred = (y_prob > 0.5).astype(int)

    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = 0.0
    try:
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auprc"] = 0.0
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Recall@K
    if len(y_true) >= k:
        topk_idx = np.argsort(y_prob)[-k:]
        metrics[f"recall@{k}"] = y_true[topk_idx].sum() / max(y_true.sum(), 1)
    else:
        metrics[f"recall@{k}"] = 0.0

    return metrics


def compute_structural_reward(evidence_graph, y_pred_labels, fraud_labels):
    """Compute structural evidence reward from the selected subgraph.

    V1 uses interpretable signals:
    - short cycle count (via cycle detector)
    - high-risk neighbor count
    - subgraph density
    - repeated shared neighbors
    """
    G = evidence_graph
    stats = {}

    # High-risk neighbor count
    nodes = list(G.nodes())
    high_risk = 0
    for n in nodes:
        if n < len(fraud_labels) and fraud_labels[n] == 1:
            high_risk += 1
    stats["high_risk_neighbors"] = high_risk

    # Subgraph density
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes > 1:
        max_edges = n_nodes * (n_nodes - 1)
        stats["density"] = n_edges / max_edges
    else:
        stats["density"] = 0.0

    # Shared neighbors (Jaccard overlap among neighbors)
    import networkx as nx
    shared_count = 0
    if n_nodes > 1 and n_nodes <= 200:
        adj = {n: set(G.neighbors(n)) for n in nodes}
        for i in range(min(len(nodes), 50)):
            for j in range(i + 1, min(len(nodes), 50)):
                inter = len(adj[nodes[i]] & adj[nodes[j]])
                if inter > 0:
                    shared_count += 1
    stats["shared_neighbors"] = shared_count

    return stats


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def subgraph_density(n_nodes, n_edges):
    if n_nodes <= 1:
        return 0.0
    return n_edges / (n_nodes * (n_nodes - 1))
