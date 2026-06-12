"""Personalized PageRank top-K retrieval."""

import torch
import networkx as nx
from typing import Set, Tuple


class PPRTopK:
    """Retrieve top-K nodes by Personalized PageRank from evidence nodes."""

    def __init__(self, alpha: float = 0.15, topk: int = 10):
        self.alpha = alpha
        self.topk = topk
        self.name = "PPRTopK"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, max_nodes: int = -1) -> Tuple[Set[int], dict]:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # PPR with personalization on current evidence nodes
        personalization = {n: 1.0 / len(evidence_nodes) for n in evidence_nodes if n < num_nodes}
        if not personalization:
            return evidence_nodes, {"action": self.name, "nodes_added": 0, "total_nodes": len(evidence_nodes)}

        try:
            ppr = nx.pagerank(G, alpha=self.alpha, personalization=personalization,
                              max_iter=100, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            return evidence_nodes, {"action": self.name, "nodes_added": 0, "total_nodes": len(evidence_nodes)}

        # Sort by PPR score, exclude already-evidence nodes
        candidates = [(n, s) for n, s in ppr.items() if n not in evidence_nodes]
        candidates.sort(key=lambda x: x[1], reverse=True)

        k = self.topk
        if max_nodes > 0:
            remaining = max_nodes - len(evidence_nodes)
            k = min(k, remaining)

        new_nodes = set()
        for n, s in candidates[:k]:
            if s > 0:
                new_nodes.add(n)

        updated = evidence_nodes | new_nodes
        info = {"action": self.name, "nodes_added": len(new_nodes),
                "total_nodes": len(updated)}
        return updated, info
