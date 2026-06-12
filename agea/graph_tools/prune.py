"""Pruning graph tool — remove low-importance nodes."""

import torch
import networkx as nx
from typing import Set, Tuple


class PruneTopK:
    """Prune evidence subgraph by keeping only top-K important nodes.

    Importance = degree within the evidence subgraph + feature magnitude.
    """

    def __init__(self, topk: int = 10):
        self.topk = topk
        self.name = "PruneTopK"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, x: torch.Tensor = None,
                 target_node: int = None) -> Tuple[Set[int], dict]:
        if len(evidence_nodes) <= self.topk:
            return evidence_nodes, {"action": self.name, "nodes_removed": 0,
                                    "total_nodes": len(evidence_nodes)}

        G = nx.DiGraph()
        G.add_nodes_from(evidence_nodes)
        edges = edge_index.t().tolist()
        for s, d in edges:
            if s in evidence_nodes and d in evidence_nodes:
                G.add_edge(s, d)

        # Score nodes by subgraph degree
        scores = {}
        for n in evidence_nodes:
            deg = G.degree(n) if n in G else 0
            feat_mag = 0.0
            if x is not None and n < x.size(0):
                feat_mag = x[n].norm().item()
            scores[n] = deg + 0.1 * feat_mag

        # Always keep target node
        protected = {target_node} if target_node is not None else set()

        # Select top-K by score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        kept = set()
        for n, s in sorted_nodes:
            if n in protected:
                kept.add(n)
            elif len(kept) < self.topk:
                kept.add(n)

        # Ensure protected nodes are kept
        kept |= protected

        removed = len(evidence_nodes) - len(kept)
        info = {"action": self.name, "nodes_removed": removed,
                "total_nodes": len(kept)}
        return kept, info
