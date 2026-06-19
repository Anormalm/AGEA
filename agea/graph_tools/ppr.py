"""Personalized PageRank top-K retrieval — random walk approximation on adj dict."""

import numpy as np
from typing import Set, Tuple


class PPRTopK:
    """Retrieve top-K nodes by Personalized PageRank from evidence nodes.

    Uses fast random-walk-with-restart approximation on the precomputed
    adjacency dict. No scipy sparse matrix needed — O(walks × walk_length).
    """

    def __init__(self, alpha: float = 0.15, topk: int = 10,
                 num_walks: int = 200, walk_length: int = 40):
        self.alpha = alpha
        self.topk = topk
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.name = "PPRTopK"
        # Precomputed adjacency stored here
        self._adj = None
        self._adj_list = None  # list version for fast random access

    def build_transition(self, edge_index, num_nodes: int):
        """No-op for compatibility. Adj is set via set_adj()."""
        pass

    def set_adj(self, adj: dict):
        """Store adjacency and pre-convert sets to lists for fast indexing."""
        self._adj = adj
        self._adj_list = {}
        for node, neighbors in adj.items():
            if neighbors:
                self._adj_list[node] = list(neighbors)

    def __call__(self, evidence_nodes: Set[int], adj: dict,
                 max_nodes: int = -1) -> Tuple[Set[int], dict]:
        if not evidence_nodes:
            return evidence_nodes, {"action": self.name, "nodes_added": 0,
                                    "total_nodes": len(evidence_nodes)}

        # Use stored adj_list if available, else build from adj
        adj_list = self._adj_list if self._adj_list is not None else {}
        if not adj_list and adj:
            for node, neighbors in adj.items():
                if neighbors:
                    adj_list[node] = list(neighbors)

        evidence_list = list(evidence_nodes)
        n_evidence = len(evidence_list)

        # Random walk with restart
        visit_counts = {}
        rng = np.random.RandomState()

        for _ in range(self.num_walks):
            current = evidence_list[rng.randint(n_evidence)]
            for _ in range(self.walk_length):
                if rng.random() < self.alpha:
                    current = evidence_list[rng.randint(n_evidence)]
                else:
                    neighbors = adj_list.get(current)
                    if neighbors:
                        current = neighbors[rng.randint(len(neighbors))]
                visit_counts[current] = visit_counts.get(current, 0) + 1

        # Sort by visit count, exclude evidence nodes
        candidates = [(n, c) for n, c in visit_counts.items()
                      if n not in evidence_nodes]
        candidates.sort(key=lambda x: x[1], reverse=True)

        k = self.topk
        if max_nodes > 0:
            k = min(k, max(0, max_nodes - len(evidence_nodes)))

        new_nodes = set()
        for n, c in candidates[:k]:
            new_nodes.add(n)

        updated = evidence_nodes | new_nodes
        info = {"action": self.name, "nodes_added": len(new_nodes),
                "total_nodes": len(updated)}
        return updated, info
