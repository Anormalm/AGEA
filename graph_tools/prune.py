"""Pruning graph tool — remove low-importance nodes (adj-dict based)."""

from typing import Set, Tuple


class PruneTopK:
    """Prune evidence subgraph by keeping only top-K important nodes.

    Importance = degree within the evidence subgraph + feature magnitude.
    """

    def __init__(self, topk: int = 10):
        self.topk = topk
        self.name = "PruneTopK"

    def __call__(self, evidence_nodes: Set[int], adj: dict,
                 x=None, target_node: int = None) -> Tuple[Set[int], dict]:
        if len(evidence_nodes) <= self.topk:
            return evidence_nodes, {"action": self.name, "nodes_removed": 0,
                                    "total_nodes": len(evidence_nodes)}

        scores = {}
        for n in evidence_nodes:
            deg = len(adj.get(n, set()) & evidence_nodes)
            feat_mag = 0.0
            if x is not None and n < x.size(0):
                feat_mag = x[n].norm().item()
            scores[n] = deg + 0.1 * feat_mag

        protected = {target_node} if target_node is not None else set()

        sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        kept = set()
        for n, s in sorted_nodes:
            if n in protected:
                kept.add(n)
            elif len(kept) < self.topk:
                kept.add(n)

        kept |= protected

        removed = len(evidence_nodes) - len(kept)
        info = {"action": self.name, "nodes_removed": removed,
                "total_nodes": len(kept)}
        return kept, info
