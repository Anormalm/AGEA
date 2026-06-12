"""Top-K node selection by feature similarity to target."""

import torch
from typing import Set, Tuple


class TopKSimilar:
    """Select top-K nodes most similar to the target by feature cosine similarity."""

    def __init__(self, topk: int = 10):
        self.topk = topk
        self.name = "TopKSimilar"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, x: torch.Tensor, target_node: int,
                 max_nodes: int = -1) -> Tuple[Set[int], dict]:
        if target_node >= x.size(0):
            return evidence_nodes, {"action": self.name, "nodes_added": 0,
                                    "total_nodes": len(evidence_nodes)}

        target_feat = x[target_node]
        target_norm = target_feat.norm()
        if target_norm < 1e-8:
            return evidence_nodes, {"action": self.name, "nodes_added": 0,
                                    "total_nodes": len(evidence_nodes)}

        # Compute cosine similarity with all nodes not in evidence
        candidates = [n for n in range(num_nodes) if n not in evidence_nodes and n != target_node]
        if not candidates:
            return evidence_nodes, {"action": self.name, "nodes_added": 0,
                                    "total_nodes": len(evidence_nodes)}

        cand_idx = torch.tensor(candidates)
        cand_feats = x[cand_idx]
        sims = torch.nn.functional.cosine_similarity(cand_feats, target_feat.unsqueeze(0), dim=1)

        k = self.topk
        if max_nodes > 0:
            k = min(k, max_nodes - len(evidence_nodes))
        k = max(k, 0)

        if k == 0:
            return evidence_nodes, {"action": self.name, "nodes_added": 0,
                                    "total_nodes": len(evidence_nodes)}

        topk_vals, topk_idx = sims.topk(min(k, len(candidates)))
        new_nodes = set(candidates[i] for i in topk_idx.tolist() if topk_vals[i] > 0.1)

        updated = evidence_nodes | new_nodes
        info = {"action": self.name, "nodes_added": len(new_nodes),
                "total_nodes": len(updated)}
        return updated, info
