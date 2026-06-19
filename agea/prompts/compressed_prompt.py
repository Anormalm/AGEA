"""Compressed evidence prompt builder — summarizes features efficiently."""

import torch
from typing import Set, Dict, Optional


class CompressedPromptBuilder:
    """Build a compressed evidence prompt that minimizes token usage."""

    def __init__(self, max_neighbor_summaries: int = 15,
                 max_edge_summaries: int = 30,
                 n_feature_bins: int = 5):
        self.max_neighbor_summaries = max_neighbor_summaries
        self.max_edge_summaries = max_edge_summaries
        self.n_feature_bins = n_feature_bins

    def _bin_value(self, val: float, vmin: float = -3.0, vmax: float = 3.0) -> int:
        clipped = max(vmin, min(vmax, val))
        ratio = (clipped - vmin) / (vmax - vmin + 1e-8)
        return int(ratio * self.n_feature_bins)

    def build(self, target_node: int, evidence_nodes: Set[int],
              edge_index: torch.Tensor, x: torch.Tensor,
              y: torch.Tensor = None,
              node_text: Dict = None, edge_text: Dict = None,
              struct_stats: Dict = None, budget_info: Dict = None,
              adj: dict = None) -> str:
        parts = []

        if target_node < x.size(0):
            feat = x[target_node]
            binned = [self._bin_value(feat[i].item()) for i in range(min(feat.size(0), 32))]
            parts.append(f"T{target_node}:f={binned}")
        else:
            parts.append(f"T{target_node}:f=?")

        neighbors = [n for n in evidence_nodes if n != target_node]
        fraud_count = 0
        legit_count = 0
        feat_norms = []
        for n in neighbors[:self.max_neighbor_summaries]:
            if n >= x.size(0):
                continue
            feat_norms.append(x[n].norm().item())
            if y is not None and n < y.size(0):
                if y[n].item() == 1:
                    fraud_count += 1
                else:
                    legit_count += 1

        if feat_norms:
            avg_norm = sum(feat_norms) / len(feat_norms)
            parts.append(f"N:{len(neighbors)}|F:{fraud_count}|L:{legit_count}|avg_n={avg_norm:.1f}")
        else:
            parts.append(f"N:{len(neighbors)}|F:0|L:0")

        # Edge count using adj dict
        if adj is not None:
            edge_count = sum(len(adj.get(n, set()) & evidence_nodes) for n in evidence_nodes)
        else:
            src, dst = edge_index
            ev_set = evidence_nodes
            edge_count = sum(1 for s, d in zip(src, dst)
                             if s.item() in ev_set and d.item() in ev_set)
        parts.append(f"E:{edge_count}")

        if struct_stats:
            parts.append(f"S:hr={struct_stats.get('high_risk_neighbors', 0)}"
                         f",d={struct_stats.get('density', 0):.2f}"
                         f",sn={struct_stats.get('shared_neighbors', 0)}"
                         f",cy={struct_stats.get('cycles_found', 0)}")

        if budget_info:
            parts.append(f"B:tk={budget_info.get('tokens', 0)}"
                         f",nd={budget_info.get('nodes', 0)}"
                         f",ed={budget_info.get('edges', 0)}")

        parts.append("PREDICT:1(fraud)or0(legit)+rationale")

        return " | ".join(parts)

    def estimate_tokens(self, prompt: str) -> int:
        return max(1, len(prompt) // 4)
