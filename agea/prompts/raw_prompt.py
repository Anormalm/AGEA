"""Raw evidence prompt builder — includes full feature details."""

import torch
from typing import Set, Dict, List, Optional


class RawPromptBuilder:
    """Build a raw evidence prompt with full feature details."""

    def __init__(self, max_neighbor_summaries: int = 15,
                 max_edge_summaries: int = 30):
        self.max_neighbor_summaries = max_neighbor_summaries
        self.max_edge_summaries = max_edge_summaries

    def build(self, target_node: int, evidence_nodes: Set[int],
              edge_index: torch.Tensor, x: torch.Tensor,
              y: torch.Tensor = None,
              node_text: Dict = None, edge_text: Dict = None,
              struct_stats: Dict = None, budget_info: Dict = None) -> str:
        parts = []

        # Target node features
        parts.append(f"=== Target Node {target_node} ===")
        if node_text and str(target_node) in node_text:
            parts.append(f"Description: {node_text[str(target_node)]}")
        if target_node < x.size(0):
            feat = x[target_node]
            # Summarize features (top-5 by magnitude)
            topk_vals, topk_idx = feat.abs().topk(min(5, feat.size(0)))
            feat_summary = ", ".join(
                f"f{idx.item()}={feat[idx].item():.3f}" for idx in topk_idx)
            parts.append(f"Top features: {feat_summary}")
            parts.append(f"Feature norm: {feat.norm().item():.3f}")

        # Neighbor summaries
        neighbors = [n for n in evidence_nodes if n != target_node]
        neighbors = sorted(neighbors)[:self.max_neighbor_summaries]
        if neighbors:
            parts.append(f"\n=== Neighbor Summaries ({len(neighbors)} shown) ===")
            for n in neighbors:
                if n >= x.size(0):
                    continue
                label_str = ""
                if y is not None and n < y.size(0):
                    label_str = f" [label={'FRAUD' if y[n].item() == 1 else 'LEGIT'}]"
                text_str = ""
                if node_text and str(n) in node_text:
                    text_str = f" | {node_text[str(n)]}"
                feat = x[n]
                parts.append(
                    f"  Node {n}{label_str}{text_str}: norm={feat.norm().item():.2f}")

        # Edge summaries
        src, dst = edge_index
        ev_list = list(evidence_nodes)
        ev_set = evidence_nodes
        edge_count = 0
        parts.append(f"\n=== Edge Summaries ===")
        for s, d in zip(src, dst):
            if s.item() in ev_set and d.item() in ev_set:
                if edge_count >= self.max_edge_summaries:
                    break
                parts.append(f"  Edge {s.item()} -> {d.item()}")
                edge_count += 1
        parts.append(f"Total evidence edges: {edge_count}")

        # Structural patterns
        if struct_stats:
            parts.append(f"\n=== Structural Patterns ===")
            parts.append(f"High-risk neighbors: {struct_stats.get('high_risk_neighbors', 0)}")
            parts.append(f"Subgraph density: {struct_stats.get('density', 0):.3f}")
            parts.append(f"Shared neighbors: {struct_stats.get('shared_neighbors', 0)}")
            parts.append(f"Cycles found: {struct_stats.get('cycles_found', 0)}")

        # Budget info
        if budget_info:
            parts.append(f"\n=== Budget Summary ===")
            parts.append(f"Token estimate: {budget_info.get('tokens', 0)}")
            parts.append(f"Nodes retrieved: {budget_info.get('nodes', 0)}")
            parts.append(f"Edges retrieved: {budget_info.get('edges', 0)}")

        # Final instruction
        parts.append("\n=== Task ===")
        parts.append("Based on the above evidence, predict whether the target node "
                      "is involved in fraud (1) or is legitimate (0).")
        parts.append("Provide your prediction and a concise rationale.")

        return "\n".join(parts)

    def estimate_tokens(self, prompt: str) -> int:
        return max(1, len(prompt) // 4)
