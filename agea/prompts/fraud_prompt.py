"""Fraud prompt builder — dispatches to raw/compressed/evidence-only modes."""

import torch
from typing import Set, Dict, Optional

from .raw_prompt import RawPromptBuilder
from .compressed_prompt import CompressedPromptBuilder


class FraudPromptBuilder:
    """Build fraud reasoning prompts from evidence subgraphs.

    Modes:
    - raw: Full feature details for each node/edge
    - compressed: Quantized, aggregated summaries
    - evidence_only: Only structural patterns, no raw features
    """

    def __init__(self, mode: str = "raw", max_neighbor_summaries: int = 15,
                 max_edge_summaries: int = 30):
        self.mode = mode
        self.raw_builder = RawPromptBuilder(max_neighbor_summaries, max_edge_summaries)
        self.compressed_builder = CompressedPromptBuilder(
            max_neighbor_summaries, max_edge_summaries)

    def build(self, target_node: int, evidence_nodes: Set[int],
              edge_index: torch.Tensor, x: torch.Tensor,
              y: torch.Tensor = None,
              node_text: Dict = None, edge_text: Dict = None,
              struct_stats: Dict = None, budget_info: Dict = None) -> str:
        if self.mode == "compressed":
            return self.compressed_builder.build(
                target_node, evidence_nodes, edge_index, x, y,
                node_text, edge_text, struct_stats, budget_info)
        elif self.mode == "evidence_only":
            return self._build_evidence_only(
                target_node, evidence_nodes, edge_index, x, y,
                struct_stats, budget_info)
        else:
            return self.raw_builder.build(
                target_node, evidence_nodes, edge_index, x, y,
                node_text, edge_text, struct_stats, budget_info)

    def _build_evidence_only(self, target_node: int, evidence_nodes: Set[int],
                             edge_index: torch.Tensor, x: torch.Tensor,
                             y: torch.Tensor = None,
                             struct_stats: Dict = None,
                             budget_info: Dict = None) -> str:
        """Evidence-only: structural patterns without raw features."""
        parts = []
        parts.append(f"Target node: {target_node}")

        # Aggregate statistics only
        neighbors = [n for n in evidence_nodes if n != target_node]
        fraud_count = sum(1 for n in neighbors if y is not None and n < y.size(0) and y[n].item() == 1)
        legit_count = len(neighbors) - fraud_count

        parts.append(f"Evidence nodes: {len(evidence_nodes)}")
        parts.append(f"Fraud neighbors: {fraud_count}, Legit neighbors: {legit_count}")

        if struct_stats:
            parts.append(f"Structural patterns:")
            parts.append(f"  High-risk neighbors: {struct_stats.get('high_risk_neighbors', 0)}")
            parts.append(f"  Subgraph density: {struct_stats.get('density', 0):.3f}")
            parts.append(f"  Shared neighbors: {struct_stats.get('shared_neighbors', 0)}")
            parts.append(f"  Short cycles: {struct_stats.get('cycles_found', 0)}")

        if budget_info:
            parts.append(f"Cost: {budget_info.get('tokens', 0)} tokens, "
                         f"{budget_info.get('nodes', 0)} nodes, "
                         f"{budget_info.get('edges', 0)} edges")

        parts.append("\nPredict: fraud (1) or legitimate (0). Provide rationale.")
        return "\n".join(parts)

    def estimate_tokens(self, prompt: str) -> int:
        return max(1, len(prompt) // 4)
