"""Neighborhood expansion graph tools."""

import torch
import networkx as nx
from typing import Set, Tuple


def _to_networkx(edge_index: torch.Tensor, num_nodes: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


class Expand1Hop:
    """Expand evidence subgraph by adding 1-hop neighbors of current nodes."""

    def __init__(self, max_neighbors: int = 20):
        self.max_neighbors = max_neighbors
        self.name = "Expand1Hop"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, max_nodes: int = -1) -> Tuple[Set[int], dict]:
        G = _to_networkx(edge_index, num_nodes)
        new_nodes = set()
        for node in evidence_nodes:
            if node >= num_nodes:
                continue
            neighbors = list(G.neighbors(node))
            # Prioritize higher-degree neighbors
            neighbors.sort(key=lambda n: G.degree(n), reverse=True)
            added = 0
            for nb in neighbors:
                if nb not in evidence_nodes and nb not in new_nodes:
                    if max_nodes > 0 and len(evidence_nodes) + len(new_nodes) >= max_nodes:
                        break
                    new_nodes.add(nb)
                    added += 1
                    if added >= self.max_neighbors:
                        break

        updated = evidence_nodes | new_nodes
        info = {"action": self.name, "nodes_added": len(new_nodes),
                "total_nodes": len(updated)}
        return updated, info


class Expand2Hop:
    """Expand evidence subgraph by adding 2-hop neighbors."""

    def __init__(self, max_neighbors: int = 20):
        self.max_neighbors = max_neighbors
        self.name = "Expand2Hop"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, max_nodes: int = -1) -> Tuple[Set[int], dict]:
        G = _to_networkx(edge_index, num_nodes)
        new_nodes = set()
        for node in evidence_nodes:
            if node >= num_nodes:
                continue
            for nb1 in G.neighbors(node):
                if nb1 in evidence_nodes or nb1 in new_nodes:
                    continue
                if max_nodes > 0 and len(evidence_nodes) + len(new_nodes) >= max_nodes:
                    break
                new_nodes.add(nb1)
                if len(new_nodes) >= self.max_neighbors:
                    break
                for nb2 in G.neighbors(nb1):
                    if nb2 not in evidence_nodes and nb2 not in new_nodes:
                        if max_nodes > 0 and len(evidence_nodes) + len(new_nodes) >= max_nodes:
                            break
                        new_nodes.add(nb2)
                        if len(new_nodes) >= self.max_neighbors:
                            break
                if len(new_nodes) >= self.max_neighbors:
                    break

        updated = evidence_nodes | new_nodes
        info = {"action": self.name, "nodes_added": len(new_nodes),
                "total_nodes": len(updated)}
        return updated, info
