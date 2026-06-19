"""Neighborhood expansion graph tools — adj-dict based (no NetworkX)."""

from typing import Set, Tuple


class Expand1Hop:
    """Expand evidence subgraph by adding 1-hop neighbors of current nodes."""

    def __init__(self, max_neighbors: int = 20):
        self.max_neighbors = max_neighbors
        self.name = "Expand1Hop"

    def __call__(self, evidence_nodes: Set[int], adj: dict,
                 max_nodes: int = -1) -> Tuple[Set[int], dict]:
        new_nodes = set()
        for node in evidence_nodes:
            neighbors = adj.get(node, set())
            sorted_neighbors = sorted(
                neighbors, key=lambda n: len(adj.get(n, set())), reverse=True)
            added = 0
            for nb in sorted_neighbors:
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

    def __call__(self, evidence_nodes: Set[int], adj: dict,
                 max_nodes: int = -1) -> Tuple[Set[int], dict]:
        new_nodes = set()
        for node in evidence_nodes:
            for nb1 in adj.get(node, set()):
                if nb1 in evidence_nodes or nb1 in new_nodes:
                    continue
                if max_nodes > 0 and len(evidence_nodes) + len(new_nodes) >= max_nodes:
                    break
                new_nodes.add(nb1)
                if len(new_nodes) >= self.max_neighbors:
                    break
                for nb2 in adj.get(nb1, set()):
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
