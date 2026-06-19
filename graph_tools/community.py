"""Community detection graph tool — adj-dict based, fast label propagation."""

from collections import defaultdict
from typing import Set, Tuple


class Community:
    """Detect community around evidence nodes and add community members.

    Uses 1-hop ego graph (capped at 300 nodes) + fast label propagation.
    """

    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        self.name = "Community"

    def __call__(self, evidence_nodes: Set[int], adj: dict,
                 max_nodes: int = -1) -> Tuple[Set[int], dict]:
        # Build 1-hop ego graph, capped
        ego_nodes = set(evidence_nodes)
        for n in evidence_nodes:
            for nb in adj.get(n, set()):
                ego_nodes.add(nb)
                if len(ego_nodes) >= 300:
                    break
            if len(ego_nodes) >= 300:
                break

        if len(ego_nodes) <= len(evidence_nodes):
            return evidence_nodes, {"action": self.name, "nodes_added": 0,
                                    "total_nodes": len(evidence_nodes)}

        # Build undirected adjacency for ego subgraph only
        adj_u = defaultdict(set)
        for n in ego_nodes:
            for nb in adj.get(n, set()):
                if nb in ego_nodes:
                    adj_u[n].add(nb)
                    adj_u[nb].add(n)

        # Label propagation (5 iterations, early stop)
        labels = {n: i for i, n in enumerate(ego_nodes)}
        for _ in range(5):
            changed = False
            for n in ego_nodes:
                nbs = adj_u.get(n)
                if not nbs:
                    continue
                counts = {}
                for nb in nbs:
                    lbl = labels[nb]
                    counts[lbl] = counts.get(lbl, 0) + 1
                best = max(counts, key=counts.get)
                if best != labels[n]:
                    labels[n] = best
                    changed = True
            if not changed:
                break

        # Group by community, add members from communities containing evidence
        communities = defaultdict(set)
        for n, lbl in labels.items():
            communities[lbl].add(n)

        new_nodes = set()
        for comm in communities.values():
            if comm & evidence_nodes:
                for n in comm:
                    if n not in evidence_nodes:
                        if max_nodes > 0 and len(evidence_nodes) + len(new_nodes) >= max_nodes:
                            break
                        new_nodes.add(n)

        updated = evidence_nodes | new_nodes
        info = {"action": self.name, "nodes_added": len(new_nodes),
                "total_nodes": len(updated), "communities_found": len(communities)}
        return updated, info
