"""Short cycle detection graph tool — adj-dict based, capped."""

from collections import defaultdict
from typing import Set, Tuple, Dict


class ShortCycle:
    """Detect short cycles (triangles by default) involving evidence nodes."""

    def __init__(self, cycle_length: int = 3):
        self.cycle_length = cycle_length
        self.name = "ShortCycle"

    def __call__(self, evidence_nodes: Set[int], adj: dict,
                 max_nodes: int = -1) -> Tuple[Set[int], Dict]:
        # Build small undirected adj for evidence + 1-hop, capped at 500
        relevant = set(evidence_nodes)
        for n in evidence_nodes:
            for nb in adj.get(n, set()):
                relevant.add(nb)
                if len(relevant) >= 500:
                    break
            if len(relevant) >= 500:
                break

        adj_u = defaultdict(set)
        for n in relevant:
            for nb in adj.get(n, set()):
                if nb in relevant:
                    adj_u[n].add(nb)
                    adj_u[nb].add(n)

        cycles_found = 0
        cycle_nodes = set()

        if self.cycle_length == 3:
            for u in evidence_nodes:
                neighbors_u = adj_u.get(u, set())
                for v in neighbors_u:
                    common = neighbors_u & adj_u.get(v, set())
                    for w in common:
                        if w != u and w != v:
                            cycles_found += 1
                            if w not in evidence_nodes:
                                cycle_nodes.add(w)
                            if len(cycle_nodes) >= 50:
                                break
                    if len(cycle_nodes) >= 50:
                        break
                if len(cycle_nodes) >= 50:
                    break

        if max_nodes > 0:
            excess = len(evidence_nodes) + len(cycle_nodes) - max_nodes
            if excess > 0:
                cycle_nodes = set(list(cycle_nodes)[:len(cycle_nodes) - excess])

        updated = evidence_nodes | cycle_nodes
        info = {"action": self.name, "nodes_added": len(cycle_nodes),
                "total_nodes": len(updated), "cycles_found": cycles_found}
        return updated, info
