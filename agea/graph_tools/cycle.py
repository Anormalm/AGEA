"""Short cycle detection graph tool."""

import torch
import networkx as nx
from typing import Set, Tuple, Dict


class ShortCycle:
    """Detect short cycles (triangles by default) involving evidence nodes."""

    def __init__(self, cycle_length: int = 3):
        self.cycle_length = cycle_length
        self.name = "ShortCycle"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, max_nodes: int = -1) -> Tuple[Set[int], Dict]:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        undirected = G.to_undirected()
        cycles_found = 0
        cycle_nodes = set()

        if self.cycle_length == 3:
            # Efficient triangle detection
            for u in evidence_nodes:
                if u >= num_nodes:
                    continue
                neighbors_u = set(undirected.neighbors(u))
                for v in neighbors_u:
                    if v in evidence_nodes or v in cycle_nodes:
                        common = neighbors_u & set(undirected.neighbors(v))
                        for w in common:
                            if w != u and w != v:
                                cycles_found += 1
                                for n in [u, v, w]:
                                    if n not in evidence_nodes:
                                        cycle_nodes.add(n)
                                if len(cycle_nodes) > 100:
                                    break
                    if len(cycle_nodes) > 100:
                        break
                if len(cycle_nodes) > 100:
                    break
        else:
            # General cycle detection (slower)
            try:
                all_cycles = nx.cycle_basis(undirected.subgraph(
                    list(evidence_nodes)[:100]))
                for cycle in all_cycles:
                    if len(cycle) <= self.cycle_length:
                        cycles_found += 1
                        for n in cycle:
                            if n not in evidence_nodes:
                                cycle_nodes.add(n)
            except Exception:
                pass

        if max_nodes > 0:
            cycle_nodes = set(list(cycle_nodes)[:max_nodes - len(evidence_nodes)])

        updated = evidence_nodes | cycle_nodes
        info = {"action": self.name, "nodes_added": len(cycle_nodes),
                "total_nodes": len(updated), "cycles_found": cycles_found}
        return updated, info
