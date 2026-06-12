"""Community detection graph tool."""

import torch
import networkx as nx
from typing import Set, Tuple


class Community:
    """Detect community around evidence nodes and add community members."""

    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        self.name = "Community"

    def __call__(self, evidence_nodes: Set[int], edge_index: torch.Tensor,
                 num_nodes: int, max_nodes: int = -1) -> Tuple[Set[int], dict]:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Get subgraph around evidence nodes (up to 3-hop for efficiency)
        ego_nodes = set()
        frontier = set(evidence_nodes)
        for _ in range(3):
            next_frontier = set()
            for n in frontier:
                if n < num_nodes:
                    next_frontier.update(G.neighbors(n))
            ego_nodes |= frontier
            frontier = next_frontier - ego_nodes
            if len(ego_nodes) > 2000:
                break
        ego_nodes |= frontier

        subG = G.subgraph([n for n in ego_nodes if n < num_nodes])
        if len(subG) == 0:
            return evidence_nodes, {"action": self.name, "nodes_added": 0, "total_nodes": len(evidence_nodes)}

        # Run Louvain on undirected version
        undirected = subG.to_undirected()
        try:
            communities = nx.community.louvain_communities(undirected, resolution=self.resolution)
        except Exception:
            return evidence_nodes, {"action": self.name, "nodes_added": 0, "total_nodes": len(evidence_nodes)}

        # Find communities containing evidence nodes and add their members
        new_nodes = set()
        for comm in communities:
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
