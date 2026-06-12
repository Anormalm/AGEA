"""Heuristic AGEA policy — rule-based action selection."""

import torch
from typing import Set, List, Dict, Optional


class HeuristicPolicy:
    """Rule-based evidence acquisition controller.

    Decision logic:
    - If evidence is empty: Expand1Hop
    - If few nodes & low density: Expand2Hop or Community
    - If many nodes & high density: PruneTopK
    - If suspicious structure: ShortCycle
    - If moderate evidence: PPRTopK
    - Stop when budget is exhausted or evidence is sufficient
    """

    ACTIONS = ["Expand1Hop", "Expand2Hop", "PPRTopK", "Community",
               "ShortCycle", "PruneTopK", "Stop"]

    def __init__(self, max_steps: int = 6, budget_tokens: int = 2048,
                 budget_nodes: int = 50, budget_edges: int = 200):
        self.max_steps = max_steps
        self.budget_tokens = budget_tokens
        self.budget_nodes = budget_nodes
        self.budget_edges = budget_edges

    def get_state_summary(self, target_node: int, evidence_nodes: Set[int],
                          edge_index: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor, prev_actions: List[str],
                          token_estimate: int) -> Dict:
        """Compute compact state summary z_t."""
        num_nodes = len(evidence_nodes)
        # Count edges within evidence
        src, dst = edge_index
        mask = [(s.item() in evidence_nodes and d.item() in evidence_nodes)
                for s, d in zip(src, dst)]
        num_edges = sum(mask)

        # Suspicious neighbors
        suspicious = 0
        for n in evidence_nodes:
            if n < len(y) and y[n] == 1:
                suspicious += 1

        # Density
        density = num_edges / max(num_nodes * (num_nodes - 1), 1)

        return {
            "target_node": target_node,
            "num_evidence_nodes": num_nodes,
            "num_evidence_edges": num_edges,
            "suspicious_neighbors": suspicious,
            "density": density,
            "token_estimate": token_estimate,
            "budget_tokens": self.budget_tokens,
            "budget_nodes": self.budget_nodes,
            "prev_actions": prev_actions,
            "steps_taken": len(prev_actions),
        }

    def select_action(self, state: Dict) -> str:
        """Select next action based on heuristic rules."""
        z = state

        # Always stop if budget exhausted
        if z["token_estimate"] >= z["budget_tokens"]:
            return "Stop"
        if z["num_evidence_nodes"] >= z["budget_nodes"]:
            return "Stop"
        if z["steps_taken"] >= self.max_steps:
            return "Stop"

        # Initial expansion
        if z["num_evidence_nodes"] <= 1:
            return "Expand1Hop"

        # If density is very high, prune
        if z["density"] > 0.5 and z["num_evidence_nodes"] > 20:
            return "PruneTopK"

        # If we have expanded but haven't looked at structure
        if z["num_evidence_nodes"] > 5 and "ShortCycle" not in z["prev_actions"]:
            return "ShortCycle"

        # If few suspicious neighbors, try PPR to find more
        if z["suspicious_neighbors"] < 2 and "PPRTopK" not in z["prev_actions"]:
            return "PPRTopK"

        # If moderate size and haven't tried community
        if 5 < z["num_evidence_nodes"] < 30 and "Community" not in z["prev_actions"]:
            return "Community"

        # Expand further if small
        if z["num_evidence_nodes"] < 15 and "Expand2Hop" not in z["prev_actions"]:
            return "Expand2Hop"

        # Expand 1-hop again if still small
        if z["num_evidence_nodes"] < 20:
            return "Expand1Hop"

        # Prune if large
        if z["num_evidence_nodes"] > 30:
            return "PruneTopK"

        return "Stop"

    def run_episode(self, target_node: int, edge_index: torch.Tensor,
                    x: torch.Tensor, y: torch.Tensor,
                    tools: dict) -> tuple:
        """Run a full acquisition episode for one target node.

        Returns: (evidence_nodes, trajectory_info)
        """
        evidence_nodes = {target_node}
        prev_actions = []
        token_estimate = 0
        trajectory = []

        for step in range(self.max_steps):
            state = self.get_state_summary(
                target_node, evidence_nodes, edge_index, x, y,
                prev_actions, token_estimate)

            action = self.select_action(state)
            if action == "Stop":
                trajectory.append({"step": step, "action": "Stop", "state": state})
                break

            tool = tools.get(action)
            if tool is None:
                break

            # Execute tool
            if action == "PruneTopK":
                new_nodes, info = tool(evidence_nodes, edge_index,
                                       x.size(0), x=x, target_node=target_node)
            else:
                new_nodes, info = tool(evidence_nodes, edge_index, x.size(0),
                                       max_nodes=self.budget_nodes)

            evidence_nodes = new_nodes
            prev_actions.append(action)

            # Update token estimate (rough: ~20 tokens per node, ~5 per edge)
            token_estimate = len(evidence_nodes) * 20 + info.get("total_nodes", 0) * 5

            trajectory.append({"step": step, "action": action, "info": info, "state": state})

        return evidence_nodes, trajectory
