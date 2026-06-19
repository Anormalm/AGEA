"""Heuristic AGEA policy — rule-based action selection (adj-dict based)."""

from typing import Set, List, Dict


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
                 budget_nodes: int = 50):
        self.max_steps = max_steps
        self.budget_tokens = budget_tokens
        self.budget_nodes = budget_nodes

    def get_state_summary(self, target_node: int, evidence_nodes: Set[int],
                          adj: dict, y, prev_actions: List[str],
                          token_estimate: int) -> Dict:
        num_nodes = len(evidence_nodes)
        num_edges = sum(len(adj.get(n, set()) & evidence_nodes) for n in evidence_nodes)
        suspicious = sum(1 for n in evidence_nodes if n < len(y) and y[n] == 1)
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
        z = state

        if z["token_estimate"] >= z["budget_tokens"]:
            return "Stop"
        if z["num_evidence_nodes"] >= z["budget_nodes"]:
            return "Stop"
        if z["steps_taken"] >= self.max_steps:
            return "Stop"

        if z["num_evidence_nodes"] <= 1:
            return "Expand1Hop"

        if z["density"] > 0.5 and z["num_evidence_nodes"] > 20:
            return "PruneTopK"

        if z["num_evidence_nodes"] > 5 and "ShortCycle" not in z["prev_actions"]:
            return "ShortCycle"

        if z["suspicious_neighbors"] < 2 and "PPRTopK" not in z["prev_actions"]:
            return "PPRTopK"

        if 5 < z["num_evidence_nodes"] < 30 and "Community" not in z["prev_actions"]:
            return "Community"

        if z["num_evidence_nodes"] < 15 and "Expand2Hop" not in z["prev_actions"]:
            return "Expand2Hop"

        if z["num_evidence_nodes"] < 20:
            return "Expand1Hop"

        if z["num_evidence_nodes"] > 30:
            return "PruneTopK"

        return "Stop"

    def run_episode(self, target_node: int, adj: dict,
                    x, y, tools: dict, tool_times=None, tool_calls=None) -> tuple:
        import time as _time
        evidence_nodes = {target_node}
        prev_actions = []
        token_estimate = 0
        trajectory = []

        for step in range(self.max_steps):
            state = self.get_state_summary(
                target_node, evidence_nodes, adj, y,
                prev_actions, token_estimate)

            action = self.select_action(state)
            if action == "Stop":
                trajectory.append({"step": step, "action": "Stop", "state": state})
                break

            tool = tools.get(action)
            if tool is None:
                break

            t0 = _time.time()
            if action == "PruneTopK":
                new_nodes, info = tool(evidence_nodes, adj,
                                       x=x, target_node=target_node)
            else:
                new_nodes, info = tool(evidence_nodes, adj,
                                       max_nodes=self.budget_nodes)
            dt = _time.time() - t0
            if tool_times is not None:
                tool_times[action] = tool_times.get(action, 0.0) + dt
            if tool_calls is not None:
                tool_calls[action] = tool_calls.get(action, 0) + 1

            evidence_nodes = new_nodes
            prev_actions.append(action)
            token_estimate = len(evidence_nodes) * 20 + info.get("total_nodes", 0) * 5

            trajectory.append({"step": step, "action": action, "info": info, "state": state})

        return evidence_nodes, trajectory
