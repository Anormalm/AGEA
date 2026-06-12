"""LLM-guided AGEA policy — uses an LLM to select graph tools."""

import json
import torch
from typing import Set, List, Dict, Optional


ACTIONS = ["Expand1Hop", "Expand2Hop", "PPRTopK", "Community",
           "ShortCycle", "PruneTopK", "Stop"]

SYSTEM_PROMPT = """You are an evidence acquisition controller for fraud detection on graphs.

At each step, you observe a state summary and choose ONE action from:
- Expand1Hop: Add 1-hop neighbors of current evidence nodes
- Expand2Hop: Add 2-hop neighbors of current evidence nodes
- PPRTopK: Retrieve top-K nodes by Personalized PageRank
- Community: Detect and add community members around evidence
- ShortCycle: Find short cycles (triangles) involving evidence nodes
- PruneTopK: Remove low-importance nodes from evidence
- Stop: Stop acquisition and proceed to reasoning

Rules:
1. Choose exactly ONE action from the list above.
2. Do NOT invent nodes, edges, or actions.
3. Respond with ONLY the action name, nothing else.
4. Stop when the evidence is sufficient or budget is exhausted.
"""


class LLMPolicy:
    """LLM-guided evidence acquisition controller.

    The LLM only chooses tools — it cannot freely invent nodes or edges.
    """

    def __init__(self, max_steps: int = 6, budget_tokens: int = 2048,
                 budget_nodes: int = 50, budget_edges: int = 200,
                 model: str = "gpt-4o-mini", api_key: str = None,
                 api_base: str = None):
        self.max_steps = max_steps
        self.budget_tokens = budget_tokens
        self.budget_nodes = budget_nodes
        self.budget_edges = budget_edges
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    def _format_state(self, state: Dict) -> str:
        return f"""Current state:
- Target node: {state['target_node']}
- Evidence nodes: {state['num_evidence_nodes']}
- Evidence edges: {state['num_evidence_edges']}
- Suspicious neighbors: {state['suspicious_neighbors']}
- Subgraph density: {state['density']:.3f}
- Token estimate: {state['token_estimate']}
- Budget: {state['budget_tokens']} tokens, {state['budget_nodes']} nodes
- Steps taken: {state['steps_taken']}/{self.max_steps}
- Previous actions: {state['prev_actions']}

Choose one action: Expand1Hop, Expand2Hop, PPRTopK, Community, ShortCycle, PruneTopK, Stop"""

    def get_state_summary(self, target_node: int, evidence_nodes: Set[int],
                          edge_index: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor, prev_actions: List[str],
                          token_estimate: int) -> Dict:
        num_nodes = len(evidence_nodes)
        src, dst = edge_index
        mask = [(s.item() in evidence_nodes and d.item() in evidence_nodes)
                for s, d in zip(src, dst)]
        num_edges = sum(mask)
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

    def _call_llm(self, messages: List[Dict]) -> str:
        """Call the LLM API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[AGEA] LLM call failed: {e}. Falling back to Stop.")
            return "Stop"

    def select_action(self, state: Dict) -> str:
        """Use LLM to select the next action."""
        # Budget check — no LLM needed
        if state["token_estimate"] >= state["budget_tokens"]:
            return "Stop"
        if state["num_evidence_nodes"] >= state["budget_nodes"]:
            return "Stop"
        if state["steps_taken"] >= self.max_steps:
            return "Stop"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_state(state)},
        ]

        action = self._call_llm(messages)

        # Validate action
        if action in ACTIONS:
            return action

        # Fallback: simple heuristic
        if state["num_evidence_nodes"] <= 1:
            return "Expand1Hop"
        return "Stop"

    def run_episode(self, target_node: int, edge_index: torch.Tensor,
                    x: torch.Tensor, y: torch.Tensor,
                    tools: dict) -> tuple:
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

            if action == "PruneTopK":
                new_nodes, info = tool(evidence_nodes, edge_index,
                                       x.size(0), x=x, target_node=target_node)
            else:
                new_nodes, info = tool(evidence_nodes, edge_index, x.size(0),
                                       max_nodes=self.budget_nodes)

            evidence_nodes = new_nodes
            prev_actions.append(action)
            token_estimate = len(evidence_nodes) * 20 + info.get("total_nodes", 0) * 5

            trajectory.append({"step": step, "action": action, "info": info, "state": state})

        return evidence_nodes, trajectory
