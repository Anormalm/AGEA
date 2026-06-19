"""GRPO-learned AGEA policy — Group Relative Policy Optimization (adj-dict based)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Set, List, Dict, Tuple


ACTIONS = ["Expand1Hop", "Expand2Hop", "PPRTopK", "Community",
           "ShortCycle", "PruneTopK", "Stop"]
ACTION_DIM = len(ACTIONS)


class PolicyNetwork(nn.Module):
    """Simple MLP policy that maps state features to action logits."""

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec)


def state_to_vector(state: Dict, state_dim: int = 16) -> torch.Tensor:
    features = [
        state.get("num_evidence_nodes", 0) / 100.0,
        state.get("num_evidence_edges", 0) / 500.0,
        state.get("suspicious_neighbors", 0) / 50.0,
        state.get("density", 0.0),
        state.get("token_estimate", 0) / 3000.0,
        state.get("budget_tokens", 2048) / 3000.0,
        state.get("budget_nodes", 50) / 100.0,
        state.get("steps_taken", 0) / 10.0,
        1.0 if state.get("num_evidence_nodes", 0) <= 1 else 0.0,
        float(state.get("density", 0.0) > 0.5),
        1.0 if "Expand1Hop" in state.get("prev_actions", []) else 0.0,
        1.0 if "Expand2Hop" in state.get("prev_actions", []) else 0.0,
        1.0 if "PPRTopK" in state.get("prev_actions", []) else 0.0,
        1.0 if "Community" in state.get("prev_actions", []) else 0.0,
        1.0 if "ShortCycle" in state.get("prev_actions", []) else 0.0,
        1.0 if "PruneTopK" in state.get("prev_actions", []) else 0.0,
    ]
    vec = torch.tensor(features[:state_dim], dtype=torch.float32)
    if vec.size(0) < state_dim:
        vec = F.pad(vec, (0, state_dim - vec.size(0)))
    return vec


class GRPOPolicy:
    """GRPO-learned evidence acquisition controller."""

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128,
                 K: int = 4, clip_eps: float = 0.2,
                 entropy_coeff: float = 0.01, lr: float = 1e-4,
                 max_steps: int = 6, budget_tokens: int = 2048,
                 budget_nodes: int = 50,
                 beta: float = 0.1, gamma: float = 0.001, eta: float = 0.01):
        self.state_dim = state_dim
        self.K = K
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_steps = max_steps
        self.budget_tokens = budget_tokens
        self.budget_nodes = budget_nodes
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        self.network = PolicyNetwork(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.old_network = PolicyNetwork(state_dim, hidden_dim)

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

    @torch.no_grad()
    def select_action(self, state: Dict, deterministic: bool = False) -> Tuple[str, float]:
        if state["token_estimate"] >= state["budget_tokens"]:
            return "Stop", 1.0
        if state["num_evidence_nodes"] >= state["budget_nodes"]:
            return "Stop", 1.0
        if state["steps_taken"] >= self.max_steps:
            return "Stop", 1.0

        state_vec = state_to_vector(state, self.state_dim)
        logits = self.network(state_vec)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action_idx = probs.argmax().item()
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        action = ACTIONS[action_idx]
        log_prob = torch.log(probs[action_idx] + 1e-8).item()
        return action, log_prob

    def compute_reward(self, y_hat_prob: float, y_true: int,
                       struct_stats: Dict, token_cost: int,
                       step_count: int) -> float:
        eps = 1e-8
        y_hat_clamped = max(eps, min(1 - eps, y_hat_prob))
        ce = -(y_true * np.log(y_hat_clamped) + (1 - y_true) * np.log(1 - y_hat_clamped))
        r_pred = -ce

        r_struct = (
            struct_stats.get("high_risk_neighbors", 0) * 0.1 +
            struct_stats.get("density", 0.0) +
            struct_stats.get("shared_neighbors", 0) * 0.05 +
            struct_stats.get("cycles_found", 0) * 0.1
        )

        r_cost = token_cost * 0.01

        reward = r_pred + self.beta * r_struct - self.gamma * r_cost - self.eta * step_count
        return reward

    def sample_trajectory(self, target_node: int, adj: dict,
                          x, y, tools: dict,
                          deterministic: bool = False) -> Tuple[Set[int], List[Dict]]:
        evidence_nodes = {target_node}
        prev_actions = []
        token_estimate = 0
        trajectory = []

        for step in range(self.max_steps):
            state = self.get_state_summary(
                target_node, evidence_nodes, adj, y,
                prev_actions, token_estimate)

            action, log_prob = self.select_action(state, deterministic=deterministic)

            if action == "Stop":
                trajectory.append({
                    "step": step, "action": "Stop", "log_prob": log_prob,
                    "state": state, "state_vec": state_to_vector(state, self.state_dim)
                })
                break

            tool = tools.get(action)
            if tool is None:
                trajectory.append({
                    "step": step, "action": action, "log_prob": log_prob,
                    "state": state, "state_vec": state_to_vector(state, self.state_dim),
                    "info": {"action": action, "nodes_added": 0}
                })
                prev_actions.append(action)
                continue

            if action == "PruneTopK":
                new_nodes, info = tool(evidence_nodes, adj,
                                       x=x, target_node=target_node)
            else:
                new_nodes, info = tool(evidence_nodes, adj,
                                       max_nodes=self.budget_nodes)

            evidence_nodes = new_nodes
            prev_actions.append(action)
            token_estimate = len(evidence_nodes) * 20 + info.get("total_nodes", 0) * 5

            trajectory.append({
                "step": step, "action": action, "log_prob": log_prob,
                "state": state, "state_vec": state_to_vector(state, self.state_dim),
                "info": info, "evidence_nodes": set(evidence_nodes)
            })

        return evidence_nodes, trajectory

    def update(self, trajectories_and_rewards: List[Tuple[List[Dict], float]]):
        if not trajectories_and_rewards:
            return

        rewards = [r for _, r in trajectories_and_rewards]
        mean_reward = np.mean(rewards)
        advantages = [r - mean_reward for r in rewards]

        total_loss = torch.tensor(0.0)
        n_steps = 0

        for (traj, _), advantage in zip(trajectories_and_rewards, advantages):
            adv_tensor = torch.tensor(advantage, dtype=torch.float32)

            for step_info in traj:
                if step_info.get("action") == "Stop" and step_info.get("step", -1) == 0:
                    continue

                state_vec = step_info["state_vec"]
                old_log_prob = step_info["log_prob"]

                logits = self.network(state_vec)
                log_probs = F.log_softmax(logits, dim=-1)
                action_idx = ACTIONS.index(step_info["action"])
                new_log_prob = log_probs[action_idx]

                ratio = torch.exp(new_log_prob - old_log_prob)

                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                obj = torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)
                total_loss = total_loss - obj

                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum()
                total_loss = total_loss - self.entropy_coeff * entropy

                n_steps += 1

        if n_steps > 0:
            total_loss = total_loss / n_steps
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        return total_loss.item() if n_steps > 0 else 0.0
