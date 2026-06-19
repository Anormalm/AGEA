"""GRPO training for learnable AGEA policy."""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm

from data.loader import load_dataset
from graph_tools.expand import Expand1Hop, Expand2Hop
from graph_tools.ppr import PPRTopK
from graph_tools.community import Community
from graph_tools.cycle import ShortCycle
from graph_tools.prune import PruneTopK
from policy.grpo_policy import GRPOPolicy
from prompts.fraud_prompt import FraudPromptBuilder
from models.classifier import GCNClassifier
from utils import load_config, compute_metrics, compute_structural_reward, estimate_tokens


def build_tools(cfg):
    gt = cfg.get("graph_tools", {})
    return {
        "Expand1Hop": Expand1Hop(max_neighbors=gt.get("expand_max_neighbors", 20)),
        "Expand2Hop": Expand2Hop(max_neighbors=gt.get("expand_max_neighbors", 20)),
        "PPRTopK": PPRTopK(alpha=gt.get("ppr_alpha", 0.15), topk=gt.get("ppr_topk", 10)),
        "Community": Community(resolution=gt.get("community_resolution", 1.0)),
        "ShortCycle": ShortCycle(cycle_length=gt.get("cycle_length", 3)),
        "PruneTopK": PruneTopK(topk=gt.get("prune_topk", 10)),
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO training for AGEA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="agea_grpo_policy.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[AGEA-GRPO] Loading dataset: {cfg['dataset']['name']}")
    dataset = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("root"))
    data = dataset.data

    tools = build_tools(cfg)

    # Train a GNN classifier as prediction proxy
    print("[AGEA-GRPO] Training GNN classifier as prediction proxy...")
    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    classifier = GCNClassifier(
        in_dim=in_dim,
        hidden_dim=tcfg.get("hidden_dim", 128),
        num_layers=tcfg.get("num_layers", 2),
        dropout=tcfg.get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=tcfg.get("lr", 1e-4))
    criterion = torch.nn.BCEWithLogitsLoss()
    classifier.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)

    for epoch in range(tcfg.get("epochs", 100)):
        optimizer.zero_grad()
        out = classifier(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    print("  GNN classifier trained.")

    # Get predictions
    classifier.eval()
    with torch.no_grad():
        all_logits = classifier(x, edge_index).cpu()
        all_probs = torch.sigmoid(all_logits)

    # Initialize GRPO policy
    grpo_cfg = cfg.get("grpo", {})
    reward_cfg = cfg.get("reward", {})
    pcfg = cfg.get("policy", {})
    prompt_builder = FraudPromptBuilder(mode=cfg.get("prompt", {}).get("mode", "raw"))

    policy = GRPOPolicy(
        state_dim=16,
        hidden_dim=tcfg.get("hidden_dim", 128),
        K=grpo_cfg.get("K", 4),
        clip_eps=grpo_cfg.get("clip_eps", 0.2),
        entropy_coeff=grpo_cfg.get("entropy_coeff", 0.01),
        lr=tcfg.get("lr", 1e-4),
        max_steps=pcfg.get("max_steps", 6),
        budget_tokens=pcfg.get("budget_tokens", 2048),
        budget_nodes=pcfg.get("budget_nodes", 50),
        beta=reward_cfg.get("beta", 0.1),
        gamma=reward_cfg.get("gamma", 0.001),
        eta=reward_cfg.get("eta", 0.01),
    ).to(device)

    # Training loop
    train_indices = data.train_mask.nonzero(as_tuple=False).squeeze(-1)
    n_epochs = args.epochs or grpo_cfg.get("epochs", 50)
    batch_size = tcfg.get("batch_size", 64)

    print(f"[AGEA-GRPO] Starting GRPO training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        perm = torch.randperm(train_indices.size(0))
        epoch_rewards = []

        for batch_start in range(0, min(train_indices.size(0), 256), batch_size):
            batch_idx = train_indices[perm[batch_start:batch_start + batch_size]]
            trajectories_and_rewards = []

            for v in batch_idx:
                v = v.item()
                # Sample K trajectories
                trajectories = []
                for _ in range(policy.K):
                    evidence_nodes, traj = policy.sample_trajectory(
                        v, data.edge_index, data.x, data.y, tools)
                    trajectories.append((evidence_nodes, traj))

                # Evaluate each trajectory
                for evidence_nodes, traj in trajectories:
                    # Prediction reward
                    prob = all_probs[v].item()
                    label = data.y[v].item()
                    r_pred = -(-(label * np.log(max(prob, 1e-8)) +
                                 (1 - label) * np.log(max(1 - prob, 1e-8))))

                    # Structural reward
                    G_ev = nx.DiGraph()
                    G_ev.add_nodes_from(evidence_nodes)
                    src, dst = data.edge_index
                    for s, d in zip(src, dst):
                        if s.item() in evidence_nodes and d.item() in evidence_nodes:
                            G_ev.add_edge(s.item(), d.item())
                    struct_stats = compute_structural_reward(G_ev, None, data.y.numpy())

                    # Token cost
                    token_cost = len(evidence_nodes) * 20
                    step_count = len([t for t in traj if t.get("action") != "Stop"])

                    reward = policy.compute_reward(
                        prob, label, struct_stats, token_cost, step_count)

                    trajectories_and_rewards.append((traj, reward))
                    epoch_rewards.append(reward)

            # GRPO update
            policy.update(trajectories_and_rewards)

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: avg_reward={avg_reward:.4f}")

    # Save policy
    torch.save(policy.network.state_dict(), args.save_path)
    print(f"[AGEA-GRPO] Policy saved to {args.save_path}")

    # Quick evaluation
    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)[:100]
    y_true, y_prob = [], []
    for v in tqdm(test_indices, desc="GRPO eval"):
        evidence_nodes, traj = policy.sample_trajectory(
            v.item(), data.edge_index, data.x, data.y, tools, deterministic=True)
        prob = all_probs[v].item()
        y_prob.append(prob)
        y_true.append(data.y[v].item())

    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
    print("\nGRPO Policy Evaluation (sample):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
