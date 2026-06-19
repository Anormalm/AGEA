#!/usr/bin/env python
"""Run AGEA experiments — heuristic + GRPO.

Key design:
- GraphSAGE backbone classifier (best GNN baseline)
- Evidence-augmented prediction: MLP fuses classifier prob + structural evidence
- Precomputed adjacency dict for fast graph tool operations
- Supports datasets with unlabeled nodes (y=-1) like FakeNews
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_dataset, GraphData
from graph_tools.expand import Expand1Hop, Expand2Hop
from graph_tools.ppr import PPRTopK
from graph_tools.community import Community
from graph_tools.cycle import ShortCycle
from graph_tools.prune import PruneTopK
from policy.heuristic_policy import HeuristicPolicy
from policy.grpo_policy import GRPOPolicy
from prompts.fraud_prompt import FraudPromptBuilder
from models.classifier import SAGEClassifier
from utils import load_config, compute_metrics, estimate_tokens


class EvidenceFuser(nn.Module):
    """Fuse GNN classifier prob with structural evidence features."""

    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def precompute_adj(edge_index, num_nodes):
    """Build adjacency list: node -> set of neighbors."""
    t0 = time.time()
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    adj = defaultdict(set)
    for i in range(len(src)):
        adj[int(src[i])].add(int(dst[i]))
    adj = dict(adj)
    print(f"  Adjacency built in {time.time()-t0:.1f}s ({len(adj)} nodes with edges)")
    return adj


def compute_evidence_features(prob, evidence_nodes, adj, fraud_labels_np, n_steps):
    """Compute evidence feature vector for a single node."""
    node_set = set(evidence_nodes)
    n_nodes = len(node_set)
    n_edges = sum(len(adj.get(n, set()) & node_set) for n in node_set)
    density = n_edges / max(n_nodes * (n_nodes - 1), 1)
    # Handle y=-1 (unlabeled) nodes: only count nodes with y==1
    high_risk = sum(1 for n in node_set
                    if n < len(fraud_labels_np) and fraud_labels_np[n] == 1)
    high_risk_ratio = high_risk / max(n_nodes, 1)
    fraud_neighbor_ratio = high_risk / max(n_nodes - 1, 1)
    avg_deg = n_edges / max(n_nodes, 1)
    return [
        prob,
        n_nodes / 100.0,
        n_edges / 500.0,
        density,
        high_risk_ratio,
        fraud_neighbor_ratio,
        n_steps / 10.0,
        avg_deg / 50.0,
    ]


def train_classifier(data, device, cfg):
    """Train GraphSAGE backbone classifier."""
    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    model = SAGEClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.get("lr", 1e-4))
    criterion = nn.BCEWithLogitsLoss()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)

    # Mask out unlabeled nodes (y=-1)
    valid = y >= 0
    train_mask = train_mask & valid

    model.train()
    for epoch in range(tcfg.get("epochs", 100)):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: train={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


def train_fuser(fuser, val_features, val_labels, device, epochs=200, lr=1e-3):
    """Train the evidence fuser MLP on validation set."""
    X = torch.tensor(val_features, dtype=torch.float32).to(device)
    Y = torch.tensor(val_labels, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(fuser.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    fuser.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = fuser(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

    fuser.eval()


def run_agea(data, cfg, device, prompt_mode="raw"):
    tools = build_tools(cfg)
    pcfg = cfg.get("policy", {})
    policy = HeuristicPolicy(
        max_steps=pcfg.get("max_steps", 6),
        budget_tokens=pcfg.get("budget_tokens", 2048),
        budget_nodes=pcfg.get("budget_nodes", 50),
    )
    prompt_builder = FraudPromptBuilder(mode=prompt_mode)
    probs_all = train_classifier(data, device, cfg)

    print("  Precomputing adjacency...")
    adj = precompute_adj(data.edge_index, data.num_nodes)
    tools["PPRTopK"].set_adj(adj)
    fraud_labels_np = data.y.numpy()

    # Build safe y for policies (replace -1 with 0 so suspicious neighbor count works)
    y_safe = data.y.clone()
    y_safe[y_safe < 0] = 0

    # Phase 1: Collect evidence
    print("  Collecting evidence for all nodes...")
    all_evidence = {}
    all_train = data.train_mask.nonzero(as_tuple=False).squeeze(-1)
    all_val = data.val_mask.nonzero(as_tuple=False).squeeze(-1)
    all_test = data.test_mask.nonzero(as_tuple=False).squeeze(-1)

    rng = np.random.RandomState(42)
    if len(all_test) > 2000:
        perm = torch.from_numpy(rng.choice(len(all_test), 2000, replace=False))
        eval_test = all_test[perm]
    else:
        eval_test = all_test
    print(f"  Evaluating on {len(eval_test)} test nodes (of {len(all_test)} total)")

    eval_nodes = torch.cat([all_val, eval_test])

    for v in tqdm(eval_nodes, desc="Collecting evidence"):
        v = v.item()
        evidence_nodes, traj = policy.run_episode(v, adj, data.x, y_safe, tools)
        n_steps = len([t for t in traj if t.get("action") != "Stop"])
        prob = probs_all[v].item()
        features = compute_evidence_features(prob, evidence_nodes, adj, fraud_labels_np, n_steps)
        all_evidence[v] = {
            "features": features,
            "prob": prob,
            "label": data.y[v].item(),
            "evidence_nodes": evidence_nodes,
            "traj": traj,
            "n_steps": n_steps,
        }

    # Phase 2: Train evidence fuser
    print("  Training evidence fuser on validation set...")
    val_features = [all_evidence[v.item()]["features"] for v in all_val if v.item() in all_evidence]
    val_labels = [all_evidence[v.item()]["label"] for v in all_val if v.item() in all_evidence]

    print("  Collecting evidence for training nodes...")
    train_subsample = all_train[torch.from_numpy(rng.choice(len(all_train), min(1000, len(all_train)), replace=False))]
    for v in tqdm(train_subsample, desc="Evidence (train)"):
        v = v.item()
        if v in all_evidence:
            continue
        evidence_nodes, traj = policy.run_episode(v, adj, data.x, y_safe, tools)
        n_steps = len([t for t in traj if t.get("action") != "Stop"])
        prob = probs_all[v].item()
        features = compute_evidence_features(prob, evidence_nodes, adj, fraud_labels_np, n_steps)
        all_evidence[v] = {"features": features, "prob": prob, "label": data.y[v].item()}

    train_features = [all_evidence[v.item()]["features"] for v in train_subsample if v.item() in all_evidence]
    train_labels = [all_evidence[v.item()]["label"] for v in train_subsample if v.item() in all_evidence]

    fuser_features = val_features + train_features
    fuser_labels = val_labels + train_labels
    print(f"  Fuser training set: {len(fuser_features)} nodes")

    fuser = EvidenceFuser(input_dim=8, hidden_dim=32).to(device)
    train_fuser(fuser, fuser_features, fuser_labels, device, epochs=300, lr=1e-3)

    # Phase 3: Predict on test set
    print("  Predicting on test set with evidence fuser...")
    y_true, y_prob = [], []
    total_steps, total_nodes, total_tokens = 0, 0, 0
    action_counts = {}

    for v in eval_test:
        v = v.item()
        ev = all_evidence.get(v)
        if ev is None:
            continue
        with torch.no_grad():
            feat = torch.tensor([ev["features"]], dtype=torch.float32).to(device)
            logit = fuser(feat)
            prob = torch.sigmoid(logit).item()

        y_prob.append(prob)
        y_true.append(ev["label"])
        total_steps += ev["n_steps"]
        total_nodes += len(ev.get("evidence_nodes", set()))
        prompt = prompt_builder.build(
            v, ev.get("evidence_nodes", set()), data.edge_index, data.x, data.y,
            struct_stats={"high_risk_neighbors": 0, "density": 0},
            budget_info={"tokens": 0, "nodes": 0, "edges": 0},
            adj=adj)
        total_tokens += estimate_tokens(prompt)
        for t in ev.get("traj", []):
            a = t.get("action", "Stop")
            action_counts[a] = action_counts.get(a, 0) + 1

    n = len(y_true)
    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
    metrics["avg_tokens"] = total_tokens / max(n, 1)
    metrics["avg_nodes"] = total_nodes / max(n, 1)
    metrics["avg_steps"] = total_steps / max(n, 1)
    metrics["action_dist"] = action_counts
    return metrics


def run_grpo(data, cfg, device):
    tools = build_tools(cfg)
    tcfg = cfg.get("training", {})
    pcfg = cfg.get("policy", {})
    reward_cfg = cfg.get("reward", {})
    grpo_cfg = cfg.get("grpo", {})

    probs_all = train_classifier(data, device, cfg)

    policy = GRPOPolicy(
        state_dim=16, hidden_dim=tcfg.get("hidden_dim", 128),
        K=grpo_cfg.get("K", 4), clip_eps=grpo_cfg.get("clip_eps", 0.2),
        entropy_coeff=grpo_cfg.get("entropy_coeff", 0.01), lr=tcfg.get("lr", 1e-4),
        max_steps=pcfg.get("max_steps", 6), budget_tokens=pcfg.get("budget_tokens", 2048),
        budget_nodes=pcfg.get("budget_nodes", 50),
        beta=reward_cfg.get("beta", 0.1), gamma=reward_cfg.get("gamma", 0.001), eta=reward_cfg.get("eta", 0.01),
    )

    train_indices = data.train_mask.nonzero(as_tuple=False).squeeze(-1)[:256]
    fraud_labels_np = data.y.numpy()

    # Build safe y (replace -1 with 0)
    y_safe = data.y.clone()
    y_safe[y_safe < 0] = 0

    print("  Precomputing adjacency...")
    adj = precompute_adj(data.edge_index, data.num_nodes)
    tools["PPRTopK"].set_adj(adj)

    # GRPO training
    n_epochs = 20
    print("\n  Training GRPO policy...")
    for epoch in range(n_epochs):
        epoch_rewards = []
        for v in train_indices:
            v = v.item()
            trajectories_and_rewards = []
            for _ in range(policy.K):
                evidence_nodes, traj = policy.sample_trajectory(v, adj, data.x, y_safe, tools)
                prob = probs_all[v].item()
                label = data.y[v].item()
                node_set = set(evidence_nodes)
                n_edges = sum(len(adj.get(n, set()) & node_set) for n in node_set)
                n_nodes = len(node_set)
                density = n_edges / max(n_nodes * (n_nodes - 1), 1)
                high_risk = sum(1 for n in node_set if n < len(fraud_labels_np) and fraud_labels_np[n] == 1)
                struct_stats = {"high_risk_neighbors": high_risk, "density": density}
                token_cost = len(evidence_nodes) * 20
                step_count = len([t for t in traj if t.get("action") != "Stop"])
                reward = policy.compute_reward(prob, label, struct_stats, token_cost, step_count)
                trajectories_and_rewards.append((traj, reward))
                epoch_rewards.append(reward)
            policy.update(trajectories_and_rewards)
        if (epoch + 1) % 5 == 0:
            print(f"    GRPO epoch {epoch+1}: avg_reward={np.mean(epoch_rewards):.4f}")

    # Collect evidence and train fuser
    print("  Collecting evidence for GRPO evaluation...")
    all_val = data.val_mask.nonzero(as_tuple=False).squeeze(-1)
    all_test = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    rng = np.random.RandomState(42)
    if len(all_test) > 2000:
        perm = torch.from_numpy(rng.choice(len(all_test), 2000, replace=False))
        eval_test = all_test[perm]
    else:
        eval_test = all_test
    print(f"  Evaluating on {len(eval_test)} test nodes")

    all_evidence = {}
    eval_nodes = torch.cat([all_val, eval_test])

    for v in tqdm(eval_nodes, desc="GRPO evidence"):
        v = v.item()
        evidence_nodes, traj = policy.sample_trajectory(v, adj, data.x, y_safe, tools, deterministic=True)
        n_steps = len([t for t in traj if t.get("action") != "Stop"])
        prob = probs_all[v].item()
        features = compute_evidence_features(prob, evidence_nodes, adj, fraud_labels_np, n_steps)
        all_evidence[v] = {"features": features, "prob": prob, "label": data.y[v].item(),
                           "evidence_nodes": evidence_nodes, "traj": traj, "n_steps": n_steps}

    val_features = [all_evidence[v.item()]["features"] for v in all_val if v.item() in all_evidence]
    val_labels = [all_evidence[v.item()]["label"] for v in all_val if v.item() in all_evidence]
    train_sub = train_indices[torch.from_numpy(rng.choice(len(train_indices), min(256, len(train_indices)), replace=False))]
    for v in train_sub:
        v = v.item()
        if v in all_evidence:
            continue
        evidence_nodes, traj = policy.sample_trajectory(v, adj, data.x, y_safe, tools, deterministic=True)
        n_steps = len([t for t in traj if t.get("action") != "Stop"])
        prob = probs_all[v].item()
        features = compute_evidence_features(prob, evidence_nodes, adj, fraud_labels_np, n_steps)
        all_evidence[v] = {"features": features, "prob": prob, "label": data.y[v].item()}

    tr_feat = [all_evidence[v.item()]["features"] for v in train_sub if v.item() in all_evidence]
    tr_lab = [all_evidence[v.item()]["label"] for v in train_sub if v.item() in all_evidence]

    fuser = EvidenceFuser(input_dim=8, hidden_dim=32).to(device)
    train_fuser(fuser, val_features + tr_feat, val_labels + tr_lab, device, epochs=300, lr=1e-3)

    # Evaluate
    y_true, y_prob = [], []
    total_nodes, total_tokens = 0, 0
    action_counts = {}

    for v in eval_test:
        v = v.item()
        ev = all_evidence.get(v)
        if ev is None:
            continue
        with torch.no_grad():
            feat = torch.tensor([ev["features"]], dtype=torch.float32).to(device)
            logit = fuser(feat)
            prob = torch.sigmoid(logit).item()
        y_prob.append(prob)
        y_true.append(ev["label"])
        total_nodes += len(ev.get("evidence_nodes", set()))
        for t in ev.get("traj", []):
            a = t.get("action", "Stop")
            action_counts[a] = action_counts.get(a, 0) + 1

    n = len(y_true)
    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
    metrics["avg_nodes"] = total_nodes / max(n, 1)
    metrics["avg_tokens"] = total_tokens / max(n, 1)
    metrics["action_dist"] = action_counts
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yelp_spam",
                        choices=["yelp_spam", "amazon", "fakenews_politifact", "fakenews_buzzfeed"])
    parser.add_argument("--mode", default="all",
                        choices=["all", "heuristic_raw", "heuristic_comp", "grpo"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_path = f"configs/{args.dataset}.yaml"
    cfg = load_config(cfg_path)
    root = cfg["dataset"].get("root")
    print(f"Loading {args.dataset} from {root}")
    dataset = load_dataset(args.dataset, root)
    data = dataset.data
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")

    # Print fraud ratio only on labeled nodes
    valid = data.y >= 0
    fraud_ratio = data.y[valid].float().mean().item()
    print(f"Fraud ratio (labeled): {fraud_ratio:.3f}")
    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    print(f"Test set size: {len(test_indices)}")

    results = {}

    if args.mode in ("all", "heuristic_raw"):
        print(f"\n=== AGEA heuristic (raw) ===")
        results["AGEA-heuristic-raw"] = run_agea(data, cfg, device, "raw")
        m = results["AGEA-heuristic-raw"]
        print(f"  MacroF1={m['macro_f1']:.4f}, AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}")
        print(f"  AvgNodes={m['avg_nodes']:.1f}, AvgSteps={m['avg_steps']:.2f}")
        print(f"  Action dist: {m['action_dist']}")

    if args.mode in ("all", "heuristic_comp"):
        print(f"\n=== AGEA heuristic (compressed) ===")
        results["AGEA-heuristic-comp"] = run_agea(data, cfg, device, "compressed")
        m = results["AGEA-heuristic-comp"]
        print(f"  MacroF1={m['macro_f1']:.4f}, AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}")
        print(f"  AvgNodes={m['avg_nodes']:.1f}, AvgSteps={m['avg_steps']:.2f}")

    if args.mode in ("all", "grpo"):
        print(f"\n=== AGEA GRPO ===")
        results["AGEA-GRPO"] = run_grpo(data, cfg, device)
        m = results["AGEA-GRPO"]
        print(f"  MacroF1={m['macro_f1']:.4f}, AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}")
        print(f"  AvgNodes={m['avg_nodes']:.1f}")
        print(f"  Action dist: {m['action_dist']}")

    print(f"\n{'='*70}")
    print(f"AGEA RESULTS: {args.dataset}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'MacroF1':>8} {'AUROC':>8} {'AUPRC':>8} {'AvgNodes':>10}")
    print(f"{'-'*70}")
    for name, m in results.items():
        print(f"{name:<25} {m['macro_f1']:>8.4f} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['avg_nodes']:>10.1f}")


if __name__ == "__main__":
    main()
