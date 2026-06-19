#!/usr/bin/env python
"""Run all AGEA experiments on real Yelp spam data — full graph, no limit."""

import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
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
from models.classifier import (
    MLPClassifier, GCNClassifier, SAGEClassifier, GATClassifier,
    HGTClassifier, PMPClassifier, ConsisGADClassifier, GAAPClassifier,
)
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


def _all_test_indices(data):
    """Return all test indices — no sampling, no limit."""
    test_mask = data.test_mask
    return test_mask.nonzero(as_tuple=False).squeeze(-1)


def _fast_subgraph_stats(evidence_nodes, edge_index, fraud_labels_np):
    """Compute structural stats using tensors instead of nx.DiGraph."""
    if not evidence_nodes:
        return 0, {"high_risk_neighbors": 0, "density": 0.0, "shared_neighbors": 0, "cycles_found": 0}

    node_set = set(evidence_nodes)
    n_nodes = len(node_set)

    src, dst = edge_index[0], edge_index[1]
    node_tensor = torch.tensor(list(node_set), dtype=torch.long)
    max_node = int(edge_index.max().item()) + 1
    in_evidence = torch.zeros(max_node, dtype=torch.bool)
    in_evidence[node_tensor] = True
    mask = in_evidence[src] & in_evidence[dst]
    n_edges = int(mask.sum().item())

    high_risk = sum(1 for n in node_set if n < len(fraud_labels_np) and fraud_labels_np[n] == 1)
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    shared_neighbors = 0
    if 1 < n_nodes <= 200:
        sub_src = src[mask].numpy()
        sub_dst = dst[mask].numpy()
        adj = {n: set() for n in node_set}
        for s, d in zip(sub_src, sub_dst):
            if s in adj:
                adj[s].add(d)
        nodes_list = list(node_set)
        lim = min(len(nodes_list), 50)
        for i in range(lim):
            for j in range(i + 1, lim):
                if adj[nodes_list[i]] & adj[nodes_list[j]]:
                    shared_neighbors += 1

    struct_stats = {
        "high_risk_neighbors": high_risk,
        "density": density,
        "shared_neighbors": shared_neighbors,
        "cycles_found": 0,
    }
    return n_edges, struct_stats


def train_classifier(model, data, device, epochs=100, lr=1e-4):
    """Standard training loop for MLP/GCN/SAGE/GAT/PMP."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)
    edge_type = data.edge_type.to(device) if "edge_type" in data else None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, MLPClassifier):
            out = model(x)
        elif isinstance(model, HGTClassifier) and edge_type is not None:
            out = model(x, edge_index, edge_type)
        else:
            out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                val_mask = data.val_mask.to(device)
                val_loss = criterion(out[val_mask], y[val_mask])
                print(f"    Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        if isinstance(model, MLPClassifier):
            logits = model(x)
        elif isinstance(model, HGTClassifier) and edge_type is not None:
            logits = model(x, edge_index, edge_type)
        else:
            logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


def train_classifier_consistency(model, data, device, epochs=100, lr=1e-4):
    """Training loop for ConsisGAD/GAAP: BCE + consistency between two augmented views."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)
    lam = model.lam

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss_bce = bce(out[train_mask], y[train_mask])

        out_v1 = model.forward_view1(x, edge_index)
        out_v2 = model.forward_view2(x, edge_index)
        loss_cons = mse(torch.sigmoid(out_v1), torch.sigmoid(out_v2))

        loss = loss_bce + lam * loss_cons
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                val_mask = data.val_mask.to(device)
                val_loss = bce(out[val_mask], y[val_mask])
                print(f"    Epoch {epoch+1}: train={loss.item():.4f} (bce={loss_bce.item():.4f}, cons={loss_cons.item():.4f}), val={val_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


def run_gnn_baselines(data, cfg, device):
    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    results = {}

    models = {
        "MLP": MLPClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "GCN": GCNClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "GraphSAGE": SAGEClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "GAT": GATClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "HGT": HGTClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "PMP": PMPClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "ConsisGAD": ConsisGADClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "GAAP": GAAPClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
    }

    test_indices = _all_test_indices(data)
    y_true = data.y[test_indices].numpy()

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model = model.to(device)
        is_consistency = isinstance(model, (ConsisGADClassifier, GAAPClassifier))
        if is_consistency:
            probs = train_classifier_consistency(model, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))
        else:
            probs = train_classifier(model, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))
        y_prob = probs[test_indices].numpy()
        metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
        results[name] = metrics
        print(f"    {name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, MacroF1={metrics['macro_f1']:.4f}")
    return results


def run_agea(data, cfg, device, policy_type="heuristic", prompt_mode="raw"):
    tools = build_tools(cfg)
    pcfg = cfg.get("policy", {})
    policy = HeuristicPolicy(
        max_steps=pcfg.get("max_steps", 6),
        budget_tokens=pcfg.get("budget_tokens", 2048),
        budget_nodes=pcfg.get("budget_nodes", 50),
    )
    prompt_builder = FraudPromptBuilder(mode=prompt_mode)

    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    classifier = GCNClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)).to(device)
    probs_all = train_classifier(classifier, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))

    test_indices = _all_test_indices(data)
    fraud_labels_np = data.y.numpy()

    y_true, y_prob = [], []
    action_counts = {}
    total_steps, total_nodes, total_edges, total_tokens = 0, 0, 0, 0
    struct_agg = {"high_risk_neighbors": 0, "density": 0.0, "shared_neighbors": 0, "cycles_found": 0}
    latencies = []

    for v in tqdm(test_indices, desc=f"AGEA-{policy_type}-{prompt_mode}"):
        t0 = time.time()
        evidence_nodes, traj = policy.run_episode(v.item(), data.edge_index, data.x, data.y, tools)
        lat = time.time() - t0
        latencies.append(lat)

        n_edges, struct_stats = _fast_subgraph_stats(evidence_nodes, data.edge_index, fraud_labels_np)

        prompt = prompt_builder.build(
            v.item(), evidence_nodes, data.edge_index, data.x, data.y,
            struct_stats=struct_stats,
            budget_info={"tokens": len(evidence_nodes) * 20, "nodes": len(evidence_nodes), "edges": n_edges})

        prob = probs_all[v].item()
        y_prob.append(prob)
        y_true.append(data.y[v].item())
        steps = len([t for t in traj if t.get("action") != "Stop"])
        total_steps += steps
        total_nodes += len(evidence_nodes)
        total_edges += n_edges
        total_tokens += estimate_tokens(prompt)
        for t in traj:
            a = t.get("action", "Stop")
            action_counts[a] = action_counts.get(a, 0) + 1
        for k in struct_agg:
            struct_agg[k] += struct_stats.get(k, 0)

    n = len(test_indices)
    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
    metrics["avg_tokens"] = total_tokens / max(n, 1)
    metrics["avg_nodes"] = total_nodes / max(n, 1)
    metrics["avg_edges"] = total_edges / max(n, 1)
    metrics["avg_latency"] = np.mean(latencies)
    metrics["avg_steps"] = total_steps / max(n, 1)
    metrics["avg_high_risk"] = struct_agg["high_risk_neighbors"] / max(n, 1)
    metrics["avg_density"] = struct_agg["density"] / max(n, 1)
    metrics["action_dist"] = action_counts
    return metrics


def run_fixed_baselines(data, cfg, device):
    tools = build_tools(cfg)
    in_dim = data.num_features
    tcfg = cfg.get("training", {})

    classifier = GCNClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)).to(device)
    probs_all = train_classifier(classifier, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))

    test_indices = _all_test_indices(data)

    results = {}
    strategies = {
        "1hop": lambda en, ei, n: tools["Expand1Hop"](en, ei, n, cfg["policy"].get("budget_nodes", 50)),
        "2hop": lambda en, ei, n: tools["Expand2Hop"](en, ei, n, cfg["policy"].get("budget_nodes", 50)),
        "ppr_topk": lambda en, ei, n: tools["PPRTopK"](en, ei, n, cfg["policy"].get("budget_nodes", 50)),
    }

    for sname, sfn in strategies.items():
        y_true, y_prob = [], []
        total_nodes = 0
        for v in tqdm(test_indices, desc=f"Fixed-{sname}"):
            evidence = {v.item()}
            evidence, _ = sfn(evidence, data.edge_index, data.x.size(0))
            prob = probs_all[v].item()
            y_prob.append(prob)
            y_true.append(data.y[v].item())
            total_nodes += len(evidence)
        n = len(test_indices)
        metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
        metrics["avg_nodes"] = total_nodes / max(n, 1)
        results[sname] = metrics
    return results


def run_grpo(data, cfg, device):
    tools = build_tools(cfg)
    tcfg = cfg.get("training", {})
    pcfg = cfg.get("policy", {})
    reward_cfg = cfg.get("reward", {})
    grpo_cfg = cfg.get("grpo", {})

    in_dim = data.num_features
    classifier = GCNClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)).to(device)
    probs_all = train_classifier(classifier, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))

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
    n_epochs = 20

    print("\n  Training GRPO policy...")
    for epoch in range(n_epochs):
        epoch_rewards = []
        for v in train_indices:
            v = v.item()
            trajectories_and_rewards = []
            for _ in range(policy.K):
                evidence_nodes, traj = policy.sample_trajectory(v, data.edge_index, data.x, data.y, tools)
                prob = probs_all[v].item()
                label = data.y[v].item()
                n_edges, struct_stats = _fast_subgraph_stats(evidence_nodes, data.edge_index, fraud_labels_np)
                token_cost = len(evidence_nodes) * 20
                step_count = len([t for t in traj if t.get("action") != "Stop"])
                reward = policy.compute_reward(prob, label, struct_stats, token_cost, step_count)
                trajectories_and_rewards.append((traj, reward))
                epoch_rewards.append(reward)
            policy.update(trajectories_and_rewards)
        if (epoch + 1) % 5 == 0:
            print(f"    GRPO epoch {epoch+1}: avg_reward={np.mean(epoch_rewards):.4f}")

    test_indices = _all_test_indices(data)

    y_true, y_prob = [], []
    total_nodes, total_edges, total_tokens = 0, 0, 0
    action_counts = {}

    for v in tqdm(test_indices, desc="AGEA-GRPO-raw"):
        evidence_nodes, traj = policy.sample_trajectory(v.item(), data.edge_index, data.x, data.y, tools, deterministic=True)
        n_edges, struct_stats = _fast_subgraph_stats(evidence_nodes, data.edge_index, fraud_labels_np)
        prob = probs_all[v].item()
        y_prob.append(prob)
        y_true.append(data.y[v].item())
        total_nodes += len(evidence_nodes)
        total_edges += n_edges
        total_tokens += len(evidence_nodes) * 20
        for t in traj:
            a = t.get("action", "Stop")
            action_counts[a] = action_counts.get(a, 0) + 1

    n = len(test_indices)
    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
    metrics["avg_tokens"] = total_tokens / max(n, 1)
    metrics["avg_nodes"] = total_nodes / max(n, 1)
    metrics["avg_edges"] = total_edges / max(n, 1)
    metrics["action_dist"] = action_counts
    return metrics


def run_dataset(name, cfg, device):
    """Run full experiment pipeline for one dataset — entire test set."""
    root = cfg["dataset"].get("root")
    print(f"\n  Loading dataset: {name} from {root}")
    dataset = load_dataset(name, root)
    data = dataset.data
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")
    print(f"  Fraud ratio: {data.y.float().mean().item():.3f}")

    test_indices = _all_test_indices(data)
    print(f"  Test set size: {len(test_indices)}")

    # GNN baselines
    print(f"\n  Training GNN baselines...")
    gnn_results = run_gnn_baselines(data, cfg, device)

    # Fixed retrieval
    print(f"\n  Running fixed-retrieval baselines...")
    fixed_results = run_fixed_baselines(data, cfg, device)

    # AGEA heuristic raw
    print(f"\n  Running AGEA heuristic (raw)...")
    agea_raw = run_agea(data, cfg, device, "heuristic", "raw")

    # AGEA heuristic compressed
    print(f"\n  Running AGEA heuristic (compressed)...")
    agea_comp = run_agea(data, cfg, device, "heuristic", "compressed")

    # AGEA GRPO
    print(f"\n  Running AGEA GRPO...")
    agea_grpo = run_grpo(data, cfg, device)

    # Print results
    print(f"\n{'='*100}")
    print(f"RESULTS: {name}")
    print(f"{'='*100}")
    print(f"{'Method':<30} {'MacroF1':>8} {'AUROC':>8} {'AUPRC':>8} {'AvgNodes':>10} {'AvgTokens':>10}")
    print(f"{'-'*100}")

    for mname, m in gnn_results.items():
        print(f"{mname:<30} {m['macro_f1']:>8.4f} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {'N/A':>10} {'N/A':>10}")
    for mname, m in fixed_results.items():
        print(f"Fixed-{mname:<24} {m['macro_f1']:>8.4f} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m.get('avg_nodes',0):>10.1f} {'N/A':>10}")
    print(f"{'AGEA (heuristic, raw)':<30} {agea_raw['macro_f1']:>8.4f} {agea_raw['auroc']:>8.4f} {agea_raw['auprc']:>8.4f} {agea_raw['avg_nodes']:>10.1f} {agea_raw['avg_tokens']:>10.1f}")
    print(f"{'AGEA (heuristic, comp)':<30} {agea_comp['macro_f1']:>8.4f} {agea_comp['auroc']:>8.4f} {agea_comp['auprc']:>8.4f} {agea_comp['avg_nodes']:>10.1f} {agea_comp['avg_tokens']:>10.1f}")
    print(f"{'AGEA (GRPO, raw)':<30} {agea_grpo['macro_f1']:>8.4f} {agea_grpo['auroc']:>8.4f} {agea_grpo['auprc']:>8.4f} {agea_grpo['avg_nodes']:>10.1f} {agea_grpo['avg_tokens']:>10.1f}")
    print(f"{'-'*100}")

    print(f"\n  Evidence & Policy (AGEA heuristic, raw):")
    print(f"    Avg high-risk neighbors: {agea_raw['avg_high_risk']:.2f}")
    print(f"    Avg density: {agea_raw['avg_density']:.3f}")
    print(f"    Avg steps: {agea_raw['avg_steps']:.2f}")
    print(f"    Action dist: {agea_raw['action_dist']}")

    return {
        "gnn": gnn_results, "fixed": fixed_results,
        "agea_raw": agea_raw, "agea_comp": agea_comp, "agea_grpo": agea_grpo,
    }


def main():
    print("=" * 70)
    print("AGEA: Adaptive Graph Evidence Acquisition — Full Graph Experiments")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run on Yelp Spam data
    cfg_yelp = load_config("configs/yelp_spam.yaml")
    yelp_results = run_dataset("yelp_spam", cfg_yelp, device)

    # Try Amazon if data exists
    amazon_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "amazon")
    if os.path.exists(amazon_root) and any(f.endswith(".jsonl.gz") for f in os.listdir(amazon_root) if not f.startswith("meta_")):
        cfg_amazon = load_config("configs/amazon.yaml")
        amazon_results = run_dataset("amazon", cfg_amazon, device)
    else:
        print("\n[AGEA] No Amazon data found at dataset/amazon/. Skipping.")
        print("  To add Amazon data, download .jsonl.gz files from:")
        print("  https://amazon-reviews-2023.github.io/")
        print("  Place them in dataset/amazon/")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
