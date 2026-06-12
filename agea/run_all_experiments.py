#!/usr/bin/env python
"""Run all AGEA experiments on real Yelp spam data."""

import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
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
from models.classifier import MLPClassifier, GCNClassifier, SAGEClassifier, GATClassifier
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


def train_classifier(model, data, device, epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, MLPClassifier):
            out = model(x)
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
        else:
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
    }

    test_mask = data.test_mask
    test_indices = test_mask.nonzero(as_tuple=False).squeeze(-1)
    y_true = data.y[test_indices].numpy()

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model = model.to(device)
        probs = train_classifier(model, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))
        y_prob = probs[test_indices].numpy()
        metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
        results[name] = metrics
        print(f"    {name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")
    return results


def run_agea(data, cfg, device, policy_type="heuristic", prompt_mode="raw", limit=500):
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

    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    if limit > 0 and len(test_indices) > limit:
        test_indices = test_indices[:limit]

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

        G_ev = nx.DiGraph()
        G_ev.add_nodes_from(evidence_nodes)
        src, dst = data.edge_index
        for s, d in zip(src, dst):
            if s.item() in evidence_nodes and d.item() in evidence_nodes:
                G_ev.add_edge(s.item(), d.item())
        struct_stats = compute_structural_reward(G_ev, None, data.y.numpy())

        n_edges = G_ev.number_of_edges()
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


def run_fixed_baselines(data, cfg, device, limit=500):
    tools = build_tools(cfg)
    in_dim = data.num_features
    tcfg = cfg.get("training", {})

    classifier = GCNClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)).to(device)
    probs_all = train_classifier(classifier, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))

    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    if limit > 0 and len(test_indices) > limit:
        test_indices = test_indices[:limit]

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


def run_grpo(data, cfg, device, limit=500):
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
                G_ev = nx.DiGraph()
                G_ev.add_nodes_from(evidence_nodes)
                src, dst = data.edge_index
                for s, d in zip(src, dst):
                    if s.item() in evidence_nodes and d.item() in evidence_nodes:
                        G_ev.add_edge(s.item(), d.item())
                struct_stats = compute_structural_reward(G_ev, None, data.y.numpy())
                token_cost = len(evidence_nodes) * 20
                step_count = len([t for t in traj if t.get("action") != "Stop"])
                reward = policy.compute_reward(prob, label, struct_stats, token_cost, step_count)
                trajectories_and_rewards.append((traj, reward))
                epoch_rewards.append(reward)
            policy.update(trajectories_and_rewards)
        if (epoch + 1) % 5 == 0:
            print(f"    GRPO epoch {epoch+1}: avg_reward={np.mean(epoch_rewards):.4f}")

    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    if limit > 0 and len(test_indices) > limit:
        test_indices = test_indices[:limit]

    y_true, y_prob = [], []
    total_nodes, total_edges, total_tokens = 0, 0, 0
    action_counts = {}

    for v in tqdm(test_indices, desc="AGEA-GRPO-raw"):
        evidence_nodes, traj = policy.sample_trajectory(v.item(), data.edge_index, data.x, data.y, tools, deterministic=True)
        G_ev = nx.DiGraph()
        G_ev.add_nodes_from(evidence_nodes)
        src, dst = data.edge_index
        for s, d in zip(src, dst):
            if s.item() in evidence_nodes and d.item() in evidence_nodes:
                G_ev.add_edge(s.item(), d.item())
        n_edges = G_ev.number_of_edges()
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
    """Run full experiment pipeline for one dataset."""
    root = cfg["dataset"].get("root")
    print(f"\n  Loading dataset: {name} from {root}")
    dataset = load_dataset(name, root)
    data = dataset.data
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")
    print(f"  Fraud ratio: {data.y.float().mean().item():.3f}")

    # GNN baselines
    print(f"\n  Training GNN baselines...")
    gnn_results = run_gnn_baselines(data, cfg, device)

    # Fixed retrieval
    print(f"\n  Running fixed-retrieval baselines...")
    fixed_results = run_fixed_baselines(data, cfg, device, limit=500)

    # AGEA heuristic raw
    print(f"\n  Running AGEA heuristic (raw)...")
    agea_raw = run_agea(data, cfg, device, "heuristic", "raw", limit=500)

    # AGEA heuristic compressed
    print(f"\n  Running AGEA heuristic (compressed)...")
    agea_comp = run_agea(data, cfg, device, "heuristic", "compressed", limit=500)

    # AGEA GRPO
    print(f"\n  Running AGEA GRPO...")
    agea_grpo = run_grpo(data, cfg, device, limit=500)

    # Print results
    print(f"\n{'='*90}")
    print(f"RESULTS: {name}")
    print(f"{'='*90}")
    print(f"{'Method':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'AvgNodes':>10} {'AvgTokens':>10}")
    print(f"{'-'*90}")

    for mname, m in gnn_results.items():
        print(f"{mname:<30} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} {'N/A':>10} {'N/A':>10}")
    for mname, m in fixed_results.items():
        print(f"Fixed-{mname:<24} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} {m.get('avg_nodes',0):>10.1f} {'N/A':>10}")
    print(f"{'AGEA (heuristic, raw)':<30} {agea_raw['auroc']:>8.4f} {agea_raw['auprc']:>8.4f} {agea_raw['f1']:>8.4f} {agea_raw['avg_nodes']:>10.1f} {agea_raw['avg_tokens']:>10.1f}")
    print(f"{'AGEA (heuristic, comp)':<30} {agea_comp['auroc']:>8.4f} {agea_comp['auprc']:>8.4f} {agea_comp['f1']:>8.4f} {agea_comp['avg_nodes']:>10.1f} {agea_comp['avg_tokens']:>10.1f}")
    print(f"{'AGEA (GRPO, raw)':<30} {agea_grpo['auroc']:>8.4f} {agea_grpo['auprc']:>8.4f} {agea_grpo['f1']:>8.4f} {agea_grpo['avg_nodes']:>10.1f} {agea_grpo['avg_tokens']:>10.1f}")
    print(f"{'-'*90}")

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
    print("AGEA: Adaptive Graph Evidence Acquisition — Real Data Experiments")
    print("=" * 70)

    device = torch.device("cpu")

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
