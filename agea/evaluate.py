"""Evaluation script for AGEA and all baselines."""

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
from graph_tools.topk import TopKSimilar
from policy.heuristic_policy import HeuristicPolicy
from policy.llm_policy import LLMPolicy
from policy.grpo_policy import GRPOPolicy
from prompts.fraud_prompt import FraudPromptBuilder
from models.classifier import MLPClassifier, GCNClassifier, SAGEClassifier, GATClassifier
from models.reasoner import LLMReasoner
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
    """Train a classifier and return predictions."""
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

    model.eval()
    with torch.no_grad():
        if isinstance(model, MLPClassifier):
            logits = model(x)
        else:
            logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


def run_gnn_baselines(data, cfg, device):
    """Train and evaluate GNN baselines."""
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
        print(f"  Training {name}...")
        model = model.to(device)
        probs = train_classifier(model, data, device,
                                 epochs=tcfg.get("epochs", 100),
                                 lr=tcfg.get("lr", 1e-4))
        y_prob = probs[test_indices].numpy()
        metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
        results[name] = metrics
        print(f"    {name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")

    return results


def run_agea_evaluation(data, cfg, device, policy_type="heuristic",
                        prompt_mode="raw", grpo_path=None, limit=-1):
    """Run full AGEA evaluation with evidence acquisition."""
    tools = build_tools(cfg)
    pcfg = cfg.get("policy", {})

    if policy_type == "llm":
        policy = LLMPolicy(
            max_steps=pcfg.get("max_steps", 6),
            budget_tokens=pcfg.get("budget_tokens", 2048),
            budget_nodes=pcfg.get("budget_nodes", 50),
        )
    elif policy_type == "grpo":
        tcfg = cfg.get("training", {})
        reward_cfg = cfg.get("reward", {})
        grpo_cfg = cfg.get("grpo", {})
        policy = GRPOPolicy(
            state_dim=16,
            hidden_dim=tcfg.get("hidden_dim", 128),
            K=grpo_cfg.get("K", 4),
            clip_eps=grpo_cfg.get("clip_eps", 0.2),
            max_steps=pcfg.get("max_steps", 6),
            budget_tokens=pcfg.get("budget_tokens", 2048),
            budget_nodes=pcfg.get("budget_nodes", 50),
            beta=reward_cfg.get("beta", 0.1),
            gamma=reward_cfg.get("gamma", 0.001),
            eta=reward_cfg.get("eta", 0.01),
        )
        if grpo_path:
            policy.network.load_state_dict(torch.load(grpo_path, weights_only=True))
    else:
        policy = HeuristicPolicy(
            max_steps=pcfg.get("max_steps", 6),
            budget_tokens=pcfg.get("budget_tokens", 2048),
            budget_nodes=pcfg.get("budget_nodes", 50),
        )

    prompt_builder = FraudPromptBuilder(mode=prompt_mode)

    # Train classifier as prediction proxy
    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    classifier = GCNClassifier(
        in_dim=in_dim, hidden_dim=tcfg.get("hidden_dim", 128),
        num_layers=tcfg.get("num_layers", 2), dropout=tcfg.get("dropout", 0.3),
    ).to(device)
    probs_all = train_classifier(classifier, data, device)

    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    if limit > 0:
        test_indices = test_indices[:limit]

    y_true, y_prob = [], []
    action_counts = {}
    total_steps, total_nodes, total_edges, total_tokens = 0, 0, 0, 0
    struct_agg = {"high_risk_neighbors": 0, "density": 0.0, "shared_neighbors": 0, "cycles_found": 0}
    latencies = []

    for v in tqdm(test_indices, desc=f"AGEA-{policy_type}-{prompt_mode}"):
        t0 = time.time()

        if policy_type == "grpo":
            evidence_nodes, traj = policy.sample_trajectory(
                v.item(), data.edge_index, data.x, data.y, tools, deterministic=True)
        else:
            evidence_nodes, traj = policy.run_episode(
                v.item(), data.edge_index, data.x, data.y, tools)

        lat = time.time() - t0
        latencies.append(lat)

        # Structural stats
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
            budget_info={"tokens": len(evidence_nodes) * 20,
                         "nodes": len(evidence_nodes), "edges": n_edges})

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
    metrics["avg_tokens"] = total_tokens / n
    metrics["avg_nodes"] = total_nodes / n
    metrics["avg_edges"] = total_edges / n
    metrics["avg_latency"] = np.mean(latencies)
    metrics["avg_steps"] = total_steps / n
    metrics["avg_high_risk"] = struct_agg["high_risk_neighbors"] / n
    metrics["avg_density"] = struct_agg["density"] / n
    metrics["action_dist"] = action_counts

    return metrics


def run_fixed_baselines(data, cfg, device, limit=-1):
    """Run fixed-retrieval baselines."""
    tools = build_tools(cfg)
    in_dim = data.num_features
    tcfg = cfg.get("training", {})

    classifier = GCNClassifier(
        in_dim=in_dim, hidden_dim=tcfg.get("hidden_dim", 128),
        num_layers=tcfg.get("num_layers", 2), dropout=tcfg.get("dropout", 0.3),
    ).to(device)
    probs_all = train_classifier(classifier, data, device)

    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    if limit > 0:
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
        metrics["avg_nodes"] = total_nodes / n
        results[sname] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="AGEA Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--grpo_path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    print(f"[AGEA] Loading dataset: {cfg['dataset']['name']}")
    dataset = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("root"))
    data = dataset.data
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    # GNN baselines
    print("\n[AGEA] Evaluating GNN baselines...")
    gnn_results = run_gnn_baselines(data, cfg, device)

    # Fixed-retrieval baselines
    print("\n[AGEA] Evaluating fixed-retrieval baselines...")
    fixed_results = run_fixed_baselines(data, cfg, device, limit=args.limit)

    # AGEA variants
    print("\n[AGEA] Evaluating AGEA (heuristic, raw)...")
    agea_raw = run_agea_evaluation(data, cfg, device, "heuristic", "raw", limit=args.limit)

    print("\n[AGEA] Evaluating AGEA (heuristic, compressed)...")
    agea_comp = run_agea_evaluation(data, cfg, device, "heuristic", "compressed", limit=args.limit)

    agea_grpo = None
    if args.grpo_path:
        print("\n[AGEA] Evaluating AGEA (GRPO, raw)...")
        agea_grpo = run_agea_evaluation(data, cfg, device, "grpo", "raw",
                                         grpo_path=args.grpo_path, limit=args.limit)

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Method':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'AvgNodes':>10} {'AvgTokens':>10}")
    print("-" * 80)

    for name, m in gnn_results.items():
        print(f"{name:<30} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} {'N/A':>10} {'N/A':>10}")

    for name, m in fixed_results.items():
        print(f"Fixed-{name:<24} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} {m.get('avg_nodes', 0):>10.1f} {'N/A':>10}")

    print(f"{'AGEA (heuristic, raw)':<30} {agea_raw['auroc']:>8.4f} {agea_raw['auprc']:>8.4f} {agea_raw['f1']:>8.4f} {agea_raw['avg_nodes']:>10.1f} {agea_raw['avg_tokens']:>10.1f}")
    print(f"{'AGEA (heuristic, comp)':<30} {agea_comp['auroc']:>8.4f} {agea_comp['auprc']:>8.4f} {agea_comp['f1']:>8.4f} {agea_comp['avg_nodes']:>10.1f} {agea_comp['avg_tokens']:>10.1f}")

    if agea_grpo:
        print(f"{'AGEA (GRPO, raw)':<30} {agea_grpo['auroc']:>8.4f} {agea_grpo['auprc']:>8.4f} {agea_grpo['f1']:>8.4f} {agea_grpo['avg_nodes']:>10.1f} {agea_grpo['avg_tokens']:>10.1f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
