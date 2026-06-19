"""Training-free AGEA: run heuristic or LLM-guided evidence acquisition."""

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


def build_policy(cfg):
    pcfg = cfg.get("policy", {})
    ptype = pcfg.get("type", "heuristic")
    if ptype == "llm":
        return LLMPolicy(
            max_steps=pcfg.get("max_steps", 6),
            budget_tokens=pcfg.get("budget_tokens", 2048),
            budget_nodes=pcfg.get("budget_nodes", 50),
            model=pcfg.get("model", "gpt-4o-mini"),
            api_key=pcfg.get("api_key"),
            api_base=pcfg.get("api_base"),
        )
    return HeuristicPolicy(
        max_steps=pcfg.get("max_steps", 6),
        budget_tokens=pcfg.get("budget_tokens", 2048),
        budget_nodes=pcfg.get("budget_nodes", 50),
    )


def train_gnn_baseline(model, data, cfg, device):
    """Train a GNN baseline classifier."""
    tcfg = cfg.get("training", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.get("lr", 1e-4))
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = tcfg.get("epochs", 100)

    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                val_mask = data.val_mask.to(device)
                val_loss = criterion(out[val_mask], y[val_mask])
                print(f"  Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

    return model


def run_fixed_retrieval(target_nodes, data, tools, prompt_builder, cfg, device):
    """Run fixed-retrieval baselines (1-hop, 2-hop, PPR)."""
    results = {}
    x = data.x
    edge_index = data.edge_index
    y = data.y

    for strategy_name, strategy_fn in [
        ("1hop", lambda nodes, ei, x, n: tools["Expand1Hop"](nodes, ei, n, cfg["policy"].get("budget_nodes", 50))),
        ("2hop", lambda nodes, ei, x, n: tools["Expand2Hop"](nodes, ei, n, cfg["policy"].get("budget_nodes", 50))),
        ("ppr", lambda nodes, ei, x, n: tools["PPRTopK"](nodes, ei, n, cfg["policy"].get("budget_nodes", 50))),
    ]:
        all_probs, all_labels = [], []
        total_nodes, total_edges, total_tokens = 0, 0, 0

        for v in tqdm(target_nodes, desc=f"Fixed-{strategy_name}"):
            evidence = {v.item()}
            evidence, info = strategy_fn(evidence, edge_index, x, x.size(0))

            # Count edges
            src, dst = edge_index
            n_edges = sum(1 for s, d in zip(src, dst)
                          if s.item() in evidence and d.item() in evidence)

            prompt = prompt_builder.build(
                v.item(), evidence, edge_index, x, y,
                budget_info={"tokens": len(evidence) * 20, "nodes": len(evidence), "edges": n_edges})

            total_nodes += len(evidence)
            total_edges += n_edges
            total_tokens += estimate_tokens(prompt)

            # Use GNN prediction as proxy for LLM (fast eval)
            all_probs.append(0.5)  # placeholder — actual LLM call in evaluate.py
            all_labels.append(y[v].item())

        results[strategy_name] = {
            "avg_nodes": total_nodes / max(len(target_nodes), 1),
            "avg_edges": total_edges / max(len(target_nodes), 1),
            "avg_tokens": total_tokens / max(len(target_nodes), 1),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Training-free AGEA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of test nodes")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[AGEA] Loading dataset: {cfg['dataset']['name']}")
    dataset = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("root"))
    data = dataset.data
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")

    tools = build_tools(cfg)
    policy = build_policy(cfg)
    prompt_builder = FraudPromptBuilder(
        mode=cfg.get("prompt", {}).get("mode", "raw"),
        max_neighbor_summaries=cfg.get("prompt", {}).get("max_neighbor_summaries", 15),
        max_edge_summaries=cfg.get("prompt", {}).get("max_edge_summaries", 30),
    )

    # Train GNN classifier for prediction
    print("[AGEA] Training GNN classifier for prediction proxy...")
    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    classifier = GCNClassifier(
        in_dim=in_dim,
        hidden_dim=tcfg.get("hidden_dim", 128),
        num_layers=tcfg.get("num_layers", 2),
        dropout=tcfg.get("dropout", 0.3),
    ).to(device)
    classifier = train_gnn_baseline(classifier, data, cfg, device)

    # Get test nodes
    test_mask = data.test_mask
    test_indices = test_mask.nonzero(as_tuple=False).squeeze(-1)
    if args.limit > 0:
        test_indices = test_indices[:args.limit]

    print(f"[AGEA] Running AGEA policy on {len(test_indices)} test nodes...")

    # Run AGEA acquisition
    y_true, y_prob = [], []
    action_counts = {a: 0 for a in ["Expand1Hop", "Expand2Hop", "PPRTopK",
                                     "Community", "ShortCycle", "PruneTopK", "Stop"]}
    total_steps, total_nodes, total_edges, total_tokens = 0, 0, 0, 0
    total_struct_stats = {"high_risk_neighbors": 0, "density": 0.0,
                          "shared_neighbors": 0, "cycles_found": 0}
    latencies = []

    classifier.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        all_logits = classifier(x, edge_index).cpu()
        all_probs = torch.sigmoid(all_logits)

    for v in tqdm(test_indices, desc="AGEA acquisition"):
        t0 = time.time()

        evidence_nodes, trajectory = policy.run_episode(
            v.item(), data.edge_index, data.x, data.y, tools)

        latency = time.time() - t0
        latencies.append(latency)

        # Compute structural stats
        G_ev = nx.DiGraph()
        G_ev.add_nodes_from(evidence_nodes)
        src, dst = data.edge_index
        for s, d in zip(src, dst):
            if s.item() in evidence_nodes and d.item() in evidence_nodes:
                G_ev.add_edge(s.item(), d.item())
        struct_stats = compute_structural_reward(G_ev, None, data.y.numpy())
        total_struct_stats["high_risk_neighbors"] += struct_stats.get("high_risk_neighbors", 0)
        total_struct_stats["density"] += struct_stats.get("density", 0)
        total_struct_stats["shared_neighbors"] += struct_stats.get("shared_neighbors", 0)
        total_struct_stats["cycles_found"] += struct_stats.get("cycles_found", 0)

        # Count edges
        n_edges = G_ev.number_of_edges()

        # Build prompt for token estimate
        prompt = prompt_builder.build(
            v.item(), evidence_nodes, data.edge_index, data.x, data.y,
            struct_stats=struct_stats,
            budget_info={"tokens": len(evidence_nodes) * 20,
                         "nodes": len(evidence_nodes), "edges": n_edges})

        # Get prediction from classifier
        prob = all_probs[v].item()
        y_prob.append(prob)
        y_true.append(data.y[v].item())

        # Track stats
        steps = len([t for t in trajectory if t.get("action") != "Stop"])
        total_steps += steps
        total_nodes += len(evidence_nodes)
        total_edges += n_edges
        total_tokens += estimate_tokens(prompt)

        for t in trajectory:
            action = t.get("action", "Stop")
            if action in action_counts:
                action_counts[action] += 1

    n = len(test_indices)
    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))

    print("\n" + "=" * 60)
    print("AGEA Training-Free Results")
    print("=" * 60)
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Policy: {cfg['policy']['type']}")
    print(f"Test nodes: {n}")
    print(f"\nPrediction Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nEfficiency Metrics:")
    print(f"  Avg tokens: {total_tokens / n:.1f}")
    print(f"  Avg nodes: {total_nodes / n:.1f}")
    print(f"  Avg edges: {total_edges / n:.1f}")
    print(f"  Avg latency: {np.mean(latencies):.3f}s")
    print(f"\nEvidence Metrics:")
    print(f"  Avg high-risk neighbors: {total_struct_stats['high_risk_neighbors'] / n:.2f}")
    print(f"  Avg density: {total_struct_stats['density'] / n:.3f}")
    print(f"  Avg shared neighbors: {total_struct_stats['shared_neighbors'] / n:.2f}")
    print(f"\nPolicy Metrics:")
    print(f"  Avg steps: {total_steps / n:.2f}")
    print(f"  Action distribution:")
    for a, c in sorted(action_counts.items()):
        print(f"    {a}: {c} ({c / max(sum(action_counts.values()), 1) * 100:.1f}%)")
    print("=" * 60)

    # Also run fixed-retrieval baselines
    print("\n[AGEA] Running fixed-retrieval baselines...")
    fixed_results = run_fixed_retrieval(test_indices, data, tools, prompt_builder, cfg, device)
    for name, res in fixed_results.items():
        print(f"  Fixed-{name}: avg_nodes={res['avg_nodes']:.1f}, "
              f"avg_edges={res['avg_edges']:.1f}, avg_tokens={res['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()
