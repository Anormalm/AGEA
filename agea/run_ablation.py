#!/usr/bin/env python
"""Ablation studies for AGEA.

Three ablation types:
1. Tool ablation: remove one graph tool at a time
2. Feature ablation: remove feature groups from the evidence fuser
3. Budget ablation: vary max_steps

Optimization: classifier and adjacency are computed once and shared.
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

from data.loader import load_dataset
from graph_tools.expand import Expand1Hop, Expand2Hop
from graph_tools.ppr import PPRTopK
from graph_tools.community import Community
from graph_tools.cycle import ShortCycle
from graph_tools.prune import PruneTopK
from policy.heuristic_policy import HeuristicPolicy
from models.classifier import SAGEClassifier
from utils import load_config, compute_metrics


class EvidenceFuser(nn.Module):
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


def build_tools(cfg, omit_tools=None):
    omit_tools = omit_tools or set()
    gt = cfg.get("graph_tools", {})
    tools = {}
    if "Expand1Hop" not in omit_tools:
        tools["Expand1Hop"] = Expand1Hop(max_neighbors=gt.get("expand_max_neighbors", 20))
    if "Expand2Hop" not in omit_tools:
        tools["Expand2Hop"] = Expand2Hop(max_neighbors=gt.get("expand_max_neighbors", 20))
    if "PPRTopK" not in omit_tools:
        tools["PPRTopK"] = PPRTopK(alpha=gt.get("ppr_alpha", 0.15), topk=gt.get("ppr_topk", 10))
    if "Community" not in omit_tools:
        tools["Community"] = Community(resolution=gt.get("community_resolution", 1.0))
    if "ShortCycle" not in omit_tools:
        tools["ShortCycle"] = ShortCycle(cycle_length=gt.get("cycle_length", 3))
    if "PruneTopK" not in omit_tools:
        tools["PruneTopK"] = PruneTopK(topk=gt.get("prune_topk", 10))
    return tools


def precompute_adj(edge_index, num_nodes):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    adj = defaultdict(set)
    for i in range(len(src)):
        adj[int(src[i])].add(int(dst[i]))
    return dict(adj)


def compute_features_full(prob, evidence_nodes, adj, fraud_labels_np, n_steps):
    node_set = set(evidence_nodes)
    n_nodes = len(node_set)
    n_edges = sum(len(adj.get(n, set()) & node_set) for n in node_set)
    density = n_edges / max(n_nodes * (n_nodes - 1), 1)
    high_risk = sum(1 for n in node_set
                    if n < len(fraud_labels_np) and fraud_labels_np[n] == 1)
    high_risk_ratio = high_risk / max(n_nodes, 1)
    fraud_neighbor_ratio = high_risk / max(n_nodes - 1, 1)
    avg_deg = n_edges / max(n_nodes, 1)
    return [
        prob, n_nodes / 100.0, n_edges / 500.0, density,
        high_risk_ratio, fraud_neighbor_ratio, n_steps / 10.0, avg_deg / 50.0,
    ]


def select_features(full_feats, feat_indices):
    if feat_indices is None:
        return full_feats
    return [full_feats[i] for i in feat_indices]


def train_classifier(data, device, cfg, seed=42):
    torch.manual_seed(seed)
    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    model = SAGEClassifier(in_dim, tcfg.get("hidden_dim", 128),
                           tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.get("lr", 1e-4))
    criterion = nn.BCEWithLogitsLoss()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)
    train_mask = data.train_mask.to(device)
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
            print(f"    Epoch {epoch+1}: train={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


def train_fuser(fuser, features, labels, device, epochs=300, lr=1e-3):
    X = torch.tensor(features, dtype=torch.float32).to(device)
    Y = torch.tensor(labels, dtype=torch.float32).to(device)
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


def run_single(data, cfg, device, probs_all, adj, fraud_labels_np,
               omit_tools=None, feat_indices=None, max_steps=6, seed=42):
    """Run AGEA-heuristic with ablation params. Shares precomputed probs/adj."""
    torch.manual_seed(seed)
    tools = build_tools(cfg, omit_tools=omit_tools)
    pcfg = cfg.get("policy", {})
    policy = HeuristicPolicy(
        max_steps=max_steps,
        budget_tokens=pcfg.get("budget_tokens", 2048),
        budget_nodes=pcfg.get("budget_nodes", 50),
    )

    if "PPRTopK" in tools:
        tools["PPRTopK"].set_adj(adj)

    y_safe = data.y.clone()
    y_safe[y_safe < 0] = 0

    all_train = data.train_mask.nonzero(as_tuple=False).squeeze(-1)
    all_val = data.val_mask.nonzero(as_tuple=False).squeeze(-1)
    all_test = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    rng = np.random.RandomState(seed)

    if len(all_test) > 2000:
        perm = torch.from_numpy(rng.choice(len(all_test), 2000, replace=False))
        eval_test = all_test[perm]
    else:
        eval_test = all_test

    eval_nodes = torch.cat([all_val, eval_test])

    # Collect evidence
    all_evidence = {}
    for v in eval_nodes:
        v = v.item()
        evidence_nodes, traj = policy.run_episode(v, adj, data.x, y_safe, tools)
        n_steps_count = len([t for t in traj if t.get("action") != "Stop"])
        prob = probs_all[v].item()
        full_feats = compute_features_full(prob, evidence_nodes, adj, fraud_labels_np, n_steps_count)
        feats = select_features(full_feats, feat_indices)
        all_evidence[v] = {
            "features": feats, "label": data.y[v].item(),
            "evidence_nodes": evidence_nodes, "n_steps": n_steps_count,
        }

    # Train evidence
    train_sub = all_train[
        torch.from_numpy(rng.choice(len(all_train), min(1000, len(all_train)), replace=False))]
    for v in train_sub:
        v = v.item()
        if v in all_evidence:
            continue
        evidence_nodes, traj = policy.run_episode(v, adj, data.x, y_safe, tools)
        n_steps_count = len([t for t in traj if t.get("action") != "Stop"])
        prob = probs_all[v].item()
        full_feats = compute_features_full(prob, evidence_nodes, adj, fraud_labels_np, n_steps_count)
        feats = select_features(full_feats, feat_indices)
        all_evidence[v] = {"features": feats, "label": data.y[v].item()}

    # Train fuser
    fuser_dim = len(feat_indices) if feat_indices else 8
    val_feats = [all_evidence[v.item()]["features"] for v in all_val if v.item() in all_evidence]
    val_labs = [all_evidence[v.item()]["label"] for v in all_val if v.item() in all_evidence]
    tr_feats = [all_evidence[v.item()]["features"] for v in train_sub if v.item() in all_evidence]
    tr_labs = [all_evidence[v.item()]["label"] for v in train_sub if v.item() in all_evidence]

    fuser = EvidenceFuser(input_dim=fuser_dim, hidden_dim=32).to(device)
    train_fuser(fuser, val_feats + tr_feats, val_labs + tr_labs, device)

    # Predict
    y_true, y_prob = [], []
    total_steps, total_nodes = 0, 0
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

    n = len(y_true)
    metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
    metrics["avg_nodes"] = total_nodes / max(n, 1)
    metrics["avg_steps"] = total_steps / max(n, 1)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yelp_spam",
                        choices=["yelp_spam", "amazon",
                                 "fakenews_politifact", "fakenews_buzzfeed"])
    parser.add_argument("--ablation", default="all",
                        choices=["all", "tool", "feature", "budget"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg_path = f"configs/{args.dataset}.yaml"
    cfg = load_config(cfg_path)
    root = cfg["dataset"].get("root")
    dataset = load_dataset(args.dataset, root)
    data = dataset.data

    print(f"Dataset: {args.dataset}, Nodes: {data.num_nodes}, Edges: {data.num_edges}", flush=True)

    # Shared precomputation (done once!)
    print("Training shared GraphSAGE classifier...", flush=True)
    probs_all = train_classifier(data, device, cfg, seed=args.seed)

    print("Precomputing adjacency...", flush=True)
    adj = precompute_adj(data.edge_index, data.num_nodes)
    fraud_labels_np = data.y.numpy()
    print("Done. Starting ablation.\n", flush=True)

    all_results = {}

    # ---- Tool Ablation ----
    if args.ablation in ("all", "tool"):
        print(f"{'='*60}", flush=True)
        print(f"TOOL ABLATION: {args.dataset}", flush=True)
        print(f"{'='*60}", flush=True)
        tool_names = ["Expand1Hop", "Expand2Hop", "PPRTopK", "Community", "ShortCycle", "PruneTopK"]
        results = {}

        print("  All-tools (full, flush=True)")
        t0 = time.time()
        results["All-tools"] = run_single(data, cfg, device, probs_all, adj, fraud_labels_np, seed=args.seed)
        print(f"    F1={results['All-tools']['macro_f1']:.4f} AUROC={results['All-tools']['auroc']:.4f} "
              f"({time.time()-t0:.0f}s)")

        for omit in tool_names:
            print(f"  w/o {omit}", flush=True)
            t0 = time.time()
            results[f"w/o-{omit}"] = run_single(data, cfg, device, probs_all, adj, fraud_labels_np,
                                                  omit_tools={omit}, seed=args.seed)
            print(f"    F1={results[f'w/o-{omit}']['macro_f1']:.4f} AUROC={results[f'w/o-{omit}']['auroc']:.4f} "
                  f"({time.time()-t0:.0f}s)")

        all_results["tool"] = results

    # ---- Feature Ablation ----
    if args.ablation in ("all", "feature"):
        print(f"\n{'='*60}", flush=True)
        print(f"FEATURE ABLATION: {args.dataset}", flush=True)
        print(f"{'='*60}", flush=True)
        feature_sets = {
            "Full-8feat": None,
            "No-structural (prob+n_steps)": [0, 6],
            "No-label (no risk ratios)": [0, 1, 2, 3, 6, 7],
            "No-classifier (no prob)": [1, 2, 3, 4, 5, 6, 7],
            "Classifier-only (prob)": [0],
        }
        results = {}
        for name, feat_indices in feature_sets.items():
            print(f"  {name}", flush=True)
            t0 = time.time()
            results[name] = run_single(data, cfg, device, probs_all, adj, fraud_labels_np,
                                        feat_indices=feat_indices, seed=args.seed)
            print(f"    F1={results[name]['macro_f1']:.4f} AUROC={results[name]['auroc']:.4f} "
                  f"AUPRC={results[name]['auprc']:.4f} ({time.time()-t0:.0f}s)")

        all_results["feature"] = results

    # ---- Budget Ablation ----
    if args.ablation in ("all", "budget"):
        print(f"\n{'='*60}", flush=True)
        print(f"BUDGET ABLATION: {args.dataset}", flush=True)
        print(f"{'='*60}", flush=True)
        results = {}
        for steps in [2, 4, 6, 8, 10]:
            print(f"  max_steps={steps}", flush=True)
            t0 = time.time()
            results[f"steps={steps}"] = run_single(data, cfg, device, probs_all, adj, fraud_labels_np,
                                                     max_steps=steps, seed=args.seed)
            m = results[f"steps={steps}"]
            print(f"    F1={m['macro_f1']:.4f} AUROC={m['auroc']:.4f} "
                  f"Nodes={m['avg_nodes']:.1f} Steps={m['avg_steps']:.2f} ({time.time()-t0:.0f}s)")

        all_results["budget"] = results

    # Summary tables
    for ablation_type, results in all_results.items():
        print(f"\n{'='*70}", flush=True)
        print(f"ABLATION ({ablation_type}, flush=True): {args.dataset}")
        print(f"{'='*70}", flush=True)
        print(f"{'Variant':<30} {'MacroF1':>8} {'AUROC':>8} {'AUPRC':>8} {'AvgNodes':>10} {'AvgSteps':>10}", flush=True)
        print(f"{'-'*70}", flush=True)
        for name, m in results.items():
            print(f"{name:<30} {m['macro_f1']:>8.4f} {m['auroc']:>8.4f} "
                  f"{m['auprc']:>8.4f} {m['avg_nodes']:>10.1f} {m['avg_steps']:>10.2f}")


if __name__ == "__main__":
    main()
