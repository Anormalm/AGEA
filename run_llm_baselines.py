#!/usr/bin/env python
"""Run LLM-based baselines with real GPT-4o-mini predictions.

Each LLM baseline uses a different fixed strategy to gather graph evidence,
formats it as an LLM-interpretable prompt, and queries GPT-4o-mini for a
fraud/legit prediction.

Baselines:
- LLM-zero: No graph evidence, zero-shot prompt with GNN prediction
- TAPE: GNN prediction + fixed 1-hop expansion prompt
- GraphGPT: Fixed BFS 2-hop expansion prompt
- HiGPT: Type-specific 1-hop expansion prompt (heterogeneous)
- InstructGLM: One-shot PPR + Community prompt
"""

import sys
import os
import time
import re
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_dataset
from graph_tools.ppr import PPRTopK
from graph_tools.community import Community
from models.classifier import SAGEClassifier
from utils import load_config, compute_metrics, estimate_tokens

# LLM config
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.5
LLM_TIMEOUT = 60
LLM_CONCURRENCY = 5

_client = None


def get_client():
    global _client
    if _client is None:
        import httpx
        from openai import OpenAI
        _client = OpenAI(
            timeout=LLM_TIMEOUT,
            http_client=httpx.Client(verify=False, timeout=LLM_TIMEOUT),
        )
    return _client


SYSTEM_PROMPT = (
    "You are a fraud detection expert analyzing a graph of users and their connections. "
    "You will be given structural evidence about a target node and its neighborhood. "
    "Use the structural patterns (fraud neighbor ratio, density, community structure) "
    "to make your prediction. "
    "Respond with exactly one line: 'Prediction: 1' (fraud) or 'Prediction: 0' (legitimate). "
    "Then provide a brief one-sentence rationale."
)


def call_llm(prompt: str) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=150,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(LLM_RETRY_DELAY * (attempt + 1))
            else:
                print(f"  LLM call failed: {e}")
                return ""


def parse_prediction(response: str) -> float:
    if not response:
        return 0.5
    match = re.search(r'[Pp]rediction\s*:\s*([01])', response)
    if match:
        return float(match.group(1))
    lower = response.lower()
    fraud_kw = ['fraud', 'fraudulent', 'fake', 'suspicious']
    legit_kw = ['legitimate', 'legit', 'normal', 'genuine']
    f_score = sum(1 for w in fraud_kw if w in lower)
    l_score = sum(1 for w in legit_kw if w in lower)
    if f_score > l_score:
        return 1.0
    elif l_score > f_score:
        return 0.0
    return 0.5


def precompute_adj(edge_index, num_nodes):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    adj = defaultdict(set)
    for i in range(len(src)):
        adj[int(src[i])].add(int(dst[i]))
    adj = dict(adj)
    print(f"  Adjacency built ({len(adj)} nodes with edges)")
    return adj


def precompute_type_adj(edge_index, edge_type):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    et = edge_type.numpy()
    type_adj = defaultdict(lambda: defaultdict(set))
    for i in range(len(src)):
        s, d, t = int(src[i]), int(dst[i]), int(et[i])
        type_adj[s][t].add(d)
        type_adj[d][t].add(s)
    return {k: dict(v) for k, v in type_adj.items()}


def train_classifier(data, device, cfg):
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
            print(f"    Epoch {epoch+1}: train={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


# ---- LLM-interpretable prompt builder ----

def build_llm_prompt(target, evidence_nodes, adj, fraud_labels_np, probs_all,
                     method_name, type_adj=None, n_steps=1):
    """Build an LLM-interpretable prompt with structural evidence.

    Instead of raw node IDs and feature norms, we provide:
    - GNN classifier probability
    - Neighborhood statistics (fraud ratio, density, degree)
    - Structural patterns the LLM can reason about
    - Per-neighbor summary (fraud/legit label if known)
    """
    prob = probs_all[target].item() if target < len(probs_all) else 0.5
    node_set = set(evidence_nodes)
    n_nodes = len(node_set)
    neighbors = [n for n in node_set if n != target]

    # Structural statistics
    n_edges = sum(len(adj.get(n, set()) & node_set) for n in node_set)
    density = n_edges / max(n_nodes * (n_nodes - 1), 1)
    high_risk = sum(1 for n in node_set
                    if n < len(fraud_labels_np) and fraud_labels_np[n] == 1)
    high_risk_ratio = high_risk / max(n_nodes, 1)
    avg_deg = n_edges / max(n_nodes, 1)

    # Count neighbors by label
    fraud_neighbors = sum(1 for n in neighbors
                          if n < len(fraud_labels_np) and fraud_labels_np[n] == 1)
    legit_neighbors = sum(1 for n in neighbors
                          if n < len(fraud_labels_np) and fraud_labels_np[n] == 0)
    unlabeled_neighbors = len(neighbors) - fraud_neighbors - legit_neighbors

    # Target degree
    target_degree = len(adj.get(target, set()))

    parts = []
    parts.append(f"=== Fraud Detection Analysis ===")
    parts.append(f"Target node: {target}")
    parts.append(f"GNN classifier fraud probability: {prob:.1%}")
    parts.append(f"Target node degree (total connections): {target_degree}")
    parts.append("")

    # Evidence summary
    parts.append(f"=== Evidence Subgraph ({method_name} retrieval) ===")
    parts.append(f"Nodes in evidence: {n_nodes}")
    parts.append(f"Internal edges: {n_edges}")
    parts.append(f"Subgraph density: {density:.3f}")
    parts.append(f"Average degree: {avg_deg:.1f}")
    parts.append("")

    # Neighbor composition — most informative signal for LLM
    parts.append(f"=== Neighbor Composition ===")
    parts.append(f"Fraud-labeled neighbors: {fraud_neighbors}")
    parts.append(f"Legitimate-labeled neighbors: {legit_neighbors}")
    parts.append(f"Unlabeled neighbors: {unlabeled_neighbors}")
    if fraud_neighbors + legit_neighbors > 0:
        fraud_pct = fraud_neighbors / (fraud_neighbors + legit_neighbors) * 100
        parts.append(f"Fraud ratio among labeled neighbors: {fraud_pct:.1f}%")
    parts.append("")

    # Structural patterns
    parts.append(f"=== Structural Patterns ===")
    if density > 0.3:
        parts.append("The evidence subgraph is DENSE — nodes are highly interconnected.")
    elif density > 0.1:
        parts.append("The evidence subgraph has MODERATE density.")
    else:
        parts.append("The evidence subgraph is SPARSE — few internal connections.")

    if high_risk_ratio > 0.3:
        parts.append("HIGH fraud concentration in the neighborhood.")
    elif high_risk_ratio > 0.1:
        parts.append("MODERATE fraud concentration in the neighborhood.")
    else:
        parts.append("LOW fraud concentration in the neighborhood.")

    # Type-specific info for HiGPT
    if type_adj is not None and target in type_adj:
        parts.append("")
        parts.append(f"=== Edge Type Breakdown ===")
        type_names = {1: "shared-reviews (R-U-R)", 2: "shared-products (R-S-R)",
                      3: "co-purchase (R-P-M-R)", 4: "shared-product-reviews"}
        for t, nbs in type_adj[target].items():
            tname = type_names.get(t, f"type-{t}")
            in_evidence = len(nbs & node_set)
            parts.append(f"  {tname}: {len(nbs)} neighbors, {in_evidence} in evidence")

    # Top neighbor details (limited)
    parts.append("")
    parts.append(f"=== Top Neighbors (up to 10) ===")
    top_neighbors = sorted(neighbors, key=lambda n: len(adj.get(n, set())), reverse=True)[:10]
    for n in top_neighbors:
        n_deg = len(adj.get(n, set()))
        label = ""
        if n < len(fraud_labels_np):
            if fraud_labels_np[n] == 1:
                label = " [FRAUD]"
            elif fraud_labels_np[n] == 0:
                label = " [LEGIT]"
        shared = len(adj.get(target, set()) & adj.get(n, set()))
        parts.append(f"  Node {n}: degree={n_deg}{label}, shared_neighbors={shared}")

    parts.append("")
    parts.append("Based on this evidence, predict whether the target node is "
                  "involved in fraud (1) or is legitimate (0).")

    return "\n".join(parts)


def build_zero_shot_prompt(target, probs_all):
    prob = probs_all[target].item() if target < len(probs_all) else 0.5
    parts = [
        "=== Fraud Detection Analysis ===",
        f"Target node: {target}",
        f"GNN classifier fraud probability: {prob:.1%}",
        "",
        "No graph evidence is available for this node.",
        "Based on the GNN prediction alone, predict whether this node is "
        "involved in fraud (1) or is legitimate (0).",
    ]
    return "\n".join(parts)


# ---- Fixed evidence collection strategies ----

def collect_evidence_llm_zero(target, adj, fraud_labels_np, budget_nodes=50):
    return {target}, 0


def collect_evidence_tape(target, adj, fraud_labels_np, probs=None, budget_nodes=50):
    evidence = {target}
    neighbors = adj.get(target, set())
    for n in sorted(neighbors, key=lambda n: len(adj.get(n, set())), reverse=True):
        if len(evidence) >= budget_nodes:
            break
        evidence.add(n)
    return evidence, 1


def collect_evidence_graphgpt(target, adj, fraud_labels_np, budget_nodes=50):
    evidence = {target}
    frontier = {target}
    for hop in range(2):
        next_frontier = set()
        for n in frontier:
            for nb in adj.get(n, set()):
                if nb not in evidence and len(evidence) < budget_nodes:
                    evidence.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    return evidence, 2


def collect_evidence_higpt(target, adj, fraud_labels_np,
                           type_adj=None, budget_nodes=50):
    if type_adj is not None and target in type_adj:
        type_neighbors = type_adj[target]
        evidence = {target}
        type_lists = {t: sorted(nbs, key=lambda n: len(adj.get(n, set())), reverse=True)
                      for t, nbs in type_neighbors.items()}
        type_idx = {t: 0 for t in type_lists}
        added = True
        while added and len(evidence) < budget_nodes:
            added = False
            for t in sorted(type_lists.keys()):
                if type_idx[t] < len(type_lists[t]):
                    n = type_lists[t][type_idx[t]]
                    if n not in evidence:
                        evidence.add(n)
                        added = True
                        if len(evidence) >= budget_nodes:
                            break
                    type_idx[t] += 1
        return evidence, 1
    else:
        return collect_evidence_tape(target, adj, fraud_labels_np, budget_nodes=budget_nodes)


def collect_evidence_instructglm(target, adj, fraud_labels_np,
                                 tools, budget_nodes=50):
    evidence = {target}
    ppr_nodes, _ = tools["PPRTopK"](evidence, adj, max_nodes=budget_nodes)
    evidence = ppr_nodes
    comm_nodes, _ = tools["Community"](evidence, adj, max_nodes=budget_nodes)
    evidence = comm_nodes
    return evidence, 1


def run_llm_baselines(data, cfg, device, max_test_nodes=500):
    gt = cfg.get("graph_tools", {})
    tools = {
        "PPRTopK": PPRTopK(alpha=gt.get("ppr_alpha", 0.15), topk=gt.get("ppr_topk", 10)),
        "Community": Community(resolution=gt.get("community_resolution", 1.0)),
    }
    pcfg = cfg.get("policy", {})
    budget_nodes = pcfg.get("budget_nodes", 50)

    print("  Training GraphSAGE classifier...")
    probs_all = train_classifier(data, device, cfg)

    print("  Precomputing adjacency...")
    adj = precompute_adj(data.edge_index, data.num_nodes)
    tools["PPRTopK"].set_adj(adj)
    fraud_labels_np = data.y.numpy()

    type_adj = None
    try:
        edge_type = data.edge_type
        print("  Precomputing type-aware adjacency for HiGPT...")
        type_adj = precompute_type_adj(data.edge_index, edge_type)
        print(f"  Type-aware adjacency built ({len(type_adj)} nodes with typed edges)")
    except AttributeError:
        pass

    # Subsample test set
    all_test = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    valid = data.y >= 0
    test_labeled = all_test[valid[all_test]]
    rng = np.random.RandomState(42)
    if len(test_labeled) > max_test_nodes:
        perm = torch.from_numpy(rng.choice(len(test_labeled), max_test_nodes, replace=False))
        eval_test = test_labeled[perm]
    else:
        eval_test = test_labeled
    print(f"  Evaluating on {len(eval_test)} test nodes")

    # Test LLM connectivity
    print("  Testing LLM connection...")
    test_resp = call_llm("Respond with: Prediction: 0")
    if not test_resp:
        print("  ERROR: Cannot reach OpenAI API. Check OPENAI_API_KEY.")
        return {}
    print(f"  LLM connected. Test response: {test_resp[:50]}")

    methods = {
        "LLM-zero": lambda t: collect_evidence_llm_zero(t, adj, fraud_labels_np, budget_nodes),
        "TAPE": lambda t: collect_evidence_tape(t, adj, fraud_labels_np, probs_all, budget_nodes),
        "GraphGPT": lambda t: collect_evidence_graphgpt(t, adj, fraud_labels_np, budget_nodes),
        "HiGPT": lambda t: collect_evidence_higpt(t, adj, fraud_labels_np,
                                                    type_adj, budget_nodes),
        "InstructGLM": lambda t: collect_evidence_instructglm(t, adj, fraud_labels_np,
                                                               tools, budget_nodes),
    }

    results = {}

    for method_name, collect_fn in methods.items():
        print(f"\n  --- {method_name} ---")
        t0 = time.time()

        # Step 1: Collect evidence and build prompts
        prompts = {}
        labels = {}
        token_counts = {}
        node_counts = {}

        for v in eval_test:
            v = v.item()
            label = data.y[v].item()
            evidence_nodes, n_steps = collect_fn(v)

            if method_name == "LLM-zero":
                prompt = build_zero_shot_prompt(v, probs_all)
            else:
                prompt = build_llm_prompt(
                    v, evidence_nodes, adj, fraud_labels_np, probs_all,
                    method_name, type_adj=type_adj, n_steps=n_steps)

            prompts[v] = prompt
            labels[v] = label
            token_counts[v] = estimate_tokens(prompt)
            node_counts[v] = len(evidence_nodes)

        avg_tok = np.mean(list(token_counts.values()))
        print(f"    Prompts built: {len(prompts)} nodes, avg tokens={avg_tok:.0f}")

        # Step 2: Call LLM in parallel
        predictions = {}
        errors = 0
        node_list = list(prompts.keys())

        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
            futures = {executor.submit(call_llm, prompts[v]): v for v in node_list}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"    {method_name} API"):
                v = futures[future]
                try:
                    response = future.result()
                    pred = parse_prediction(response)
                    if not response:
                        errors += 1
                except Exception:
                    pred = 0.5
                    errors += 1
                predictions[v] = pred

        # Step 3: Compute metrics
        y_true = [labels[v] for v in node_list]
        y_prob = [predictions[v] for v in node_list]
        total_tokens = sum(token_counts[v] for v in node_list)
        total_nodes = sum(node_counts[v] for v in node_list)

        n = len(y_true)
        metrics = compute_metrics(y_true, y_prob,
                                  k=cfg.get("evaluation", {}).get("recall_k", 100))
        metrics["avg_tokens"] = total_tokens / max(n, 1)
        metrics["avg_nodes"] = total_nodes / max(n, 1)
        metrics["api_errors"] = errors
        results[method_name] = metrics

        elapsed = time.time() - t0
        print(f"    MacroF1={metrics['macro_f1']:.4f}, AUROC={metrics['auroc']:.4f}, "
              f"AUPRC={metrics['auprc']:.4f}")
        print(f"    AvgNodes={metrics['avg_nodes']:.1f}, "
              f"AvgTokens={metrics['avg_tokens']:.0f}, "
              f"Errors={errors}/{n}, Time={elapsed:.0f}s")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "yelp_spam", "amazon",
                                 "fakenews_politifact", "fakenews_buzzfeed"])
    parser.add_argument("--max_test_nodes", type=int, default=500,
                        help="Max test nodes to evaluate (LLM calls are expensive)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"Concurrency: {LLM_CONCURRENCY}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    datasets = []
    if args.dataset == "all":
        datasets = ["yelp_spam", "amazon",
                     "fakenews_politifact", "fakenews_buzzfeed"]
    else:
        datasets = [args.dataset]

    all_results = {}
    for ds_name in datasets:
        cfg_path = f"configs/{ds_name}.yaml"
        cfg = load_config(cfg_path)
        root = cfg["dataset"].get("root")
        print(f"\n{'='*60}")
        print(f"LLM BASELINES: {ds_name}")
        print(f"{'='*60}")
        dataset = load_dataset(ds_name, root)
        data = dataset.data
        print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
              f"Features: {data.num_features}")
        valid = data.y >= 0
        fraud_ratio = data.y[valid].float().mean().item()
        print(f"Fraud ratio (labeled): {fraud_ratio:.3f}")

        results = run_llm_baselines(data, cfg, device,
                                    max_test_nodes=args.max_test_nodes)
        all_results[ds_name] = results

    # Summary
    print(f"\n{'='*80}")
    print(f"LLM BASELINES SUMMARY (model={LLM_MODEL})")
    print(f"{'='*80}")
    print(f"{'Dataset':<22} {'Method':<15} {'MacroF1':>8} {'AUROC':>8} "
          f"{'AUPRC':>8} {'AvgNodes':>10} {'AvgTokens':>10}")
    print(f"{'-'*80}")
    for ds_name, results in all_results.items():
        for method, m in results.items():
            print(f"{ds_name:<22} {method:<15} {m['macro_f1']:>8.4f} "
                  f"{m['auroc']:>8.4f} {m['auprc']:>8.4f} "
                  f"{m['avg_nodes']:>10.1f} {m['avg_tokens']:>10.0f}")


if __name__ == "__main__":
    main()
