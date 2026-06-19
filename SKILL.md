---
name: agea
version: 1.0.0
description: Adaptive Graph Evidence Acquisition for LLM-Based Fraud Reasoning. Run evidence acquisition policies, evaluate baselines, train GRPO, and manage graph tools.
license: MIT
compatibility: claude-code
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# AGEA: Adaptive Graph Evidence Acquisition

You are an expert in graph-based fraud detection research. AGEA is a framework that actively selects which graph evidence to acquire before LLM-based fraud reasoning. Unlike fixed-retrieval approaches (1-hop, 2-hop, PPR), AGEA adapts evidence acquisition to the target node's context.

## Core Concepts

**Pipeline:** Graph G → AGEA policy selects evidence subgraph G* → prompt construction → LLM reasoner → prediction ŷ

**Objective:** G* = argmax_{G' ⊆ G} [ I(Y; G') - λ C(G') ]

**Reward:** R = R_pred + β R_struct - γ R_cost - η step_count

## Action Space (Fixed)

The controller chooses exactly ONE action per step from:

| Action | Graph Operator | Description |
|--------|---------------|-------------|
| Expand1Hop | `graph_tools/expand.py` | Add 1-hop neighbors of evidence nodes |
| Expand2Hop | `graph_tools/expand.py` | Add 2-hop neighbors of evidence nodes |
| PPRTopK | `graph_tools/ppr.py` | Retrieve top-K nodes by Personalized PageRank |
| Community | `graph_tools/community.py` | Detect communities around evidence via Louvain |
| ShortCycle | `graph_tools/cycle.py` | Find short cycles (triangles) involving evidence |
| PruneTopK | `graph_tools/prune.py` | Remove low-importance nodes from evidence |
| Stop | — | Stop acquisition and proceed to reasoning |

The LLM **only chooses tools**. It cannot invent nodes or edges.

## State Summary z_t

Each step, the controller observes:
- Target node ID
- Current evidence graph size (nodes, edges)
- Current token estimate
- Suspicious neighbor count
- Subgraph density
- Available budget (tokens, nodes)
- Previous actions
- Steps taken

## Policy Types

### Heuristic Policy (`policy/heuristic_policy.py`)
Rule-based controller. No training required.
- Empty evidence → Expand1Hop
- High density → PruneTopK
- Few suspicious neighbors → PPRTopK
- Haven't checked cycles → ShortCycle
- Moderate size → Community
- Budget exhausted → Stop

### LLM Policy (`policy/llm_policy.py`)
Uses an LLM to select actions. The LLM receives z_t as text and outputs one action name. Falls back to heuristic on API failure.

### GRPO Policy (`policy/grpo_policy.py`)
Learnable policy trained with Group Relative Policy Optimization:
1. Sample K trajectories per target node
2. Compute reward for each trajectory
3. Group-relative advantage: A_i = R_i - mean_j(R_j)
4. Clipped objective: L = E[min(r_i A_i, clip(r_i, 1-ε, 1+ε) A_i)]

## Prompt Modes

| Mode | Builder | Description |
|------|---------|-------------|
| raw | `prompts/raw_prompt.py` | Full feature details, individual neighbor summaries |
| compressed | `prompts/compressed_prompt.py` | Quantized features, aggregated statistics |
| evidence_only | `prompts/fraud_prompt.py` | Structural patterns only, no raw features |

## Structural Reward Signals (v1)

Interpretable signals — no spectral methods in v1:
- Short cycle count (triangles)
- High-risk neighbor count
- Subgraph density
- Shared neighbor count (Jaccard overlap)
- Temporal burst count (if timestamps exist)

## Datasets

- **YelpChi**: 32 features, ~14% fraud ratio
- **DGraphFin**: 17 features, ~5% fraud ratio
- **Elliptic Bitcoin**: 165 features, ~10% fraud ratio
- **Generic**: Load from .pt files (x.pt, edge_index.pt, y.pt, *_mask.pt)
- **Synthetic**: Auto-generated if no data found

## Commands

### Run training-free AGEA
```bash
python run_training_free.py --config configs/yelpchi.yaml
python run_training_free.py --config configs/yelpchi.yaml --limit 50  # debug
```

### Full evaluation (all baselines)
```bash
python evaluate.py --config configs/yelpchi.yaml
```

### GRPO training
```bash
python train_grpo.py --config configs/yelpchi.yaml --save_path agea_grpo_policy.pt
```

### With GRPO policy in evaluation
```bash
python evaluate.py --config configs/yelpchi.yaml --grpo_path agea_grpo_policy.pt
```

## Baselines

| # | Method | Type |
|---|--------|------|
| 1 | MLP | Feature-only |
| 2 | GCN | GNN |
| 3 | GraphSAGE | GNN |
| 4 | GAT | GNN |
| 5 | Text-only LLM | LLM |
| 6 | Fixed 1-hop GraphRAG | Fixed retrieval |
| 7 | Fixed 2-hop GraphRAG | Fixed retrieval |
| 8 | PPR Top-K | Fixed retrieval |
| 9 | AGEA (heuristic, raw) | Adaptive |
| 10 | AGEA (heuristic, compressed) | Adaptive |
| 11 | AGEA (GRPO, raw) | Adaptive + learned |

## Evaluation Metrics

**Prediction:** AUROC, AUPRC, F1, Recall@K
**Efficiency:** avg tokens, avg nodes, avg edges, latency
**Evidence:** cycle count, high-risk neighbors, density, sparsity
**Policy:** action distribution, avg steps, stop rate

## Key Experimental Goal

Show that AGEA beats fixed 1-hop, fixed 2-hop, PPR, and fixed GraphRAG retrieval **under the same token budget**.

## Configuration

All settings are in `configs/*.yaml`. Key sections:

```yaml
dataset:       # name, root, feature_dim
policy:        # type, max_steps, budget_tokens, budget_nodes
graph_tools:   # ppr_alpha, ppr_topk, community_resolution, cycle_length, prune_topk
prompt:        # mode (raw/compressed/evidence_only), max_neighbor_summaries
reward:        # beta, gamma, eta
training:      # lr, epochs, batch_size, hidden_dim, num_layers, dropout
grpo:          # K, clip_eps, entropy_coeff
evaluation:    # metrics, recall_k, eval_split
```

## Repository Structure

```
agea/
  configs/                # Dataset YAML configs
  data/loader.py          # Dataset loading + synthetic generation
  graph_tools/
    expand.py             # Expand1Hop, Expand2Hop
    ppr.py                # PPRTopK
    community.py          # Community (Louvain)
    cycle.py              # ShortCycle (triangles)
    prune.py              # PruneTopK
    topk.py               # TopKSimilar (feature cosine)
  policy/
    heuristic_policy.py   # Rule-based controller
    llm_policy.py         # LLM-guided controller
    grpo_policy.py        # GRPO-learned controller
  prompts/
    fraud_prompt.py       # FraudPromptBuilder (dispatcher)
    raw_prompt.py         # RawPromptBuilder (full features)
    compressed_prompt.py  # CompressedPromptBuilder (quantized)
  models/
    classifier.py         # MLP, GCN, GraphSAGE, GAT
    reasoner.py           # LLMReasoner (OpenAI API)
  run_training_free.py    # Stage 1 entry point
  train_grpo.py           # Stage 3 GRPO training
  evaluate.py             # Full evaluation
  utils.py                # Metrics, config loading, helpers
```
