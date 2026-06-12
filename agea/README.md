# AGEA: Adaptive Graph Evidence Acquisition for LLM-Based Fraud Reasoning

## Problem

Existing graph-enhanced fraud reasoning systems retrieve graph evidence using **fixed rules** — 1-hop neighborhoods, 2-hop neighborhoods, PPR, or fixed GraphRAG retrieval. These approaches are not task-adaptive: they either retrieve too little evidence (missing fraud signals) or too much (wasting tokens and diluting reasoning).

## Method

AGEA is a **sequential graph evidence acquisition framework** that actively selects which graph evidence to acquire before LLM-based fraud reasoning.

```
Graph G → AGEA policy selects evidence subgraph G* → prompt construction → LLM reasoner → prediction ŷ
```

**Core objective:**

G* = argmax_{G' ⊆ G} [ I(Y; G') - λ C(G') ]

Approximated by reward:

R = R_pred + β R_struct - γ R_cost

where:
- R_pred = negative classification loss
- R_struct = structural evidence reward (cycles, high-risk neighbors, density)
- R_cost = token / latency / edge / node cost

## Architecture

### Action Space

At each step, the AGEA controller chooses exactly one action from:

| Action | Description |
|--------|-------------|
| Expand1Hop | Add 1-hop neighbors of current evidence nodes |
| Expand2Hop | Add 2-hop neighbors of current evidence nodes |
| PPRTopK | Retrieve top-K nodes by Personalized PageRank |
| Community | Detect community around evidence and add members |
| ShortCycle | Find short cycles (triangles) involving evidence |
| PruneTopK | Remove low-importance nodes from evidence |
| Stop | Stop acquisition and proceed to reasoning |

The LLM **only chooses tools** — it cannot freely invent nodes or edges.

### State Summary

The controller observes a compact state summary z_t including:
- Target node ID
- Current evidence graph size (nodes, edges)
- Current token estimate
- Suspicious neighbor statistics
- Available budget
- Previous actions taken
- Subgraph density

### Three Stages

**Stage 1: Training-Free AGEA**
- LLM-guided or heuristic controller
- Deterministic graph tool execution
- No training required

**Stage 2: Prompt Construction**
- Raw evidence prompt (full features)
- Compressed evidence prompt (quantized, aggregated)
- Evidence-only prompt (structural patterns only)

**Stage 3: Learnable AGEA with GRPO**
- Group Relative Policy Optimization
- Sample K trajectories per target node
- Compute group-relative advantage: A_i = R_i - mean_j(R_j)
- Clipped objective: L = E[min(r_i A_i, clip(r_i, 1-ε, 1+ε) A_i)]

### Reward (v1)

R = -CE(ŷ, y) + β · structural_signal(G*) - γ · token_cost(G*) - η · step_count

Structural signals:
- Short cycle count
- High-risk neighbor count
- Subgraph density
- Shared neighbors
- Temporal burst count (if timestamps exist)

## Datasets

- YelpChi
- DGraphFin
- Elliptic Bitcoin

If official loaders are unavailable, AGEA falls back to a generic `.pt` format or generates synthetic data for development.

## Baselines

1. MLP on node features
2. GCN
3. GraphSAGE
4. GAT
5. Text-only LLM prompting
6. Fixed 1-hop GraphRAG
7. Fixed 2-hop GraphRAG
8. PPR Top-K retrieval
9. AGEA without compression (raw prompt)
10. AGEA with compressed prompt
11. AGEA with learned policy (GRPO)

## Evaluation Metrics

**Prediction:** AUROC, AUPRC, F1, Recall@K

**Efficiency:** avg tokens, avg retrieved nodes, avg retrieved edges, latency

**Evidence:** short-cycle count, high-risk neighbor count, evidence sparsity, subgraph density

**Policy:** action distribution, average steps, stop rate

## Installation

```bash
cd agea
pip install -r requirements.txt
```

## Usage

### Training-Free AGEA (heuristic policy)

```bash
python run_training_free.py --config configs/yelpchi.yaml
```

### Full Evaluation (all baselines)

```bash
python evaluate.py --config configs/yelpchi.yaml
```

### GRPO Training

```bash
python train_grpo.py --config configs/yelpchi.yaml --save_path agea_grpo_policy.pt
```

### With LLM Policy

Set `policy.type: llm` in the config and provide API credentials:

```yaml
policy:
  type: llm
  model: gpt-4o-mini
  api_key: YOUR_KEY
```

### Limit Test Nodes (for debugging)

```bash
python run_training_free.py --config configs/yelpchi.yaml --limit 50
```

## Contribution

AGEA jointly optimizes graph evidence acquisition and LLM-based fraud prediction. Unlike fixed-retrieval approaches, AGEA adapts which graph evidence to acquire based on the target node's context, achieving better fraud detection under the same token budget.

## Repository Structure

```
agea/
  configs/          # Dataset configurations (YAML)
  data/             # Data loaders
  graph_tools/      # Deterministic graph operators
  policy/           # Evidence acquisition controllers
  prompts/          # Prompt construction (raw/compressed/evidence-only)
  models/           # Classifiers and LLM reasoner
  run_training_free.py   # Stage 1: training-free AGEA
  train_grpo.py          # Stage 3: GRPO training
  evaluate.py            # Full evaluation with baselines
  utils.py               # Shared utilities
```
