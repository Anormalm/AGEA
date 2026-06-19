#!/usr/bin/env python
"""Multi-seed AGEA experiments for mean±std reporting.

Runs AGEA (heuristic + GRPO) with multiple random seeds and reports
mean ± standard deviation for MacroF1, AUROC, AUPRC.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_agea import run_agea, run_grpo
from data.loader import load_dataset
from utils import load_config


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yelp_spam",
                        choices=["yelp_spam", "amazon",
                                 "fakenews_politifact", "fakenews_buzzfeed"])
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated seeds")
    parser.add_argument("--mode", default="all",
                        choices=["all", "heuristic_raw", "heuristic_comp", "grpo"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}")

    cfg_path = f"configs/{args.dataset}.yaml"
    cfg = load_config(cfg_path)
    root = cfg["dataset"].get("root")
    dataset = load_dataset(args.dataset, root)
    data = dataset.data

    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    methods = []
    if args.mode in ("all", "heuristic_raw"):
        methods.append(("AGEA-heuristic-raw", "raw"))
    if args.mode in ("all", "heuristic_comp"):
        methods.append(("AGEA-heuristic-comp", "compressed"))
    if args.mode in ("all", "grpo"):
        methods.append(("AGEA-GRPO", None))

    all_results = {}  # method -> list of metric dicts

    for method_name, prompt_mode in methods:
        all_results[method_name] = []
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"{method_name} | seed={seed} | {args.dataset}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            if method_name == "AGEA-GRPO":
                metrics = run_grpo(data, cfg, device)
            else:
                metrics = run_agea(data, cfg, device, prompt_mode=prompt_mode)

            all_results[method_name].append(metrics)
            print(f"  MacroF1={metrics['macro_f1']:.4f}, "
                  f"AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")

    # Summary with mean ± std
    print(f"\n{'='*70}")
    print(f"MULTI-SEED RESULTS: {args.dataset} ({len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'MacroF1':>20} {'AUROC':>20} {'AUPRC':>20}")
    print(f"{'-'*70}")

    metric_keys = ["macro_f1", "auroc", "auprc"]
    for method_name, runs in all_results.items():
        row = [method_name]
        for key in metric_keys:
            vals = [r[key] for r in runs]
            mean = np.mean(vals)
            std = np.std(vals)
            row.append(f"{mean:.4f}±{std:.4f}")
        print(f"{row[0]:<25} {row[1]:>20} {row[2]:>20} {row[3]:>20}")


if __name__ == "__main__":
    main()
