#!/usr/bin/env python
"""Generate all AGEA paper figures (PDF) from experiment results.

Update the DATA dictionaries below with values from real-data runs,
then run:  python generate_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agea", "AuthorKit27", "Figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────
C_GNN  = "#4C72B0"   # blue
C_LLM  = "#DD8452"   # orange
C_AGEA_H = "#55A868" # green
C_AGEA_G = "#C44E52" # red
C_DGP  = "#8172B3"   # purple

METHOD_COLORS = {
    "MLP": C_GNN, "GCN": C_GNN, "GraphSAGE": C_GNN, "GAT": C_GNN,
    "PMP": C_GNN, "ConsisGAD": C_GNN,
    "LLM-zero": C_LLM, "TAPE": C_LLM, "GraphGPT": C_LLM,
    "HiGPT": C_LLM, "InstructGLM": C_LLM,
    "DGP†": C_DGP,
    "AGEA-Heur.": C_AGEA_H, "AGEA-GRPO": C_AGEA_G,
}

# ── Data: update AUPRC "---" values after real-data runs ───────────────
# Format: {method: (MacroF1, AUROC, AUPRC)}
# Use None for missing AUPRC (will show as lighter bar)

YELP = {
    "MLP":          (0.5068, 0.5582, 0.5480),
    "GCN":          (0.4206, 0.4879, 0.4464),
    "GraphSAGE":    (0.5733, 0.6957, 0.6371),
    "GAT":          (0.4999, 0.5037, 0.4867),
    "PMP":          (0.5515, 0.5966, 0.5536),
    "ConsisGAD":    (0.4975, 0.5249, 0.5098),
    "LLM-zero":     (0.4310, 0.5557, 0.4425),
    "TAPE":         (0.5461, 0.5809, 0.4807),
    "GraphGPT":     (0.5781, 0.5880, 0.4767),
    "HiGPT":        (0.5737, 0.5999, 0.5024),
    "InstructGLM":  (0.5912, 0.5952, 0.4825),
    "DGP†":         (0.6907, 0.8428, 0.4887),
    "AGEA-Heur.":   (0.7609, 0.7914, 0.7066),
    "AGEA-GRPO":    (0.7758, 0.8065, 0.7150),
}

AMAZON = {
    "MLP":          (0.5220, 0.6194, 0.1040),
    "GCN":          (0.6255, 0.7053, 0.3203),
    "GraphSAGE":    (0.5710, 0.7650, 0.2038),
    "GAT":          (0.6525, 0.7839, 0.5093),
    "PMP":          (0.4964, 0.5080, 0.1863),
    "ConsisGAD":    (0.4791, 0.3946, 0.1621),
    "LLM-zero":     (0.4395, 0.5426, 0.5033),
    "TAPE":         (0.8271, 0.8234, 0.7553),
    "GraphGPT":     (0.8222, 0.8181, 0.7597),
    "HiGPT":        (0.8329, 0.8289, 0.7545),
    "InstructGLM":  (0.8211, 0.8164, 0.7557),
    "DGP†":         (0.6691, 0.7732, 0.3463),
    "AGEA-Heur.":   (0.9124, 0.9711, 0.9623),
    "AGEA-GRPO":    (0.9145, 0.9758, 0.9701),
}

POLITIFACT = {
    "MLP":          (0.6473, 0.6815, 0.7036),
    "GCN":          (0.6266, 0.6720, 0.6614),
    "GraphSAGE":    (0.6381, 0.6891, 0.7049),
    "GAT":          (0.4399, 0.5775, 0.5872),
    "PMP":          (0.6198, 0.6403, 0.6805),
    "ConsisGAD":    (0.4757, 0.4453, 0.4677),
    "LLM-zero":     (0.5856, 0.6057, 0.5466),
    "TAPE":         (0.6501, 0.6511, 0.6251),
    "GraphGPT":     (0.6493, 0.6500, 0.6599),
    "HiGPT":        (0.5585, 0.5679, 0.5811),
    "InstructGLM":  (0.7139, 0.7157, 0.7167),
    "DGP†":         (None, None, None),
    "AGEA-Heur.":   (0.7488, 0.8107, 0.7809),
    "AGEA-GRPO":    (0.7819, 0.8256, 0.8324),
}

BUZZFEED = {
    "MLP":          (0.4695, 0.5284, 0.5527),
    "GCN":          (0.5835, 0.6021, 0.5801),
    "GraphSAGE":    (0.5396, 0.5247, 0.5562),
    "GAT":          (0.5223, 0.5572, 0.5742),
    "PMP":          (0.5453, 0.5300, 0.5490),
    "ConsisGAD":    (0.4740, 0.4443, 0.4996),
    "LLM-zero":     (0.4960, 0.5129, 0.5285),
    "TAPE":         (0.4411, 0.5547, 0.5694),
    "GraphGPT":     (0.4952, 0.5277, 0.5334),
    "HiGPT":        (0.4411, 0.5547, 0.5618),
    "InstructGLM":  (0.7161, 0.7339, 0.7059),
    "DGP†":         (None, None, None),
    "AGEA-Heur.":   (0.9391, 0.9470, 0.8753),
    "AGEA-GRPO":    (0.8301, 0.8747, 0.7908),
}

# ── Figure 1: Main comparison (parallel coordinates) ──────────────────

# Category labels for row grouping
_METHOD_CAT = {
    "MLP": "GNN", "GCN": "GNN", "GraphSAGE": "GNN", "GAT": "GNN",
    "PMP": "GNN", "ConsisGAD": "GNN",
    "LLM-zero": "LLM", "TAPE": "LLM", "GraphGPT": "LLM",
    "HiGPT": "LLM", "InstructGLM": "LLM", "DGP†": "LLM",
    "AGEA-Heur.": "AGEA", "AGEA-GRPO": "AGEA",
}

_CAT_LINESTYLE = {"GNN": "--", "LLM": ":", "AGEA": "-"}


def plot_main_comparison():
    """Parallel coordinates: one line per method across metric×dataset axes."""
    metrics = ["MacroF1", "AUROC", "AUPRC"]
    datasets = [("Yelp-Chi", YELP), ("Amazon", AMAZON),
                ("PolitiFact", POLITIFACT), ("BuzzFeed", BUZZFEED)]

    # Build axis labels and data
    methods = list(YELP.keys())
    # Two-line labels: metric on top, dataset abbreviation below
    ds_abbr = {"Yelp-Chi": "Yelp", "Amazon": "Amz",
               "PolitiFact": "PF", "BuzzFeed": "BF"}
    axis_labels = []
    for ds_name, _ in datasets:
        for met in metrics:
            axis_labels.append(f"{met}\n{ds_abbr[ds_name]}")

    n_axes = len(axis_labels)
    xs = np.arange(n_axes)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Plot each method
    for m in methods:
        cat = _METHOD_CAT[m]
        vals = []
        for _, data in datasets:
            if m in data and data[m][0] is not None:
                for mi in range(3):
                    v = data[m][mi]
                    vals.append(v if v is not None else np.nan)
            else:
                vals.extend([np.nan, np.nan, np.nan])

        color = METHOD_COLORS.get(m, "#888888")
        lw = 2.5 if m.startswith("AGEA") else 1.0
        alpha = 1.0 if m.startswith("AGEA") else 0.45
        ls = _CAT_LINESTYLE.get(cat, "-")
        zorder = 10 if m.startswith("AGEA") else 3

        ax.plot(xs, vals, color=color, linewidth=lw, alpha=alpha,
                linestyle=ls, marker="o", markersize=3 if not m.startswith("AGEA") else 6,
                label=m, zorder=zorder)

    # Axis styling
    ax.set_xticks(xs)
    ax.set_xticklabels(axis_labels, fontsize=6.5)
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Score", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelsize=7)

    # Vertical lines to separate datasets
    for sep in [2.5, 5.5, 8.5]:
        ax.axvline(x=sep, color="gray", linestyle="-", linewidth=0.8, alpha=0.4)

    # Dataset labels at top
    dataset_centers = [1.0, 4.0, 7.0, 10.0]
    for center, (ds_name, _) in zip(dataset_centers, datasets):
        ax.text(center, 1.02, ds_name, ha="center", fontsize=8, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Legend — compact, outside plot
    from matplotlib.lines import Line2D
    legend_handles = []
    for m in methods:
        if _METHOD_CAT[m] == "GNN":
            legend_handles.append(Line2D([0], [0], color=METHOD_COLORS[m],
                                         linestyle="--", linewidth=1, alpha=0.7, label=m))
    for m in methods:
        if _METHOD_CAT[m] == "LLM":
            legend_handles.append(Line2D([0], [0], color=METHOD_COLORS[m],
                                         linestyle=":", linewidth=1, alpha=0.7, label=m))
    for m in methods:
        if m.startswith("AGEA"):
            legend_handles.append(Line2D([0], [0], color=METHOD_COLORS[m],
                                         linestyle="-", linewidth=2.5, label=m))

    ax.legend(handles=legend_handles, loc="lower left", fontsize=5.5,
              ncol=3, framealpha=0.9, columnspacing=0.8,
              handletextpad=0.4, borderpad=0.4)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "main_comparison.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Tool ablation ───────────────────────────────────────────

TOOL_ABLATION = {
    "Yelp-Chi": {
        "All tools":  0.7624,
        "w/o Expand1Hop": None,  # degenerate: no evidence gathered
        "w/o Expand2Hop": 0.7588,
        "w/o PPRTopK": 0.7623,
        "w/o Community": 0.7513,
        "w/o ShortCycle": 0.9372,
        "w/o PruneTopK": 0.8344,
    },
    "Amazon": {
        "All tools":  0.9129,
        "w/o Expand1Hop": None,
        "w/o Expand2Hop": 0.9176,
        "w/o PPRTopK": 0.9448,
        "w/o Community": 0.9243,
        "w/o ShortCycle": 0.9233,
        "w/o PruneTopK": 0.9022,
    },
    "PolitiFact": {
        "All tools":  0.7409,
        "w/o Expand1Hop": None,
        "w/o Expand2Hop": 0.7405,
        "w/o PPRTopK": 0.7708,
        "w/o Community": 0.7105,
        "w/o ShortCycle": 0.9940,
        "w/o PruneTopK": 0.7468,
    },
    "BuzzFeed": {
        "All tools":  0.9443,
        "w/o Expand1Hop": None,
        "w/o Expand2Hop": 0.8965,
        "w/o PPRTopK": 0.9762,
        "w/o Community": 0.8882,
        "w/o ShortCycle": 0.9762,
        "w/o PruneTopK": 0.9124,
    },
}

def plot_tool_ablation():
    datasets = list(TOOL_ABLATION.items())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, (ds, data) in zip(axes.flat, datasets):
        tools = [t for t in data if data[t] is not None]
        vals = [data[t] for t in tools]
        colors = [C_AGEA_H if t == "All tools" else "#BBBBBB" for t in tools]

        bars = ax.barh(range(len(tools)), vals, color=colors, edgecolor="white")
        ax.set_yticks(range(len(tools)))
        ax.set_yticklabels(tools, fontsize=7)
        ax.set_xlabel("MacroF1", fontsize=8)
        ax.set_title(ds, fontsize=9)
        ax.set_xlim(0.4, 1.05)
        ax.axvline(x=data["All tools"], color=C_AGEA_H, linestyle="--", linewidth=0.8, alpha=0.5)

        for i, v in enumerate(vals):
            delta = ((v - data["All tools"]) / data["All tools"]) * 100
            sign = "+" if delta >= 0 else ""
            ax.text(v + 0.005, i, f"{v:.3f} ({sign}{delta:.1f}%)", va="center", fontsize=6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tool_ablation.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Feature ablation ────────────────────────────────────────

FEATURE_ABLATION = {
    "Yelp-Chi": {
        "Full-8feat":    0.7633,
        "No-structural": 0.6736,
        "No-label":      0.6590,
        "No-classifier": 0.6812,
        "Classifier-only": 0.6500,
    },
    "Amazon": {
        "Full-8feat":    0.9134,
        "No-structural": 0.8740,
        "No-label":      0.8851,
        "No-classifier": 0.9112,
        "Classifier-only": 0.8901,
    },
    "PolitiFact": {
        "Full-8feat":    0.7346,
        "No-structural": 0.6455,
        "No-label":      0.6560,
        "No-classifier": 0.7348,
        "Classifier-only": 0.6455,
    },
    "BuzzFeed": {
        "Full-8feat":    0.9123,
        "No-structural": 0.4821,
        "No-label":      0.4694,
        "No-classifier": 0.9602,
        "Classifier-only": 0.4888,
    },
}

def plot_feature_ablation():
    datasets = list(FEATURE_ABLATION.items())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, (ds, data) in zip(axes.flat, datasets):
        feats = list(data.keys())
        vals = list(data.values())
        colors = [C_AGEA_H if f == "Full-8feat" else "#BBBBBB" for f in feats]

        bars = ax.barh(range(len(feats)), vals, color=colors, edgecolor="white")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=7)
        ax.set_xlabel("MacroF1", fontsize=8)
        ax.set_title(ds, fontsize=9)
        ax.set_xlim(0.4, 1.05)

        for i, v in enumerate(vals):
            delta = ((v - data["Full-8feat"]) / data["Full-8feat"]) * 100
            sign = "+" if delta >= 0 else ""
            ax.text(v + 0.005, i, f"{v:.3f} ({sign}{delta:.1f}%)", va="center", fontsize=6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feature_ablation.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Budget ablation ─────────────────────────────────────────

BUDGET_ABLATION = {
    "Yelp-Chi":     {2: 0.9301, 4: 0.7719, 6: 0.7643, 8: 0.7435, 10: 0.7299},
    "Amazon":       {2: 0.9248, 4: 0.9193, 6: 0.9119, 8: 0.9104, 10: 0.9138},
    "PolitiFact":   {2: 0.7826, 4: 0.7098, 6: 0.7287, 8: 0.7343, 10: 0.7225},
    "BuzzFeed":     {2: 0.9036, 4: 0.9523, 6: 0.9206, 8: 0.8722, 10: 0.9280},
}

DS_COLORS = {
    "Yelp-Chi": "#4C72B0",
    "Amazon": "#DD8452",
    "PolitiFact": "#55A868",
    "BuzzFeed": "#C44E52",
}

def plot_budget_ablation():
    fig, ax = plt.subplots(figsize=(6, 4))
    for ds, data in BUDGET_ABLATION.items():
        steps = sorted(data.keys())
        vals = [data[s] for s in steps]
        ax.plot(steps, vals, "o-", color=DS_COLORS[ds], label=ds, linewidth=1.5, markersize=5)

    ax.set_xlabel("max\_steps", fontsize=10)
    ax.set_ylabel("MacroF1", fontsize=10)
    ax.set_xticks(list(BUDGET_ABLATION["Yelp-Chi"].keys()))
    ax.legend(fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "budget_ablation.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating AGEA paper figures...")
    print()
    print("1/4  Main comparison (MacroF1 + AUROC + AUPRC)")
    plot_main_comparison()
    print()
    print("2/4  Tool ablation")
    plot_tool_ablation()
    print()
    print("3/4  Feature ablation")
    plot_feature_ablation()
    print()
    print("4/4  Budget ablation")
    plot_budget_ablation()
    print()
    print("Done. Update None values in YELP/AMAZON dicts after real-data runs, then re-run.")
