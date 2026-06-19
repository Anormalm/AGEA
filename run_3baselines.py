#!/usr/bin/env python
"""Run PMP, ConsisGAD, GAAP baselines only."""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_dataset
from models.classifier import PMPClassifier, ConsisGADClassifier, GAAPClassifier
from utils import load_config, compute_metrics


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
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits).cpu()
    return probs


def train_classifier_consistency(model, data, device, epochs=100, lr=1e-4):
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


def main():
    device = torch.device("cpu")
    cfg = load_config("configs/yelp_spam.yaml")

    root = cfg["dataset"].get("root")
    print(f"Loading yelp_spam from {root}")
    dataset = load_dataset("yelp_spam", root)
    data = dataset.data
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")
    print(f"Fraud ratio: {data.y.float().mean().item():.3f}")

    in_dim = data.num_features
    tcfg = cfg.get("training", {})
    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze(-1)
    y_true = data.y[test_indices].numpy()
    print(f"Test set size: {len(test_indices)}")

    models = {
        "PMP": PMPClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "ConsisGAD": ConsisGADClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
        "GAAP": GAAPClassifier(in_dim, tcfg.get("hidden_dim", 128), tcfg.get("num_layers", 2), tcfg.get("dropout", 0.3)),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        is_consistency = isinstance(model, (ConsisGADClassifier, GAAPClassifier))
        if is_consistency:
            probs = train_classifier_consistency(model, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))
        else:
            probs = train_classifier(model, data, device, epochs=tcfg.get("epochs", 100), lr=tcfg.get("lr", 1e-4))
        y_prob = probs[test_indices].numpy()
        metrics = compute_metrics(y_true, y_prob, k=cfg.get("evaluation", {}).get("recall_k", 100))
        results[name] = metrics
        print(f"  {name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, MacroF1={metrics['macro_f1']:.4f}")

    print(f"\n{'='*50}")
    print(f"{'Method':<15} {'MacroF1':>8} {'AUROC':>8} {'AUPRC':>8}")
    print(f"{'-'*50}")
    for name, m in results.items():
        print(f"{name:<15} {m['macro_f1']:>8.4f} {m['auroc']:>8.4f} {m['auprc']:>8.4f}")


if __name__ == "__main__":
    main()
