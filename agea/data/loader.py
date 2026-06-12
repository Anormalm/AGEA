"""Data loading for fraud detection benchmarks.

Pure PyTorch — no torch_geometric dependency required.
Uses a simple GraphData container instead of PyG Data objects.
"""

import os
import json
import torch
import numpy as np


class GraphData:
    """Simple graph data container (replaces torch_geometric.data.Data)."""

    def __init__(self, x=None, edge_index=None, y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self._store = kwargs

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __contains__(self, key):
        return key in self._store

    @property
    def num_nodes(self):
        if self.x is not None:
            return self.x.size(0)
        if self.edge_index is not None:
            return int(self.edge_index.max().item()) + 1
        return 0

    @property
    def num_edges(self):
        if self.edge_index is not None:
            return self.edge_index.size(1)
        return 0

    @property
    def num_features(self):
        if self.x is not None:
            return self.x.size(1)
        return 0


class FraudGraphDataset:
    """Container for a fraud graph dataset."""

    def __init__(self, data: GraphData, name: str, node_text: dict = None, edge_text: dict = None):
        self.data = data
        self.name = name
        self.node_text = node_text or {}
        self.edge_text = edge_text or {}

    @property
    def num_nodes(self):
        return self.data.num_nodes

    @property
    def num_edges(self):
        return self.data.num_edges

    @property
    def num_features(self):
        return self.data.num_features

    @property
    def num_classes(self):
        return int(self.data.y.max().item()) + 1

    def get_split(self, split: str):
        mask = self.data[f"{split}_mask"]
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return indices


def _load_generic(root: str) -> FraudGraphDataset:
    """Load from generic .pt format: x.pt, edge_index.pt, y.pt, masks."""
    x = torch.load(os.path.join(root, "x.pt"), weights_only=True)
    edge_index = torch.load(os.path.join(root, "edge_index.pt"), weights_only=True).long()
    y = torch.load(os.path.join(root, "y.pt"), weights_only=True).long()

    data = GraphData(x=x, edge_index=edge_index, y=y)

    for split in ["train", "val", "test"]:
        path = os.path.join(root, f"{split}_mask.pt")
        if os.path.exists(path):
            data[f"{split}_mask"] = torch.load(path, weights_only=True).bool()
        else:
            data[f"{split}_mask"] = _default_split(y, split)

    node_text, edge_text = {}, {}
    nt_path = os.path.join(root, "node_text.json")
    et_path = os.path.join(root, "edge_text.json")
    if os.path.exists(nt_path):
        with open(nt_path) as f:
            node_text = json.load(f)
    if os.path.exists(et_path):
        with open(et_path) as f:
            edge_text = json.load(f)

    return FraudGraphDataset(data, name="generic", node_text=node_text, edge_text=edge_text)


def _default_split(y: torch.Tensor, split: str) -> torch.Tensor:
    """Create a 60/20/20 random split if masks are not provided."""
    n = y.size(0)
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    mask = torch.zeros(n, dtype=torch.bool)
    if split == "train":
        mask[perm[:train_end]] = True
    elif split == "val":
        mask[perm[train_end:val_end]] = True
    else:
        mask[perm[val_end:]] = True
    return mask


def _load_yelpchi(root: str) -> FraudGraphDataset:
    """Load YelpChi dataset. Falls back to generic, then synthetic."""
    processed = os.path.join(root, "yelpchi_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if not isinstance(data, GraphData):
            # Migrate old PyG Data
            data = GraphData(x=data.x, edge_index=data.edge_index, y=data.y,
                             train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        return FraudGraphDataset(data, name="yelpchi")

    # Try PyG dataset
    try:
        from torch_geometric.datasets import YelpChi
        dataset = YelpChi(root=root)
        d = dataset[0]
        data = GraphData(x=d.x, edge_index=d.edge_index, y=d.y)
        data["train_mask"] = _default_split(d.y, "train")
        data["val_mask"] = _default_split(d.y, "val")
        data["test_mask"] = _default_split(d.y, "test")
        os.makedirs(root, exist_ok=True)
        torch.save(data, processed)
        return FraudGraphDataset(data, name="yelpchi")
    except (ImportError, Exception):
        pass

    if os.path.exists(os.path.join(root, "x.pt")):
        ds = _load_generic(root)
        ds.name = "yelpchi"
        return ds

    return _synthetic_fraud_graph("yelpchi", n_nodes=5000, n_features=32, fraud_ratio=0.14)


def _load_dgraphfin(root: str) -> FraudGraphDataset:
    """Load DGraphFin dataset. Falls back to generic, then synthetic."""
    processed = os.path.join(root, "dgraphfin_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if not isinstance(data, GraphData):
            data = GraphData(x=data.x, edge_index=data.edge_index, y=data.y,
                             train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        return FraudGraphDataset(data, name="dgraphfin")

    try:
        from torch_geometric.datasets import DGraphFin
        dataset = DGraphFin(root=root)
        d = dataset[0]
        data = GraphData(x=d.x, edge_index=d.edge_index, y=d.y)
        data["train_mask"] = _default_split(d.y, "train")
        data["val_mask"] = _default_split(d.y, "val")
        data["test_mask"] = _default_split(d.y, "test")
        os.makedirs(root, exist_ok=True)
        torch.save(data, processed)
        return FraudGraphDataset(data, name="dgraphfin")
    except (ImportError, Exception):
        pass

    if os.path.exists(os.path.join(root, "x.pt")):
        ds = _load_generic(root)
        ds.name = "dgraphfin"
        return ds

    return _synthetic_fraud_graph("dgraphfin", n_nodes=8000, n_features=17, fraud_ratio=0.05)


def _load_elliptic(root: str) -> FraudGraphDataset:
    """Load Elliptic Bitcoin dataset. Falls back to generic, then synthetic."""
    processed = os.path.join(root, "elliptic_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if not isinstance(data, GraphData):
            data = GraphData(x=data.x, edge_index=data.edge_index, y=data.y,
                             train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        return FraudGraphDataset(data, name="elliptic")

    try:
        from torch_geometric.datasets import EllipticBitcoin
        dataset = EllipticBitcoin(root=root)
        d = dataset[0]
        data = GraphData(x=d.x, edge_index=d.edge_index, y=d.y)
        data["train_mask"] = _default_split(d.y, "train")
        data["val_mask"] = _default_split(d.y, "val")
        data["test_mask"] = _default_split(d.y, "test")
        os.makedirs(root, exist_ok=True)
        torch.save(data, processed)
        return FraudGraphDataset(data, name="elliptic")
    except (ImportError, Exception):
        pass

    if os.path.exists(os.path.join(root, "x.pt")):
        ds = _load_generic(root)
        ds.name = "elliptic"
        return ds

    return _synthetic_fraud_graph("elliptic", n_nodes=10000, n_features=165, fraud_ratio=0.10)


def _synthetic_fraud_graph(name: str, n_nodes: int, n_features: int, fraud_ratio: float) -> FraudGraphDataset:
    """Generate a synthetic fraud graph for development and testing."""
    torch.manual_seed(42)
    x = torch.randn(n_nodes, n_features)
    y = torch.zeros(n_nodes, dtype=torch.long)
    fraud_count = int(n_nodes * fraud_ratio)
    fraud_idx = torch.randperm(n_nodes)[:fraud_count]
    y[fraud_idx] = 1

    # Build edges: higher connectivity within fraud group
    src, dst = [], []
    for i in range(n_nodes):
        n_edges = torch.poisson(torch.tensor(3.0 if y[i] == 0 else 6.0)).item()
        neighbors = torch.randint(0, n_nodes, (int(n_edges),))
        for j in neighbors:
            if j.item() != i:
                src.append(i)
                dst.append(j.item())
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)

    data = GraphData(x=x, edge_index=edge_index, y=y)
    data["train_mask"] = _default_split(y, "train")
    data["val_mask"] = _default_split(y, "val")
    data["test_mask"] = _default_split(y, "test")

    return FraudGraphDataset(data, name=name)


def _load_yelp_spam(root: str) -> FraudGraphDataset:
    """Load Yelp spam review dataset (Hotel + Restaurant).

    Expects directory structure:
      root/output_meta_yelpHotelData_NRYRcleaned.txt  (structured: date rev_id reviewer_id product_id label ...)
      root/output_meta_yelpResData_NRYRcleaned.txt
      root/output_review_yelpHotelData_NRYRcleaned.txt (review text)
      root/output_review_yelpResData_NRYRcleaned.txt

    Nodes = reviews. Edges connect reviews sharing the same reviewer (R-U-R),
    same product in the same month (R-T-R), or same product with same rating (R-S-R).
    """
    processed = os.path.join(root, "yelp_spam_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        return FraudGraphDataset(data, name="yelp_spam")

    # Parse structured metadata from both Hotel and Restaurant
    reviews = []  # list of (date, rev_id, reviewer_id, product_id, label, star)
    node_text = {}

    for subset in ["Hotel", "Res"]:
        meta_path = os.path.join(root, f"output_meta_yelp{subset}Data_NRYRcleaned.txt")
        text_path = os.path.join(root, f"output_review_yelp{subset}Data_NRYRcleaned.txt")

        if not os.path.exists(meta_path):
            continue

        texts = {}
        if os.path.exists(text_path):
            with open(text_path) as f:
                for line in f:
                    texts[len(texts)] = line.strip()

        with open(meta_path) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                date = parts[0]
                # Some IDs have spaces; the last 5 fields are: label, col6, col7, col8, star
                # Parse from right
                star = int(parts[-1])
                label = parts[-5]
                # reviewer_id and product_id are between date and label
                # Format: date rev_id reviewer_id product_id label N1 N2 N3 star
                reviewer_id = parts[2] if len(parts) >= 9 else parts[1]
                product_id = parts[3] if len(parts) >= 9 else parts[2]
                rev_id = parts[1] if len(parts) >= 9 else ""

                reviews.append({
                    "date": date, "rev_id": rev_id, "reviewer_id": reviewer_id,
                    "product_id": product_id, "label": 1 if label == "Y" else 0,
                    "star": star,
                })
                if idx in texts:
                    node_text[len(reviews) - 1] = texts[idx]

    if not reviews:
        return _synthetic_fraud_graph("yelp_spam", 5000, 32, 0.13)

    n = len(reviews)

    # Build feature matrix: star_rating (1-5), one-hot star, month, fraud ratio proxies
    import re
    from collections import defaultdict

    x_list = []
    reviewer_counts = defaultdict(int)
    product_counts = defaultdict(int)
    for r in reviews:
        reviewer_counts[r["reviewer_id"]] += 1
        product_counts[r["product_id"]] += 1

    for r in reviews:
        # Features: star (normalized), star one-hot(5), reviewer_activity, product_popularity
        star = r["star"]
        star_onehot = [1 if star == i else 0 for i in range(1, 6)]
        reviewer_act = min(reviewer_counts[r["reviewer_id"]] / 10.0, 1.0)
        product_pop = min(product_counts[r["product_id"]] / 50.0, 1.0)
        # Parse month
        try:
            parts_date = r["date"].split("/")
            month = int(parts_date[0]) / 12.0
        except (ValueError, IndexError):
            month = 0.5
        feat = [star / 5.0] + star_onehot + [reviewer_act, product_pop, month]
        x_list.append(feat)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor([r["label"] for r in reviews], dtype=torch.long)

    # Build edges: R-U-R (same reviewer), R-T-R (same product same month), R-S-R (same product same star)
    reviewer_to_reviews = defaultdict(list)
    product_month_to_reviews = defaultdict(list)
    product_star_to_reviews = defaultdict(list)

    for i, r in enumerate(reviews):
        reviewer_to_reviews[r["reviewer_id"]].append(i)
        try:
            parts_date = r["date"].split("/")
            month_key = f"{r['product_id']}_{parts_date[0]}"
        except (ValueError, IndexError):
            month_key = r["product_id"]
        product_month_to_reviews[month_key].append(i)
        star_key = f"{r['product_id']}_{r['star']}"
        product_star_to_reviews[star_key].append(i)

    src, dst = [], []
    for rev_list in reviewer_to_reviews.values():
        for i in range(len(rev_list)):
            for j in range(i + 1, len(rev_list)):
                src.extend([rev_list[i], rev_list[j]])
                dst.extend([rev_list[j], rev_list[i]])
    for rev_list in product_month_to_reviews.values():
        if len(rev_list) > 100:
            continue
        for i in range(len(rev_list)):
            for j in range(i + 1, len(rev_list)):
                src.extend([rev_list[i], rev_list[j]])
                dst.extend([rev_list[j], rev_list[i]])
    for rev_list in product_star_to_reviews.values():
        if len(rev_list) > 100:
            continue
        for i in range(len(rev_list)):
            for j in range(i + 1, len(rev_list)):
                src.extend([rev_list[i], rev_list[j]])
                dst.extend([rev_list[j], rev_list[i]])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    data = GraphData(x=x, edge_index=edge_index, y=y)
    data["train_mask"] = _default_split(y, "train")
    data["val_mask"] = _default_split(y, "val")
    data["test_mask"] = _default_split(y, "test")

    os.makedirs(os.path.dirname(processed) if os.path.dirname(processed) else ".", exist_ok=True)
    torch.save(data, processed)

    return FraudGraphDataset(data, name="yelp_spam", node_text=node_text)


def _load_amazon(root: str) -> FraudGraphDataset:
    """Load Amazon review fraud dataset (2023 format, JSONL).

    Expects: root/All_Beauty.jsonl.gz or similar category files.
    Fraud heuristics: unverified purchase, extreme ratings from low-activity users,
    temporal burst reviews on the same product.

    Nodes = reviews. Edges: same-user (R-U-R), same-product (R-T-R),
    same-product same-rating (R-S-R).
    """
    processed = os.path.join(root, "amazon_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        return FraudGraphDataset(data, name="amazon")

    import gzip

    # Find JSONL files
    jsonl_files = []
    if os.path.exists(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".jsonl.gz") and not f.startswith("meta_"):
                jsonl_files.append(os.path.join(root, f))

    if not jsonl_files:
        print("[AGEA] No Amazon JSONL files found. Generating synthetic graph.")
        return _synthetic_fraud_graph("amazon", 5000, 32, 0.10)

    reviews = []
    node_text = {}
    from collections import defaultdict

    for fpath in jsonl_files:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    import json as _json
                    r = _json.loads(line.strip())
                except Exception:
                    continue
                reviews.append({
                    "user_id": r.get("user_id", ""),
                    "product_id": r.get("parent_asin", ""),
                    "rating": r.get("rating", 3.0),
                    "verified": r.get("verified_purchase", False),
                    "timestamp": r.get("timestamp", 0),
                    "helpful_vote": r.get("helpful_vote", 0),
                    "text": r.get("text", ""),
                })
                node_text[len(reviews) - 1] = r.get("text", "")[:500]

        # Limit to first 50K reviews per category for tractability
        if len(reviews) >= 50000:
            break

    if not reviews:
        return _synthetic_fraud_graph("amazon", 5000, 32, 0.10)

    n = len(reviews)

    # Fraud heuristic: unverified + extreme rating + low helpful votes
    reviewer_counts = defaultdict(int)
    product_counts = defaultdict(int)
    for r in reviews:
        reviewer_counts[r["user_id"]] += 1
        product_counts[r["product_id"]] += 1

    x_list = []
    y_list = []
    for r in reviews:
        rating = r["rating"]
        verified = float(r["verified"])
        helpful = min(r["helpful_vote"] / 10.0, 1.0)
        reviewer_act = min(reviewer_counts[r["user_id"]] / 10.0, 1.0)
        product_pop = min(product_counts[r["product_id"]] / 50.0, 1.0)
        star_onehot = [1 if rating == i else 0 for i in range(1, 6)]
        is_extreme = 1.0 if rating in [1.0, 5.0] else 0.0

        # Fraud label heuristic: unverified + extreme rating + low helpful + low activity user
        fraud_score = (1.0 - verified) * 0.3 + is_extreme * 0.2 + (1.0 - helpful) * 0.2 + (1.0 - reviewer_act) * 0.3
        fraud_label = 1 if fraud_score > 0.7 else 0

        feat = [rating / 5.0, verified, helpful, reviewer_act, product_pop, is_extreme] + star_onehot
        x_list.append(feat)
        y_list.append(fraud_label)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    # Build edges
    user_to_reviews = defaultdict(list)
    product_to_reviews = defaultdict(list)
    product_star_to_reviews = defaultdict(list)

    for i, r in enumerate(reviews):
        user_to_reviews[r["user_id"]].append(i)
        product_to_reviews[r["product_id"]].append(i)
        product_star_to_reviews[f"{r['product_id']}_{int(r['rating'])}"].append(i)

    src, dst = [], []
    for rev_list in user_to_reviews.values():
        if len(rev_list) > 50:
            continue
        for i in range(len(rev_list)):
            for j in range(i + 1, len(rev_list)):
                src.extend([rev_list[i], rev_list[j]])
                dst.extend([rev_list[j], rev_list[i]])
    for rev_list in product_to_reviews.values():
        if len(rev_list) > 50:
            continue
        for i in range(len(rev_list)):
            for j in range(i + 1, len(rev_list)):
                src.extend([rev_list[i], rev_list[j]])
                dst.extend([rev_list[j], rev_list[i]])
    for rev_list in product_star_to_reviews.values():
        if len(rev_list) > 50:
            continue
        for i in range(len(rev_list)):
            for j in range(i + 1, len(rev_list)):
                src.extend([rev_list[i], rev_list[j]])
                dst.extend([rev_list[j], rev_list[i]])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    data = GraphData(x=x, edge_index=edge_index, y=y)
    data["train_mask"] = _default_split(y, "train")
    data["val_mask"] = _default_split(y, "val")
    data["test_mask"] = _default_split(y, "test")

    os.makedirs(os.path.dirname(processed) if os.path.dirname(processed) else ".", exist_ok=True)
    torch.save(data, processed)

    return FraudGraphDataset(data, name="amazon", node_text=node_text)


_LOADERS = {
    "yelpchi": _load_yelpchi,
    "dgraphfin": _load_dgraphfin,
    "elliptic": _load_elliptic,
    "yelp_spam": _load_yelp_spam,
    "amazon": _load_amazon,
}


def load_dataset(name: str, root: str = None) -> FraudGraphDataset:
    """Load a fraud detection dataset by name.

    Supported: yelpchi, dgraphfin, elliptic.
    Falls back to generic .pt loader, then synthetic data.
    """
    if name in _LOADERS:
        root = root or f"./data/{name}"
        return _LOADERS[name](root)

    if root and os.path.exists(os.path.join(root, "x.pt")):
        return _load_generic(root)
    print(f"[AGEA] Dataset '{name}' not found. Generating synthetic graph for development.")
    return _synthetic_fraud_graph(name, n_nodes=5000, n_features=32, fraud_ratio=0.10)
