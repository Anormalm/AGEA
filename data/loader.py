"""Data loading for fraud detection benchmarks.

Pure PyTorch — no torch_geometric dependency required.
Uses a simple GraphData container instead of PyG Data objects.
"""

import os
import json
import pickle
import random
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime


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

    def __getattr__(self, key):
        if key.startswith('_') or key in ('x', 'edge_index', 'y', 'num_nodes', 'num_edges', 'num_features'):
            raise AttributeError(key)
        try:
            return self._store[key]
        except KeyError:
            raise AttributeError(key)

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
    for name, fname in [("node_text", "node_text.json"), ("edge_text", "edge_text.json")]:
        fpath = os.path.join(root, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                locals()[name] = json.load(f)
    return FraudGraphDataset(data, name="generic", node_text=node_text, edge_text=edge_text)


def _default_split(y: torch.Tensor, split: str) -> torch.Tensor:
    n = y.size(0)
    g = torch.Generator()
    g.manual_seed(42)
    perm = torch.randperm(n, generator=g)
    train_end = int(0.1 * n)
    val_end = int(0.2 * n)
    mask = torch.zeros(n, dtype=torch.bool)
    if split == "train":
        mask[perm[:train_end]] = True
    elif split == "val":
        mask[perm[train_end:val_end]] = True
    else:
        mask[perm[val_end:]] = True
    return mask


def _balanced_split(labels, seed=42, val_neg_per_pos=1.2, test_neg_per_pos=1.2):
    """Balanced split following split.py: train 1:1, val/test 1:1.2 pos:neg.

    Positive class: 20/10/70 split.
    Train negatives sampled 1:1 with positives.
    Val/Test negatives at 1:1.2 ratio.
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(labels)
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]

    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    # Positive: 20/10/70
    pos_train_size = int(len(pos_idx) * 0.2)
    pos_val_size = int(len(pos_idx) * 0.1)

    pos_train = pos_idx[:pos_train_size]
    pos_val = pos_idx[pos_train_size:pos_train_size + pos_val_size]
    pos_test = pos_idx[pos_train_size + pos_val_size:]

    # Train negatives: 1:1
    neg_train_size = len(pos_train)
    neg_train = neg_idx[:neg_train_size]
    neg_remaining = neg_idx[neg_train_size:]

    # Val negatives: 1:1.2
    neg_val_size = round(len(pos_val) * val_neg_per_pos)
    neg_val = neg_remaining[:neg_val_size]
    neg_remaining = neg_remaining[neg_val_size:]

    # Test negatives: 1:1.2
    neg_test_size = round(len(pos_test) * test_neg_per_pos)
    neg_test = neg_remaining[:neg_test_size]

    train_idx = pos_train + neg_train
    val_idx = pos_val + neg_val
    test_idx = pos_test + neg_test

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def _load_yelpchi(root: str) -> FraudGraphDataset:
    processed = os.path.join(root, "yelpchi_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if not isinstance(data, GraphData):
            data = GraphData(x=data.x, edge_index=data.edge_index, y=data.y,
                             train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        return FraudGraphDataset(data, name="yelpchi")
    try:
        from torch_geometric.datasets import YelpChi
        d = YelpChi(root=root)[0]
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
    return _synthetic_fraud_graph("yelpchi", 5000, 32, 0.14)


def _load_dgraphfin(root: str) -> FraudGraphDataset:
    processed = os.path.join(root, "dgraphfin_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if not isinstance(data, GraphData):
            data = GraphData(x=data.x, edge_index=data.edge_index, y=data.y,
                             train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        return FraudGraphDataset(data, name="dgraphfin")
    try:
        from torch_geometric.datasets import DGraphFin
        d = DGraphFin(root=root)[0]
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
    return _synthetic_fraud_graph("dgraphfin", 8000, 17, 0.05)


def _load_elliptic(root: str) -> FraudGraphDataset:
    processed = os.path.join(root, "elliptic_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if not isinstance(data, GraphData):
            data = GraphData(x=data.x, edge_index=data.edge_index, y=data.y,
                             train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        return FraudGraphDataset(data, name="elliptic")
    try:
        from torch_geometric.datasets import EllipticBitcoin
        d = EllipticBitcoin(root=root)[0]
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
    return _synthetic_fraud_graph("elliptic", 10000, 165, 0.10)


def _synthetic_fraud_graph(name, n_nodes, n_features, fraud_ratio):
    torch.manual_seed(42)
    x = torch.randn(n_nodes, n_features)
    y = torch.zeros(n_nodes, dtype=torch.long)
    fraud_count = int(n_nodes * fraud_ratio)
    y[torch.randperm(n_nodes)[:fraud_count]] = 1
    src, dst = [], []
    for i in range(n_nodes):
        n_edges = torch.poisson(torch.tensor(3.0 if y[i] == 0 else 6.0)).item()
        neighbors = torch.randint(0, n_nodes, (int(n_edges),))
        for j in neighbors:
            if j.item() != i:
                src.append(i)
                dst.append(j.item())
    edge_index = torch.unique(torch.tensor([src, dst], dtype=torch.long), dim=1)
    data = GraphData(x=x, edge_index=edge_index, y=y)
    data["train_mask"] = _default_split(y, "train")
    data["val_mask"] = _default_split(y, "val")
    data["test_mask"] = _default_split(y, "test")
    return FraudGraphDataset(data, name=name)


def _add_edges(indices, edge_flag, edge_map):
    """Add edges within a group with bitmask type encoding."""
    m = len(indices)
    for i in range(m):
        for j in range(i + 1, m):
            u, v = indices[i], indices[j]
            edge_map[(u, v)] |= edge_flag
            edge_map[(v, u)] |= edge_flag


EDGE_SAME_USER = 1
EDGE_SAME_PRODUCT_SAME_STAR = 2
EDGE_SAME_PRODUCT_SAME_MONTH = 4


def _load_yelp_spam(root: str) -> FraudGraphDataset:
    """Load Yelp spam review dataset following yelp_process.py + split.py.

    Features: [useful, funny, cool, star] (4-dim, from reference pipeline).
    Edges: R-U-R (type 1), R-S-R (type 2), R-P-M-R (type 4), full clique per group.
    Split: balanced — train 1:1, val/test 1:1.2 pos:neg.
    """
    processed = os.path.join(root, "yelp_spam_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if isinstance(data, GraphData):
            return FraudGraphDataset(data, name="yelp_spam")
        os.remove(processed)

    # --- Phase 1: read meta + review files (same as yelp_process.py) ---
    meta_files = [
        os.path.join(root, "output_meta_yelpHotelData_NRYRcleaned.txt"),
        os.path.join(root, "output_meta_yelpResData_NRYRcleaned.txt"),
    ]
    review_files = [
        os.path.join(root, "output_review_yelpHotelData_NRYRcleaned.txt"),
        os.path.join(root, "output_review_yelpResData_NRYRcleaned.txt"),
    ]

    num_features = []
    labels = []
    node_text = {}
    edge_build_nodes = []

    for meta_path, review_path in zip(meta_files, review_files):
        if not os.path.exists(meta_path):
            continue
        # Read review texts
        texts = {}
        if os.path.exists(review_path):
            with open(review_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    texts[idx] = line.rstrip("\n")

        with open(meta_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.rstrip("\n").split()
                if len(parts) < 9:
                    continue
                date_str, review_id, reviewer_id, product_id = parts[0], parts[1], parts[2], parts[3]
                label_str = parts[4]
                useful, funny, cool = int(parts[5]), int(parts[6]), int(parts[7])
                star = int(float(parts[8]))

                dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")

                node_idx = len(num_features)
                num_features.append([useful, funny, cool, star])
                labels.append(1 if label_str == "Y" else 0)

                if idx in texts:
                    node_text[node_idx] = texts[idx][:500]

                edge_build_nodes.append({
                    "idx": node_idx,
                    "reviewer_id": reviewer_id,
                    "product_id": product_id,
                    "star": star,
                    "year": dt.year,
                    "month": dt.month,
                })

    if not labels:
        return _synthetic_fraud_graph("yelp_spam", 5000, 4, 0.13)

    n = len(labels)

    # --- Phase 2: build edges with bitmask types (same as yelp_process.py) ---
    by_user = defaultdict(list)
    by_product_star = defaultdict(list)
    by_product_month = defaultdict(list)

    for node in edge_build_nodes:
        idx = node["idx"]
        by_user[node["reviewer_id"]].append(idx)
        by_product_star[(node["product_id"], node["star"])].append(idx)
        by_product_month[(node["product_id"], node["year"], node["month"])].append(idx)

    edge_map = defaultdict(int)
    for indices in by_user.values():
        if len(indices) > 1:
            _add_edges(indices, EDGE_SAME_USER, edge_map)
    for indices in by_product_star.values():
        if len(indices) > 1:
            _add_edges(indices, EDGE_SAME_PRODUCT_SAME_STAR, edge_map)
    for indices in by_product_month.values():
        if len(indices) > 1:
            _add_edges(indices, EDGE_SAME_PRODUCT_SAME_MONTH, edge_map)

    # Convert edge_map to edge_index
    src_list, dst_list, edge_type_list = [], [], []
    for (u, v), etype in edge_map.items():
        src_list.append(u)
        dst_list.append(v)
        edge_type_list.append(etype)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    x = torch.tensor(num_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # --- Phase 3: balanced split (same as split.py) ---
    train_mask, val_mask, test_mask = _balanced_split(labels)

    data = GraphData(x=x, edge_index=edge_index, y=y,
                     train_mask=torch.from_numpy(train_mask),
                     val_mask=torch.from_numpy(val_mask),
                     test_mask=torch.from_numpy(test_mask))
    data["edge_type"] = edge_type

    os.makedirs(os.path.dirname(processed) or ".", exist_ok=True)
    torch.save(data, processed)
    return FraudGraphDataset(data, name="yelp_spam", node_text=node_text)


def _load_amazon(root: str) -> FraudGraphDataset:
    """Load Amazon review fraud dataset (2023 format, JSONL).

    Edges: R-U-R (type 1), R-S-R (type 2) with bitmask + capping.
    Split: balanced (1:1 train, 1:1.2 val/test).
    """
    processed = os.path.join(root, "amazon_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if isinstance(data, GraphData):
            return FraudGraphDataset(data, name="amazon")
        os.remove(processed)

    import gzip
    jsonl_files = []
    if os.path.exists(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".jsonl.gz") and not f.startswith("meta_"):
                jsonl_files.append(os.path.join(root, f))
    if not jsonl_files:
        print("[AGEA] No Amazon JSONL files found. Generating synthetic graph.")
        return _synthetic_fraud_graph("amazon", 5000, 11, 0.10)

    reviews = []
    node_text = {}
    for fpath in jsonl_files:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                except Exception:
                    continue
                reviews.append({
                    "user_id": r.get("user_id", ""),
                    "product_id": r.get("parent_asin", ""),
                    "rating": r.get("rating", 3.0),
                    "verified": r.get("verified_purchase", False),
                    "helpful_vote": r.get("helpful_vote", 0),
                })
                node_text[len(reviews) - 1] = r.get("text", "")[:500]
        if len(reviews) >= 10000:
            break

    if not reviews:
        return _synthetic_fraud_graph("amazon", 5000, 11, 0.10)

    reviewer_counts = defaultdict(int)
    product_counts = defaultdict(int)
    for r in reviews:
        reviewer_counts[r["user_id"]] += 1
        product_counts[r["product_id"]] += 1

    x_list, y_list = [], []
    for r in reviews:
        rating = r["rating"]
        verified = float(r["verified"])
        helpful = min(r["helpful_vote"] / 10.0, 1.0)
        reviewer_act = min(reviewer_counts[r["user_id"]] / 10.0, 1.0)
        product_pop = min(product_counts[r["product_id"]] / 50.0, 1.0)
        star_onehot = [1 if rating == i else 0 for i in range(1, 6)]
        is_extreme = 1.0 if rating in [1.0, 5.0] else 0.0
        fraud_score = (1.0 - verified) * 0.3 + is_extreme * 0.2 + (1.0 - helpful) * 0.2 + (1.0 - reviewer_act) * 0.3
        x_list.append([rating / 5.0, verified, helpful, reviewer_act, product_pop, is_extreme] + star_onehot)
        y_list.append(1 if fraud_score > 0.7 else 0)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    # Build edges with bitmask + capping (same approach as Yelp)
    user_to_reviews = defaultdict(list)
    product_star_to_reviews = defaultdict(list)
    for i, r in enumerate(reviews):
        user_to_reviews[r["user_id"]].append(i)
        product_star_to_reviews[f"{r['product_id']}_{int(r['rating'])}"].append(i)

    edge_map = defaultdict(int)
    for indices in user_to_reviews.values():
        if 1 < len(indices) <= 30:
            _add_edges(indices, EDGE_SAME_USER, edge_map)
    for indices in product_star_to_reviews.values():
        if 1 < len(indices) <= 30:
            _add_edges(indices, EDGE_SAME_PRODUCT_SAME_STAR, edge_map)

    src_list, dst_list, edge_type_list = [], [], []
    for (u, v), etype in edge_map.items():
        src_list.append(u)
        dst_list.append(v)
        edge_type_list.append(etype)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    train_mask, val_mask, test_mask = _balanced_split(y_list)

    data = GraphData(x=x, edge_index=edge_index, y=y,
                     train_mask=torch.from_numpy(train_mask),
                     val_mask=torch.from_numpy(val_mask),
                     test_mask=torch.from_numpy(test_mask))
    data["edge_type"] = edge_type
    os.makedirs(os.path.dirname(processed) or ".", exist_ok=True)
    torch.save(data, processed)
    return FraudGraphDataset(data, name="amazon", node_text=node_text)




def _load_fakenews(root, sub_dataset="PolitiFact"):
    """Load UPFD FakeNews dataset (PolitiFact or BuzzFeed).

    Graph: news nodes (targets) + user nodes, with bipartite news-user edges
    and user-user social edges.
    Labels: news nodes only (fake=1, real=0).
    Features: try scipy .mat user features, fallback to degree-based.
    Split: balanced (1:1 train, 1:1.2 val/test) on news nodes only.
    """
    import struct
    sub_dir = os.path.join(root, sub_dataset)
    if not os.path.exists(sub_dir):
        sub_dir = root

    prefix = sub_dataset
    news_file = os.path.join(sub_dir, f"{prefix}News.txt")
    news_user_file = os.path.join(sub_dir, f"{prefix}NewsUser.txt")
    user_user_file = os.path.join(sub_dir, f"{prefix}UserUser.txt")
    user_feature_file = os.path.join(sub_dir, f"{prefix}UserFeature.mat")

    # Fallback: check parent dir
    if not os.path.exists(news_file):
        news_file = os.path.join(root, f"{prefix}News.txt")
        news_user_file = os.path.join(root, f"{prefix}NewsUser.txt")
        user_user_file = os.path.join(root, f"{prefix}UserUser.txt")
        user_feature_file = os.path.join(root, f"{prefix}UserFeature.mat")

    if not os.path.exists(news_file):
        print(f"[AGEA] FakeNews {sub_dataset} files not found. Generating synthetic.")
        return _synthetic_fraud_graph(f"fakenews_{sub_dataset.lower()}", 2000, 10, 0.5)

    processed = os.path.join(root, f"fakenews_{sub_dataset.lower()}_processed.pt")
    if os.path.exists(processed):
        data = torch.load(processed, weights_only=False)
        if isinstance(data, GraphData):
            return FraudGraphDataset(data, name=f"fakenews_{sub_dataset.lower()}")
        os.remove(processed)

    # --- Read news list and labels ---
    news_labels = []
    with open(news_file, "r") as f:
        for line in f:
            name = line.strip()
            label = 1 if "Fake" in name else 0
            news_labels.append(label)

    n_news = len(news_labels)

    # --- Read news-user bipartite edges ---
    news_to_users = defaultdict(list)
    n_users_from_nu = 0
    with open(news_user_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                news_id = int(parts[0]) - 1  # 0-indexed
                user_id = int(parts[1])       # 1-indexed, will offset by n_news
                if 0 <= news_id < n_news:
                    news_to_users[news_id].append(user_id)
                    n_users_from_nu = max(n_users_from_nu, user_id)

    # --- Read user-user social edges ---
    user_edges_src, user_edges_dst = [], []
    n_users_from_uu = 0
    if os.path.exists(user_user_file):
        with open(user_user_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u1, u2 = int(parts[0]), int(parts[1])
                    user_edges_src.append(u1)
                    user_edges_dst.append(u2)
                    n_users_from_uu = max(n_users_from_uu, u1, u2)

    n_users = max(n_users_from_nu, n_users_from_uu)
    total_nodes = n_news + n_users

    # --- Build edge_index ---
    src_list, dst_list = [], []

    # News <-> User bipartite edges (bidirectional)
    for news_id, users in news_to_users.items():
        for uid in users:
            user_node = n_news + uid - 1  # offset by n_news, 0-indexed
            if user_node < total_nodes:
                src_list.append(news_id)
                dst_list.append(user_node)
                src_list.append(user_node)
                dst_list.append(news_id)

    # User <-> User social edges
    for u1, u2 in zip(user_edges_src, user_edges_dst):
        n1 = n_news + u1 - 1
        n2 = n_news + u2 - 1
        if n1 < total_nodes and n2 < total_nodes:
            src_list.append(n1)
            dst_list.append(n2)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Remove duplicates
    if edge_index.size(1) > 0:
        edge_index = torch.unique(edge_index, dim=1)

    # --- Build features ---
    # Try scipy .mat for user features
    user_features = None
    if os.path.exists(user_feature_file):
        try:
            from scipy.io import loadmat
            mat = loadmat(user_feature_file)
            for key in mat:
                if not key.startswith('_') and isinstance(mat[key], np.ndarray):
                    if mat[key].ndim == 2 and mat[key].shape[0] == n_users:
                        user_features = mat[key].astype(np.float32)
                        break
        except Exception:
            pass

    if user_features is not None:
        # News features: average of engaged user features
        news_feat_dim = user_features.shape[1]
        news_feats = np.zeros((n_news, news_feat_dim), dtype=np.float32)
        for nid, users in news_to_users.items():
            uidxs = [uid - 1 for uid in users if uid - 1 < n_users]
            if uidxs:
                news_feats[nid] = user_features[uidxs].mean(axis=0)
        x = torch.tensor(np.vstack([news_feats, user_features]), dtype=torch.float32)
    else:
        # Fallback: degree-based features
        # Compute degree per node
        deg = np.zeros(total_nodes, dtype=np.float32)
        if edge_index.size(1) > 0:
            for i in range(edge_index.size(1)):
                deg[edge_index[0, i].item()] += 1

        feat_dim = 10
        x = torch.zeros(total_nodes, feat_dim, dtype=torch.float32)
        x[:, 0] = torch.from_numpy(deg)
        x[:, 1] = torch.from_numpy(np.log1p(deg))
        # News-specific: number of engaged users
        for nid, users in news_to_users.items():
            x[nid, 2] = len(users)
        # User-specific: number of news engaged with
        user_news_count = defaultdict(int)
        for nid, users in news_to_users.items():
            for uid in users:
                user_news_count[uid] += 1
        for uid, count in user_news_count.items():
            node = n_news + uid - 1
            if node < total_nodes:
                x[node, 3] = count

    # --- Labels: news nodes only, user nodes = -1 (unlabeled) ---
    y = torch.full((total_nodes,), -1, dtype=torch.long)
    for i, label in enumerate(news_labels):
        y[i] = label

    # --- Split on news nodes only ---
    train_mask, val_mask, test_mask = _balanced_split(news_labels)
    # Extend masks to full graph
    full_train = torch.zeros(total_nodes, dtype=torch.bool)
    full_val = torch.zeros(total_nodes, dtype=torch.bool)
    full_test = torch.zeros(total_nodes, dtype=torch.bool)
    full_train[:n_news] = torch.from_numpy(train_mask)
    full_val[:n_news] = torch.from_numpy(val_mask)
    full_test[:n_news] = torch.from_numpy(test_mask)

    data = GraphData(x=x, edge_index=edge_index, y=y,
                     train_mask=full_train, val_mask=full_val, test_mask=full_test)
    data["n_news"] = n_news
    data["n_users"] = n_users

    os.makedirs(os.path.dirname(processed) or ".", exist_ok=True)
    torch.save(data, processed)
    return FraudGraphDataset(data, name=f"fakenews_{sub_dataset.lower()}")

_LOADERS = {
    "yelpchi": _load_yelpchi,
    "dgraphfin": _load_dgraphfin,
    "elliptic": _load_elliptic,
    "yelp_spam": _load_yelp_spam,
    "amazon": _load_amazon,
    "fakenews_politifact": lambda root: _load_fakenews(root, "PolitiFact"),
    "fakenews_buzzfeed": lambda root: _load_fakenews(root, "BuzzFeed"),
}


def load_dataset(name: str, root: str = None) -> FraudGraphDataset:
    if name in _LOADERS:
        root = root or f"./data/{name}"
        return _LOADERS[name](root)
    if root and os.path.exists(os.path.join(root, "x.pt")):
        return _load_generic(root)
    print(f"[AGEA] Dataset '{name}' not found. Generating synthetic graph.")
    return _synthetic_fraud_graph(name, 5000, 32, 0.10)
