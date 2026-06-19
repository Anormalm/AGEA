"""GNN and MLP classifiers for fraud detection baselines.

Pure PyTorch implementations — no torch_geometric compiled extensions required.
Uses sparse tensor operations from torch.sparse for message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _normalize_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
    """Convert edge_index to a normalized sparse adjacency matrix (symmetric)."""
    src, dst = edge_index[0], edge_index[1]
    n = num_nodes
    loop_src = torch.arange(n, device=src.device)
    src = torch.cat([src, loop_src])
    dst = torch.cat([dst, loop_src])

    deg = torch.zeros(n, device=src.device)
    deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=src.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([dst, src])
    adj = torch.sparse_coo_tensor(indices, norm, (n, n))
    return adj.coalesce()


def _mean_aggregate(edge_index: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean aggregation of neighbor features (GraphSAGE-style)."""
    src, dst = edge_index[0], edge_index[1]
    n = num_nodes

    deg = torch.zeros(n, device=dst.device)
    deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=dst.device))
    deg = deg.clamp(min=1)

    msg = x[src]
    agg = torch.zeros(n, x.size(1), device=x.device)
    agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), msg)
    agg = agg / deg.unsqueeze(1)
    return agg


def _ppr_adj(edge_index: torch.Tensor, num_nodes: int, alpha: float = 0.15) -> torch.sparse.FloatTensor:
    """Compute PPR-derived adjacency: alpha * I + (1-alpha) * D^{-1} A."""
    src, dst = edge_index[0], edge_index[1]
    n = num_nodes

    # Row-normalized adjacency: D^{-1} A
    deg = torch.zeros(n, device=src.device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=src.device))
    deg_inv = deg.float().pow(-1)
    deg_inv[deg_inv == float('inf')] = 0.0

    norm = deg_inv[src]
    indices = torch.stack([dst, src])
    adj = torch.sparse_coo_tensor(indices, norm, (n, n)).coalesce()

    # PPR: alpha * I + (1-alpha) * D^{-1} A
    I_indices = torch.arange(n, device=src.device).unsqueeze(0).repeat(2, 1)
    I_vals = torch.full((n,), alpha, device=src.device)
    ppr_indices = torch.cat([adj.indices(), I_indices], dim=1)
    ppr_vals = torch.cat([(1 - alpha) * adj.values(), I_vals])
    ppr = torch.sparse_coo_tensor(ppr_indices, ppr_vals, (n, n)).coalesce()
    return ppr


class GCNConvPure(nn.Module):
    """Pure PyTorch GCN convolution layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        x = self.lin(x)
        return torch.sparse.mm(adj, x)


class SAGEConvPure(nn.Module):
    """Pure PyTorch GraphSAGE convolution layer (mean aggregator)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neigh_agg = _mean_aggregate(edge_index, x, x.size(0))
        out = self.lin_self(x) + self.lin_neigh(neigh_agg) + self.bias
        return out


class GATConvPure(nn.Module):
    """Pure PyTorch GAT convolution layer with chunked attention for memory efficiency."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim // heads
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, heads * self.out_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * self.out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)

        h = self.lin(x).view(n, self.heads, self.out_dim)
        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        CHUNK = 500000
        out = torch.zeros(n, self.heads, self.out_dim, device=x.device)
        e_sum = torch.zeros(n, self.heads, device=x.device)

        for start in range(0, E, CHUNK):
            end = min(start + CHUNK, E)
            s_idx = src[start:end]
            d_idx = dst[start:end]

            e_chunk = self.leaky_relu(alpha_src[s_idx] + alpha_dst[d_idx])
            e_max = torch.full((n, self.heads), -1e9, device=x.device)
            e_max.scatter_reduce_(0, d_idx.unsqueeze(1).expand_as(e_chunk), e_chunk, reduce='amax', include_self=True)
            e_chunk = torch.exp(e_chunk - e_max[d_idx])

            e_sum_chunk = torch.zeros(n, self.heads, device=x.device)
            e_sum_chunk.scatter_add_(0, d_idx.unsqueeze(1).expand_as(e_chunk), e_chunk)
            e_sum += e_sum_chunk

            alpha_chunk = e_chunk / (e_sum[d_idx] + 1e-16)
            msg_chunk = h[s_idx] * alpha_chunk.unsqueeze(-1)
            out.scatter_add_(0,
                             d_idx.unsqueeze(1).unsqueeze(2).expand_as(msg_chunk),
                             msg_chunk)

        out = out / (e_sum.unsqueeze(-1) + 1e-16)
        out = out.view(n, self.heads * self.out_dim) + self.bias
        return out


class HGTConvPure(nn.Module):
    """Heterogeneous Graph Transformer convolution — type-specific projections + attention."""

    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int = 8, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim // heads
        self.num_edge_types = num_edge_types
        # Type-specific key/query/value projections
        self.lin_q = nn.Linear(in_dim, heads * self.out_dim, bias=False)
        self.lin_k = nn.Linear(in_dim, heads * self.out_dim, bias=False)
        self.lin_v = nn.Linear(in_dim, heads * self.out_dim, bias=False)
        self.type_emb = nn.Embedding(num_edge_types, heads * self.out_dim)
        self.bias = nn.Parameter(torch.zeros(heads * self.out_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)

        Q = self.lin_q(x).view(n, self.heads, self.out_dim)
        K = self.lin_k(x).view(n, self.heads, self.out_dim)
        V = self.lin_v(x).view(n, self.heads, self.out_dim)
        T = self.type_emb(edge_type.clamp(0, self.num_edge_types - 1)).view(E, self.heads, self.out_dim)

        # Attention: Q_dst * (K_src + type)
        alpha_src = (K * Q).sum(dim=-1)  # per-node score component
        q_dst = Q  # [N, H, D]

        CHUNK = 500000
        out = torch.zeros(n, self.heads, self.out_dim, device=x.device)
        e_sum = torch.zeros(n, self.heads, device=x.device)

        for start in range(0, E, CHUNK):
            end = min(start + CHUNK, E)
            s_idx = src[start:end]
            d_idx = dst[start:end]
            t_chunk = T[start:end]  # [chunk, H, D]

            # e = Q[d] · (K[s] + T) per edge
            e_chunk = (q_dst[d_idx] * (K[s_idx] + t_chunk)).sum(dim=-1) / math.sqrt(self.out_dim)
            e_chunk = F.leaky_relu(e_chunk, 0.2)

            e_max = torch.full((n, self.heads), -1e9, device=x.device)
            e_max.scatter_reduce_(0, d_idx.unsqueeze(1).expand_as(e_chunk), e_chunk, reduce='amax', include_self=True)
            e_chunk = torch.exp(e_chunk - e_max[d_idx])

            e_sum_chunk = torch.zeros(n, self.heads, device=x.device)
            e_sum_chunk.scatter_add_(0, d_idx.unsqueeze(1).expand_as(e_chunk), e_chunk)
            e_sum += e_sum_chunk

            alpha_chunk = e_chunk / (e_sum[d_idx] + 1e-16)
            msg_chunk = V[s_idx] * alpha_chunk.unsqueeze(-1)
            out.scatter_add_(0,
                             d_idx.unsqueeze(1).unsqueeze(2).expand_as(msg_chunk),
                             msg_chunk)

        out = out / (e_sum.unsqueeze(-1) + 1e-16)
        out = out.view(n, self.heads * self.out_dim) + self.bias
        return out


# ─── Classifier wrappers ───────────────────────────────────────────

class MLPClassifier(nn.Module):
    """Simple MLP on node features."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GCNClassifier(nn.Module):
    """GCN-based node classifier (pure PyTorch). Caches normalized adjacency."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvPure(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvPure(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self._adj = None
        self._adj_device = None

    def _get_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
        if self._adj is None or self._adj_device != edge_index.device:
            self._adj = _normalize_edge_index(edge_index, num_nodes)
            self._adj_device = edge_index.device
        return self._adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = self._get_adj(edge_index, x.size(0))
        for conv in self.convs:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class SAGEClassifier(nn.Module):
    """GraphSAGE-based node classifier (pure PyTorch)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConvPure(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConvPure(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class GATClassifier(nn.Module):
    """GAT-based node classifier (pure PyTorch). Reduced heads for memory."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, heads: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConvPure(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConvPure(hidden_dim, hidden_dim, heads=heads, dropout=dropout))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class HGTClassifier(nn.Module):
    """Heterogeneous Graph Transformer classifier with edge-type-aware attention."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, heads: int = 2, num_edge_types: int = 8):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HGTConvPure(in_dim, hidden_dim, num_edge_types=num_edge_types, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(HGTConvPure(hidden_dim, hidden_dim, num_edge_types=num_edge_types, heads=heads))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class PMPClassifier(nn.Module):
    """Personalized PageRank Message Passing classifier.

    Uses PPR-derived propagation: adj_ppr = alpha * I + (1-alpha) * D^{-1}A.
    Caches the PPR adjacency for efficiency.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, alpha: float = 0.15):
        super().__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lin_out = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.alpha = alpha
        self._ppr = None
        self._ppr_device = None

    def _get_ppr(self, edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
        if self._ppr is None or self._ppr_device != edge_index.device:
            self._ppr = _ppr_adj(edge_index, num_nodes, alpha=self.alpha)
            self._ppr_device = edge_index.device
        return self._ppr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = self._get_ppr(edge_index, x.size(0))
        for lin in self.lins:
            x = lin(x)
            x = torch.sparse.mm(adj, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x).squeeze(-1)


class ConsisGADClassifier(nn.Module):
    """Consistency-based anomaly detection: GCN backbone + two-view agreement loss."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, lam: float = 0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvPure(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvPure(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.lam = lam
        self._adj = None
        self._adj_device = None

    def _get_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
        if self._adj is None or self._adj_device != edge_index.device:
            self._adj = _normalize_edge_index(edge_index, num_nodes)
            self._adj_device = edge_index.device
        return self._adj

    def _encode(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = self._get_adj(edge_index, x.size(0))
        return self._encode(x, adj)

    def forward_view1(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """View 1: feature masking (zero out 20% of features)."""
        if self.training:
            mask = torch.rand(x.size(), device=x.device) > 0.2
            x = x * mask.float()
        adj = self._get_adj(edge_index, x.size(0))
        return self._encode(x, adj)

    def forward_view2(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """View 2: edge dropout (remove 20% of edges)."""
        if self.training:
            keep = torch.rand(edge_index.size(1), device=x.device) > 0.2
            edge_index = edge_index[:, keep]
        adj = self._get_adj(edge_index, x.size(0))
        return self._encode(x, adj)


class GAAPClassifier(nn.Module):
    """Graph Augmentation + Adaptive Attention: GAT backbone + consistency loss."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, heads: int = 2, lam: float = 0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConvPure(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConvPure(hidden_dim, hidden_dim, heads=heads, dropout=dropout))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.lam = lam

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self._encode(x, edge_index)

    def forward_view1(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """View 1: feature masking."""
        if self.training:
            mask = torch.rand(x.size(), device=x.device) > 0.2
            x = x * mask.float()
        return self._encode(x, edge_index)

    def forward_view2(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """View 2: edge dropout."""
        if self.training:
            keep = torch.rand(edge_index.size(1), device=x.device) > 0.2
            edge_index = edge_index[:, keep]
        return self._encode(x, edge_index)
