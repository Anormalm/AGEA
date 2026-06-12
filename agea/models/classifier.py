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
    # Add self-loops
    n = num_nodes
    loop_src = torch.arange(n, device=src.device)
    src = torch.cat([src, loop_src])
    dst = torch.cat([dst, loop_src])

    # Compute degree
    deg = torch.zeros(n, device=src.device)
    deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=src.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # D^{-1/2} A D^{-1/2} with self-loops
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([dst, src])
    adj = torch.sparse_coo_tensor(indices, norm, (n, n))
    return adj.coalesce()


def _mean_aggregate(edge_index: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean aggregation of neighbor features (GraphSAGE-style)."""
    src, dst = edge_index[0], edge_index[1]
    n = num_nodes

    # Compute neighbor mean for each node
    deg = torch.zeros(n, device=dst.device)
    deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=dst.device))
    deg = deg.clamp(min=1)

    # Aggregate: scatter_add then divide by degree
    msg = x[src]  # [E, F]
    agg = torch.zeros(n, x.size(1), device=x.device)
    agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), msg)
    agg = agg / deg.unsqueeze(1)
    return agg


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
    """Pure PyTorch GAT convolution layer (single-head for simplicity)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
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

        # Linear transform
        h = self.lin(x).view(n, self.heads, self.out_dim)  # [N, H, D]

        # Attention coefficients
        alpha_src = (h * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        # Edge attention: e_ij = LeakyReLU(a_src_i + a_dst_j)
        e = self.leaky_relu(alpha_src[src] + alpha_dst[dst])  # [E, H]

        # Softmax per destination node
        # Subtract max for numerical stability
        e_max = torch.zeros(n, self.heads, device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=True)
        e = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(n, self.heads, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        alpha = e / (e_sum[dst] + 1e-16)

        # Weighted aggregation
        out = torch.zeros(n, self.heads, self.out_dim, device=x.device)
        msg = h[src] * alpha.unsqueeze(-1)  # [E, H, D]
        out.scatter_add_(0,
                         dst.unsqueeze(1).unsqueeze(2).expand_as(msg),
                         msg)

        out = out.view(n, self.heads * self.out_dim) + self.bias
        return out


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
    """GCN-based node classifier (pure PyTorch)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvPure(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvPure(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = _normalize_edge_index(edge_index, x.size(0))
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
    """GAT-based node classifier (pure PyTorch)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, heads: int = 4):
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
