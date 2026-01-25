from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_gnn import GNNModelConfig, GNNLayerConfig


# ============================================================================
# ============================================================================

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softmax": lambda: nn.Softmax(dim=-1),
    "none": nn.Identity,
}


def get_activation(name: str) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Global mean pooling over nodes."""
    out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
    count = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
    
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    count.scatter_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    
    return out / count.unsqueeze(1).clamp(min=1)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Global max pooling over nodes."""
    out = torch.full((num_graphs, x.size(1)), float('-inf'), device=x.device, dtype=x.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(1).expand_as(x), x, reduce='amax')
    return out


def global_add_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Global sum pooling over nodes."""
    out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    return out


POOLING = {
    "mean": global_mean_pool,
    "global_mean": global_mean_pool,
    "max": global_max_pool,
    "global_max": global_max_pool,
    "add": global_add_pool,
    "sum": global_add_pool,
    "global_add": global_add_pool,
}



class GCNConv(nn.Module):
    """
    Graph Convolutional Network layer.
    
    Implements: X' = D^{-1/2} A D^{-1/2} X W
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, row, edge_weight)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        

        x = self.linear(x)
        

        out = torch.zeros_like(x)
        out.scatter_add_(0, row.unsqueeze(1).expand_as(x[col]), x[col] * norm.unsqueeze(1))
        
        return out


class GATConv(nn.Module):
    """
    Graph Attention Network layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        
        x = self.linear(x).view(-1, self.heads, self.out_channels)
        
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        
        alpha = alpha_src[row] + alpha_dst[col]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = alpha * edge_weight.unsqueeze(1)
        
        alpha_max = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        alpha_max.scatter_reduce_(0, row.unsqueeze(1).expand_as(alpha), alpha, reduce='amax')
        alpha = alpha - alpha_max[row]
        alpha = alpha.exp()
        
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        
        alpha_sum = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        alpha_sum.scatter_add_(0, row.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[row] + 1e-16)
        
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
        msg = x[col] * alpha.unsqueeze(-1)
        out.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class EdgeConv(nn.Module):
    """
    EdgeConv layer (Dynamic Graph CNN).
    
    For each node, computes edge features with k nearest neighbors and aggregates.
    For fully connected graphs, we use all edges with edge weights.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 k: int = 7, aggr: str = 'max'):
        super().__init__()
        self.k = k
        self.aggr = aggr.lower()
        
        # MLP for edge features: [x_i, x_j - x_i] -> edge_feature
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        row, col = edge_index
        
        # Compute edge features: [x_i, x_j - x_i]
        x_i = x[row]
        x_j = x[col]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        

        edge_features = self.mlp(edge_features)
        
        # Weight by edge weights
        edge_features = edge_features * edge_weight.unsqueeze(-1)
        

        if self.aggr == 'max':
            out = torch.full((num_nodes, edge_features.size(1)), float('-inf'), 
                           device=x.device, dtype=x.dtype)
            out = out.scatter_reduce(0, row.unsqueeze(1).expand_as(edge_features), 
                              edge_features, reduce='amax')
            out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
        elif self.aggr == 'mean':
            out = torch.zeros(num_nodes, edge_features.size(1), device=x.device, dtype=x.dtype)
            count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            out = out.scatter_add(0, row.unsqueeze(1).expand_as(edge_features), edge_features)
            count = count.scatter_add(0, row.unsqueeze(1), edge_weight.unsqueeze(1))
            out = out / count.clamp(min=1e-16)
        else:  # add/sum
            out = torch.zeros(num_nodes, edge_features.size(1), device=x.device, dtype=x.dtype)
            out = out.scatter_add(0, row.unsqueeze(1).expand_as(edge_features), edge_features)
        
        return out



class FlexibleGNN(nn.Module):
    """
    Flexible GNN model that can be configured with different layer types.
    Compatible with the GUI main.py interface.
    """
    
    def __init__(self, num_features: int, cfg: GNNModelConfig):
        super().__init__()
        
        self.gnn_type = cfg.type.upper()
        self.pooling = cfg.pooling
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_channels = num_features
        
        for i, layer_cfg in enumerate(cfg.layers):
            out_channels = layer_cfg.out_channels
            
            # Create convolution layer based on type
            if self.gnn_type == "GCN":
                conv = GCNConv(in_channels, out_channels)
            elif self.gnn_type == "GAT":
                conv = GATConv(
                    in_channels, out_channels, 
                    heads=layer_cfg.heads,
                    concat=layer_cfg.concat,
                    dropout=layer_cfg.dropout
                )
                if layer_cfg.concat:
                    out_channels = out_channels * layer_cfg.heads
            elif self.gnn_type == "EDGECONV":
                conv = EdgeConv(in_channels, out_channels, k=layer_cfg.k, aggr=layer_cfg.aggr)
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")
            
            self.convs.append(conv)
            
            if layer_cfg.batchnorm:
                self.bns.append(nn.BatchNorm1d(out_channels))
            else:
                self.bns.append(nn.Identity())
            
            self.activations.append(get_activation(layer_cfg.activation))
            
     
            if layer_cfg.dropout > 0:
                self.dropouts.append(nn.Dropout(layer_cfg.dropout))
            else:
                self.dropouts.append(nn.Identity())
            
            in_channels = out_channels
        

        self.output_dim = in_channels
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_channels // 2, cfg.output_units)
        )
        

        if cfg.output_activation:
            self.output_activation = get_activation(cfg.output_activation)
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor, batch: torch.Tensor,
                num_graphs: int) -> torch.Tensor:
        # Apply GNN layers
        for conv, bn, act, drop in zip(self.convs, self.bns, self.activations, self.dropouts):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = act(x)
            x = drop(x)
        

        pool_fn = POOLING.get(self.pooling, global_mean_pool)
        x = pool_fn(x, batch, num_graphs)
        

        x = self.classifier(x)
        x = self.output_activation(x)
        
        return x


# ============================================================================
# ============================================================================

def build_gnn_model(num_features: int, cfg: GNNModelConfig) -> nn.Module:
    """
    Build a GNN model from configuration.
    
    Args:
        num_features: Number of input node features
        cfg: GNNModelConfig with type, layers, pooling, etc.
    """
    model = FlexibleGNN(num_features, cfg)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.type}")
    print(f"  Input features: {num_features}")
    print(f"  Layers: {len(cfg.layers)}")
    for i, layer in enumerate(cfg.layers):
        layer_info = f"    [{i}] out_channels={layer.out_channels}, activation={layer.activation}"
        if cfg.type.upper() == "GAT":
            layer_info += f", heads={layer.heads}, concat={layer.concat}"
        elif cfg.type.upper() == "EDGECONV":
            layer_info += f", k={layer.k}, aggr={layer.aggr}"
        if layer.batchnorm:
            layer_info += ", batchnorm"
        if layer.dropout > 0:
            layer_info += f", dropout={layer.dropout}"
        print(layer_info)
    print(f"  Pooling: {cfg.pooling}")
    print(f"  Output units: {cfg.output_units}")
    print(f"  Total parameters: {n_params:,}")
    
    return model
