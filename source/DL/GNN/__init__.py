"""
GNN Training Package

Graph Neural Network training framework for graph classification.
Supports: GCN, GAT, EdgeConv

Compatible with the GUI main.py interface.
"""

from .config_gnn import (
    load_gnn_config, 
    GNNConfig, 
    GNNModelConfig, 
    GNNLayerConfig,
    DataConfig, 
    TrainConfig,
    SchedulerConfig
)
from .data_gnn import (
    load_graph_data, 
    create_graph_dataloaders, 
    GraphData, 
    Batch,
    GraphDataset,
    StandardScaler
)
from .model_gnn import (
    build_gnn_model, 
    FlexibleGNN,
    GCNConv,
    GATConv,
    EdgeConv
)
from .train_gnn import train, evaluate, get_device

__all__ = [
    # Config
    'load_gnn_config', 'GNNConfig', 'GNNModelConfig', 'GNNLayerConfig',
    'DataConfig', 'TrainConfig', 'SchedulerConfig',
    # Data
    'load_graph_data', 'create_graph_dataloaders', 'GraphData', 'Batch',
    'GraphDataset', 'StandardScaler',
    # Model
    'build_gnn_model', 'FlexibleGNN', 'GCNConv', 'GATConv', 'EdgeConv',
    # Training
    'train', 'evaluate', 'get_device',
]
