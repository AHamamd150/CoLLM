from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    signal_path: str
    background_path: str
    train_size: int
    test_size: int
    nodes_per_graph: int = 0      # Number of nodes per graph (inferred if 0)
    val_ratio: float = 0.15
    normalize: bool = True


@dataclass
class GNNLayerConfig:
    """Single GNN layer configuration."""
    out_channels: int = 64
    activation: str = "relu"
    # EdgeConv specific
    k: int = 7
    aggr: str = "max"
    # GAT specific
    heads: int = 4
    concat: bool = True
    # GCN specific
    cached: bool = False
    # Common
    batchnorm: bool = False
    dropout: float = 0.0


@dataclass
class GNNModelConfig:
    """GNN model configuration."""
    type: str = "GCN"              # GCN, GAT, EdgeConv
    layers: List[GNNLayerConfig] = field(default_factory=list)
    pooling: str = "global_mean"   # global_mean, global_max, global_add
    output_units: int = 2
    output_activation: Optional[str] = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5
    min_lr: float = 1e-6
    max_lr: Optional[float] = None


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "adam"
    device: str = "auto"
    
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    
    precision: str = "float32"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    eval_metric: str = "accuracy"


@dataclass
class GNNConfig:
    """Complete GNN configuration."""
    data: DataConfig
    model: GNNModelConfig
    train: TrainConfig
    seed: int = 42
    output_dir: str = "output"


def parse_layer(raw: Dict[str, Any]) -> GNNLayerConfig:
    """Parse a single layer configuration."""
    return GNNLayerConfig(
        out_channels=raw.get("out_channels", 64),
        activation=raw.get("activation", "relu").lower(),
        k=raw.get("k", 7),
        aggr=raw.get("aggr", "max").lower(),
        heads=raw.get("heads", 4),
        concat=raw.get("concat", True),
        cached=raw.get("cached", False),
        batchnorm=raw.get("batchnorm", False),
        dropout=raw.get("dropout", 0.0),
    )


def parse_scheduler(raw: Optional[Dict[str, Any]]) -> SchedulerConfig:
    """Parse scheduler configuration."""
    if raw is None:
        return SchedulerConfig()
    return SchedulerConfig(
        type=raw.get("type", "none").lower(),
        step_size=raw.get("step_size", 10),
        gamma=raw.get("gamma", 0.1),
        patience=raw.get("patience", 5),
        min_lr=raw.get("min_lr", 1e-6),
        max_lr=raw.get("max_lr"),
    )


def load_gnn_config(path: str) -> GNNConfig:
    """Load GNN configuration from YAML file."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Data config
    data_raw = raw['data']
    data_cfg = DataConfig(
        signal_path=data_raw['signal_path'],
        background_path=data_raw['background_path'],
        train_size=data_raw['train_size'],
        test_size=data_raw['test_size'],
        nodes_per_graph=data_raw.get('nodes_per_graph', 0),
        val_ratio=data_raw.get('val_ratio', 0.15),
        normalize=data_raw.get('normalize', True),
    )
    
    # Model config
    model_raw = raw['model']
    layers = [parse_layer(l) for l in model_raw.get('layers', [])]
    

    pooling = model_raw.get('pooling', 'global_mean')
    pooling_map = {
        'global_mean': 'mean',
        'global_max': 'max', 
        'global_add': 'add',
        'mean': 'mean',
        'max': 'max',
        'add': 'add',
        'sum': 'add',
    }
    pooling = pooling_map.get(pooling.lower(), 'mean')
    
    model_cfg = GNNModelConfig(
        type=model_raw.get('type', 'GCN'),
        layers=layers,
        pooling=pooling,
        output_units=model_raw.get('output_units', 2),
        output_activation=model_raw.get('output_activation'),
    )
    
    # Training config
    train_raw = raw['train']
    train_cfg = TrainConfig(
        epochs=train_raw.get('epochs', 50),
        batch_size=train_raw.get('batch_size', 32),
        learning_rate=train_raw.get('learning_rate', 0.001),
        weight_decay=train_raw.get('weight_decay', 0.0),
        optimizer=train_raw.get('optimizer', 'adam').lower(),
        device=train_raw.get('device', 'auto').lower(),
        early_stopping=train_raw.get('early_stopping', False),
        early_stopping_patience=train_raw.get('early_stopping_patience', 10),
        early_stopping_metric=train_raw.get('early_stopping_metric', 'val_loss'),
        precision=train_raw.get('precision', 'float32').lower(),
        scheduler=parse_scheduler(train_raw.get('scheduler')),
        eval_metric=train_raw.get('eval_metric', 'accuracy').lower(),
    )
    
    return GNNConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        seed=raw.get('seed', 42),
        output_dir=raw.get('output_dir', 'output'),
    )
