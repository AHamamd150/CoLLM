"""
Transformer Configuration Module

Dataclasses and parser for Particle Cloud Transformer training configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    signal_path: str
    background_path: str
    train_size: int
    test_size: int
    particles_per_cloud: int  # Number of particles (rows) per cloud
    val_ratio: float = 0.15
    normalize: bool = True


@dataclass
class TransformerModelConfig:
    """Transformer model configuration."""
    embed_dim: int = 128           # Embedding dimension
    num_heads: int = 8             # Number of attention heads
    num_layers: int = 4            # Number of transformer layers
    ffn_dim: int = 256             # Feed-forward network dimension
    dropout: float = 0.1           # Dropout rate
    attention_dropout: float = 0.1 # Attention dropout
    pooling: str = "cls"           # cls, mean, max
    num_classes: int = 2
    
    # Positional encoding
    use_pos_encoding: bool = False  # Particle clouds are permutation invariant
    
    # Layer norm
    pre_norm: bool = True          # Pre-normalization (more stable)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5
    min_lr: float = 1e-6
    max_lr: Optional[float] = None
    warmup_steps: int = 0          # Warmup steps for transformer


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0001  # Transformers typically use smaller LR
    weight_decay: float = 0.01     # Transformers benefit from weight decay
    optimizer: str = "adamw"       # AdamW is standard for transformers
    device: str = "auto"
    
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    
    precision: str = "float32"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    eval_metric: str = "accuracy"
    
    # Gradient clipping (important for transformers)
    gradient_clip: float = 1.0


@dataclass
class TransformerConfig:
    """Complete Transformer configuration."""
    data: DataConfig
    model: TransformerModelConfig
    train: TrainConfig
    seed: int = 42
    output_dir: str = "output"


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
        warmup_steps=raw.get("warmup_steps", 0),
    )


def load_transformer_config(path: str) -> TransformerConfig:
    """Load Transformer configuration from YAML file."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Data config
    data_cfg = DataConfig(
        signal_path=raw['data']['signal_path'],
        background_path=raw['data']['background_path'],
        train_size=raw['data']['train_size'],
        test_size=raw['data']['test_size'],
        particles_per_cloud=raw['data']['particles_per_cloud'],
        val_ratio=raw['data'].get('val_ratio', 0.15),
        normalize=raw['data'].get('normalize', True),
    )
    
    # Model config
    model_raw = raw['model']
    model_cfg = TransformerModelConfig(
        embed_dim=model_raw.get('embed_dim', 128),
        num_heads=model_raw.get('num_heads', 8),
        num_layers=model_raw.get('num_layers', 4),
        ffn_dim=model_raw.get('ffn_dim', 256),
        dropout=model_raw.get('dropout', 0.1),
        attention_dropout=model_raw.get('attention_dropout', 0.1),
        pooling=model_raw.get('pooling', 'cls').lower(),
        num_classes=model_raw.get('num_classes', 2),
        use_pos_encoding=model_raw.get('use_pos_encoding', False),
        pre_norm=model_raw.get('pre_norm', True),
    )
    
    # Train config
    train_raw = raw['train']
    train_cfg = TrainConfig(
        epochs=train_raw.get('epochs', 50),
        batch_size=train_raw.get('batch_size', 32),
        learning_rate=train_raw.get('learning_rate', 0.0001),
        weight_decay=train_raw.get('weight_decay', 0.01),
        optimizer=train_raw.get('optimizer', 'adamw').lower(),
        device=train_raw.get('device', 'auto').lower(),
        early_stopping=train_raw.get('early_stopping', False),
        early_stopping_patience=train_raw.get('early_stopping_patience', 10),
        early_stopping_metric=train_raw.get('early_stopping_metric', 'val_loss'),
        precision=train_raw.get('precision', 'float32').lower(),
        scheduler=parse_scheduler(train_raw.get('scheduler')),
        eval_metric=train_raw.get('eval_metric', 'accuracy').lower(),
        gradient_clip=train_raw.get('gradient_clip', 1.0),
    )
    
    return TransformerConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        seed=raw.get('seed', 42),
        output_dir=raw.get('output_dir', 'output'),
    )
