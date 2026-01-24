from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import yaml
import numpy as np
########################
def read_conf(file_): 
    with open(file_, 'r') as file:
        data = yaml.safe_load(file)
        
    #######Read general setup ###########
    output_dir = data["Output_dir"]
    DEFAULT_MODEL = data["DEFAULT_MODEL"]
    MAX_RETRIES = data["MAX_RETRIES"]
    input_test  = data["Input_file"]
    user_input = data["User_input"]
    api = data["Use_api"]
    api_key = data["Api_key"]
    return (output_dir,
              DEFAULT_MODEL,
              MAX_RETRIES,
              input_test,
              user_input,
              api,
              api_key)
             
##=========================
##  Read the MLP configuration ==
##=========================
@dataclass
class LayerConfig:
    """Single layer configuration."""
    type: str                    
    units: Optional[int] = None  
    activation: Optional[str] = None  
    rate: Optional[float] = None  


@dataclass
class DataConfig:
    """Data configuration."""
    signal_path: str
    background_path: str
    train_size: int
    test_size: int
    val_ratio: float = 0.15
    normalize: bool = True
    feature_columns: Optional[List[str]] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    layers: List[LayerConfig] = field(default_factory=list)
    output_units: int = 2
    output_activation: Optional[str] = None  # None for logits, or softmax/sigmoid


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "none"  # none, step, plateau, cosine, onecycle
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5
    min_lr: float = 1e-6
    max_lr: Optional[float] = None  # for onecycle


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "adam"
    device: str = "auto"
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"  # val_loss or eval_metric
    
    # Training precision
    precision: str = "float32"  # float32, float16, mixed
    
    # LR scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Evaluation metric for model selection
    eval_metric: str = "accuracy"  # accuracy, auc, f1, recall, precision


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    seed: int = 42
    output_dir: str = "output"


def parse_layers(raw_layers: List[Dict[str, Any]]) -> List[LayerConfig]:
    """Parse layer configurations from YAML."""
    layers = []
    for layer in raw_layers:
        layers.append(LayerConfig(
            type=layer["type"].lower(),
            units=layer.get("units"),
            activation=layer.get("activation"),
            rate=layer.get("rate"),
        ))
    return layers


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


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Data config
    data_cfg = DataConfig(
        signal_path=raw['data']['signal_path'],
        background_path=raw['data']['background_path'],
        train_size=raw['data']['train_size'],
        test_size=raw['data']['test_size'],
        val_ratio=raw['data'].get('val_ratio', 0.15),
        normalize=raw['data'].get('normalize', True),
        feature_columns=raw['data'].get('feature_columns'),
    )
    
    # Model config
    model_cfg = ModelConfig(
        layers=parse_layers(raw['model']['layers']),
        output_units=raw['model'].get('output_units', 2),
        output_activation=raw['model'].get('output_activation'),
    )
    
    # Train config
    train_raw = raw['train']
    train_cfg = TrainConfig(
        epochs=train_raw.get('epochs', 50),
        batch_size=train_raw.get('batch_size', 256),
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
    
    return Config(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        seed=raw.get('seed', 42),
        output_dir=raw.get('output_dir', 'output'),
    )              
