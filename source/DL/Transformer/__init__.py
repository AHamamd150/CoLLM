"""
Particle Cloud Transformer Training Package

Transformer-based model for particle cloud classification.
"""

from .config_transformer import (
    load_transformer_config, 
    TransformerConfig, 
    TransformerModelConfig, 
    DataConfig, 
    TrainConfig
)
from .data_transformer import (
    load_cloud_data, 
    create_cloud_dataloaders, 
    ParticleCloud, 
    ParticleCloudDataset
)
from .model_transformer import (
    build_transformer_model, 
    ParticleCloudTransformer,
    MultiHeadSelfAttention,
    TransformerBlock
)
from .train_transformer import train, evaluate, get_device

__all__ = [
    # Config
    'load_transformer_config', 'TransformerConfig', 'TransformerModelConfig', 
    'DataConfig', 'TrainConfig',
    # Data
    'load_cloud_data', 'create_cloud_dataloaders', 'ParticleCloud', 'ParticleCloudDataset',
    # Model
    'build_transformer_model', 'ParticleCloudTransformer', 
    'MultiHeadSelfAttention', 'TransformerBlock',
    # Training
    'train', 'evaluate', 'get_device',
]
