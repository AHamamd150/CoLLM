from typing import List
import torch
import torch.nn as nn

from source.utils.read_configs import ModelConfig, LayerConfig


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
    """Get activation module by name."""
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with flexible layer configuration.
    
    Supported layer types:
        - dense: Linear layer with optional activation
        - dropout: Dropout layer
        - batchnorm: Batch normalization
        - flatten: Flatten layer
    """
    
    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        
        modules = []
        current_dim = input_dim
        
        for layer_cfg in cfg.layers:
            layer_type = layer_cfg.type.lower()
            
            if layer_type == "dense":
                if layer_cfg.units is None:
                    raise ValueError("Dense layer requires 'units' parameter")
                modules.append(nn.Linear(current_dim, layer_cfg.units))
                if layer_cfg.activation:
                    modules.append(get_activation(layer_cfg.activation))
                current_dim = layer_cfg.units
                
            elif layer_type == "dropout":
                rate = layer_cfg.rate if layer_cfg.rate is not None else 0.5
                modules.append(nn.Dropout(rate))
                
            elif layer_type == "batchnorm":
                modules.append(nn.BatchNorm1d(current_dim))
                
            elif layer_type == "flatten":
                modules.append(nn.Flatten())
                
            else:
                raise ValueError(f"Unknown layer type: {layer_type}. "
                               f"Choose from: dense, dropout, batchnorm, flatten")
        
        self.layers = nn.Sequential(*modules)
        self.output = nn.Linear(current_dim, cfg.output_units)
        
        # Output activation
        self.output_activation = get_activation(cfg.output_activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.output(x)
        x = self.output_activation(x)
        return x


def build_model(input_dim: int, cfg: ModelConfig) -> MLP:
    """Build MLP from config."""
    model = MLP(input_dim, cfg)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Print architecture
    print(f"Model architecture:")
    for i, layer_cfg in enumerate(cfg.layers):
        if layer_cfg.type == "dense":
            act_str = f" -> {layer_cfg.activation}" if layer_cfg.activation else ""
            print(f"  [{i}] Dense({layer_cfg.units}){act_str}")
        elif layer_cfg.type == "dropout":
            print(f"  [{i}] Dropout({layer_cfg.rate})")
        elif layer_cfg.type == "batchnorm":
            print(f"  [{i}] BatchNorm")
        elif layer_cfg.type == "flatten":
            print(f"  [{i}] Flatten")
    print(f"  [out] Dense({cfg.output_units})")
    print(f"Total parameters: {n_params:,}")
    
    return model
