import os
import subprocess
import sys
import html
import logging
import warnings
import tempfile
import yaml
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __package__ is None or __package__ == '':
    sys.path.insert(0, _project_root)
#=======================================
#=======================================
#=======================================

# Suppress Streamlit warnings
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*to view this Streamlit app.*')

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import streamlit as st
import threading
from queue import Queue

#=======================================
#   PAGE CONFIGURATION
#=======================================
st.set_page_config(
    page_title="CoLLM • ML Toolbox",
    page_icon=":dizzy:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules AFTER page config to avoid Streamlit issues
from source.utils.requirements_check import ensure_packages 
from source.runs.run_preselection_GUI import run_LLM
LLM_RUNNER_AVAILABLE = True

#=======================================
#   Helper functions for MLP
#=======================================

def generate_mlp_config(
    seed, output_dir, signal_path, background_path, train_size, test_size, 
    val_ratio, normalize, layers_config, output_units, output_activation,
    epochs, batch_size, learning_rate, weight_decay, optimizer, device,
    early_stopping, early_stopping_patience, early_stopping_metric,
    precision, scheduler_type, scheduler_params, eval_metric
):
    """Generate MLP configuration dictionary matching config.yaml format."""

    model_layers = []
    for layer in layers_config:
        layer_type = layer.get("type", "").lower()
        
        if "dense" in layer_type:
            layer_dict = {
                "type": "dense",
                "units": layer.get("neurons", 128),
                "activation": layer.get("activation", "relu").lower()
            }
        elif "dropout" in layer_type:
            layer_dict = {
                "type": "dropout",
                "rate": layer.get("rate", 0.2)
            }
        elif "batchnorm" in layer_type:
            layer_dict = {"type": "batchnorm"}
        elif "flatten" in layer_type:
            layer_dict = {"type": "flatten"}
        else:
            continue
        
        model_layers.append(layer_dict)
    
    
    scheduler_config = {"type": scheduler_type}
    if scheduler_type == "step":
        scheduler_config["step_size"] = scheduler_params.get("step_size", 10)
        scheduler_config["gamma"] = scheduler_params.get("gamma", 0.1)
    elif scheduler_type == "plateau":
        scheduler_config["patience"] = scheduler_params.get("patience", 5)
        scheduler_config["min_lr"] = scheduler_params.get("min_lr", 1e-6)
    elif scheduler_type == "onecycle":
        scheduler_config["max_lr"] = scheduler_params.get("max_lr", learning_rate * 10)
    
    
    config = {
        "seed": seed,
        "output_dir": output_dir,
        
        "data": {
            "signal_path": signal_path,
            "background_path": background_path,
            "train_size": train_size,
            "test_size": test_size,
            "val_ratio": val_ratio,
            "normalize": normalize
        },
        
        "model": {
            "layers": model_layers,
            "output_units": output_units,
            "output_activation": output_activation if output_activation != "None" else None
        },
        
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "device": device,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_metric": early_stopping_metric,
            "precision": precision,
            "scheduler": scheduler_config,
            "eval_metric": eval_metric
        }
    }
    
    return config


def config_to_yaml(config):
    """Convert config dictionary to YAML string with comments."""
    
    yaml_str = """# ============================================================================
# MLP Classifier Configuration
# ============================================================================

"""
    yaml_str += f"seed: {config['seed']}\n"
    yaml_str += f'output_dir: "{config["output_dir"]}"\n\n'
    
    yaml_str += """# ============================================================================
# Data Configuration
# ============================================================================
data:
"""
    yaml_str += f'  signal_path: "{config["data"]["signal_path"]}"\n'
    yaml_str += f'  background_path: "{config["data"]["background_path"]}"\n'
    yaml_str += f'  train_size: {config["data"]["train_size"]}          # samples per class for training\n'
    yaml_str += f'  test_size: {config["data"]["test_size"]}            # samples per class for testing\n'
    yaml_str += f'  val_ratio: {config["data"]["val_ratio"]}             # fraction of training data for validation\n'
    yaml_str += f'  normalize: {str(config["data"]["normalize"]).lower()}             # normalize features with StandardScaler\n\n'
    
    yaml_str += """# ============================================================================
# Model Architecture
# ============================================================================
# Layer types: dense, dropout, batchnorm, flatten
# Activations: relu, gelu, tanh, sigmoid, leaky_relu, elu, selu, softplus
model:
  layers:
"""
    for layer in config["model"]["layers"]:
        yaml_str += f'    - type: {layer["type"]}\n'
        if layer["type"] == "dense":
            yaml_str += f'      units: {layer["units"]}\n'
            yaml_str += f'      activation: {layer["activation"]}\n'
        elif layer["type"] == "dropout":
            yaml_str += f'      rate: {layer["rate"]}\n'
    
    yaml_str += f'\n  output_units: {config["model"]["output_units"]}               # 2 for binary classification\n'
    out_act = config["model"]["output_activation"]
    yaml_str += f'  output_activation: {out_act if out_act else "null"}       # null for logits (use with cross_entropy loss)\n\n'
    
    yaml_str += """# ============================================================================
# Training Configuration
# ============================================================================
train:
"""
    train = config["train"]
    yaml_str += f'  epochs: {train["epochs"]}\n'
    yaml_str += f'  batch_size: {train["batch_size"]}\n'
    yaml_str += f'  learning_rate: {train["learning_rate"]}\n'
    yaml_str += f'  weight_decay: {train["weight_decay"]}          # L2 regularization\n'
    yaml_str += f'  optimizer: {train["optimizer"]}               # adam, adamw, sgd\n'
    yaml_str += f'  device: {train["device"]}                  # auto, cpu, cuda, cuda:0, cuda:1\n'
    yaml_str += f'\n  # Early Stopping\n'
    yaml_str += f'  early_stopping: {str(train["early_stopping"]).lower()}\n'
    yaml_str += f'  early_stopping_patience: {train["early_stopping_patience"]}   # stop if no improvement for N epochs\n'
    yaml_str += f'  early_stopping_metric: {train["early_stopping_metric"]}  # val_loss or eval_metric\n'
    yaml_str += f'\n  # Training Precision\n'
    yaml_str += f'  precision: {train["precision"]}            # float32, float16, mixed\n'
    yaml_str += f'\n  # Learning Rate Scheduler\n'
    yaml_str += f'  scheduler:\n'
    yaml_str += f'    type: {train["scheduler"]["type"]}               # none, step, plateau, cosine, onecycle\n'
    
    sched = train["scheduler"]
    if sched["type"] == "step":
        yaml_str += f'    step_size: {sched.get("step_size", 10)}               # reduce LR every N epochs\n'
        yaml_str += f'    gamma: {sched.get("gamma", 0.1)}                  # LR multiplier\n'
    elif sched["type"] == "plateau":
        yaml_str += f'    patience: {sched.get("patience", 5)}                 # reduce LR if no improvement for N epochs\n'
        yaml_str += f'    min_lr: {sched.get("min_lr", 0.000001)}            # minimum learning rate\n'
    elif sched["type"] == "onecycle":
        yaml_str += f'    max_lr: {sched.get("max_lr", 0.01)}                # maximum learning rate\n'
    elif sched["type"] == "cosine":
        yaml_str += f'    # Cosine annealing uses epochs as T_max\n'
    
    yaml_str += f'\n  # Evaluation Metric (for model selection and reporting)\n'
    yaml_str += f'  eval_metric: {train["eval_metric"]}              # accuracy, auc, f1, recall, precision\n'
    
    return yaml_str


#=======================================
#   Helper functions for GNN networks configurations 
#=======================================

def generate_gnn_config(
    seed, output_dir, signal_path, background_path, train_size, test_size,
    val_ratio, normalize, nodes_per_graph,
    gnn_type, gnn_layers_config, pooling, output_units, output_activation,
    epochs, batch_size, learning_rate, weight_decay, optimizer, device,
    early_stopping, early_stopping_patience, early_stopping_metric,
    precision, scheduler_type, scheduler_params, eval_metric
):
    """Generate GNN configuration dictionary matching config.yaml format."""
    
    # Build GNN layers list
    gnn_layers = []
    for layer in gnn_layers_config:
        layer_dict = {
            "out_channels": layer.get("out_channels", 64),
            "activation": layer.get("activation", "relu").lower()
        }
        
        
        if gnn_type == "EdgeConv":
            layer_dict["k"] = layer.get("k", 7)  
            layer_dict["aggr"] = layer.get("aggr", "max")
        elif gnn_type == "GAT":
            layer_dict["heads"] = layer.get("heads", 4)
            layer_dict["concat"] = layer.get("concat", True)
            layer_dict["dropout"] = layer.get("dropout", 0.0)
        # Modified: we don't consider the self connected loops    
        #elif gnn_type == "GCN":
         #   layer_dict["improved"] = layer.get("improved", False)
          #  layer_dict["cached"] = layer.get("cached", False)
        
        if layer.get("add_batchnorm", False):
            layer_dict["batchnorm"] = True
        if layer.get("dropout_rate", 0) > 0:
            layer_dict["dropout"] = layer.get("dropout_rate", 0)
        
        gnn_layers.append(layer_dict)
    

    scheduler_config = {"type": scheduler_type}
    if scheduler_type == "step":
        scheduler_config["step_size"] = scheduler_params.get("step_size", 10)
        scheduler_config["gamma"] = scheduler_params.get("gamma", 0.1)
    elif scheduler_type == "plateau":
        scheduler_config["patience"] = scheduler_params.get("patience", 5)
        scheduler_config["min_lr"] = scheduler_params.get("min_lr", 1e-6)
    elif scheduler_type == "onecycle":
        scheduler_config["max_lr"] = scheduler_params.get("max_lr", learning_rate * 10)
    

    config = {
        "seed": seed,
        "output_dir": output_dir,
        
        "data": {
            "signal_path": signal_path,
            "background_path": background_path,
            "train_size": train_size,
            "test_size": test_size,
            "val_ratio": val_ratio,
            "normalize": normalize,
            "nodes_per_graph": nodes_per_graph
        },
        
        "model": {
            "type": gnn_type,
            "layers": gnn_layers,
            "pooling": pooling,
            "output_units": output_units,
            "output_activation": output_activation if output_activation != "None" else None
        },
        
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "device": device,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_metric": early_stopping_metric,
            "precision": precision,
            "scheduler": scheduler_config,
            "eval_metric": eval_metric
        }
    }
    
    return config


def gnn_config_to_yaml(config):
    """Convert GNN config dictionary to YAML string with comments."""
    
    gnn_type = config["model"]["type"]
    
    yaml_str = f"""# ============================================================================
# GNN Classifier Configuration - {gnn_type}
# ============================================================================

"""
    yaml_str += f"seed: {config['seed']}\n"
    yaml_str += f'output_dir: "{config["output_dir"]}"\n\n'
    
    yaml_str += """# ============================================================================
# Data Configuration
# ============================================================================
data:
"""
    yaml_str += f'  signal_path: "{config["data"]["signal_path"]}"\n'
    yaml_str += f'  background_path: "{config["data"]["background_path"]}"\n'
    yaml_str += f'  train_size: {config["data"]["train_size"]}          # samples per class for training\n'
    yaml_str += f'  test_size: {config["data"]["test_size"]}            # samples per class for testing\n'
    yaml_str += f'  val_ratio: {config["data"]["val_ratio"]}             # fraction of training data for validation\n'
    yaml_str += f'  normalize: {str(config["data"]["normalize"]).lower()}             # normalize features\n'
    yaml_str += f'  nodes_per_graph: {config["data"]["nodes_per_graph"]}      # number of particles in the graph\n\n'
    
    yaml_str += f"""# ============================================================================
# Model Architecture - {gnn_type}
# ============================================================================
# GNN Types: EdgeConv, GCN, GAT
# Pooling: global_mean, global_max, global_add
# Activations: relu, gelu, tanh, sigmoid, leaky_relu, elu, selu
model:
  type: {gnn_type}
  layers:
"""
    for i, layer in enumerate(config["model"]["layers"]):
        yaml_str += f'    - out_channels: {layer["out_channels"]}\n'
        yaml_str += f'      activation: {layer["activation"]}\n'
        
        if gnn_type == "EdgeConv":
            yaml_str += f'      k: {layer.get("k", 20)}                    # k nearest neighbors\n'
            yaml_str += f'      aggr: {layer.get("aggr", "max")}              # aggregation: max, mean, add\n'
        elif gnn_type == "GAT":
            yaml_str += f'      heads: {layer.get("heads", 4)}                 # number of attention heads\n'
            yaml_str += f'      concat: {str(layer.get("concat", True)).lower()}              # concatenate heads or average\n'
            if layer.get("dropout", 0) > 0:
                yaml_str += f'      dropout: {layer.get("dropout", 0.0)}            # attention dropout\n'
        #elif gnn_type == "GCN":
        #    yaml_str += f'      improved: {str(layer.get("improved", False)).lower()}           # use improved GCN\n'
        
        if layer.get("batchnorm"):
            yaml_str += f'      batchnorm: true\n'
        if layer.get("dropout", 0) > 0 and gnn_type != "GAT":
            yaml_str += f'      dropout: {layer.get("dropout", 0.0)}\n'
    
    yaml_str += f'\n  pooling: {config["model"]["pooling"]}         # global pooling method\n'
    yaml_str += f'  output_units: {config["model"]["output_units"]}               # 2 for binary classification\n'
    out_act = config["model"]["output_activation"]
    yaml_str += f'  output_activation: {out_act if out_act else "null"}       # null for logits\n\n'
    
    yaml_str += """# ============================================================================
# Training Configuration
# ============================================================================
train:
"""
    train = config["train"]
    yaml_str += f'  epochs: {train["epochs"]}\n'
    yaml_str += f'  batch_size: {train["batch_size"]}\n'
    yaml_str += f'  learning_rate: {train["learning_rate"]}\n'
    yaml_str += f'  weight_decay: {train["weight_decay"]}          # L2 regularization\n'
    yaml_str += f'  optimizer: {train["optimizer"]}               # adam, adamw, sgd\n'
    yaml_str += f'  device: {train["device"]}                  # auto, cpu, cuda, cuda:0, cuda:1\n'
    yaml_str += f'\n  # Early Stopping\n'
    yaml_str += f'  early_stopping: {str(train["early_stopping"]).lower()}\n'
    yaml_str += f'  early_stopping_patience: {train["early_stopping_patience"]}   # stop if no improvement for N epochs\n'
    yaml_str += f'  early_stopping_metric: {train["early_stopping_metric"]}  # val_loss or eval_metric\n'
    yaml_str += f'\n  # Training Precision\n'
    yaml_str += f'  precision: {train["precision"]}            # float32, float16, mixed\n'
    yaml_str += f'\n  # Learning Rate Scheduler\n'
    yaml_str += f'  scheduler:\n'
    yaml_str += f'    type: {train["scheduler"]["type"]}               # none, step, plateau, cosine, onecycle\n'
    
    sched = train["scheduler"]
    if sched["type"] == "step":
        yaml_str += f'    step_size: {sched.get("step_size", 10)}               # reduce LR every N epochs\n'
        yaml_str += f'    gamma: {sched.get("gamma", 0.1)}                  # LR multiplier\n'
    elif sched["type"] == "plateau":
        yaml_str += f'    patience: {sched.get("patience", 5)}                 # reduce LR if no improvement for N epochs\n'
        yaml_str += f'    min_lr: {sched.get("min_lr", 0.000001)}            # minimum learning rate\n'
    elif sched["type"] == "onecycle":
        yaml_str += f'    max_lr: {sched.get("max_lr", 0.01)}                # maximum learning rate\n'
    elif sched["type"] == "cosine":
        yaml_str += f'    # Cosine annealing uses epochs as T_max\n'
    
    yaml_str += f'\n  # Evaluation Metric (for model selection and reporting)\n'
    yaml_str += f'  eval_metric: {train["eval_metric"]}              # accuracy, auc, f1, recall, precision\n'
    
    return yaml_str


#=======================================
# Helper functions for Transformer network 
#=======================================

def generate_transformer_config(
    seed, output_dir, signal_path, background_path, train_size, test_size,
    val_ratio, normalize, particles_per_cloud,
    embed_dim, num_heads, num_layers, ffn_dim, dropout, attention_dropout,
    pooling, num_classes, pre_norm,
    epochs, batch_size, learning_rate, weight_decay, optimizer, device,
    early_stopping, early_stopping_patience, early_stopping_metric,
    precision, scheduler_type, scheduler_params, eval_metric, gradient_clip
):
    """Generate Transformer configuration dictionary matching config.yaml format."""
    
    # Build scheduler config
    scheduler_config = {"type": scheduler_type}
    if scheduler_type == "step":
        scheduler_config["step_size"] = scheduler_params.get("step_size", 10)
        scheduler_config["gamma"] = scheduler_params.get("gamma", 0.1)
    elif scheduler_type == "plateau":
        scheduler_config["patience"] = scheduler_params.get("patience", 5)
        scheduler_config["min_lr"] = scheduler_params.get("min_lr", 1e-6)
    elif scheduler_type == "onecycle":
        scheduler_config["max_lr"] = scheduler_params.get("max_lr", learning_rate * 10)
    elif scheduler_type == "cosine_warmup":
        scheduler_config["warmup_steps"] = scheduler_params.get("warmup_steps", 100)
        scheduler_config["min_lr"] = scheduler_params.get("min_lr", 1e-6)
    
    # Build complete config
    config = {
        "seed": seed,
        "output_dir": output_dir,
        
        "data": {
            "signal_path": signal_path,
            "background_path": background_path,
            "train_size": train_size,
            "test_size": test_size,
            "val_ratio": val_ratio,
            "normalize": normalize,
            "particles_per_cloud": particles_per_cloud
        },
        
        "model": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ffn_dim": ffn_dim,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "pooling": pooling,
            "num_classes": num_classes,
            "pre_norm": pre_norm
        },
        
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "device": device,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_metric": early_stopping_metric,
            "precision": precision,
            "scheduler": scheduler_config,
            "eval_metric": eval_metric,
            "gradient_clip": gradient_clip
        }
    }
    
    return config


def transformer_config_to_yaml(config):
    """Convert Transformer config dictionary to YAML string with comments."""
    
    yaml_str = """# ============================================================================
# Transformer Classifier Configuration
# ============================================================================

"""
    yaml_str += f"seed: {config['seed']}\n"
    yaml_str += f'output_dir: "{config["output_dir"]}"\n\n'
    
    yaml_str += """# ============================================================================
# Data Configuration
# ============================================================================
data:
"""
    yaml_str += f'  signal_path: "{config["data"]["signal_path"]}"\n'
    yaml_str += f'  background_path: "{config["data"]["background_path"]}"\n'
    yaml_str += f'  train_size: {config["data"]["train_size"]}          # samples per class for training\n'
    yaml_str += f'  test_size: {config["data"]["test_size"]}            # samples per class for testing\n'
    yaml_str += f'  val_ratio: {config["data"]["val_ratio"]}             # fraction of training data for validation\n'
    yaml_str += f'  normalize: {str(config["data"]["normalize"]).lower()}             # normalize features\n'
    yaml_str += f'  particles_per_cloud: {config["data"]["particles_per_cloud"]}      # number of particles per cloud\n\n'
    
    yaml_str += """# ============================================================================
# Model Architecture - Transformer
# ============================================================================
# Pooling: mean, max, attention
model:
"""
    model = config["model"]
    yaml_str += f'  embed_dim: {model["embed_dim"]}                # embedding dimension\n'
    yaml_str += f'  num_heads: {model["num_heads"]}                 # number of attention heads\n'
    yaml_str += f'  num_layers: {model["num_layers"]}                # number of transformer layers\n'
    yaml_str += f'  ffn_dim: {model["ffn_dim"]}                  # feed-forward network dimension\n'
    yaml_str += f'  dropout: {model["dropout"]}                 # dropout rate\n'
    yaml_str += f'  attention_dropout: {model["attention_dropout"]}       # attention dropout rate\n'
    yaml_str += f'  pooling: {model["pooling"]}                  # pooling method: mean, max, attention\n'
    yaml_str += f'  num_classes: {model["num_classes"]}                # number of output classes\n'
    yaml_str += f'  pre_norm: {str(model["pre_norm"]).lower()}               # use pre-normalization\n\n'
    
    yaml_str += """# ============================================================================
# Training Configuration
# ============================================================================
train:
"""
    train = config["train"]
    yaml_str += f'  epochs: {train["epochs"]}\n'
    yaml_str += f'  batch_size: {train["batch_size"]}\n'
    yaml_str += f'  learning_rate: {train["learning_rate"]}\n'
    yaml_str += f'  weight_decay: {train["weight_decay"]}          # L2 regularization\n'
    yaml_str += f'  optimizer: {train["optimizer"]}               # adam, adamw, sgd\n'
    yaml_str += f'  device: {train["device"]}                  # auto, cpu, cuda, cuda:0, cuda:1\n'
    yaml_str += f'\n  # Early Stopping\n'
    yaml_str += f'  early_stopping: {str(train["early_stopping"]).lower()}\n'
    yaml_str += f'  early_stopping_patience: {train["early_stopping_patience"]}   # stop if no improvement for N epochs\n'
    yaml_str += f'  early_stopping_metric: {train["early_stopping_metric"]}  # val_loss or eval_metric\n'
    yaml_str += f'\n  # Training Precision\n'
    yaml_str += f'  precision: {train["precision"]}            # float32, float16, mixed\n'
    yaml_str += f'\n  # Gradient Clipping\n'
    yaml_str += f'  gradient_clip: {train["gradient_clip"]}             # gradient clipping value\n'
    yaml_str += f'\n  # Learning Rate Scheduler\n'
    yaml_str += f'  scheduler:\n'
    yaml_str += f'    type: {train["scheduler"]["type"]}               # none, step, plateau, cosine, cosine_warmup, onecycle\n'
    
    sched = train["scheduler"]
    if sched["type"] == "step":
        yaml_str += f'    step_size: {sched.get("step_size", 10)}               # reduce LR every N epochs\n'
        yaml_str += f'    gamma: {sched.get("gamma", 0.1)}                  # LR multiplier\n'
    elif sched["type"] == "plateau":
        yaml_str += f'    patience: {sched.get("patience", 5)}                 # reduce LR if no improvement for N epochs\n'
        yaml_str += f'    min_lr: {sched.get("min_lr", 0.000001)}            # minimum learning rate\n'
    elif sched["type"] == "onecycle":
        yaml_str += f'    max_lr: {sched.get("max_lr", 0.01)}                # maximum learning rate\n'
    elif sched["type"] == "cosine_warmup":
        yaml_str += f'    warmup_steps: {sched.get("warmup_steps", 100)}          # warmup steps\n'
        yaml_str += f'    min_lr: {sched.get("min_lr", 0.000001)}            # minimum learning rate\n'
    elif sched["type"] == "cosine":
        yaml_str += f'    # Cosine annealing uses epochs as T_max\n'
    
    yaml_str += f'\n  # Evaluation Metric (for model selection and reporting)\n'
    yaml_str += f'  eval_metric: {train["eval_metric"]}              # accuracy, auc, f1, recall, precision\n'
    
    return yaml_str


#=======================================
#  Style for CoLLM page. This style is created by Opus 4.5
#=======================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&family=Sora:wght@300;400;500;600&display=swap');
    
    /* ═══════════════════ ROOT VARIABLES ═══════════════════ */
    :root {
        --bg-primary: #1e1e2e;
        --bg-secondary: #262640;
        --bg-card: #2a2a45;
        --bg-hover: #353550;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #a855f7;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #f1f5f9;
        --text-secondary: #fff;
        --text-title: #94a3b8;
        --text-muted: #64748b;
        --border-color: #454560;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        --gradient-2: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    }
    
    /* ═══════════════════ GLOBAL STYLES ═══════════════════ */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Sora', sans-serif;
    }
    
    .stApp > header {
        background: transparent;
    }
    
    /* ═══════════════════ SIDEBAR STYLING ═══════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #262640 0%, #1e1e2e 100%);
        border-right: 1px solid var(--border-color);
        width: 340px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--text-secondary);
    }
    
    /* ═══════════════════ MAIN CONTENT ═══════════════════ */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ═══════════════════ TYPOGRAPHY ═══════════════════ */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }
    
    p, span, label, div {
        color: var(--text-secondary);
    }
    
    /* ═══════════════════ CUSTOM HERO SECTION ═══════════════════ */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-1);
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size:15rem;
        font-weight: 1700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-family: 'Sora', sans-serif;
        font-size: 6.1rem;
        color: white; /*var(--text-title);*/
        max-width: 900px;
        line-height: 1.7;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: var(--accent-primary);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* ═══════════════════ SECTION HEADERS ═══════════════════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .section-icon {
        width: 44px;
        height: 44px;
        background: var(--gradient-1);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5.6rem;
        font-weight: 600;
        color:  var(--text-primary);
        margin: 0;
    }
    
    .section-desc {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    /* ═══════════════════ CARD COMPONENTS ═══════════════════ */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 0 30px var(--accent-glow);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* ═══════════════════ INPUT STYLING ═══════════════════ */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: white; #var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }
    
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSelectbox label,
    .stRadio label {
        font-family: 'Sora', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        font-size: 4.9rem !important;
    }
    
    /* ═══════════════════ SELECT BOX STYLING ═══════════════════ */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    
    /* ═══════════════════ BUTTON STYLING ═══════════════════ */
    .stButton > button {
        background: var(--gradient-1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 2.9rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--accent-glow) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px var(--accent-glow) !important;
    }
    
    /* ═══════════════════ RADIO BUTTONS ═══════════════════ */
    .stRadio > div {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stRadio > div > div > label {
        color: var(--text-secondary) !important;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stRadio > div > div > label:hover {
        background: var(--bg-hover);
    }
    
    /* ═══════════════════ CHECKBOX STYLING ═══════════════════ */
    .stCheckbox > label {
        color: var(--text-secondary) !important;
    }
    
    /* ═══════════════════ EXPANDER STYLING ═══════════════════ */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ═══════════════════ TABS STYLING ═══════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 24px;
        padding: 18px;
        gap: 20px;
        border: 2px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 20px !important;
        color: var(--text-secondary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 5.2rem !important;
        padding: 32px 60px !important;
        transition: all 0.2s ease !important;
        min-height: 90px !important;

    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
    }
    
    
    /* ═══════════════════ PROGRESS BAR ═══════════════════ */
    .stProgress > div > div > div {
        background: var(--gradient-1) !important;
    }
    
    /* ═══════════════════ SUCCESS/ERROR MESSAGES ═══════════════════ */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* ═══════════════════ DIVIDER ═══════════════════ */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* ═══════════════════ METRIC CARDS ═══════════════════ */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }
    
    /* ═══════════════════ AUTHOR CARDS ═══════════════════ */
    .author-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .author-card:hover {
        border-color: var(--accent-primary);
    }
    
    .author-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.15rem;
    }
    
    .author-email {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.00rem;
        color: var(--text-muted);
    }
    
    /* ═══════════════════ SCROLLBAR ═══════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
    
    /* ═══════════════════ ANIMATION KEYFRAMES ═══════════════════ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes terminalPulse {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(99, 102, 241, 0.3);
        }
        50% { 
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.6), 0 0 30px rgba(168, 85, 247, 0.3);
        }
    }
    
    @keyframes cursorBlink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .terminal-output {
        animation: terminalPulse 1.5s ease-in-out infinite;
        border: 1px solid var(--accent-primary) !important;
        border-radius: 10px;
    }
    
    .terminal-cursor {
        display: inline-block;
        width: 8px;
        height: 16px;
        background: var(--accent-primary);
        margin-left: 2px;
        animation: cursorBlink 1s infinite;
    }
    
    /* ═══════════════════ YAML CODE DISPLAY ═══════════════════ */
    .yaml-preview {
        background: #2a2a45;
        border: 1px solid #454560;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ====================================
#            Sidbar
# ====================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">CoLLM</div>
        <p style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">
            ML Toolbox for collider analyses
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Authors  
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <p style="
            color: white;
            font-size: 1.2rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        ">Authors</p>
    </div>
    """, unsafe_allow_html=True)
    
    authors = [
        {"name": "Ahmed Hammad", "email": "ahammad115566@gmail.com"},
        {"name": "Waleed Esmail", "email": "waleed.physics@gmail.com"},
        {"name": "Mihoko Nojiri", "email": "mihoko.nojiri@gmail.com "},
    ]
    
    for author in authors:
        st.markdown(f"""
        <div class="author-card">
            <div class="author-name">{author['name']}</div>
            <div class="author-email">{author['email']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # HuggingFace API Instructions
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <p style="
            color: #94a3b8;
            font-size: 1.15rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        ">🤗 HuggingFace API Setup</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 How to create a HuggingFace API Token", expanded=True):
        st.markdown("""
        **Step 1: Create a HuggingFace Account**
        - Go to [huggingface.co](https://huggingface.co)
        - Click **Sign Up** and create an account
        - Verify your email address
        
        **Step 2: Generate an API Token**
        1. Log in to your HuggingFace account
        2. Click on your profile picture (top right)
        3. Select **Settings**
        4. Navigate to **Access Tokens** in the left sidebar
        5. Click **New token**
        6. Give your token a name (e.g., "CoLLM")
        7. Select **Read** permission (or **Write** if needed)
        8. Click **Generate token**
        9. **Copy the token immediately** (it won't be shown again!)
        
        **Step 3: Use the Token**
        - Store your token securely
        
        **⚠️ Important:**
        - Never share your token publicly
        """)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # HuggingFace API Instructions
    with st.expander("🤗 HuggingFace API Setup", expanded=True):
        st.markdown("""
        <div style="font-size: 1.15rem; color: white;">
        
        **Steps to create a HuggingFace API Token:**
        
        1. **Create an account**
           - Go to [huggingface.co](https://huggingface.co)
           - Click "Sign Up" and create a free account
        
        2. **Access Settings**
           - Click on your profile picture (top right)
           - Select "Settings" from the dropdown
        
        3. **Generate Access Token**
           - In the left sidebar, click "Access Tokens"
           - Click "New token" button
           - Give your token a name (e.g., "CoLLM")
           - Select token type: **Read** (for inference)
           - Click "Generate token"
        
        4. **Copy and Save**
           - Copy the generated token immediately
           - Store it securely (you won't see it again!)
        
        5. **Use in CoLLM**
           - Paste your token in the API Key field
           - The token format: `hf_xxxxxxxxxxxxxxxxx`
        
        </div>
        
        <div style="
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.75rem;
        ">
            
        </div>
        """, unsafe_allow_html=True)

# ===================================
#  Main part
# ===================================

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">CoLLM Toolbox</h1>
    <p class="hero-subtitle">
        An integrated framework combining LLM-powered selection analysis code generation 
        with advanced deep learning classifiers for high energy physics research.
    </p>
        <p class="hero-subtitle">
⚠️   For LLM code generation on your laptop, we strongly recommend using the Hugging Face API.
This enables inference via Hugging Face Claude. 
Please refer to the sidebar for detailed instructions on how to create your API key.
 </p>

</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs([" Selection Analysis", " Deep Learning", " Results"])
start_training = False
gnn_start_training = False
tr_start_training = False
#===================================================
#     Selection analysis configuration
#===================================================

 #   st.markdown("""
 #   <div class="section-header">
 #    </div>
#    """, unsafe_allow_html=True)
#    
#    # Signal Files Section
#    with st.container():
#        col1, col2 = st.columns([1, 1])
#        
#        with col1:
#            st.markdown("""
#            <div class="custom-card">
#                <div class="card-header">
#                    <span style="font-size: 1.2rem;">⚡</span>
#                    <h4 class="card-title">Signal Files</h4>
#                </div>
#            """, unsafe_allow_html=True)
 #           
 #           num_sig = st.number_input(
 #               label="Number of signal types",
#                min_value=1,
#                max_value=100,
#                step=1,
#                value=1,
#                key="num_signals",
#                help="Specify the number of different signal samples"
#            )
 #           
 #           st.markdown("</div>", unsafe_allow_html=True)
 #           
 #           sig_dirs = ['sig_' + str(i) for i in range(int(num_sig))]
 #           sigma_sig = []
 #           
 #           for i, item in enumerate(sig_dirs):
 #               with st.expander(f"📂 Signal {i+1} Configuration", expanded=(i==0)):
 #                   sig_dirs[i] = st.text_input(
 #                       label=f"Path to signal-{i+1} directory",
 #                       placeholder=f"/path/to/signal_{i+1}/files",
 #                       key=item
 #                   )
 #                   sigma_1 = st.number_input(
 #                       label=f"Cross section (pb)",
 #                       min_value=1.0e-08,
 #                       step=1.0e-08,
 #                       format="%.8f",
 #                       value=1.0,
 #                       key=f"sigma_sig_{i}",
 #                       help="Cross section value in picobarns"
 #                   )
 #                   sigma_sig.append(sigma_1)
 #       
 #       with col2:
 #           st.markdown("""
 #           <div class="custom-card">
 #               <div class="card-header">
 #                   <span style="font-size: 1.2rem;">🌫️</span>
 #                   <h4 class="card-title">Background Files</h4>
 #               </div>
 #           """, unsafe_allow_html=True)
 #           
 #           num_bkg = st.number_input(
  #              label="Number of background types",
 #               min_value=1,
 #               max_value=100,
 #               step=1,
 #               value=1,
 #               key="num_backgrounds",
 #               help="Specify the number of different background samples"
 #           )
  #          
 #           st.markdown("</div>", unsafe_allow_html=True)
 #           
 #           bkg_dirs = ['bkg_' + str(i) for i in range(int(num_bkg))]
#            sigma_bkg = []
            
#            for i, item in enumerate(bkg_dirs):
#                with st.expander(f"📂 Background {i+1} Configuration", expanded=(i==0)):
#                    bkg_dirs[i] = st.text_input(
#                        label=f"Path to background-{i+1} directory",
 #                       placeholder=f"/path/to/background_{i+1}/files",
 #                       key=item
 #                   )
 #                   sigma_ = st.number_input(
 #                       label=f"Cross section (pb)",
 #                       min_value=1.0e-08,
 #                       step=1.0e-08,
 #                       format="%.8f",
 #                       value=1.0,
 #                       key=f"sigma_bkg_{i}",
 #                       help="Cross section value in picobarns"
 #                   )
 #                   sigma_bkg.append(sigma_)
    
    # File Validation
  #  st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
 #   col_val1, col_val2, col_val3 = st.columns([1, 1, 2])
 #   with col_val1:
 #       check_files = st.button(" Validate Files", use_container_width=True)
 #   
 #   if check_files:
#        with st.spinner("Validating file paths..."):
#            time.sleep(0.5)
#            all_valid = True
#            
#            for path_ in sig_dirs:
#                if path_ and not os.path.exists(path_):
#                    st.error(f" Signal path not found: `{path_}`")
#                    all_valid = False
#                elif path_:
#                    st.success(f" Signal files verified: `{path_}`")
 #           
#            for path_ in bkg_dirs:
 #               if path_ and not os.path.exists(path_):
  #                  st.error(f" Background path not found: `{path_}`")
 #                   all_valid = False
 #               elif path_:
 #                   st.success(f" Background files verified: `{path_}`")
with tab1:    

    st.markdown("""
    <div class="section-header">
        <div class=""></div>
        <div>
            <h3 class="section-title">LLM  Analysis Generation</h3>
            <p class="section-desc">Describe your analysis in natural language</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    HEADERS = [
        "### SELECTION CUTS",
        "### PLOTS FOR VALIDATION",
        "### OUTPUT STRUCTURE"
    ]
    
    template = "\n\n".join([h + "\n" for h in HEADERS])
    
    col_llm1, col_llm2 = st.columns([2, 1])
    
    with col_llm1:
        text = st.text_area(
            "Analysis Specification (Do not change the naming tag after ###.)",
            value=template,
            height=350,
            key="preselection_analysis_specification",
            help="Describe your analysis cuts, validation plots, and output format"
        )
        
        # Restore missing headers
        for h in HEADERS:
            if h not in text:
                text = h + "\n\n" + text
    
    with col_llm2:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <div class="card-header">
                <span style="font-size: 1.2rem;">💡</span>
                <h4 class="card-title">Example Template</h4>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">
                <p><strong style="color: #a855f7;">### SELECTION CUTS</strong></p>
                <p>- Require at least 2 jets with pT > 30 GeV </p>
                <p>- Select MET > 30 GeV</p>
                <br>
                <p><strong style="color: #a855f7;">### PLOTS FOR VALIDATION</strong></p>
                <p>- Plot the MET distribution</p>
                <p>- Plot delta R between the two leading jets</p>
                <br>
                <p><strong style="color: #a855f7;">### OUTPUT STRUCTURE</strong></p>
                <p>- Save plots in png format</p>
                <p>- print summary statistics </p>
                 <p>- save the following  in a single  csv file for MLP analysis: </p>
                 <p>1- Transverse mass of lepton + MET </p>
                 <p>2- Delta R between the two jets </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    

    st.markdown("""
    <div class="section-header">
        <div class=""></div>
        <div>
            <h3 class="section-title">LLM Configuration</h3>
            <p class="section-desc">Configure the language model and execution settings</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;"></span>
                <h4 class="card-title">Paths Configuration</h4>
            </div>
        """, unsafe_allow_html=True)
        
        output_dir = st.text_input(
            "Output Directory (Please enter the full path)",
            value="/Users/hammad/CoLLM-main/output",
            help="Directory where generated analysis and plots will be saved"
        )
        
        input_file = st.text_input(
            "Input LHCO File (Please enter the full path)",
            value="/Users/hammad/CoLLM-main/data/signal_1.lhco",
            help="Path to the LHCO file for testing the generated analysis"
        )
        
       # user_input = st.text_input(
        #    "User Input Template",
         #   value="/Users/hammad/work/CoLLM/templates/user_input_1.txt",
          #  help="Path to save the user input template"
       # )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_cfg2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;"></span>
                <h4 class="card-title">Model Settings</h4>
            </div>
        """, unsafe_allow_html=True)
        
        default_model = st.selectbox(
            "LLM Model (recommend: meta-llama/Llama-3.3-70B-Instruct)",
            options=[
          "Qwen/Qwen2.5-Coder-7B-Instruct",      # Best balance of speed/quality
          "Qwen/Qwen2.5-Coder-32B-Instruct",     # Higher quality
          "Qwen/Qwen3-Coder-30B-A3B-Instruct",   # Latest MoE coder
         # General purpose (Good at code too)
         "meta-llama/Llama-3.1-8B-Instruct",
         "meta-llama/Llama-3.3-70B-Instruct",
         "Qwen/Qwen2.5-72B-Instruct",
   
        # Lightweight/Fast options
        "Qwen/Qwen2.5-Coder-3B-Instruct",
       "meta-llama/Llama-3.2-3B-Instruct",
       # Reasoning models (good for complex code)
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
         "Qwen/QwQ-32B",         
            ],
            index=0,
            help="Select the Hugging Face model for code generation"
        )
        
        max_retries = st.number_input(
            "Max Retries",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of attempts to fix generated code"
        )
        
        use_api = st.checkbox(
            "Use HuggingFace API (recommended when working on a laptop)",
            value=False,
            help="Use Hugging Face Inference API instead of local model. If not, LLM will be downloaded and decoded locally."
        )
        
        api_key = st.text_input(
            "API Key",
            value="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",
            type="password",
            help="Your Hugging Face API key (required if using API)",
            disabled=not use_api
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col_run1, col_run2, col_run3 = st.columns([1, 1, 2])
    
    with col_run1:
        run_analysis = st.button("Run Preselection Analysis", use_container_width=True)
    
    #with col_run2:
     #   save_config = st.button("💾 Save Configuration", use_container_width=True)
    
     
    if run_analysis:
        # Convert text format from GUI to expected format
        user_input_text = text.replace("### SELECTION CUTS", "[SELECTION_CUTS]")
        user_input_text = user_input_text.replace("### PLOTS FOR VALIDATION", "[PLOTS_FOR_VALIDATION]")
        user_input_text = user_input_text.replace("### OUTPUT STRUCTURE", "[OUTPUT_STRUCTURE]")
        
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📟</div>
            <div>
                <h3 class="section-title">Terminal Output</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a placeholder for terminal output
        terminal_output = st.empty()
        output_messages = []
        
        def update_terminal(message: str, status_type: str = 'info'):
            """Callback to update terminal output."""
            output_messages.append(message)
            output_text = '\n '.join(output_messages)
            escaped_output = html.escape(output_text)
            
            # Color based on status type
            color_map = {
                'info': '#10b981',
                'success': '#22c55e', 
                'warning': '#f59e0b',
                'error': '#ef4444'
            }
            color = color_map.get(status_type, '#10b981')
            
            terminal_output.markdown(f"""
            <div class="terminal-output" style="
                background: #0a0a12;
                padding: 1rem;
                border-radius: 10px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                max-height: 400px;
                overflow-y: auto;
            ">
                <pre style="margin: 0; color: {color}; white-space: pre-wrap;">{escaped_output}<span class="terminal-cursor"></span></pre>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Running analysis pipeline..."):
            status_placeholder = st.empty()
            
            try:
                
                script_content = f'''
import sys
import os
sys.path.insert(0, {repr(_project_root)})
os.chdir({repr(_project_root)})

from source.runs.run_preselection_GUI import run_LLM

run_LLM(
    {repr(output_dir)},
    {repr(default_model)},
    {repr(input_file)},
    {repr(user_input_text)},
    {repr(output_dir + "generated_lhco_analysis.py")},
    {max_retries},
    {use_api},
    {repr(api_key)}
)
'''
                
                # Write the script to a temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(script_content)
                    temp_script = f.name 

                process = subprocess.Popen(
                    [sys.executable, '-u', temp_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=_project_root,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                )
                
                
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.rstrip('\n ')
                        update_terminal(line, 'info')
                
                process.wait()
                
                # Clean up temp file
                try:
                    os.unlink(temp_script)
                except:
                    pass
                
                if process.returncode == 0:
                    update_terminal("Analysis completed successfully", "success")
                    status_placeholder.success("Analysis completed successfully!")
                    status_placeholder.text('''To proceed with the deep learning analysis, run the generated code on the signal and background events using:
python generated_lhco_analysis.py   full/path/to/file.lhco
                    
                    ''')
                                     
                else:
                    update_terminal(f"Process exited with code {process.returncode}", "error")
                    status_placeholder.error("Analysis failed")
                    
            except Exception as e:
                status_placeholder.error(f"Error occurred: {e}")
                st.error(f"Error running analysis: {e}")
                import traceback
                st.code(traceback.format_exc(), language="bash")
            
            # Display generated files if output directory exists
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                if files:
                    st.markdown("### 📁 Generated Files")
                    for f in files:
                        file_path = os.path.join(output_dir, f)
                        if f.endswith('.png'):
                            st.image(file_path, caption=f)
                        elif f.endswith('.py'):
                            st.markdown(f"📄 **{f}**")
                            with st.expander("View generated code"):
                                with open(file_path, 'r') as code_file:
                                    st.code(code_file.read(), language="python")

# ====================================
# Tab 2: Deep Learnign
# ====================================
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class=""></div>
        <div>
            <h3 class="section-title">Network Architecture</h3>
            <p class="section-desc">Select and configure your deep learning network</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ML_type = st.radio(
        "Select Model Type",
        options=[
            "Multi-Layer Perceptron (MLP)",
            "Graph Neural Networks (GNNs)",
            "Transformer"
        ],
        index=0,
        horizontal=True
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # MLP Configuration
    if ML_type == "Multi-Layer Perceptron (MLP)":
        
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;"></span>
                <h4 class="card-title">MLP Configuration</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col_arch1, col_arch2 = st.columns([1, 3])
        
        with col_arch1:
            num_layers = st.number_input(
                "Number of hidden layers",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="Number of hidden layers (excluding output)"
            )
        
        with col_arch2:
            output_units = st.selectbox(
                "Output Layer",
                options= 2,
                help=" 2 for binary classification with softmax/cross-entropy"
            )
            
            output_activation_options = ["None", "softmax"]
            
            output_activation = st.selectbox(
                "Output Activation",
                options=output_activation_options,
                index=0,
                help="None for logits (recommended with cross_entropy loss)"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Build layer configurations
        mlp_layers_config = []
        
        for i in range(int(num_layers)):
            with st.expander(f"Layer {i+1} Configuration", expanded=(i < 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    layer_options = ["Dense Layer", "Dropout Layer", "BatchNorm Layer"]
                    
                    layer_type = st.selectbox(
                        "Layer Type",
                        options=layer_options,
                        key=f"mlp_{i}_type"
                    )
                
                layer_dict = {"type": layer_type}
                
                if layer_type == "Dense Layer":
                    with col2:
                        neurons = st.number_input(
                            "Neurons",
                            min_value=1,
                            max_value=10000,
                            step=1,
                            value=128 if i == 0 else 64,
                            key=f"mlp_{i}_neurons"
                        )
                        layer_dict["neurons"] = neurons
                    
                    with col3:
                        activation_options = [
                            "relu", "leaky_relu", "gelu", "elu", "selu",
                            "tanh", "softplus", "sigmoid"
                        ]
                        
                        activation = st.selectbox(
                            "Activation",
                            options=activation_options,
                            key=f"mlp_{i}_activation"
                        )
                        layer_dict["activation"] = activation
                
                elif layer_type == "Dropout Layer":
                    with col2:
                        rate = st.slider(
                            "Dropout Rate",
                            min_value=0.0,
                            max_value=0.9,
                            value=0.2,
                            step=0.05,
                            key=f"mlp_{i}_drop_rate"
                        )
                        layer_dict["rate"] = rate
                
                mlp_layers_config.append(layer_dict)
        
        # ====================================
        #   Configure Training
        # ====================================
        st.markdown("""
        <div class="section-header">
            <div class=""></div>
            <div>
                <h3 class="section-title">Training Configuration</h3>
                <p class="section-desc">Configure training hyperparameters</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Data Settings</h4>
                </div>
            """, unsafe_allow_html=True)
            
            sig_events = st.text_input(
                "Signal Events File",
                placeholder="/path/to/signal_events.csv",
                help="Full path to the signal dataset file (CSV)"
            )
            
            if sig_events:
                if os.path.exists(sig_events):
                    st.success("File found")
                else:
                    st.error("File not found")
            
            bkg_events = st.text_input(
                "Background Events File",
                placeholder="/path/to/background_events.csv",
                help="Full path to the background dataset file (CSV)"
            )
            
            if bkg_events:
                if os.path.exists(bkg_events):
                    st.success("File found")
                else:
                    st.error("File not found")
            
            train_size = st.number_input(
                "Training Size (per class)",
                min_value=1000,
                max_value=5000000,
                value=1000,
                step=10,
                help="Number of samples per class for training"
            )
            
            test_size = st.number_input(
                "Test Size (per class)",
                min_value=1000,
                max_value=5000000,
                value=1000,
                step=10,
                help="Number of samples per class for testing"
            )
            
            normalize_data = st.checkbox("Normalize Features", value=True, help="Apply StandardScaler normalization")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Training Parameters</h4>
                </div>
            """, unsafe_allow_html=True)
            
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=10000,
                value=10,
                help="Number of training epochs"
            )
            
            batch_size = st.select_slider(
                "Batch Size",
                options=np.arange(10,2001,1),
                value=128,
                help="Samples per gradient update"
            )
            
            optimizer = st.selectbox(
                "Optimizer",
                options=["adam", "adamw", "sgd"],
                index=0,
                help="Optimization algorithm"
            )
            
            lr = st.select_slider(
                "Learning Rate",
                options=np.arange(1e-6,1e-2,1e-6),
                value=1e-3,
                format_func=lambda x: f"{x:.0e}"
            )
            
            weight_decay = st.select_slider(
                "Weight Decay (L2)",
                options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}" if x > 0 else "0"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Resources & Optimization</h4>
                </div>
            """, unsafe_allow_html=True)
            
            device = st.selectbox(
                "Hardware Device",
                options=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                help="Select computing device"
            )
            
            precision = st.selectbox(
                "Training Precision",
                options=["float32", "float16", "mixed"],
                help="Numerical precision for training"
            )
            
            scheduler = st.selectbox(
                "LR Scheduler",
                options=["none", "step", "plateau", "cosine", "onecycle"],
                index=2,
                help="Learning rate scheduler type"
            )
            
            # Scheduler-specific parameters
            scheduler_params = {}
            if scheduler == "step":
                scheduler_params["step_size"] = st.number_input("Step Size", min_value=1, value=10, help="Reduce LR every N epochs")
                scheduler_params["gamma"] = st.number_input("Gamma", min_value=0.01, max_value=1.0, value=0.1, help="LR multiplier")
            elif scheduler == "plateau":
                scheduler_params["patience"] = st.number_input("Scheduler Patience", min_value=1, value=5, help="Reduce LR if no improvement for N epochs")
                scheduler_params["min_lr"] = st.number_input("Min LR", min_value=1e-8, value=1e-6, format="%.0e", help="Minimum learning rate")
            elif scheduler == "onecycle":
                scheduler_params["max_lr"] = st.number_input("Max LR", min_value=1e-4, value=0.01, format="%.0e", help="Maximum learning rate")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
     
        col_add1, col_add2, col_add3, col_add4 = st.columns(4)
        
        with col_add1:
            eval_metric = st.selectbox(
                "Evaluation Metric",
                options=["accuracy", "auc", "f1", "recall", "precision"],
                index=1,
                help="Metric for model selection"
            )
        
        with col_add2:
            early_stopping = st.checkbox("Early Stopping", value=True)
            if early_stopping:
                patience = st.number_input(
                    "⏱️ Patience",
                    min_value=1,
                    max_value=50,
                    value=3,
                    help="Epochs to wait before stopping"
                )
            else:
                patience = 5
        
        with col_add3:
            early_stopping_metric = st.selectbox(
                "Stop Metric",
                options=["val_loss", "eval_metric"],
                help="Metric to monitor for early stopping"
            )
        
        with col_add4:
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=9999,
                value=42,
                help="For reproducibility"
            )
        
        validation_ratio = st.slider(
            "Validation Split Ratio",
            min_value=0.05,
            max_value=0.3,
            value=0.15,
            step=0.005,
            help="Fraction of training data for validation"
        )
        
        mlp_output_dir = st.text_input(
            "Output Directory",
            value="output",
            help="Directory for saving model and results"
        )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================
        #  GNNs configuration
       # ====================================
        col_train1, col_train2, col_train3, col_train4 = st.columns([1, 1, 1, 1])
        
        with col_train1:
            show_config = st.button("Preview Config", use_container_width=True)
        
        with col_train2:
            save_config = st.button("Save Config", use_container_width=True)
        
        with col_train3:
            start_training = st.button("Start Training", use_container_width=True)
        
        # Generate the configuration
        mlp_config = generate_mlp_config(
            seed=seed,
            output_dir=mlp_output_dir,
            signal_path=sig_events or "full/path/to/signal.csv",
            background_path=bkg_events or "full/path/to/background.csv",
            train_size=train_size,
            test_size=test_size,
            val_ratio=validation_ratio,
            normalize=normalize_data,
            layers_config=mlp_layers_config,
            output_units=output_units,
            output_activation=output_activation if output_activation != "None" else None,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            device=device,
            early_stopping=early_stopping,
            early_stopping_patience=patience,
            early_stopping_metric=early_stopping_metric,
            precision=precision,
            scheduler_type=scheduler,
            scheduler_params=scheduler_params,
            eval_metric=eval_metric
        )
        
        yaml_content = config_to_yaml(mlp_config)
        
        if show_config:
            st.markdown("### 📄 Configuration Preview")
            st.code(yaml_content, language="yaml")
        
        if save_config:
            # Save config to file
            config_filename = "input_config_mlp.yml"
            config_path = os.path.join(mlp_output_dir, config_filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(mlp_output_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(yaml_content)
            
            st.success(f"Configuration saved to: `{config_path}`")
            
            # Provide download button
            st.download_button(
                label="Download Config",
                data=yaml_content,
                file_name="input_config_mlp.yml",
                mime="text/yaml"
            )
        
        if start_training:
            # First save the config
            config_filename = "input_config_mlp.yml"
            config_path = os.path.join(mlp_output_dir, config_filename)
            os.makedirs(mlp_output_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(yaml_content)
            
            st.info(f"Configuration saved to: `{config_path}`")
            
            # Run training
            with st.spinner("Starting MLP training..."):
                try:
                
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'source.DL.MLP.main_mlp', '--config', config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=_project_root,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                    )
                    
                    # Terminal output
                    terminal_output = st.empty()
                    output_lines = []
                    
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_lines.append(line.rstrip())
                            terminal_output.code('\n'.join(output_lines[-30:]), language="bash")
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        st.success("Training completed successfully. Please click the Results section to see the results.")
                    else:
                        st.error(f"Training failed with exit code {process.returncode}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="bash")
    
    ##=======
    #   GNN  ==
    ##=======
    elif ML_type == "Graph Neural Networks (GNNs)":
        
      
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;"></span>
                <h4 class="card-title">GNN Model Selection</h4>
            </div>
        """, unsafe_allow_html=True)
        
        gnn_type = st.selectbox(
            "Select GNN Architecture",
            options=["EdgeConv", "GCN", "GAT"],
            index=0,
            help="EdgeConv: Dynamic Graph CNN, GCN: Graph Convolutional Network, GAT: Graph Attention Network"
        )
        
        # Model descriptions
        gnn_descriptions = {
            "EdgeConv": "**Edge Convolution (EdgeConv)** - Dynamic graph CNN that constructs graphs dynamically using k-nearest neighbors.",
            "GCN": "**Graph Convolutional Network (GCN)** - Classic spectral graph convolution with efficient layer-wise propagation. ",
            "GAT": "**Graph Attention Network (GAT)** - Uses attention mechanisms to weight neighbor contributions."
        }
        st.info(gnn_descriptions[gnn_type])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
       
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;"></span>
                <h4 class="card-title">GNN Configuration</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col_gnn_arch1, col_gnn_arch2, col_gnn_arch3 = st.columns([1, 1, 1])
        
        with col_gnn_arch1:
            gnn_num_layers = st.number_input(
                "Number of GNN layers",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="gnn_num_layers",
                help="Number of graph convolution layers"
            )
        
        with col_gnn_arch2:
            gnn_pooling = st.selectbox(
                "Global Pooling",
                options=["global_mean", "global_max", "global_add"],
                index=0,
                help="Method to aggregate node features into graph-level representation"
            )
        
        with col_gnn_arch3:
            gnn_output_units = st.selectbox(
                "Output Layer",
                options=2,
                key="gnn_output_units",
                help="2 for softmax/cross-entropy"
            )
            
            gnn_output_activation_options = ["None", "softmax"]
            
            gnn_output_activation = st.selectbox(
                "Output Activation",
                options=gnn_output_activation_options,
                index=0,
                key="gnn_output_activation",
                help="None for logits (recommended with cross_entropy loss)"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Build GNN layer configurations
        gnn_layers_config = []
        
        for i in range(int(gnn_num_layers)):
            with st.expander(f"GNN Layer {i+1} Configuration", expanded=(i < 2)):
                col1, col2, col3, col4 = st.columns(4)
                
                layer_dict = {}
                
                with col1:
                    out_channels = st.number_input(
                        "Output Channels",
                        min_value=8,
                        max_value=512,
                        value=64 if i == 0 else 128 if i == 1 else 64,
                        step=8,
                        key=f"gnn_{i}_out_channels",
                        help="Number of output features per node"
                    )
                    layer_dict["out_channels"] = out_channels
                
                with col2:
                    activation = st.selectbox(
                        "Activation",
                        options=["relu", "leaky_relu", "gelu", "elu", "selu", "tanh"],
                        index=0,
                        key=f"gnn_{i}_activation"
                    )
                    layer_dict["activation"] = activation
                
                # Model-specific parameters
                if gnn_type == "EdgeConv":
                    with col3:
                        k_neighbors = st.number_input(
                            "K Neighbors",
                            min_value=3,
                            max_value=50,
                            value=7,
                            step=1,
                            key=f"gnn_{i}_k",
                            help="Number of nearest neighbors for edge construction"
                        )
                        layer_dict["k"] = k_neighbors
                    
                    with col4:
                        aggr = st.selectbox(
                            "Aggregation",
                            options=["max", "mean", "add"],
                            index=0,
                            key=f"gnn_{i}_aggr",
                            help="Method to aggregate neighbor messages"
                        )
                        layer_dict["aggr"] = aggr
                
                elif gnn_type == "GAT":
                    with col3:
                        heads = st.number_input(
                            "Attention Heads",
                            min_value=1,
                            max_value=16,
                            value=4,
                            step=1,
                            key=f"gnn_{i}_heads",
                            help="Number of attention heads"
                        )
                        layer_dict["heads"] = heads
                    
                    with col4:
                        concat_heads = st.checkbox(
                            "Concatenate Heads",
                            value=True if i < gnn_num_layers - 1 else False,
                            key=f"gnn_{i}_concat",
                            help="Concatenate or average attention head outputs"
                        )
                        layer_dict["concat"] = concat_heads
                        
                        attn_dropout = st.slider(
                            "Attention Dropout",
                            min_value=0.0,
                            max_value=0.5,
                            value=0.0,
                            step=0.01,
                            key=f"gnn_{i}_attn_dropout",
                            help="Dropout on attention coefficients"
                        )
                        layer_dict["dropout"] = attn_dropout
                
                #elif gnn_type == "GCN":
                   # with col3:
                    #    improved = st.checkbox(
                     #       "Improved GCN",
                      #      value=False,
                       #     key=f"gnn_{i}_improved",
                        #    help="Use improved self-loop formulation"
                        #)
                        #layer_dict["improved"] = improved
                    
                    #with col4:
                     #   cached = st.checkbox(
                      #      "Cache Computation",
                       #     value=False,
                        #    key=f"gnn_{i}_cached",
                         #   help="Cache normalized adjacency matrix (only for fixed graphs)"
                       # )
                        #layer_dict["cached"] = cached
                
                # Common options for all GNN types
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    add_batchnorm = st.checkbox(
                        "Add BatchNorm",
                        value=True,
                        key=f"gnn_{i}_batchnorm"
                    )
                    layer_dict["add_batchnorm"] = add_batchnorm
                
                with col_opt2:
                    dropout_rate = st.slider(
                        "Dropout Rate",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.1 if i < gnn_num_layers - 1 else 0.0,
                        step=0.05,
                        key=f"gnn_{i}_dropout"
                    )
                    layer_dict["dropout_rate"] = dropout_rate
                
                gnn_layers_config.append(layer_dict)
        
        # ====================================
        #   GNN training configuration
        # ====================================
        st.markdown("""
        <div class="section-header">
            <div class=""></div>
            <div>
                <h3 class="section-title">Training Configuration</h3>
                <p class="section-desc">Configure training hyperparameters</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Data Settings</h4>
                </div>
            """, unsafe_allow_html=True)
            
            gnn_sig_events = st.text_input(
                "Signal Events File",
                placeholder="/path/to/signal_graphs.csv",
                key="gnn_sig_events",
                help="Path to signal graph data (PyG format or CSV)"
            )
            
            if gnn_sig_events:
                if os.path.exists(gnn_sig_events):
                    st.success("File found")
                else:
                    st.error("File not found")
            
            gnn_bkg_events = st.text_input(
                "Background Events File",
                placeholder="/path/to/background_graphs.csv",
                key="gnn_bkg_events",
                help="Path to background graph data (PyG format or CSV)"
            )
            
            if gnn_bkg_events:
                if os.path.exists(gnn_bkg_events):
                    st.success("File found")
                else:
                    st.error("File not found")
            
            gnn_train_size = st.number_input(
                "Training Size (per class)",
                min_value=1000,
                max_value=1000000,
                value=1000,
                step=10,
                key="gnn_train_size",
                help="Number of graphs per class for training"
            )
            
            gnn_test_size = st.number_input(
                "Test Size (per class)",
                min_value=1000,
                max_value=500000,
                value=1000,
                step=10,
                key="gnn_test_size",
                help="Number of graphs per class for testing"
            )
            
            gnn_normalize = st.checkbox("Normalize Features", value=True, key="gnn_normalize")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Training Parameters</h4>
                </div>
            """, unsafe_allow_html=True)
            
            gnn_epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=10,
                key="gnn_epochs",
                help="Number of training epochs"
            )
            
            gnn_batch_size = st.select_slider(
                "Batch Size",
                options=np.arange(10,2001,1),
                value=64,
                key="gnn_batch_size",
                help="Number of graphs per batch"
            )
            
            gnn_optimizer = st.selectbox(
                "Optimizer",
                options=["adam", "adamw", "sgd"],
                index=0,
                key="gnn_optimizer"
            )
            
            gnn_lr = st.select_slider(
                "Learning Rate",
                options=np.arange(1e-6,1e-2,1e-6),
                value=1e-3,
                key="gnn_lr",
                format_func=lambda x: f"{x:.0e}"
            )
            
            gnn_weight_decay = st.select_slider(
                "Weight Decay (L2)",
                options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                value=1e-4,
                key="gnn_weight_decay",
                format_func=lambda x: f"{x:.0e}" if x > 0 else "0"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>🖥️</span>
                    <h4 class="card-title">Resources & Optimization</h4>
                </div>
            """, unsafe_allow_html=True)
            
            gnn_device = st.selectbox(
                "Hardware Device",
                options=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                key="gnn_device",
                help="Select computing device"
            )
            
            gnn_precision = st.selectbox(
                "Training Precision",
                options=["float32", "float16", "mixed"],
                key="gnn_precision",
                help="Numerical precision for training"
            )
            
            gnn_scheduler = st.selectbox(
                "LR Scheduler",
                options=["none", "step", "plateau", "cosine", "onecycle"],
                index=2,
                key="gnn_scheduler",
                help="Learning rate scheduler type"
            )
            
            # Scheduler-specific parameters
            gnn_scheduler_params = {}
            if gnn_scheduler == "step":
                gnn_scheduler_params["step_size"] = st.number_input("Step Size", min_value=1, value=10, key="gnn_sched_step")
                gnn_scheduler_params["gamma"] = st.number_input("Gamma", min_value=0.01, max_value=1.0, value=0.1, key="gnn_sched_gamma")
            elif gnn_scheduler == "plateau":
                gnn_scheduler_params["patience"] = st.number_input("Scheduler Patience", min_value=1, value=5, key="gnn_sched_patience")
                gnn_scheduler_params["min_lr"] = st.number_input("Min LR", min_value=1e-8, value=1e-6, format="%.0e", key="gnn_sched_minlr")
            elif gnn_scheduler == "onecycle":
                gnn_scheduler_params["max_lr"] = st.number_input("Max LR", min_value=1e-4, value=0.01, format="%.0e", key="gnn_sched_maxlr")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        col_add1, col_add2, col_add3, col_add4 = st.columns(4)
        
        with col_add1:
            gnn_eval_metric = st.selectbox(
                "📊 Evaluation Metric",
                options=["accuracy", "auc", "f1", "recall", "precision"],
                index=1,
                key="gnn_eval_metric"
            )
        
        with col_add2:
            gnn_early_stopping = st.checkbox("Early Stopping", value=True, key="gnn_early_stopping")
            if gnn_early_stopping:
                gnn_patience = st.number_input(
                    "Patience",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="gnn_patience"
                )
            else:
                gnn_patience = 5
        
        with col_add3:
            gnn_early_stopping_metric = st.selectbox(
                "Stop Metric",
                options=["val_loss", "eval_metric"],
                key="gnn_es_metric"
            )
        
        with col_add4:
            gnn_seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=9999,
                value=42,
                key="gnn_seed"
            )
        
        # Graph-specific settings : To be modified 
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            nodes_per_graph = st.number_input(
                "Number of particles per graph",
                min_value=1,
                max_value=100,
                value=4,
                key="nodes_per_graph",
                help="Number of particles in the graph"
            )
        
        
        
        with col_graph2:
            gnn_validation_ratio = st.slider(
                "Validation Split Ratio",
                min_value=0.05,
                max_value=0.3,
                value=0.15,
                step=0.005,
                key="gnn_val_ratio"
            )
        
        gnn_output_dir = st.text_input(
            "Output Directory",
            value="output",
            key="gnn_output_dir",
            help="Directory for saving model and results"
        )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================
        #     Gnn Training
        # ====================================
        col_train1, col_train2, col_train3, col_train4 = st.columns([1, 1, 1, 1])
        
        with col_train1:
            gnn_show_config = st.button("Preview Config", use_container_width=True, key="gnn_preview")
        
        with col_train2:
            gnn_save_config = st.button("Save Config", use_container_width=True, key="gnn_save")
        
        with col_train3:
            gnn_start_training = st.button("Start Training", use_container_width=True, key="gnn_train")
        
        # Generate the GNN configuration
        gnn_config = generate_gnn_config(
            seed=gnn_seed,
            output_dir=gnn_output_dir,
            signal_path=gnn_sig_events or "/path/to/signal_graphs.csv",
            background_path=gnn_bkg_events or "/path/to/background_graphs.csv",
            train_size=gnn_train_size,
            test_size=gnn_test_size,
            val_ratio=gnn_validation_ratio,
            normalize=gnn_normalize,
            nodes_per_graph=nodes_per_graph,
            gnn_type=gnn_type,
            gnn_layers_config=gnn_layers_config,
            pooling=gnn_pooling,
            output_units=gnn_output_units,
            output_activation=gnn_output_activation if gnn_output_activation != "None" else None,
            epochs=gnn_epochs,
            batch_size=gnn_batch_size,
            learning_rate=gnn_lr,
            weight_decay=gnn_weight_decay,
            optimizer=gnn_optimizer,
            device=gnn_device,
            early_stopping=gnn_early_stopping,
            early_stopping_patience=gnn_patience,
            early_stopping_metric=gnn_early_stopping_metric,
            precision=gnn_precision,
            scheduler_type=gnn_scheduler,
            scheduler_params=gnn_scheduler_params,
            eval_metric=gnn_eval_metric
        )
        
        gnn_yaml_content = gnn_config_to_yaml(gnn_config)
        
        if gnn_show_config:
            st.markdown("### 📄 GNN Configuration Preview")
            st.code(gnn_yaml_content, language="yaml")
        
        if gnn_save_config:
            config_filename = "input_config_gnn.yml"
            config_path = os.path.join(gnn_output_dir, config_filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(gnn_output_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(gnn_yaml_content)
            
            st.success(f"Configuration saved to: `{config_path}`")
            
            # Provide download button
            st.download_button(
                label="📥 Download Config",
                data=gnn_yaml_content,
                file_name="input_config_gnn.yml",
                mime="text/yaml",
                key="gnn_download"
            )
        
        if gnn_start_training:
            config_filename = "input_config_gnn.yml"
            config_path = os.path.join(gnn_output_dir, config_filename)
            os.makedirs(gnn_output_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(gnn_yaml_content)
            
            st.info(f"Configuration saved to: `{config_path}`")
            
            # Run training
            with st.spinner("Starting GNN training..."):
                try:
                    # Run the GNN training script
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'source.DL.GNN.main_gnn', '--config', config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=_project_root,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                    )
                    
                    # Terminal output
                    terminal_output = st.empty()
                    output_lines = []
                    
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_lines.append(line.rstrip())
                            terminal_output.code('\n'.join(output_lines[-30:]), language="bash")
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        st.success("Training completed successfully. . Please click the Results section to see the results.")
                    else:
                        st.error(f"Training failed with exit code {process.returncode}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="bash")
    
    ##=======
    #   TRANSFORMER  ==
    ##=======
    elif ML_type == "Transformer":
        
        #st.markdown("""
        #<div class="custom-card">
         #   <div class="card-header">
          #      <span style="font-size: 1.2rem;"></span>
           #     <h4 class="card-title">Transformer  Configuration</h4>
            #</div>
        #""", unsafe_allow_html=True)
        #
        #st.info("**Particle Cloud Transformer** - Uses self-attention over particles followed by pooling for cloud-level prediction.")
        #
       # st.markdown("</div>", unsafe_allow_html=True)
        
        
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;"></span>
                <h4 class="card-title">Transformer Configuration</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col_tr_arch1, col_tr_arch2, col_tr_arch3 = st.columns([1, 1, 1])
        
        with col_tr_arch1:
            tr_embed_dim = st.selectbox(
                "Embedding Dimension",
                options=[64, 128, 256, 512],
                index=1,
                key="tr_embed_dim",
                help="Dimension of the embedding space"
            )
            
            tr_num_heads = st.selectbox(
                "Number of Attention Heads",
                options=[1, 2, 4, 8, 16],
                index=3,
                key="tr_num_heads",
                help="Number of attention heads (must divide embed_dim)"
            )
        
        with col_tr_arch2:
            tr_num_layers = st.number_input(
                "Number of Transformer Layers",
                min_value=1,
                max_value=12,
                value=4,
                step=1,
                key="tr_num_layers",
                help="Number of transformer encoder layers"
            )
            
            tr_ffn_dim = st.selectbox(
                "FFN Dimension",
                options=[128, 256, 512, 1024, 2048],
                index=1,
                key="tr_ffn_dim",
                help="Feed-forward network hidden dimension"
            )
        
        with col_tr_arch3:
            tr_pooling = st.selectbox(
                "Pooling Method",
                options=["mean", "max", "attention"],
                index=0,
                key="tr_pooling",
                help="mean/max: pooling over sequence, attention: learnable query"
            )
            
            tr_num_classes = st.selectbox(
                "Output Classes",
                options=2,
                key="tr_num_classes",
                help="2 for binary classification"
            )
        
        # Advanced architecture options
        col_tr_adv1, col_tr_adv2, col_tr_adv3 = st.columns(3)
        
        with col_tr_adv1:
            tr_dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                key="tr_dropout",
                help="General dropout rate"
            )
        
        with col_tr_adv2:
            tr_attention_dropout = st.slider(
                "Attention Dropout",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                key="tr_attention_dropout",
                help="Dropout on attention weights"
            )
        
        with col_tr_adv3:
            tr_pre_norm = st.checkbox(
                "Pre-Normalization",
                value=True,
                key="tr_pre_norm",
                help="Use pre-normalization (more stable training)"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        

        st.markdown("""
        <div class="section-header">
            <div class=""></div>
            <div>
                <h3 class="section-title">Training Configuration</h3>
                <p class="section-desc">Configure training hyperparameters</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Data Settings</h4>
                </div>
            """, unsafe_allow_html=True)
            
            tr_sig_events = st.text_input(
                "Signal Events File",
                placeholder="/path/to/signal_clouds.csv",
                key="tr_sig_events",
                help="Path to signal particle cloud data (CSV)"
            )
            
            if tr_sig_events:
                if os.path.exists(tr_sig_events):
                    st.success("File found")
                else:
                    st.error("File not found")
            
            tr_bkg_events = st.text_input(
                "Background Events File",
                placeholder="/path/to/background_clouds.csv",
                key="tr_bkg_events",
                help="Path to background particle cloud data (CSV)"
            )
            
            if tr_bkg_events:
                if os.path.exists(tr_bkg_events):
                    st.success("File found")
                else:
                    st.error("File not found")
            
            tr_train_size = st.number_input(
                "Training Size (per class)",
                min_value=1000,
                max_value=1000000,
                value=1000,
                step=10,
                key="tr_train_size",
                help="Number of clouds per class for training"
            )
            
            tr_test_size = st.number_input(
                "Test Size (per class)",
                min_value=1000,
                max_value=500000,
                value=1000,
                step=10,
                key="tr_test_size",
                help="Number of clouds per class for testing"
            )
            
            tr_normalize = st.checkbox("Normalize Features", value=True, key="tr_normalize")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Training Parameters</h4>
                </div>
            """, unsafe_allow_html=True)
            
            tr_epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=10,
                key="tr_epochs",
                help="Number of training epochs"
            )
            
            tr_batch_size = st.select_slider(
                "Batch Size",
                options=np.arange(10,2001,1),
                value=64,
                key="tr_batch_size",
                help="Number of clouds per batch"
            )
            
            tr_optimizer = st.selectbox(
                "Optimizer",
                options=["adamw", "adam", "sgd"],
                index=0,
                key="tr_optimizer",
                help="AdamW is recommended for Transformers"
            )
            
            tr_lr = st.select_slider(
                "Learning Rate",
                options=np.arange(1e-6,1e-2,1e-6),
                value=1e-4,
                key="tr_lr",
                format_func=lambda x: f"{x:.0e}"
            )
            
            tr_weight_decay = st.select_slider(
                "Weight Decay (L2)",
                options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
                value=1e-2,
                key="tr_weight_decay",
                format_func=lambda x: f"{x:.0e}" if x > 0 else "0"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span></span>
                    <h4 class="card-title">Resources & Optimization</h4>
                </div>
            """, unsafe_allow_html=True)
            
            tr_device = st.selectbox(
                "Hardware Device",
                options=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                key="tr_device",
                help="Select computing device"
            )
            
            tr_precision = st.selectbox(
                "Training Precision",
                options=["float32", "float16", "mixed"],
                key="tr_precision",
                help="Numerical precision for training"
            )
            
            tr_gradient_clip = st.slider(
                "Gradient Clipping",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="tr_gradient_clip",
                help="Gradient clipping value (important for Transformers)"
            )
            
            tr_scheduler = st.selectbox(
                "LR Scheduler",
                options=["none", "step", "plateau", "cosine", "cosine_warmup", "onecycle"],
                index=3,
                key="tr_scheduler",
                help="Learning rate scheduler type"
            )
            
            
            tr_scheduler_params = {}
            if tr_scheduler == "step":
                tr_scheduler_params["step_size"] = st.number_input("Step Size", min_value=1, value=10, key="tr_sched_step")
                tr_scheduler_params["gamma"] = st.number_input("Gamma", min_value=0.01, max_value=1.0, value=0.1, key="tr_sched_gamma")
            elif tr_scheduler == "plateau":
                tr_scheduler_params["patience"] = st.number_input("Scheduler Patience", min_value=1, value=5, key="tr_sched_patience")
                tr_scheduler_params["min_lr"] = st.number_input("Min LR", min_value=1e-8, value=1e-6, format="%.0e", key="tr_sched_minlr")
            elif tr_scheduler == "onecycle":
                tr_scheduler_params["max_lr"] = st.number_input("Max LR", min_value=1e-4, value=0.01, format="%.0e", key="tr_sched_maxlr")
            elif tr_scheduler == "cosine_warmup":
                tr_scheduler_params["warmup_steps"] = st.number_input("Warmup Steps", min_value=0, value=100, key="tr_sched_warmup")
                tr_scheduler_params["min_lr"] = st.number_input("Min LR", min_value=1e-8, value=1e-6, format="%.0e", key="tr_sched_minlr_cw")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
       
        col_add1, col_add2, col_add3, col_add4 = st.columns(4)
        
        with col_add1:
            tr_eval_metric = st.selectbox(
                "Evaluation Metric",
                options=["accuracy", "auc", "f1", "recall", "precision"],
                index=1,
                key="tr_eval_metric"
            )
        
        with col_add2:
            tr_early_stopping = st.checkbox("Early Stopping", value=True, key="tr_early_stopping")
            if tr_early_stopping:
                tr_patience = st.number_input(
                    "Patience",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="tr_patience"
                )
            else:
                tr_patience = 5
        
        with col_add3:
            tr_early_stopping_metric = st.selectbox(
                "Stop Metric",
                options=["val_loss", "eval_metric"],
                key="tr_es_metric"
            )
        
        with col_add4:
            tr_seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=9999,
                value=42,
                key="tr_seed"
            )
        
        # Cloud-specific settings
        col_cloud1, col_cloud2 = st.columns(2)
        
        with col_cloud1:
            particles_per_cloud = st.number_input(
                "Number of particles per cloud",
                min_value=1,
                max_value=100,
                value=4,
                key="particles_per_cloud",
                help="Number of particles in each cloud"
            )
        
        with col_cloud2:
            tr_validation_ratio = st.slider(
                "Validation Split Ratio",
                min_value=0.05,
                max_value=0.3,
                value=0.15,
                step=0.005,
                key="tr_val_ratio"
            )
        
        tr_output_dir = st.text_input(
            "Output Directory",
            value="output",
            key="tr_output_dir",
            help="Directory for saving model and results"
        )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================
        #                      Transformer training
        # ====================================
        col_train1, col_train2, col_train3, col_train4 = st.columns([1, 1, 1, 1])
        
        with col_train1:
            tr_show_config = st.button("Preview Config", use_container_width=True, key="tr_preview")
        
        with col_train2:
            tr_save_config = st.button("Save Config", use_container_width=True, key="tr_save")
        
        with col_train3:
            tr_start_training = st.button("Start Training", use_container_width=True, key="tr_train")
        
        # Generate the Transformer configuration
        tr_config = generate_transformer_config(
            seed=tr_seed,
            output_dir=tr_output_dir,
            signal_path=tr_sig_events or "/path/to/signal_clouds.csv",
            background_path=tr_bkg_events or "/path/to/background_clouds.csv",
            train_size=tr_train_size,
            test_size=tr_test_size,
            val_ratio=tr_validation_ratio,
            normalize=tr_normalize,
            particles_per_cloud=particles_per_cloud,
            embed_dim=tr_embed_dim,
            num_heads=tr_num_heads,
            num_layers=tr_num_layers,
            ffn_dim=tr_ffn_dim,
            dropout=tr_dropout,
            attention_dropout=tr_attention_dropout,
            pooling=tr_pooling,
            num_classes=tr_num_classes,
            pre_norm=tr_pre_norm,
            epochs=tr_epochs,
            batch_size=tr_batch_size,
            learning_rate=tr_lr,
            weight_decay=tr_weight_decay,
            optimizer=tr_optimizer,
            device=tr_device,
            early_stopping=tr_early_stopping,
            early_stopping_patience=tr_patience,
            early_stopping_metric=tr_early_stopping_metric,
            precision=tr_precision,
            scheduler_type=tr_scheduler,
            scheduler_params=tr_scheduler_params,
            eval_metric=tr_eval_metric,
            gradient_clip=tr_gradient_clip
        )
        
        tr_yaml_content = transformer_config_to_yaml(tr_config)
        
        if tr_show_config:
            st.markdown("### 📄 Transformer Configuration Preview")
            st.code(tr_yaml_content, language="yaml")
        
        if tr_save_config:
            config_filename = "input_config_transformer.yml"
            config_path = os.path.join(tr_output_dir, config_filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(tr_output_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(tr_yaml_content)
            
            st.success(f"Configuration saved to: `{config_path}`")
            
            # Provide download button
            st.download_button(
                label="Download Config",
                data=tr_yaml_content,
                file_name="input_config_transformer.yml",
                mime="text/yaml",
                key="tr_download"
            )
        
        if tr_start_training:
            config_filename = "input_config_transformer.yml"
            config_path = os.path.join(tr_output_dir, config_filename)
            os.makedirs(tr_output_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(tr_yaml_content)
            
            st.info(f"Configuration saved to: `{config_path}`")
            
            # Run training
            with st.spinner("Starting Transformer training..."):
                try:
                    
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'source.DL.Transformer.main_transformer', '--config', config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=_project_root,
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                    )
                    
                    # Terminal output
                    terminal_output = st.empty()
                    output_lines = []
                    
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_lines.append(line.rstrip())
                            terminal_output.code('\n'.join(output_lines[-30:]), language="bash")
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        st.success("Training completed successfully. Please click the Results section to see the results.")
                    else:
                        st.error(f"Training failed with exit code {process.returncode}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="bash")

# =====================================
#                          Results Tab
# =====================================
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class=""></div>
        <div>
            <h3 class="section-title">Analysis Results</h3>
            <p class="section-desc">View training metrics and model performance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Results file selection
    col_res1, col_res2 = st.columns([3, 1])
    
    with col_res1:
        results_dir = st.text_input(
            "📁 Results Directory",
            value="output",
            help="Directory containing results_mlp.json"
        )
    
    with col_res2:
        load_results = st.button("Load Results", use_container_width=True)
    results_file = None
    results_data = None
    
    if start_training:
        results_file = os.path.join(results_dir, "results_mlp.json")
        results_data = None
    elif gnn_start_training:
        results_file = os.path.join(results_dir, "results_gnn.json")
        results_data = None
    elif tr_start_training:
        results_file = os.path.join(results_dir, "results_transformer.json")
        results_data = None
    
    # Auto-load or load on button click
    if results_file is not None:
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
            
                if load_results:
                    st.success(f"Results loaded from: `{results_file}`")
            except Exception as e:
                st.error(f"Error loading results: {e}")
        elif load_results:
            st.warning(f"Results file not found: `{results_file}`")
    else:
        if load_results:
            st.warning("No training mode selected, so no results file to load.")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if results_data:
        # ====================================
        #                      Test results
        # ====================================
        test_metrics = results_data.get("test_metrics", {})
        history = results_data.get("history", {})
        
        # Metric cards row 1
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        accuracy = test_metrics.get("accuracy", 0) * 100
        auc_score = test_metrics.get("auc", 0) * 100
        precision_val = test_metrics.get("precision", 0) * 100
        recall_val = test_metrics.get("recall", 0) * 100
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{accuracy:.2f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{auc_score:.2f}%</div>
                <div class="metric-label">AUC Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{precision_val:.2f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{recall_val:.2f}%</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Metric cards row 2
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        
        f1_score = test_metrics.get("f1", 0) * 100
        test_loss = test_metrics.get("loss", 0)
        best_epoch = history.get("best_epoch", 0)
        best_val_loss = history.get("best_val_loss", 0)
        
        with col_m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{f1_score:.2f}%</div>
                <div class="metric-label">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{test_loss:.4f}</div>
                <div class="metric-label">Test Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_epoch}</div>
                <div class="metric-label">Best Epoch</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m8:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_val_loss:.4f}</div>
                <div class="metric-label">Best Val Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================
        #                      Result plots
        # ====================================
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📈</div>
            <div>
                <h3 class="section-title">Training History</h3>
                <p class="section-desc">Visualize training progress over epochs</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get history data
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("train_acc", [])
        val_acc = history.get("val_acc", [])
        val_auc = history.get("val_auc", [])
        val_f1 = history.get("val_f1", [])
        val_precision = history.get("val_precision", [])
        val_recall = history.get("val_recall", [])
        
        epochs_range = list(range(1, len(train_loss) + 1))
        

        plt.style.use('dark_background')
        legend_font = 13

        colors = {
            'train': '#6366f1',      # Primary accent (indigo)
            'val': '#a855f7',        # Secondary accent (purple)
            'auc': '#10b981',        # Success green
            'f1': '#f59e0b',         # Warning orange
            'precision': '#ec4899',  # Pink
            'recall': '#0ea5e9',     # Cyan
            'grid': '#454560',       # Grid color
            'bg': '#2a2a45'          # Card background
        }
        
        # Row 1: Loss and Accuracy plots
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>📉</span>
                    <h4 class="card-title">Loss Curves</h4>
                </div>
            """, unsafe_allow_html=True)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5), facecolor=colors['bg'])
            ax1.set_facecolor(colors['bg'])
            
            ax1.plot(epochs_range, train_loss, color=colors['train'], linewidth=2.5, 
                    label='Training Loss', marker='o', markersize=6)
            ax1.plot(epochs_range, val_loss, color=colors['val'], linewidth=2.5, 
                    label='Validation Loss', marker='s', markersize=6)
            
            # Get the  best epoch
            best_ep = history.get("best_epoch", 0)
            if best_ep > 0 and best_ep <= len(val_loss):
                ax1.axvline(x=best_ep, color='#10b981', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_ep})')
                ax1.scatter([best_ep], [val_loss[best_ep-1]], color='#10b981', s=150, zorder=5, marker='*')
            
            ax1.set_xlabel('Epoch', fontsize=12, color='#f1f5f9')
            ax1.set_ylabel('Loss', fontsize=12, color='#f1f5f9')
            ax1.set_title('Training & Validation Loss', fontsize=16, color='#f1f5f9', fontweight='bold')
            ax1.legend(loc='upper right', facecolor=colors['bg'], edgecolor=colors['grid'],fontsize=legend_font,frameon=False)
            ax1.grid(True, alpha=0.3, color=colors['grid'])
            ax1.tick_params(colors='#94a3b8')
            ax1.spines['bottom'].set_color(colors['grid'])
            ax1.spines['top'].set_color(colors['grid'])
            ax1.spines['left'].set_color(colors['grid'])
            ax1.spines['right'].set_color(colors['grid'])
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
       
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_plot2:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>📈</span>
                    <h4 class="card-title">Accuracy Curves</h4>
                </div>
            """, unsafe_allow_html=True)
            
            fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor=colors['bg'])
            ax2.set_facecolor(colors['bg'])
            
            ax2.plot(epochs_range, [a * 100 for a in train_acc], color=colors['train'], 
                    linewidth=2.5, label='Training Accuracy', marker='o', markersize=6)
            ax2.plot(epochs_range, [a * 100 for a in val_acc], color=colors['val'], 
                    linewidth=2.5, label='Validation Accuracy', marker='s', markersize=6)
            
            if best_ep > 0 and best_ep <= len(val_acc):
                ax2.axvline(x=best_ep, color='#10b981', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_ep})')
            
            ax2.set_xlabel('Epoch', fontsize=12, color='#f1f5f9')
            ax2.set_ylabel('Accuracy (%)', fontsize=12, color='#f1f5f9')
            ax2.set_title('Training & Validation Accuracy', fontsize=16, color='#f1f5f9', fontweight='bold')
            ax2.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'],fontsize=legend_font,frameon=False)
            ax2.grid(True, alpha=0.3, color=colors['grid'])
            ax2.tick_params(colors='#94a3b8')
            ax2.spines['bottom'].set_color(colors['grid'])
            ax2.spines['top'].set_color(colors['grid'])
            ax2.spines['left'].set_color(colors['grid'])
            ax2.spines['right'].set_color(colors['grid'])
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Row 2: AUC and F1 plots
        col_plot3, col_plot4 = st.columns(2)
        
        with col_plot3:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>🎯</span>
                    <h4 class="card-title">Validation AUC</h4>
                </div>
            """, unsafe_allow_html=True)
            
            fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor=colors['bg'])
            ax3.set_facecolor(colors['bg'])
            
            ax3.plot(epochs_range, [a * 100 for a in val_auc], color=colors['auc'], 
                    linewidth=2.5, label='Validation AUC', marker='D', markersize=6)
            ax3.fill_between(epochs_range, [a * 100 for a in val_auc], alpha=0.2, color=colors['auc'])
            
            if best_ep > 0 and best_ep <= len(val_auc):
                ax3.axvline(x=best_ep, color='#f59e0b', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_ep})')
                ax3.scatter([best_ep], [val_auc[best_ep-1] * 100], color='#f59e0b', s=150, zorder=5, marker='*')
            
            ax3.set_xlabel('Epoch', fontsize=12, color='#f1f5f9')
            ax3.set_ylabel('AUC (%)', fontsize=12, color='#f1f5f9')
            ax3.set_title('Validation AUC Score', fontsize=16, color='#f1f5f9', fontweight='bold')
            ax3.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'],fontsize=legend_font,frameon=False)
            ax3.grid(True, alpha=0.3, color=colors['grid'])
            ax3.tick_params(colors='#94a3b8')
            ax3.spines['bottom'].set_color(colors['grid'])
            ax3.spines['top'].set_color(colors['grid'])
            ax3.spines['left'].set_color(colors['grid'])
            ax3.spines['right'].set_color(colors['grid'])
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_plot4:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>⚖️</span>
                    <h4 class="card-title">Validation F1 Score</h4>
                </div>
            """, unsafe_allow_html=True)
            
            fig4, ax4 = plt.subplots(figsize=(8, 5), facecolor=colors['bg'])
            ax4.set_facecolor(colors['bg'])
            
            ax4.plot(epochs_range, [f * 100 for f in val_f1], color=colors['f1'], 
                    linewidth=2.5, label='Validation F1', marker='^', markersize=6)
            ax4.fill_between(epochs_range, [f * 100 for f in val_f1], alpha=0.2, color=colors['f1'])
            
            if best_ep > 0 and best_ep <= len(val_f1):
                ax4.axvline(x=best_ep, color='#6366f1', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_ep})')
            
            ax4.set_xlabel('Epoch', fontsize=12, color='#f1f5f9')
            ax4.set_ylabel('F1 Score (%)', fontsize=12, color='#f1f5f9')
            ax4.set_title('Validation F1 Score', fontsize=16, color='#f1f5f9', fontweight='bold')
            ax4.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'],fontsize=legend_font,frameon=False)
            ax4.grid(True, alpha=0.3, color=colors['grid'])
            ax4.tick_params(colors='#94a3b8')
            ax4.spines['bottom'].set_color(colors['grid'])
            ax4.spines['top'].set_color(colors['grid'])
            ax4.spines['left'].set_color(colors['grid'])
            ax4.spines['right'].set_color(colors['grid'])
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Row 3: Precision and Recall plots
        col_plot5, col_plot6 = st.columns(2)
        
        with col_plot5:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>🎯</span>
                    <h4 class="card-title">Validation Precision</h4>
                </div>
            """, unsafe_allow_html=True)
            
            fig5, ax5 = plt.subplots(figsize=(8, 5), facecolor=colors['bg'])
            ax5.set_facecolor(colors['bg'])
            
            ax5.plot(epochs_range, [p * 100 for p in val_precision], color=colors['precision'], 
                    linewidth=2.5, label='Validation Precision', marker='p', markersize=6)
            ax5.fill_between(epochs_range, [p * 100 for p in val_precision], alpha=0.2, color=colors['precision'])
            
            ax5.set_xlabel('Epoch', fontsize=12, color='#f1f5f9')
            ax5.set_ylabel('Precision (%)', fontsize=12, color='#f1f5f9')
            ax5.set_title('Validation Precision', fontsize=16, color='#f1f5f9', fontweight='bold')
            ax5.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'],fontsize=legend_font,frameon=False)
            ax5.grid(True, alpha=0.3, color=colors['grid'])
            ax5.tick_params(colors='#94a3b8')
            ax5.spines['bottom'].set_color(colors['grid'])
            ax5.spines['top'].set_color(colors['grid'])
            ax5.spines['left'].set_color(colors['grid'])
            ax5.spines['right'].set_color(colors['grid'])
            
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_plot6:
            st.markdown("""
            <div class="custom-card">
                <div class="card-header">
                    <span>🔍</span>
                    <h4 class="card-title">Validation Recall</h4>
                </div>
            """, unsafe_allow_html=True)
            
            fig6, ax6 = plt.subplots(figsize=(8, 5), facecolor=colors['bg'])
            ax6.set_facecolor(colors['bg'])
            
            ax6.plot(epochs_range, [r * 100 for r in val_recall], color=colors['recall'], 
                    linewidth=2.5, label='Validation Recall', marker='h', markersize=6)
            ax6.fill_between(epochs_range, [r * 100 for r in val_recall], alpha=0.2, color=colors['recall'])
            
            ax6.set_xlabel('Epoch', fontsize=12, color='#f1f5f9')
            ax6.set_ylabel('Recall (%)', fontsize=12, color='#f1f5f9')
            ax6.set_title('Validation Recall', fontsize=16, color='#f1f5f9', fontweight='bold')
            ax6.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'],fontsize=legend_font,frameon=False)
            ax6.grid(True, alpha=0.3, color=colors['grid'])
            ax6.tick_params(colors='#94a3b8')
            ax6.spines['bottom'].set_color(colors['grid'])
            ax6.spines['top'].set_color(colors['grid'])
            ax6.spines['left'].set_color(colors['grid'])
            ax6.spines['right'].set_color(colors['grid'])
            
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
       
        #COMBINED METRICS PLOT
        
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div>
                <h3 class="section-title">All Validation Metrics</h3>
                <p class="section-desc">Compare all metrics in a single view</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
        """, unsafe_allow_html=True)
        
        fig7, ax7 = plt.subplots(figsize=(14, 6), facecolor=colors['bg'])
        ax7.set_facecolor(colors['bg'])
        
        ax7.plot(epochs_range, [a * 100 for a in val_acc], color=colors['val'], 
                linewidth=2.5, label='Accuracy', marker='o', markersize=5)
        ax7.plot(epochs_range, [a * 100 for a in val_auc], color=colors['auc'], 
                linewidth=2.5, label='AUC', marker='s', markersize=5)
        ax7.plot(epochs_range, [f * 100 for f in val_f1], color=colors['f1'], 
                linewidth=2.5, label='F1 Score', marker='^', markersize=5)
        ax7.plot(epochs_range, [p * 100 for p in val_precision], color=colors['precision'], 
                linewidth=2.5, label='Precision', marker='D', markersize=5)
        ax7.plot(epochs_range, [r * 100 for r in val_recall], color=colors['recall'], 
                linewidth=2.5, label='Recall', marker='p', markersize=5)
        
        if best_ep > 0:
            ax7.axvline(x=best_ep, color='#ef4444', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_ep})')
        
        ax7.set_xlabel('Epoch', fontsize=14, color='#f1f5f9')
        ax7.set_ylabel('Score (%)', fontsize=14, color='#f1f5f9')
        ax7.set_title('All Validation Metrics Over Training', fontsize=16, color='#f1f5f9', fontweight='bold')
        ax7.legend(loc='lower right', facecolor=colors['bg'], edgecolor=colors['grid'], ncol=3,fontsize=legend_font,frameon=False)
        ax7.grid(True, alpha=0.3, color=colors['grid'])
        ax7.tick_params(colors='#94a3b8')
        ax7.spines['bottom'].set_color(colors['grid'])
        ax7.spines['top'].set_color(colors['grid'])
        ax7.spines['left'].set_color(colors['grid'])
        ax7.spines['right'].set_color(colors['grid'])
        ax7.set_ylim([min(min([a * 100 for a in val_acc]), min([p * 100 for p in val_precision])) - 5, 100])
        
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
       
        with st.expander("📋 View Raw Results Data"):
            st.json(results_data)
        
        # Config info
        config_file = results_data.get("config", "")
        if config_file:
            st.info(f"📄 Configuration file used: `{config_file}`")
    
    else:
        # No results placeholder
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
            border: 1px dashed #454560;
            border-radius: 16px;
            padding: 4rem;
            text-align: center;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #f1f5f9;
                margin-bottom: 0.5rem;
            ">No Results Yet</h3>
            <p style="color: #64748b; max-width: 400px; margin: 0 auto;">
                Run a training job to see performance metrics, learning curves, and model evaluation results here.
                <br><br>
                Results will be loaded from: <code style="color: #a855f7;">results_mlp.json</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
