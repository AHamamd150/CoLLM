"""
Main GNN Training Script

Train graph neural networks for graph classification.
Supports: GCN, GAT, EdgeConv

Usage:
    python -m source.DL.GNN.main_gnn --config config.yaml
"""

import argparse
import os
import json
import numpy as np
import torch

from .config_gnn import load_gnn_config
from .data_gnn import load_graph_data, create_graph_dataloaders
from .model_gnn import build_gnn_model
from .train_gnn import train, evaluate, get_device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train GNN classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    print("=" * 40)
    print("GNN graph classifier")
    print("=" * 40)
    print(f"Config: {args.config}")
    cfg = load_gnn_config(args.config)
    
    # Set seed
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")
    
    # Device
    device = get_device(cfg.train.device)
    
    # Load data
    print("\n" + "=" * 40)
    print("Data")
    print("=" * 40)
    train_graphs, val_graphs, test_graphs, num_features, nodes_per_graph, scaler = load_graph_data(cfg.data)
    
    train_loader, val_loader, test_loader = create_graph_dataloaders(
        train_graphs, val_graphs, test_graphs,
        batch_size=cfg.train.batch_size,
        use_cuda=(device.type == "cuda"),
    )
    
    # Build model
    print("\n" + "=" * 40)
    print("Model")
    print("=" * 40)
    
    model = build_gnn_model(num_features, cfg.model).to(device)
    
    # Train
    print("\n" + "=" * 40)
    print("Training")
    print("=" * 40)
    history = train(model, train_loader, val_loader, cfg.train, device)
    
    # Test
    print("\n" + "=" * 40)
    print("Test result")
    print("=" * 40)
    test_metrics = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(cfg.output_dir, "model_gnn.pt")
    torch.save({
        "model": model.state_dict(),
        "scaler": scaler,
        "num_features": num_features,
        "nodes_per_graph": nodes_per_graph,
        "config": {
            "type": cfg.model.type,
            "layers": [(l.out_channels, l.activation) for l in cfg.model.layers],
            "pooling": cfg.model.pooling,
            "output_units": cfg.model.output_units,
        }
    }, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metrics and history
    results = {
        "test_metrics": test_metrics,
        "history": history,
        "config": args.config,
    }
    results_path = os.path.join(cfg.output_dir, "results_gnn.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
