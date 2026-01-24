"""
Main Particle Cloud Transformer Training Script

Train Transformer model for particle cloud classification.

Usage:
    python -m transformer.main_transformer --config config.yaml
"""

import argparse
import os
import json
import numpy as np
import torch

from .config_transformer import load_transformer_config
from .data_transformer import load_cloud_data, create_cloud_dataloaders
from .model_transformer import build_transformer_model
from .train_transformer import train, evaluate, get_device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Particle Cloud Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    print("=" * 70)
    print("PARTICLE CLOUD TRANSFORMER")
    print("=" * 70)
    print(f"Config: {args.config}")
    cfg = load_transformer_config(args.config)
    
    # Set seed
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")
    
    # Device
    device = get_device(cfg.train.device)
    
    # Load data
    print("\n" + "=" * 70)
    print("DATA")
    print("=" * 70)
    train_clouds, val_clouds, test_clouds, num_features, scaler = load_cloud_data(
        cfg.data, cfg.data.particles_per_cloud
    )
    
    train_loader, val_loader, test_loader = create_cloud_dataloaders(
        train_clouds, val_clouds, test_clouds,
        batch_size=cfg.train.batch_size,
        use_cuda=(device.type == "cuda"),
    )
    
    # Build model
    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)
    
    model = build_transformer_model(
        num_features=num_features,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ffn_dim=cfg.model.ffn_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        pooling=cfg.model.pooling,
        use_pos_encoding=cfg.model.use_pos_encoding,
        pre_norm=cfg.model.pre_norm,
    ).to(device)
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    history = train(model, train_loader, val_loader, cfg.train, device)
    
    # Test
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    test_metrics = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    
    # Save results
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(cfg.output_dir, "model_transformer.pt")
    torch.save({
        "model": model.state_dict(),
        "scaler": scaler,
        "num_features": num_features,
        "particles_per_cloud": cfg.data.particles_per_cloud,
        "config": {
            "embed_dim": cfg.model.embed_dim,
            "num_heads": cfg.model.num_heads,
            "num_layers": cfg.model.num_layers,
            "ffn_dim": cfg.model.ffn_dim,
            "dropout": cfg.model.dropout,
            "pooling": cfg.model.pooling,
            "num_classes": cfg.model.num_classes,
        }
    }, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metrics and history
    results = {
        "test_metrics": test_metrics,
        "history": history,
        "config": args.config,
    }
    results_path = os.path.join(cfg.output_dir, "results_transformer.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
