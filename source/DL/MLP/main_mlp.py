
import argparse
import os
import json
import numpy as np
import torch

from source.utils.read_configs import load_config,TrainConfig, SchedulerConfig
from .data import load_data, create_dataloaders
from .model import build_model
from .train import train, evaluate, get_device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    print("=" * 40)
    print("MLP Classifier")
    print("=" * 40)
    print(f"Config: {args.config}")
    cfg = load_config(args.config)
    
    # Set seed
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")
    
    # Device
    device = get_device(cfg.train.device)
    
    # Load data
    print("\n" + "=" * 40)
    print("Data")
    print("=" * 40)
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim, scaler = load_data(cfg.data)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=cfg.train.batch_size,
        use_cuda=(device.type == "cuda"),
    )
    
    # Build model
    print("\n" + "=" * 40)
    print("Model")
    print("=" * 40)
    model = build_model(input_dim, cfg.model).to(device)
    
    # Train
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    history = train(model, train_loader, val_loader, cfg.train, device)
    
    # Test
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    test_metrics = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    

    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(cfg.output_dir, "model_mlp.pt")
    torch.save({
        "model": model.state_dict(),
        "scaler": scaler,
        "input_dim": input_dim,
        "config": {
            "model": {
                "layers": [(l.type, l.units, l.activation, l.rate) for l in cfg.model.layers],
                "output_units": cfg.model.output_units,
            }
        }
    }, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metrics and history
    results = {
        "test_metrics": test_metrics,
        "history": history,
        "config": args.config,
    }
    results_path = os.path.join(cfg.output_dir, "results_mlp.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
 
if __name__ == "__main__":
    main()
