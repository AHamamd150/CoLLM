"""
GNN Training Module

Training loop and evaluation functions for graph neural networks.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from .data_gnn import Batch


# ============================================================================
# Metrics
# ============================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    prev_prob = None
    
    for label, prob in zip(y_true_sorted, y_prob_sorted):
        if prev_prob is not None and prob != prev_prob:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_prob = prob
    
    tpr_list.append(tp / n_pos)
    fpr_list.append(fp / n_neg)
    
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
    }


def get_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    metric_name = metric_name.lower()
    if metric_name in metrics:
        return metrics[metric_name]
    raise ValueError(f"Unknown metric: {metric_name}")


# ============================================================================
# Optimizer and Scheduler
# ============================================================================

def get_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    lr = float(lr)
    weight_decay = float(weight_decay)
    
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(cfg, optimizer, train_loader_len: int, epochs: int):
    """Create learning rate scheduler from config."""
    sched_type = cfg.type.lower()
    
    if sched_type == "none":
        return None
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.step_size), gamma=float(cfg.gamma))
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(cfg.gamma), 
                                                          patience=int(cfg.patience), min_lr=float(cfg.min_lr))
    elif sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=float(cfg.min_lr))
    elif sched_type == "onecycle":
        max_lr = float(cfg.max_lr) if cfg.max_lr else optimizer.param_groups[0]['lr'] * 10
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=int(epochs), 
                                                    steps_per_epoch=int(train_loader_len))
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


# ============================================================================
# Device Selection
# ============================================================================

def get_device(device_config: str) -> torch.device:
    device_config = device_config.lower()
    
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    elif device_config == "cpu":
        device = torch.device("cpu")
        print("Using CPU")
    elif device_config.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device_config)
            gpu_id = 0 if device_config == "cuda" else int(device_config.split(":")[1])
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"Warning: CUDA not available. Using CPU.")
            device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_config}")
    
    return device


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                logits = model(batch.x, batch.edge_index, batch.edge_weight, 
                              batch.batch, batch.num_graphs)
                loss = loss_fn(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch.x, batch.edge_index, batch.edge_weight,
                          batch.batch, batch.num_graphs)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        total_loss += loss.item() * batch.num_graphs
        correct += (logits.argmax(1) == batch.y).sum().item()
        total += batch.num_graphs
    
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate model and compute all metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        if use_amp:
            with autocast():
                logits = model(batch.x, batch.edge_index, batch.edge_weight,
                              batch.batch, batch.num_graphs)
                loss = loss_fn(logits, batch.y)
        else:
            logits = model(batch.x, batch.edge_index, batch.edge_weight,
                          batch.batch, batch.num_graphs)
            loss = loss_fn(logits, batch.y)
        
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        
        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / total
    
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg,
    device: torch.device,
) -> Dict:
    """Full training loop."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(cfg.optimizer, model.parameters(), cfg.learning_rate, cfg.weight_decay)
    scheduler = get_scheduler(cfg.scheduler, optimizer, len(train_loader), cfg.epochs)
    
    use_amp = cfg.precision in ["float16", "mixed"] and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    
    if cfg.precision == "float16" and device.type == "cuda":
        model = model.half()
    
    print(f"Precision: {cfg.precision}" + (f", AMP: {use_amp}" if device.type == "cuda" else ""))
    print(f"Optimizer: {cfg.optimizer}, LR: {cfg.learning_rate}")
    print(f"Scheduler: {cfg.scheduler.type}")
    print(f"Eval metric: {cfg.eval_metric}")
    if cfg.early_stopping:
        print(f"Early stopping: patience={cfg.early_stopping_patience}, metric={cfg.early_stopping_metric}")
    print("-" * 70)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_auc": [], "val_f1": [], "val_precision": [], "val_recall": []
    }
    
    best_val_loss = float("inf")
    best_eval_metric = 0.0 if cfg.eval_metric != "loss" else float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, cfg.epochs + 1):
        train_m = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, scheduler, use_amp)
        val_m = evaluate(model, val_loader, loss_fn, device, use_amp)
        
        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["acc"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_auc"].append(val_m["auc"])
        history["val_f1"].append(val_m["f1"])
        history["val_precision"].append(val_m["precision"])
        history["val_recall"].append(val_m["recall"])
        
        current_eval_metric = get_metric_value(val_m, cfg.eval_metric)
        higher_is_better = cfg.eval_metric in ["accuracy", "auc", "f1", "precision", "recall"]
        
        if cfg.early_stopping_metric == "val_loss":
            improved = val_m["loss"] < best_val_loss
        else:
            if higher_is_better:
                improved = current_eval_metric > best_eval_metric
            else:
                improved = current_eval_metric < best_eval_metric
        
        if improved:
            best_val_loss = val_m["loss"]
            best_eval_metric = current_eval_metric
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch:03d}/{cfg.epochs} | "
              f"loss: {train_m['loss']:.4f}/{val_m['loss']:.4f} | "
              f"acc: {train_m['acc']:.4f}/{val_m['accuracy']:.4f} | "
              f"{cfg.eval_metric}: {current_eval_metric:.4f}")
        
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_m["loss"])
            else:
                scheduler.step()
        
        if cfg.early_stopping and patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")
    
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history[f"best_{cfg.eval_metric}"] = best_eval_metric
    
    return history
