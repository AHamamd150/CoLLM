from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from source.utils.read_configs import DataConfig


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute mean and std from training data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


def stratified_split(X: np.ndarray, y: np.ndarray, test_ratio: float):
    """
    Stratified train/test split maintaining class balance.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Get indices for each class
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]
    
    # Shuffle each class
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    
    # Calculate test size per class
    n_test_0 = int(len(class_0_idx) * test_ratio)
    n_test_1 = int(len(class_1_idx) * test_ratio)
    
    # Split indices
    test_idx = np.concatenate([class_0_idx[:n_test_0], class_1_idx[:n_test_1]])
    train_idx = np.concatenate([class_0_idx[n_test_0:], class_1_idx[n_test_1:]])
    
    # Shuffle
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_csv_old(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load CSV file without pandas dependency.
    
    Returns:
        columns: List of column names
        data: numpy array of data
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    columns = [col.strip() for col in lines[0].strip().split(',')]
    
    # Parse data
    data = []
    for line in lines[1:]:
        if line.strip():
            row = [float(x.strip()) for x in line.strip().split(',')]
            data.append(row)
    
    return columns, np.array(data, dtype=np.float32)

def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load a CSV file WITHOUT headers.

    Returns:
        columns : auto-generated column names (f0, f1, ...)
        data    : numpy array of shape (N, D)
    """
    data = np.loadtxt(path, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_features = data.shape[1]
    columns = [f"f{i}" for i in range(n_features)]

    return columns, np.array(data, dtype=np.float32)



def load_data(cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, int, Optional[StandardScaler]]:
    """
    Load signal and background CSV files and split into train/val/test.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, input_dim, scaler
    """
    # Load CSVs
    sig_cols, sig_data = load_csv(cfg.signal_path)
    bkg_cols, bkg_data = load_csv(cfg.background_path)
    
    # Determine feature columns
    if cfg.feature_columns:
        features = cfg.feature_columns
        sig_idx = [sig_cols.index(f) for f in features]
        bkg_idx = [bkg_cols.index(f) for f in features]
    else:
        # Find common columns
        common = set(sig_cols) & set(bkg_cols)
        features = sorted(list(common))
        sig_idx = [sig_cols.index(f) for f in features]
        bkg_idx = [bkg_cols.index(f) for f in features]
    
    #print(f"Features ({len(features)}): {features}")
    
    # Extract features
    sig_x = sig_data[:, sig_idx]
    bkg_x = bkg_data[:, bkg_idx]
    
    # Shuffle
    np.random.shuffle(sig_x)
    np.random.shuffle(bkg_x)
    
    # Split into train/test per class
    sig_train = sig_x[:cfg.train_size]
    sig_test = sig_x[cfg.train_size:cfg.train_size + cfg.test_size]
    bkg_train = bkg_x[:cfg.train_size]
    bkg_test = bkg_x[cfg.train_size:cfg.train_size + cfg.test_size]
    
    # Combine
    X_train_full = np.vstack([sig_train, bkg_train])
    y_train_full = np.array([1]*len(sig_train) + [0]*len(bkg_train), dtype=np.int64)
    
    X_test = np.vstack([sig_test, bkg_test])
    y_test = np.array([1]*len(sig_test) + [0]*len(bkg_test), dtype=np.int64)
    
    # Split train into train/val (stratified)
    X_train, X_val, y_train, y_val = stratified_split(
        X_train_full, y_train_full, test_ratio=cfg.val_ratio
    )
    
    # Normalize
    scaler = None
    if cfg.normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[1], scaler


def create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int,
    use_cuda: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from numpy arrays."""
    
    kwargs = {"batch_size": batch_size, "pin_memory": use_cuda}
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader
