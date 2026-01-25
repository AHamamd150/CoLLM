from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class ParticleCloud:
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x    # [num_particles, num_features]
        self.y = y    #  labels
        self.num_particles = x.size(0)
    
    def to(self, device):
        return ParticleCloud(
            x=self.x.to(device),
            y=self.y.to(device)
        )


class ParticleCloudDataset(Dataset):    
    def __init__(self, clouds: List[ParticleCloud]):
        self.clouds = clouds
    
    def __len__(self):
        return len(self.clouds)
    
    def __getitem__(self, idx):
        cloud = self.clouds[idx]
        return cloud.x, cloud.y


def collate_clouds(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DataLoader.
    
    Returns:
        x: [batch_size, num_particles, num_features]
        y: [batch_size]
    """
    x_list, y_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y


def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    
    data = np.loadtxt(path, delimiter=",")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_features = data.shape[1]
    columns = [f"f{i}" for i in range(n_features)]
    
    return columns, np.array(data, dtype=np.float32)


def parse_clouds_from_csv(data: np.ndarray, particles_per_cloud: int) -> Tuple[List[np.ndarray], List[int]]:

    if particles_per_cloud <= 0:
        raise ValueError(f"Invalid number for particles per cloud: {particles_per_cloud}")
    
    total_rows = data.shape[0]
    num_clouds = total_rows // particles_per_cloud
    
    if total_rows % particles_per_cloud != 0:
        print(f"Warning: Total rows ({total_rows}) not divisible by particles_per_cloud ({particles_per_cloud}). "
              f"Ignoring last {total_rows % particles_per_cloud} rows.")
    
    features_list = []
    id_list = []
    
    for cid in range(num_clouds):
        start_idx = cid * particles_per_cloud
        end_idx = start_idx + particles_per_cloud
        cloud_features = data[start_idx:end_idx]
        features_list.append(cloud_features)
        id_list.append(cid)
    
    return features_list, id_list


def create_cloud_objects(features_list: List[np.ndarray], labels: np.ndarray) -> List[ParticleCloud]:
    """
    Create ParticleCloud objects from features and labels.
    
    Args:
        features_list: list of [num_particles, num_features] arrays
        labels: [num_clouds] array of labels
        
    Returns:
        List of ParticleCloud objects
    """
    clouds = []
    for feat, label in zip(features_list, labels):
        x = torch.from_numpy(feat.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)
        clouds.append(ParticleCloud(x, y))
    
    return clouds


def stratified_split_clouds(clouds: List[ParticleCloud], test_ratio: float):
    
    labels = np.array([c.y.item() for c in clouds])
    
    class_0_idx = np.where(labels == 0)[0]
    class_1_idx = np.where(labels == 1)[0]
    
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    
    n_test_0 = int(len(class_0_idx) * test_ratio)
    n_test_1 = int(len(class_1_idx) * test_ratio)
    
    test_idx = np.concatenate([class_0_idx[:n_test_0], class_1_idx[:n_test_1]])
    train_idx = np.concatenate([class_0_idx[n_test_0:], class_1_idx[n_test_1:]])
    
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)
    
    train_clouds = [clouds[i] for i in train_idx]
    test_clouds = [clouds[i] for i in test_idx]
    
    return train_clouds, test_clouds


def load_cloud_data(cfg, particles_per_cloud: int) -> Tuple[
    List[ParticleCloud], List[ParticleCloud], List[ParticleCloud], int, int, Optional[StandardScaler]]:
  
   
    _, sig_data = load_csv(cfg.signal_path)
    _, bkg_data = load_csv(cfg.background_path)
    
    sig_features, sig_ids = parse_clouds_from_csv(sig_data, particles_per_cloud)
    bkg_features, bkg_ids = parse_clouds_from_csv(bkg_data, particles_per_cloud)
    
    print(f"Loaded {len(sig_features)} signal clouds, {len(bkg_features)} background clouds")
    
    
    sig_perm = np.random.permutation(len(sig_features))
    bkg_perm = np.random.permutation(len(bkg_features))
    sig_features = [sig_features[i] for i in sig_perm]
    bkg_features = [bkg_features[i] for i in bkg_perm]
    
    # Split into train/test per class
    sig_train = sig_features[:cfg.train_size]
    sig_test = sig_features[cfg.train_size:cfg.train_size + cfg.test_size]
    bkg_train = bkg_features[:cfg.train_size]
    bkg_test = bkg_features[cfg.train_size:cfg.train_size + cfg.test_size]
    
    all_train_features = np.vstack([np.vstack(sig_train), np.vstack(bkg_train)])
    
    
    scaler = None
    if cfg.normalize:
        scaler = StandardScaler()
        scaler.fit(all_train_features)
        
        # Apply to all clouds
        sig_train = [scaler.transform(f).astype(np.float32) for f in sig_train]
        sig_test = [scaler.transform(f).astype(np.float32) for f in sig_test]
        bkg_train = [scaler.transform(f).astype(np.float32) for f in bkg_train]
        bkg_test = [scaler.transform(f).astype(np.float32) for f in bkg_test]
    
  
    train_features = sig_train + bkg_train
    train_labels = np.array([1]*len(sig_train) + [0]*len(bkg_train), dtype=np.int64)
    
    test_features = sig_test + bkg_test
    test_labels = np.array([1]*len(sig_test) + [0]*len(bkg_test), dtype=np.int64)
    
    all_train_clouds = create_cloud_objects(train_features, train_labels)
    test_clouds = create_cloud_objects(test_features, test_labels)

    train_clouds, val_clouds = stratified_split_clouds(all_train_clouds, cfg.val_ratio)
    
    num_features = train_features[0].shape[1]
    
    print(f"Train: {len(train_clouds)} | Val: {len(val_clouds)} | Test: {len(test_clouds)}")
    print(f"Particles per cloud: {particles_per_cloud} | Features per particle: {num_features}")
    
    return train_clouds, val_clouds, test_clouds, num_features, particles_per_cloud, scaler


def create_cloud_dataloaders(
    train_clouds: List[ParticleCloud],
    val_clouds: List[ParticleCloud],
    test_clouds: List[ParticleCloud],
    batch_size: int,
    use_cuda: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for particle cloud datasets."""
    
    kwargs = {"batch_size": batch_size, "collate_fn": collate_clouds}
    if use_cuda:
        kwargs["pin_memory"] = True
    
    train_ds = ParticleCloudDataset(train_clouds)
    val_ds = ParticleCloudDataset(val_clouds)
    test_ds = ParticleCloudDataset(test_clouds)
    
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader
