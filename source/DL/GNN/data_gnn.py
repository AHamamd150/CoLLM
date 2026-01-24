from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys

class StandardScaler:
    """Standard scaler for node features."""
    
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


def create_fully_connected_edges(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create edge index for a fully connected graph (excluding self-loops).
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        edge_index: [2, num_edges] array of edge connections
        edge_weight: [num_edges] array of edge weights (all ones)
    """
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  
                src.append(i)
                dst.append(j)
    
    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    
    return edge_index, edge_weight
class GraphData:
    
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, 
                 edge_weight: torch.Tensor, y: torch.Tensor):
        self.x = x                    
        self.edge_index = edge_index  
        self.edge_weight = edge_weight  
        self.y = y                    
        self.num_nodes = x.size(0)
    
    def to(self, device):

        return GraphData(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            y=self.y.to(device)
        )


class Batch:
    
    def __init__(self, graphs: List[GraphData]):
        """Create a batch from list of graphs."""
        x_list = []
        edge_index_list = []
        edge_weight_list = []
        y_list = []
        batch_list = []
        
        node_offset = 0
        for i, g in enumerate(graphs):
            x_list.append(g.x)
            edge_index_list.append(g.edge_index + node_offset)
            edge_weight_list.append(g.edge_weight)
            y_list.append(g.y)
            batch_list.append(torch.full((g.num_nodes,), i, dtype=torch.long))
            node_offset += g.num_nodes
        
        self.x = torch.cat(x_list, dim=0)
        self.edge_index = torch.cat(edge_index_list, dim=1)
        self.edge_weight = torch.cat(edge_weight_list, dim=0)
        self.y = torch.stack(y_list)
        self.batch = torch.cat(batch_list)
        self.num_graphs = len(graphs)
    
    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.y = self.y.to(device)
        self.batch = self.batch.to(device)
        return self


class GraphDataset(Dataset):
    """Dataset of graphs."""
    
    def __init__(self, graphs: List[GraphData]):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(graphs: List[GraphData]) -> Batch:

    return Batch(graphs)


def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load CSV file without headers.
    
    Returns:
        columns: auto-generated column names
        data: numpy array of shape (N, D)
    """
    data = np.loadtxt(path, delimiter=",")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_features = data.shape[1]
    columns = [f"f{i}" for i in range(n_features)]
    
    return columns, np.array(data, dtype=np.float32)


def parse_graphs_from_csv(data: np.ndarray, nodes_per_graph: int = 0) -> Tuple[List[np.ndarray], List[int], int]:
    """
    Parse CSV data into individual graphs by grouping consecutive rows.
    
    Args:
        data: [N, D] array where all columns are node features
        nodes_per_graph: number of nodes per graph (required)
        
    Returns:
        features_list: list of [nodes_per_graph, D] arrays
        graph_ids: list of graph identifiers
        detected_nodes_per_graph: number of nodes per graph
    """
    if nodes_per_graph <= 0:
        print(f"Invalid number for graph nodes: {nodes_per_graph}")
        sys.exit()
    
    total_rows = data.shape[0]
    num_graphs = total_rows // nodes_per_graph
    
    if total_rows % nodes_per_graph != 0:
        print(f"Warning: Total rows ({total_rows}) not divisible by nodes_per_graph ({nodes_per_graph}). "
              f"Ignoring last {total_rows % nodes_per_graph} rows.")
    
    features_list = []
    id_list = []
    
    for gid in range(num_graphs):
        start_idx = gid * nodes_per_graph
        end_idx = start_idx + nodes_per_graph
        graph_features = data[start_idx:end_idx]
        features_list.append(graph_features)
        id_list.append(gid)
    
    return features_list, id_list, nodes_per_graph


def create_graph_objects(features_list: List[np.ndarray], labels: np.ndarray, 
                         nodes_per_graph: int) -> List[GraphData]:
    """
    Create GraphData objects from features and labels.
    """
    edge_index, edge_weight = create_fully_connected_edges(nodes_per_graph)
    edge_index_t = torch.from_numpy(edge_index)
    edge_weight_t = torch.from_numpy(edge_weight)
    
    graphs = []
    for feat, label in zip(features_list, labels):
        x = torch.from_numpy(feat.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)
        graphs.append(GraphData(x, edge_index_t.clone(), edge_weight_t.clone(), y))
    
    return graphs


def stratified_split_graphs(graphs: List[GraphData], test_ratio: float):
    """
    Stratified split of graphs maintaining class balance.
    """
    labels = np.array([g.y.item() for g in graphs])
    
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
    
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    return train_graphs, test_graphs


def load_graph_data(cfg) -> Tuple[
    List[GraphData], List[GraphData], List[GraphData], int, int, Optional[StandardScaler]]:
    """
    Load signal and background CSV files and create graph datasets.
    
    Args:
        cfg: DataConfig object with signal_path, background_path, train_size, test_size,
             val_ratio, normalize, and nodes_per_graph attributes
        
    Returns:
        train_graphs, val_graphs, test_graphs, num_features, nodes_per_graph, scaler
    """
    # Load CSVs
    _, sig_data = load_csv(cfg.signal_path)
    _, bkg_data = load_csv(cfg.background_path)
    
    # Get nodes_per_graph (from config or auto-detect)
    nodes_per_graph = getattr(cfg, 'nodes_per_graph', 0)
    
    # Parse into graphs
    sig_features, sig_ids, nodes_per_graph = parse_graphs_from_csv(sig_data, nodes_per_graph)
    bkg_features, bkg_ids, _ = parse_graphs_from_csv(bkg_data, nodes_per_graph)
    
    print(f"Loaded {len(sig_features)} signal graphs, {len(bkg_features)} background graphs")
    
    # Shuffle
    sig_perm = np.random.permutation(len(sig_features))
    bkg_perm = np.random.permutation(len(bkg_features))
    sig_features = [sig_features[i] for i in sig_perm]
    bkg_features = [bkg_features[i] for i in bkg_perm]
    
    # Split into train/test per class
    sig_train = sig_features[:cfg.train_size]
    sig_test = sig_features[cfg.train_size:cfg.train_size + cfg.test_size]
    bkg_train = bkg_features[:cfg.train_size]
    bkg_test = bkg_features[cfg.train_size:cfg.train_size + cfg.test_size]
    
    # Combine features for normalization
    all_train_features = np.vstack([np.vstack(sig_train), np.vstack(bkg_train)])
    
    # Normalize
    scaler = None
    if cfg.normalize:
        scaler = StandardScaler()
        scaler.fit(all_train_features)
        
        sig_train = [scaler.transform(f).astype(np.float32) for f in sig_train]
        sig_test = [scaler.transform(f).astype(np.float32) for f in sig_test]
        bkg_train = [scaler.transform(f).astype(np.float32) for f in bkg_train]
        bkg_test = [scaler.transform(f).astype(np.float32) for f in bkg_test]
    
    # Create graph objects
    train_features = sig_train + bkg_train
    train_labels = np.array([1]*len(sig_train) + [0]*len(bkg_train), dtype=np.int64)
    
    test_features = sig_test + bkg_test
    test_labels = np.array([1]*len(sig_test) + [0]*len(bkg_test), dtype=np.int64)
    
    all_train_graphs = create_graph_objects(train_features, train_labels, nodes_per_graph)
    test_graphs = create_graph_objects(test_features, test_labels, nodes_per_graph)
    
    # Split train into train/val
    train_graphs, val_graphs = stratified_split_graphs(all_train_graphs, cfg.val_ratio)
    
    num_features = train_features[0].shape[1]
    
    print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
    print(f"Nodes per graph: {nodes_per_graph} | Features per node: {num_features}")
    
    return train_graphs, val_graphs, test_graphs, num_features, nodes_per_graph, scaler


def create_graph_dataloaders(
    train_graphs: List[GraphData],
    val_graphs: List[GraphData], 
    test_graphs: List[GraphData],
    batch_size: int,
    use_cuda: bool = False
):
    
    kwargs = {"batch_size": batch_size, "collate_fn": collate_graphs}
    if use_cuda:
        kwargs["pin_memory"] = True
    
    train_ds = GraphDataset(train_graphs)
    val_ds = GraphDataset(val_graphs)
    test_ds = GraphDataset(test_graphs)
    
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader
