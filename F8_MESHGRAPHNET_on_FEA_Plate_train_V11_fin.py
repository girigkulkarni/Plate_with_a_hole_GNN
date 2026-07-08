#%%
from pathlib import Path
import pandas as pd
import numpy as np
import torch, os, time, itertools
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MetaLayer, global_mean_pool
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# NOTE: Assuming SF1_save_artifacts.py is in the same directory or accessible.
try:
    from SF1_save_artifacts import save_training_artifacts
except ImportError:
    print("Warning: SF1_save_artifacts.py not found. Saving artifacts will be disabled.")
    def save_training_artifacts(**kwargs):
        print("Artifact saving is disabled.")
        return None

# --- Hyperparameters ---
# The scale factor for target normalization. 50.0 is chosen since von_mises peaks around 45-50 MPa.
TARGET_SCALE = 50.0 

# --- Data Preparation Functions ---
def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def add_material_features(df):
    df = df.copy()
    if "material" in df.columns:
        df["material"] = df["material"].astype(str).str.strip().str.upper()
        df["mat_steel"] = (df["material"] == "STEEL").astype(float)
        df["mat_alu"] = (df["material"] == "ALU").astype(float)
    else:
        df["mat_steel"] = 0.0
        df["mat_alu"] = 0.0
    return df

def add_hole_features(df, hole_center=None, hole_radius=None):
    df = df.copy()
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("Missing x/y columns for hole features")
    if hole_center is None:
        xc = float(df["x"].median())
        yc = float(df["y"].median())
    else:
        xc, yc = hole_center
    if hole_radius is None:
        r = float(np.nanpercentile(np.sqrt((df["x"] - xc) ** 2 + (df["y"] - yc) ** 2), 20))
    else:
        r = float(hole_radius)
    dx = df["x"] - xc
    dy = df["y"] - yc
    rr = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
    df["hole_dist"] = np.abs(rr - r)
    df["hole_angle_sin"] = np.sin(np.arctan2(dy, dx))
    df["hole_angle_cos"] = np.cos(np.arctan2(dy, dx))
    return df, (xc, yc, r)

def add_distance_features(df):
    """
    Computes distance to applied loads and boundary conditions to solve long-range dependency issues.
    """
    df = df.copy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    
    # Identify Load Nodes
    if "load_info" in df.columns:
        load_mask = df["load_info"] > 0.0
        if load_mask.any():
            lx, ly = x[load_mask], y[load_mask]
            # Distance to nearest load
            dist_to_load = np.min(np.sqrt((x[:, None] - lx)**2 + (y[:, None] - ly)**2), axis=1)
        else:
            dist_to_load = np.zeros_like(x)
    else:
        dist_to_load = np.zeros_like(x)
        
    # Identify BC Nodes
    if "bc_info" in df.columns:
        bc_mask = df["bc_info"] > 0.0
        if bc_mask.any():
            bx, by = x[bc_mask], y[bc_mask]
            dist_to_bc = np.min(np.sqrt((x[:, None] - bx)**2 + (y[:, None] - by)**2), axis=1)
        else:
            dist_to_bc = np.zeros_like(x)
    else:
        dist_to_bc = np.zeros_like(x)

    df["dist_to_load"] = dist_to_load
    df["dist_to_bc"] = dist_to_bc
    return df

def build_edge_index(connectivity_df, num_nodes):
    edges = set()
    connectivity_df = clean_columns(connectivity_df)
    if "ele_id" in connectivity_df.columns:
        for _, g in connectivity_df.groupby("ele_id", sort=False):
            idxs = [i for i in g.index if i < num_nodes]
            for a, b in itertools.combinations(idxs, 2):
                edges.add((a, b))
                edges.add((b, a))
    node_columns = [c for c in connectivity_df.columns if c.startswith("node_")]
    for node_col in node_columns:
        tmp = connectivity_df.copy()
        tmp["node_tmp"] = tmp[node_col]
        for _, g in tmp.groupby("node_tmp", sort=False):
            idxs = [i for i in g.index if i < num_nodes]
            for a, b in itertools.combinations(idxs, 2):
                edges.add((a, b))
                edges.add((b, a))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

def build_edge_attr_2d(pos, edge_index):
    row, col = edge_index
    rel = pos[col] - pos[row]
    length = torch.norm(rel, dim=1, keepdim=True)
    unit = rel / (length.clamp(min=1e-12))
    return torch.cat([rel, length, unit], dim=1)

def build_sample_weights(df, hole_center, hole_radius, alpha=6.0, tau=0.06):
    xc, yc = hole_center
    dx = df["x"].to_numpy(dtype=float) - xc
    dy = df["y"].to_numpy(dtype=float) - yc
    rr = np.sqrt(dx**2 + dy**2)
    d = np.abs(rr - hole_radius)
    w_geom = 1.0 + alpha * np.exp(-d / tau)
    return w_geom.astype(np.float32)

def build_graph(flat_path, conn_path, feature_cols, target_cols):
    df = clean_columns(pd.read_csv(flat_path))
    conn = clean_columns(pd.read_csv(conn_path))
    df = add_material_features(df)
    df, hole_info = add_hole_features(df)
    df = add_distance_features(df)

    x = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float)
    
    # NORMALIZATION: Scale target down to stabilize gradients
    y_raw = df[target_cols].to_numpy()
    y_scaled = y_raw / TARGET_SCALE
    y = torch.tensor(y_scaled, dtype=torch.float)
    
    pos = torch.tensor(df[["x", "y"]].to_numpy(), dtype=torch.float)

    edge_index = build_edge_index(conn, len(df))
    edge_attr = build_edge_attr_2d(pos, edge_index)
    sample_weights = build_sample_weights(df, hole_center=(hole_info[0], hole_info[1]), hole_radius=hole_info[2])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
    data.num_nodes = len(df)
    data.case_name = Path(flat_path).stem
    data.sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    if "node_id" in df.columns:
        data.node_id = torch.tensor(df["node_id"].to_numpy(), dtype=torch.long)

    return data

class FEAFlatDataset(Dataset):
    def __init__(self, flat_paths, conn_paths, feature_cols, target_cols):
        super().__init__()
        assert len(flat_paths) == len(conn_paths)
        self.flat_paths = list(flat_paths)
        self.conn_paths = list(conn_paths)
        self.feature_cols = feature_cols
        self.target_cols = target_cols

    def len(self):
        return len(self.flat_paths)

    def get(self, idx):
        return build_graph(self.flat_paths[idx], self.conn_paths[idx], self.feature_cols, self.target_cols)

def scatter_mean_pytorch(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype).unsqueeze(-1))
    return out / count.clamp(min=1.0)

# --- Model Architecture ---
class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        # Added input dimension to account for global context broadcast
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, src, dst, edge_attr, u, batch):
        # PyG MetaLayer passes batch[row] to EdgeModel internally, 
        # so 'batch' here naturally aligns with the edges.
        out = torch.cat([src, dst, edge_attr], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        # 'batch' here naturally aligns with the nodes.
        _, col = edge_index
        agg = scatter_mean_pytorch(edge_attr, col, x.size(0))
        out = torch.cat([x, agg], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.node_mlp(out)

class ProcessorBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.meta = MetaLayer(
            edge_model=EdgeModel(hidden_dim, edge_dim),
            node_model=NodeModel(hidden_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, batch, u):
        x_res, e_res = x, edge_attr
        # Pass the standard node batch. PyG MetaLayer automatically routes 
        # batch to NodeModel and batch[row] to EdgeModel.
        x, edge_attr, _ = self.meta(x, edge_index, edge_attr, u=u, batch=batch)
        return x + x_res, edge_attr + e_res

class MeshGraphNet2D(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_attr_dim, num_layers=8, dropout=0.05):
        super().__init__()
        self.dropout = dropout
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.processor = nn.ModuleList([
            ProcessorBlock(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_channels, out_channels),
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") else x.new_zeros(x.size(0), dtype=torch.long)
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for block in self.processor:
            # Generate Global Context (Virtual Node representation)
            u = global_mean_pool(x, batch)
            x, edge_attr = block(x, edge_index, edge_attr, batch, u)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return self.head(x)


# --- Loss and Training/Evaluation Functions ---
def graphwise_topk_mean_loss(pred, target, batch_idx, sample_weight=None, top_k=0.05):
    node_loss = F.huber_loss(pred, target, reduction="none").squeeze(-1)
    if sample_weight is not None:
        node_loss = node_loss * sample_weight
    total = 0.0
    num_graphs = int(batch_idx.max().item()) + 1
    for g in range(num_graphs):
        mask = (batch_idx == g)
        g_loss = node_loss[mask]
        k = max(1, int(top_k * g_loss.numel()))
        total = total + torch.topk(g_loss, k, largest=True).values.mean()
    return total / num_graphs

def graph_edge_gradient_loss(pred, target, edge_index):
    if edge_index.numel() == 0:
        return pred.new_tensor(0.0)
    row, col = edge_index
    pred_diff = pred[row] - pred[col]
    true_diff = target[row] - target[col]
    return F.smooth_l1_loss(pred_diff, true_diff)

def train_epoch(model, loader, optimizer, device, w_main=0.9, w_topk=0.7, w_grad=0.15):
    model.train()
    total_loss = 0.0
    huber_fn = torch.nn.HuberLoss(delta=1.0, reduction="none")
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        sample_weight = batch.sample_weights.to(device)
        main_loss = (huber_fn(pred, batch.y).squeeze(-1) * sample_weight).mean()
        topk_loss = graphwise_topk_mean_loss(
            pred, batch.y, batch.batch, sample_weight=sample_weight, top_k=0.1
        )
        grad_loss = graph_edge_gradient_loss(pred.squeeze(-1), batch.y.squeeze(-1), batch.edge_index)
        loss = w_main * main_loss + w_topk * topk_loss + w_grad * grad_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_mae, total_rmse, total_r2, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        
        # INVERSE TRANSFORM: Multiply by TARGET_SCALE for metrics in original unit
        y_true = (batch.y * TARGET_SCALE).detach().cpu().numpy().reshape(-1)
        y_pred = (pred * TARGET_SCALE).detach().cpu().numpy().reshape(-1)
        
        total_mae += mean_absolute_error(y_true, y_pred)
        total_rmse += np.sqrt(mean_squared_error(y_true, y_pred))
        total_r2 += r2_score(y_true, y_pred)
        n += 1
    n = max(n, 1)
    return total_mae / n, total_rmse / n, total_r2 / n

@torch.no_grad()
def run_single_test_case_fixed(model, test_flat, test_conn, feature_cols, target_cols, device):
    model.eval()
    test_graph = build_graph(test_flat, test_conn, feature_cols, target_cols)
    test_graph = test_graph.to(device)
    pred = model(test_graph)

    y_true = (test_graph.y.detach().cpu().numpy().reshape(-1)) * TARGET_SCALE
    y_pred = (pred.detach().cpu().numpy().reshape(-1)) * TARGET_SCALE

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print("Case name:", test_graph.case_name)
    print(f"MAE  : {mae:.6f}, RMSE : {rmse:.6f}, R2   : {r2:.6f}")
    return y_true, y_pred


if __name__ == "__main__":
    # --- !! IMPORTANT !! User Configuration ---
    TRAIN_DATA_ROOT = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Train_data\flat_csvs_1"
    TEST_DATA_ROOT = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data\flat_csvs_1"
    SAVE_ARTIFACTS_PATH = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_1"
    
    # --- Hyperparameters ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    HIDDEN_CHANNELS = 128
    NUM_LAYERS = 8
    DROPOUT = 0.05

    # ADDED dist_to_load and dist_to_bc to features
    feature_cols = [
        "x", "y", "bc_info", "load_info", "mat_steel", "mat_alu",
        "hole_dist", "hole_angle_sin", "hole_angle_cos", 
        "dist_to_load", "dist_to_bc"
    ]
    target_cols = ["von_mises"]

    list_files = os.listdir(TRAIN_DATA_ROOT)
    flat_files = sorted([os.path.join(TRAIN_DATA_ROOT, f) for f in list_files if f.endswith("flat.csv")])
    connectivity_files = sorted([os.path.join(TRAIN_DATA_ROOT, f) for f in list_files if f.endswith("element_details.csv")])

    print("flat_files:", len(flat_files))
    assert len(flat_files) > 0, "No training files found. Check TRAIN_DATA_ROOT."

    dataset = FEAFlatDataset(flat_files, connectivity_files, feature_cols, target_cols)
    sample = dataset[0]

    train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(0.8 * len(dataset)))))
    val_dataset = torch.utils.data.Subset(dataset, list(range(int(0.8 * len(dataset)), len(dataset))))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MeshGraphNet2D(
        in_channels=sample.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=1,
        edge_attr_dim=sample.edge_attr.shape[1],
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Ensure T_max matches NUM_EPOCHS so learning rate scales down appropriately
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_time = time.time()
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, device)
        scheduler.step()

        # EARLY STOPPING LOGIC: Keep the best state based on Validation RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6e} | val_rmse={val_rmse:.6f} | val_r2={val_r2:.6f} | LR={lr:.2e}")

    # Load best state dict before saving/testing
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Training took {(time.time() - start_time)/60:.3f} minutes.")
    print(f"Best validation RMSE: {best_val_rmse:.6f}")
    
    run_config = {
        "model_name": "MeshGraphNet2D",
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "in_channels": int(sample.x.shape[1]),
        "edge_attr_dim": int(sample.edge_attr.shape[1]),
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": 1,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NUM_EPOCHS,
        "train_root": TRAIN_DATA_ROOT,
        "notes": "Target normalized /50. Added dist_to_bc and dist_to_load. Added Global Context pool. 50 epochs.",
    }

    path = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_11_fin"
    os.makedirs(path, exist_ok=True)

    artifacts = save_training_artifacts(
        model=model,
        optimizer=optimizer,
        metrics={
            "train_loss_last": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2
        },
        run_config=run_config,
        model_pt_path=os.path.join(path, "meshgraphnet_model_state_vonmises.pt"),
        checkpoint_pt_path=os.path.join(path, "meshgraphnet_checkpoint_vonmises.pt"),
        run_pickle_path=os.path.join(path, "meshgraphnet_run_info_vonmises.pkl")
    )

    test_root = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data\flat_csvs_1"
    test_list_files = os.listdir(test_root)

    test_flat_files = sorted([os.path.join(test_root, f) for f in test_list_files if f.endswith("flat.csv")])
    test_connectivity_files = sorted([os.path.join(test_root, f) for f in test_list_files if f.endswith("element_details.csv")])

    for files in range(len(test_flat_files)):
        y_true, y_pred = run_single_test_case_fixed(
            model=model,
            test_flat=test_flat_files[files],
            test_conn=test_connectivity_files[files],
            feature_cols=feature_cols,
            target_cols=target_cols,
            device=device
        )

        t1 = clean_columns(pd.read_csv(test_flat_files[files]))
        t1 = add_material_features(t1)
        t1, _ = add_hole_features(t1)

        export = pd.DataFrame({
            "node_id": t1["node_id"].to_numpy(),
            "von_mises_true": y_true,
            "von_mises_pred": y_pred
        })

        name = os.path.basename(test_flat_files[files]).split(".")[0] + "_results_vonmises.csv"
        export.to_csv(os.path.join(path, name), index=False)

# %%