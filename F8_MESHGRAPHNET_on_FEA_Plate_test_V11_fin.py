# ============================================================================
# TEST RELOAD SCRIPT - Load MeshGraphNet2D from .pkl and run test inference
# UPDATED FOR VON_MISES-ONLY TRAINING SETUP
# - matches updated training script architectures (LayerNorm, SiLU)
# - matches updated data preparation functions (edge_index, hole features)
# - reports hotspot-region metrics separately from global metrics
# ============================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer, global_mean_pool
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import itertools

TARGET_SCALE = 50.0

# ============================================================================
# 1. MODEL DEFINITION (MUST MATCH TRAINING EXACTLY)
# ============================================================================
def scatter_mean_pytorch(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype).unsqueeze(-1))
    return out / count.clamp(min=1.0)

class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        # Matches the updated training script exactly: dim * 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, src, dst, edge_attr, u, batch):
        out = torch.cat([src, dst, edge_attr], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Matches the updated training script exactly: dim * 3
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
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
        
        # Safely handle single unbatched graph
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for block in self.processor:
            u = global_mean_pool(x, batch)
            x, edge_attr = block(x, edge_index, edge_attr, batch, u)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return self.head(x)


# ============================================================================
# 2. DATA PREPROCESSING FOR FLAT CSVs
# ============================================================================
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

def normalize_xy_columns(df):
    df = df.copy()
    df = df.rename(columns={"x_coord": "x", "y_coord": "y"})
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
        
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

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
    """Computes distance to applied loads and boundary conditions."""
    df = df.copy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    
    if "load_info" in df.columns:
        load_mask = df["load_info"] > 0.0
        if load_mask.any():
            lx, ly = x[load_mask], y[load_mask]
            dist_to_load = np.min(np.sqrt((x[:, None] - lx)**2 + (y[:, None] - ly)**2), axis=1)
        else:
            dist_to_load = np.zeros_like(x)
    else:
        dist_to_load = np.zeros_like(x)
        
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

def build_hotspot_mask(df, target_col, hole_center, hole_radius, geom_tau=0.08, stress_pct=90):
    xc, yc = hole_center
    dx = df["x"].to_numpy(dtype=float) - xc
    dy = df["y"].to_numpy(dtype=float) - yc
    rr = np.sqrt(dx ** 2 + dy ** 2)
    d = np.abs(rr - hole_radius)
    near_hole = d < geom_tau
    stress_vals = df[target_col].to_numpy(dtype=float)
    thresh = np.nanpercentile(stress_vals, stress_pct)
    high_stress = stress_vals >= thresh
    return (near_hole | high_stress).astype(np.float32)

def report_nan_rows(df, cols, label, max_rows=50):
    mask = df[cols].isna().any(axis=1)
    if mask.any():
        print(f"\n{label}: NaN rows found")
        cols_show = [c for c in ["node_id", "x", "y"] + cols if c in df.columns]
        print(df.loc[mask, cols_show].head(max_rows).to_string(index=False))
        return True
    return False

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
    if edge_index.numel() == 0:
        return torch.empty((0, 5), dtype=pos.dtype)
    row, col = edge_index
    rel = pos[col] - pos[row]
    length = torch.norm(rel, dim=1, keepdim=True)
    unit = rel / (length.clamp(min=1e-12))
    return torch.cat([rel, length, unit], dim=1)

def build_graph(flat_path, conn_path, feature_cols, target_cols, debug_nan=True):
    df = clean_columns(pd.read_csv(flat_path))
    conn = clean_columns(pd.read_csv(conn_path))

    df = normalize_xy_columns(df)
    df = add_material_features(df)
    df, hole_info = add_hole_features(df)
    df = add_distance_features(df)

    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols if c not in df.columns]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}. Available: {df.columns.tolist()}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}. Available: {df.columns.tolist()}")

    for c in ["node_id", "x", "y"] + feature_cols + target_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if debug_nan:
        feat_has_nan = report_nan_rows(df, feature_cols, "Feature check", max_rows=50)
        targ_has_nan = report_nan_rows(df, target_cols, "Target check", max_rows=50)
        if feat_has_nan:
            raise ValueError(f"NaN found in feature columns for file: {flat_path}")
        if targ_has_nan:
            raise ValueError(f"NaN found in target columns for file: {flat_path}")

    x = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float)
    y = torch.tensor(df[target_cols].to_numpy(), dtype=torch.float)
    pos = torch.tensor(df[["x", "y"]].to_numpy(), dtype=torch.float)

    edge_index = build_edge_index(conn, len(df))
    edge_attr = build_edge_attr_2d(pos, edge_index)

    hotspot_mask = build_hotspot_mask(
        df, target_cols[0], hole_center=(hole_info[0], hole_info[1]), hole_radius=hole_info[2]
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
    data.num_nodes = len(df)
    data.case_name = Path(flat_path).stem
    data.hotspot_mask = torch.tensor(hotspot_mask, dtype=torch.float)
    
    if "node_id" in df.columns:
        data.node_id = torch.tensor(df["node_id"].to_numpy(), dtype=torch.long)
        
    return data

# ============================================================================
# EXTRA HELPERS FOR SAVING CSV OUTPUTS
# ============================================================================
def get_node_ids_and_xy(flat_path, expected_len):
    df = clean_columns(pd.read_csv(flat_path))
    df = normalize_xy_columns(df)
    node_ids = df["node_id"].to_numpy() if "node_id" in df.columns else np.arange(expected_len)
    x_vals = df["x"].to_numpy() if "x" in df.columns else np.full(expected_len, np.nan)
    y_vals = df["y"].to_numpy() if "y" in df.columns else np.full(expected_len, np.nan)
    return node_ids, x_vals, y_vals

def save_case_predictions_to_csv(case_name, flat_path, y_true, y_pred, target_cols, output_dir, hotspot_mask=None):
    y_true = np.asarray(y_true).reshape(-1, len(target_cols))
    y_pred = np.asarray(y_pred).reshape(-1, len(target_cols))
    node_ids, x_vals, y_vals = get_node_ids_and_xy(flat_path, len(y_true))

    out_df = pd.DataFrame({
        "case_name": case_name,
        "node_id": node_ids,
        "x": x_vals,
        "y": y_vals,
    })

    for j, col in enumerate(target_cols):
        out_df[f"{col}_true"] = y_true[:, j]
        out_df[f"{col}_pred"] = y_pred[:, j]
        out_df[f"{col}_err"] = y_pred[:, j] - y_true[:, j]
        out_df[f"{col}_abs_err"] = np.abs(y_pred[:, j] - y_true[:, j])

    if hotspot_mask is not None:
        out_df["hotspot_mask"] = np.asarray(hotspot_mask).reshape(-1)

    csv_path = os.path.join(output_dir, f"{case_name}_predictions.csv")
    out_df.to_csv(csv_path, index=False)
    return out_df, csv_path

# ============================================================================
# 3. LOAD MODEL FROM .PKL FILE
# ============================================================================
def load_model_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)

    cfg = meta["run_config"]
    model_pt_path = meta["model_pt_path"]

    print(f"Loading model from: {model_pt_path}")
    print(f"Original Model config: {cfg}")

    state_dict = torch.load(model_pt_path, map_location="cpu")
    
    actual_hidden = state_dict['node_encoder.0.weight'].shape[0]
    actual_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('processor.')]) + 1
    
    print(f"Correcting config based on state_dict -> hidden_channels: {actual_hidden}, num_layers: {actual_layers}")
    cfg["hidden_channels"] = actual_hidden
    cfg["num_layers"] = actual_layers

    edge_attr_dim = cfg.get("edge_attr_dim", 5)

    model = MeshGraphNet2D(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        edge_attr_dim=edge_attr_dim,
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.05)
    )

    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print(f"Device: {device}")

    return model, cfg, device

# ============================================================================
# 4. RUN TEST CASE
# ============================================================================
# ============================================================================
# 4. RUN TEST CASE
# ============================================================================
@torch.no_grad()
def run_test_case(model, test_flat, test_conn, feature_cols, target_cols, device):
    model.eval()

    try:
        test_graph = build_graph(test_flat, test_conn, feature_cols, target_cols, debug_nan=True)
    except Exception as e:
        print(f"Error building graph: {e}")
        return None, None, None, None

    test_graph = test_graph.to(device)
    pred = model(test_graph)

    if torch.isnan(pred).any():
        print(f"WARNING: NaN predictions generated for file: {test_flat}")
        return None, None, None, None

    # FIX: test_graph.y is already in original physical units (MPa) because 
    # we did not divide by TARGET_SCALE in this script's build_graph function.
    y_true = test_graph.y.cpu().numpy().reshape(-1, len(target_cols))
    
    # The model outputs normalized values [~0 to 1], so we must scale ONLY 
    # the predictions back up to physical units.
    y_pred = pred.cpu().numpy().reshape(-1, len(target_cols)) * TARGET_SCALE
    
    hotspot_mask = test_graph.hotspot_mask.cpu().numpy().reshape(-1)

    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    rmse = np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    r2 = r2_score(y_true.reshape(-1), y_pred.reshape(-1))

    hot_idx = hotspot_mask > 0.5
    if hot_idx.sum() >= 2:
        hot_mae = mean_absolute_error(y_true.reshape(-1)[hot_idx], y_pred.reshape(-1)[hot_idx])
        hot_rmse = np.sqrt(mean_squared_error(y_true.reshape(-1)[hot_idx], y_pred.reshape(-1)[hot_idx]))
    else:
        hot_mae = np.nan
        hot_rmse = np.nan

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "hotspot_mae": hot_mae,
        "hotspot_rmse": hot_rmse,
        "num_hotspot_nodes": int(hot_idx.sum()),
        "num_nodes": test_graph.num_nodes,
        "case_name": test_graph.case_name
    }

    print(f"\n{'='*60}")
    print(f"Case: {test_graph.case_name}")
    print(f"MAE          : {mae:.6f}")
    print(f"RMSE         : {rmse:.6f}")
    print(f"Hotspot MAE  : {hot_mae:.6f}  ({int(hot_idx.sum())} nodes)")
    print(f"{'='*60}")

    return y_true, y_pred, metrics, hotspot_mask

# ============================================================================
# 5. RUN ON TEST DATA
# ============================================================================
if __name__ == "__main__":
    pkl_path = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Runs\MeshGraphNet_20260702_11_fin\meshgraphnet_run_info_vonmises.pkl"
    model, cfg, device = load_model_from_pkl(pkl_path)

    feature_cols = cfg["feature_cols"]
    target_cols = cfg["target_cols"]

    test_root = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data\flat_csvs_1"
    test_files = os.listdir(test_root)

    test_flat_files = sorted([os.path.join(test_root, f) for f in test_files if f.endswith("flat.csv")])
    test_connectivity = sorted([os.path.join(test_root, f) for f in test_files if f.endswith("element_details.csv")])

    print(f"\nFound: {len(test_flat_files)} flat files, {len(test_connectivity)} connectivity files")
    
    if not test_flat_files:
        print("No flat test files found!")
        raise SystemExit(0)

    output_dir = os.path.join(os.path.dirname(pkl_path), "test_case_csv_outputs_vonmises_11_fin")
    os.makedirs(output_dir, exist_ok=True)

    combined_case_dfs = []
    metrics_rows = []

    n_cases = min(len(test_flat_files), len(test_connectivity))
    for i in range(n_cases):
        y_true, y_pred, metrics, hotspot_mask = run_test_case(
            model=model,
            test_flat=test_flat_files[i],
            test_conn=test_connectivity[i],
            feature_cols=feature_cols,
            target_cols=target_cols,
            device=device
        )

        if y_true is None:
            continue

        case_name = Path(test_flat_files[i]).stem
        case_df, case_csv_path = save_case_predictions_to_csv(
            case_name=case_name,
            flat_path=test_flat_files[i],
            y_true=y_true,
            y_pred=y_pred,
            target_cols=target_cols,
            output_dir=output_dir,
            hotspot_mask=hotspot_mask
        )

        combined_case_dfs.append(case_df)
        metrics_rows.append(metrics)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_path = os.path.join(output_dir, "test_case_metrics_summary.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics summary CSV: {metrics_csv_path}")

        overall_df = pd.DataFrame([{
            "num_cases": len(metrics_df),
            "mean_case_mae": metrics_df["mae"].mean(),
            "mean_case_rmse": metrics_df["rmse"].mean(),
            "mean_case_r2": metrics_df["r2"].mean(),
            "mean_case_hotspot_mae": metrics_df["hotspot_mae"].mean(),
            "mean_case_hotspot_rmse": metrics_df["hotspot_rmse"].mean()
        }])
        overall_csv_path = os.path.join(output_dir, "overall_summary.csv")
        overall_df.to_csv(overall_csv_path, index=False)

    print("\nDone.")