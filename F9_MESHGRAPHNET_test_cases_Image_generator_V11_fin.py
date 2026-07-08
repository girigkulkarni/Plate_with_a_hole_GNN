#%%
import pandas as pd
import numpy as np
import pyvista as pv
from pathlib import Path
import re

# ============================================================
# Paths
# ============================================================
ROOT_DIR = Path(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model")
RESULTS_DIR = ROOT_DIR / "Runs" / "MeshGraphNet_20260702_11_fin" / "test_case_csv_outputs_vonmises_11_fin"
ELEM_DIR = ROOT_DIR / "Test_data" / "flat_csvs_1"
OUTPUT_DIR = ROOT_DIR / "Runs" / "MeshGraphNet_20260702_11_fin"/ "generated_images_vonmises_11_fin_3"

SHARE_SCALE = False
SHOW_EDGES = False
WINDOW_SIZE = (1800, 650)

RESULTS_SUFFIX = "_predictions.csv"
ELEMENTS_SUFFIX = "_element_details.csv"

# ============================================================
# File helpers
# ============================================================
def extract_plate_key(path: Path):
    m = re.search(r"(plate_\d+)", path.name.lower())
    return m.group(1) if m else None


def find_results_file(files, key):
    matches = [f for f in files if key in f.name.lower() and f.name.lower().endswith(RESULTS_SUFFIX)]
    return matches[0] if matches else None


def find_elem_file(files, key):
    matches = [f for f in files if key in f.name.lower() and f.name.lower().endswith(ELEMENTS_SUFFIX)]
    return matches[0] if matches else None


# ============================================================
# Load nodal + von Mises prediction data
# ============================================================
def load_and_prepare_data(results_csv: Path):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "x_coord": "x",
        "y_coord": "y"
    })

    required_cols = [
        "node_id", "x", "y",
        "von_mises_true", "von_mises_pred"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in {results_csv.name}: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["node_id", "x", "y", "von_mises_true", "von_mises_pred"]).copy()
    df["node_id"] = df["node_id"].astype(np.int64)

    df_nodes = df[["node_id", "x", "y"]].drop_duplicates(subset=["node_id"]).copy()
    df_vals = df.groupby("node_id", as_index=False)[["von_mises_true", "von_mises_pred"]].mean()
    df_vals["von_mises_diff"] = df_vals["von_mises_pred"] - df_vals["von_mises_true"]

    return df_vals, df_nodes


# ============================================================
# Load connectivity
# ============================================================
def load_connectivity(elem_csv: Path):
    df_elem = pd.read_csv(elem_csv)
    df_elem.columns = df_elem.columns.str.strip()

    node_cols = [c for c in df_elem.columns if c.lower().startswith("node_")]
    if len(node_cols) < 3:
        raise ValueError(f"Need at least 3 node_ columns in {elem_csv.name}. Found: {node_cols}")

    node_cols = node_cols[:3]
    for col in node_cols:
        df_elem[col] = pd.to_numeric(df_elem[col], errors="coerce")

    df_elem = df_elem.dropna(subset=node_cols).copy()
    df_elem[node_cols] = df_elem[node_cols].astype(np.int64)

    return df_elem, node_cols


# ============================================================
# Build PyVista grid
# ============================================================
def build_grid(df_vals, df_nodes, df_elem, node_cols):
    node_ids = df_nodes["node_id"].to_numpy(dtype=np.int64)

    points = np.column_stack([
        df_nodes["x"].to_numpy(dtype=np.float64),
        df_nodes["y"].to_numpy(dtype=np.float64),
        np.zeros(len(df_nodes), dtype=np.float64)
    ])

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    cells = []
    cell_types = []
    skipped = 0

    n1_col, n2_col, n3_col = node_cols[:3]

    for _, row in df_elem.iterrows():
        n1 = int(row[n1_col])
        n2 = int(row[n2_col])
        n3 = int(row[n3_col])

        if n1 in id_to_idx and n2 in id_to_idx and n3 in id_to_idx:
            cells.extend([3, id_to_idx[n1], id_to_idx[n2], id_to_idx[n3]])
            cell_types.append(pv.CellType.TRIANGLE)
        else:
            skipped += 1

    if len(cell_types) == 0:
        raise ValueError("No valid triangular elements could be built.")

    grid = pv.UnstructuredGrid(
        np.array(cells, dtype=np.int64),
        np.array(cell_types, dtype=np.uint8),
        points
    )

    geom_ids = df_nodes["node_id"].to_numpy(dtype=np.int64)
    val_map = df_vals.set_index("node_id")

    true_arr = pd.Series(geom_ids).map(val_map["von_mises_true"]).to_numpy(dtype=np.float64)
    pred_arr = pd.Series(geom_ids).map(val_map["von_mises_pred"]).to_numpy(dtype=np.float64)
    diff_arr = pd.Series(geom_ids).map(val_map["von_mises_diff"]).to_numpy(dtype=np.float64)

    grid.point_data["von_mises_true"] = true_arr
    grid.point_data["von_mises_pred"] = pred_arr
    grid.point_data["von_mises_diff"] = diff_arr

    return grid, skipped


# ============================================================
# Save von Mises comparison plot
# ============================================================
def save_vonmises_plot(grid, out_file, share_scale=False):
    true_vals = grid.point_data["von_mises_true"]
    pred_vals = grid.point_data["von_mises_pred"]
    diff_vals = grid.point_data["von_mises_diff"]

    if share_scale:
        clim = [
            float(np.nanmin([np.nanmin(true_vals), np.nanmin(pred_vals)])),
            float(np.nanmax([np.nanmax(true_vals), np.nanmax(pred_vals)]))
        ]
        clim_true = clim
        clim_pred = clim
    else:
        clim_true = [float(np.nanmin(true_vals)), float(np.nanmax(true_vals))]
        clim_pred = [float(np.nanmin(pred_vals)), float(np.nanmax(pred_vals))]

    abs_diff = float(np.nanmax(np.abs(diff_vals)))
    clim_diff = [-abs_diff, abs_diff]

    imax_true = int(np.nanargmax(true_vals))
    imax_pred = int(np.nanargmax(pred_vals))
    imax_diff = int(np.nanargmax(np.abs(diff_vals)))

    g1 = grid.copy(deep=True)
    g2 = grid.copy(deep=True)
    g3 = grid.copy(deep=True)

    pv.set_plot_theme("document")
    pl = pv.Plotter(shape=(1, 3), window_size=WINDOW_SIZE, border=True, off_screen=True)
    pl.link_views()

    sargs = dict(
        vertical=True,
        position_x=0.03,
        position_y=0.12,
        height=0.72,
        width=0.08,
        fmt="%.3g",
        title_font_size=14,
        label_font_size=12,
    )

    pl.subplot(0, 0)
    pl.add_text("VON MISES TRUE", position="upper_left", font_size=12, color="black")
    pl.add_mesh(
        g1,
        scalars="von_mises_true",
        preference="point",
        cmap="jet",
        clim=clim_true,
        lighting=False,
        interpolate_before_map=True,
        show_edges=SHOW_EDGES,
        scalar_bar_args={**sargs, "title": "Von Mises True", "label_font_size": 18}
    )
    pl.add_points(
        g1.points[imax_true].reshape(1, 3),
        color="white",
        point_size=12,
        render_points_as_spheres=True
    )
    pl.view_xy()
    pl.camera.parallel_projection = True

    pl.subplot(0, 1)
    pl.add_text("VON MISES PRED", position="upper_left", font_size=12, color="black")
    pl.add_mesh(
        g2,
        scalars="von_mises_pred",
        preference="point",
        cmap="jet",
        clim=clim_pred,
        lighting=False,
        interpolate_before_map=True,
        show_edges=SHOW_EDGES,
        scalar_bar_args={**sargs, "title": "Von Mises Pred","label_font_size": 18}
    )
    pl.add_points(
        g2.points[imax_pred].reshape(1, 3),
        color="white",
        point_size=12,
        render_points_as_spheres=True
    )
    pl.view_xy()
    pl.camera.parallel_projection = True

    pl.subplot(0, 2)
    pl.add_text("VON MISES DIFF (Pred - True)", position="upper_left", font_size=12, color="black")
    pl.add_mesh(
        g3,
        scalars="von_mises_diff",
        preference="point",
        cmap="jet",
        clim=clim_diff,
        lighting=False,
        interpolate_before_map=True,
        show_edges=SHOW_EDGES,
        scalar_bar_args={**sargs, "title": "Von Mises Diff","label_font_size": 18}
    )
    pl.add_points(
        g3.points[imax_diff].reshape(1, 3),
        color="black",
        point_size=12,
        render_points_as_spheres=True
    )
    pl.view_xy()
    pl.camera.parallel_projection = True

    out_file.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out_file))
    pl.close()


# ============================================================
# Process one plate
# ============================================================
def process_plate(key, result_files, elem_files):
    results_csv = find_results_file(result_files, key)
    elem_csv = find_elem_file(elem_files, key)

    print(f"\n[CASE] {key}")
    print("  results:", results_csv.name if results_csv else None)
    print("  elems  :", elem_csv.name if elem_csv else None)

    if not results_csv or not elem_csv:
        print("  -> skipped")
        return

    try:
        df_vals, df_nodes = load_and_prepare_data(results_csv)
        df_elem, node_cols = load_connectivity(elem_csv)
        grid, skipped = build_grid(df_vals, df_nodes, df_elem, node_cols)

        print("  nodes       :", len(df_nodes))
        print("  mesh points :", grid.n_points)
        print("  mesh cells  :", grid.n_cells)
        print("  skipped elem:", skipped)

        out_file = OUTPUT_DIR / f"{key}_vonmises_compare_shared.png"
        save_vonmises_plot(grid, out_file, share_scale=SHARE_SCALE)
        print("  saved:", out_file.name)

    except Exception as e:
        print("  ERROR:", repr(e))


# ============================================================
# Main
# ============================================================
def main():
    print("ROOT_DIR   :", ROOT_DIR)
    print("RESULTS_DIR:", RESULTS_DIR)
    print("ELEM_DIR   :", ELEM_DIR)
    print("OUTPUT_DIR :", OUTPUT_DIR)
    print("RESULTS_DIR exists:", RESULTS_DIR.exists())
    print("ELEM_DIR exists   :", ELEM_DIR.exists())

    result_files = list(RESULTS_DIR.glob("*.csv"))
    elem_files = list(ELEM_DIR.glob("*.csv"))

    print("\n=== RESULTS FILES ===")
    for f in result_files[:10]:
        print(" ", f.name)

    print("\n=== ELEMENT FILES ===")
    for f in elem_files[:10]:
        print(" ", f.name)

    keys = sorted(set(filter(None, [extract_plate_key(f) for f in result_files + elem_files])))
    print(f"\nDetected {len(keys)} keys")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key in keys:
        process_plate(key, result_files, elem_files)


if __name__ == "__main__":
    main()