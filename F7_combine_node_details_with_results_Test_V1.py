from pathlib import Path
import pandas as pd

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

def make_flat_csv(node_csv, result_csv, output_csv):
    nodes = clean_columns(pd.read_csv(node_csv))
    results = clean_columns(pd.read_csv(result_csv))

    # Start from results, then add node info only if needed
    df = results.copy().reset_index(drop=True)

    if "node_id" not in df.columns and "node_id" in nodes.columns:
        df["node_id"] = nodes["node_id"].values

    # Add x/y/material from nodes if missing
    for col in ["x", "y", "material"]:
        if col not in df.columns and col in nodes.columns:
            df[col] = nodes[col].values

    # If x/y are named differently in node file
    if "x" not in df.columns and "x_coord" in nodes.columns:
        df["x"] = nodes["x_coord"].values
    if "y" not in df.columns and "y_coord" in nodes.columns:
        df["y"] = nodes["y_coord"].values

    # Reorder preferred columns first
    preferred = [c for c in [
        "node_id", "x", "y", "bc_info", "load_info", "material",
         "von_mises"
    ] if c in df.columns]

    rest = [c for c in df.columns if c not in preferred]
    df = df[preferred + rest]

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

#%%

input_root = Path(r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data")
output_root = input_root / "flat_csvs_1"
output_root.mkdir(exist_ok=True)

result_files = sorted(input_root.glob("*results_test_1.csv"))
node_files = sorted(input_root.glob("*nodal_details.csv"))

for result_csv, node_csv in zip(result_files, node_files):
    out_name = result_csv.name.replace("results_test_1.csv", "flat.csv")
    output_csv = output_root / out_name
    make_flat_csv(node_csv, result_csv, output_csv)