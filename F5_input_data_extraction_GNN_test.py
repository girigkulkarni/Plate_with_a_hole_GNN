#%%
import numpy as np
import os,glob
import pandas as pd
import re, csv
from pathlib import Path
#############################
root = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test"
save_path = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data"
list_files = os.listdir(root)

sta_files = [file for file in list_files if file.endswith(".sta")]
inp_files = [file for file in list_files if file.endswith(".inp")]

req_inp_files = []

for sta_file in sta_files:
    for inp_file in inp_files:
        if sta_file.split(".")[0] == inp_file.split(".")[0]:
            req_inp_files.append(inp_file)

# %%
inp_files = [os.path.join(root, file) for file in req_inp_files]

#%%
# ---------- parse INP ----------

for inp_path in inp_files:
    nodes = {}
    elements = {}
    reading_nodes = False
    reading_elements = False
    with open(inp_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("**"):
                continue

            if s.startswith("*"):
                reading_nodes = False
                reading_elements = False
                su = s.upper()
                if su.startswith("*NODE"):
                    reading_nodes = True
                elif su.startswith("*ELEMENT") and "CPS3" in su:
                    reading_elements = True
                continue

            if reading_nodes:
                vals = [v.strip() for v in s.split(",")]
                if len(vals) >= 3:
                    nid = int(vals[0])
                    nodes[nid] = [float(vals[1]), float(vals[2])]

            elif reading_elements:
                vals = [v.strip() for v in s.split(",")]
                if len(vals) >= 4:
                    eid = int(vals[0])
                    conn = [int(vals[1]), int(vals[2]), int(vals[3])]
                    elements[eid] = conn
# %%
    input_data_nodes = pd.DataFrame()
    input_data_nodes['node_id'] = nodes.keys()
    input_data_nodes['x_coord'] = [sublist[0] for sublist in list(nodes.values())]
    input_data_nodes['y_coord'] = [sublist[1] for sublist in list(nodes.values())]
    input_data_nodes.to_csv(save_path+'\\'+os.path.basename(inp_path).split('.')[0]+'_nodal_details.csv', index=False)
    input_data_ele = pd.DataFrame()
    input_data_ele['ele_id'] = elements.keys()
    input_data_ele['Node_1'] = [sublist[0] for sublist in list(elements.values())]
    input_data_ele['Node_2'] = [sublist[1] for sublist in list(elements.values())]
    input_data_ele['Node_3'] = [sublist[2] for sublist in list(elements.values())]
    input_data_ele.to_csv(save_path+'\\'+os.path.basename(inp_path).split('.')[0]+'_element_details.csv', index=False)

