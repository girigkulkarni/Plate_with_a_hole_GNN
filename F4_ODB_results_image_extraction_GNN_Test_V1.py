#%%
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import displayGroupOdbToolset as dgo
#####################
import __main__
import os, sys, pathlib
###################
sys.path.append(r"C:\Users\girig\site-packages\python3.10")
########################
import numpy as np
import os,glob
import pandas as pd
import re, csv
#############################
root = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test"
save_path = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data"
list_files = os.listdir(root)

sta_files = [file for file in list_files if file.endswith(".sta")]
odb_files = [file for file in list_files if file.endswith(".odb")]

req_odb_files = []

for sta_file in sta_files:
    for odb_file in odb_files:
        if sta_file.split(".")[0] == odb_file.split(".")[0]:
            req_odb_files.append(odb_file)

# %%
odb_files = [os.path.join(root, file) for file in req_odb_files]
#print(odb_files)

# %%
#import session
session.graphicsOptions.setValues(backgroundStyle=SOLID, backgroundColor='#FFFFFF')
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(
    legend=ON, title=OFF, state=OFF, annotations=OFF)
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=FREE)

all_max_results = []
###################
for odb_file in odb_files:
    print(os.path.basename(odb_file).split('.')[0])
    rows = []
    odb = openOdb(path=odb_file, readOnly=True)
    req_nodeset = 'SET-1'
    req_surf = 'SURF-1'
    req_node_id1 = []
    req_node_id2 = []
    ######################
    req_nodes_1 = odb.rootAssembly.nodeSets[req_nodeset].nodes[0]
    for x in req_nodes_1:
        req_node_id1.append(x.label)
    #######################
    req_nodes_2 = odb.rootAssembly.surfaces[req_surf].nodes[0]
    for x in req_nodes_2:
        req_node_id2.append(x.label)
    ########################
    # Get the frame you want (frames[1] or frames[-1] for last)
    frame = odb.steps['Step-1'].frames[-1]
    ###################################
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
    session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    path_1 = os.path.join(save_path, os.path.basename(odb_file).split('.')[0]) 
    session.printToFile(
        fileName=path_1, 
        format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))
    #####################################

    # Get nodal averaged stress (matches Abaqus visualization)
    stress_field = frame.fieldOutputs['S'].getSubset(position=ELEMENT_NODAL)
    #disp_field = frame.fieldOutputs['U'].getSubset(position=NODAL)
    coords_field = frame.fieldOutputs['NFORC1'].getSubset(position=ELEMENT_NODAL)
    
    # ========== FILE 1: All nodal data for PyVista ==========

    for stress_val, coord_val in zip(
        stress_field.values, 
        #disp_field.values, 
        coords_field.values
    ):
        material = odb.materials.keys()[0]
        ele_id = stress_val.elementLabel
        node_id = stress_val.nodeLabel
        #x, y = coord_val.data
        #u1, u2 = disp_val.data
        s11, s22, s33, s12 = stress_val.data
        #########################
        if node_id in req_node_id1:
            bc_info = 1
        else:
            bc_info = 0
        ############################
        if node_id in req_node_id2:
            load_info = coord_val.data
        else:
            load_info = 0
        von_mises = stress_val.mises
        #rows.append([node_id, x, y, u1, u2, s11, s22, s33, s12, von_mises])
        rows.append([ele_id, node_id, s11, s22, s33, s12, bc_info, load_info, von_mises, material])
    ##########################
    rows.sort(key=lambda r: r[0])
    
    csv_path = os.path.join(save_path, os.path.basename(odb_file).split('.')[0]+"_nodal_results.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ele_id', 'node_id', 
                            's11', 's22', 's33', 's12', 'bc_info', 'load_info', 'von_mises', 'material'])
        writer.writerows(rows)

    # ============================================================
    # ========== Find max von Mises and node ==========
    node_labels = np.array([v.nodeLabel for v in stress_field.values])
    mises = np.array([v.mises for v in stress_field.values])
    
    # Average by node (if there are duplicates)
    uniq_nodes, inv = np.unique(node_labels, return_inverse=True)
    avg_mises = np.bincount(inv, weights=mises) / np.bincount(inv)
    
    imax = np.argmax(avg_mises)
    max_mises = round(avg_mises[imax], 2)
    max_node = uniq_nodes[imax]
    
    # Store for summary CSV
    all_max_results.append({
        'odb_name': os.path.basename(odb_file),
        'max_von_mises': max_mises,
        'node_id': max_node
    })
    
    odb.close()
    print(f"Extracted {len(uniq_nodes)} nodes from {odb_file} | Max VM={max_mises} at Node {max_node}")
# ========== FILE 2: Max results summary ==========

summary_csv_path = os.path.join(save_path,"max_mises_summary.csv")
with open(summary_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['odb_name', 'max_von_mises', 'node_id'])
    #############
    for result in all_max_results:
        writer.writerow([
            result['odb_name'],
            result['max_von_mises'],
            result['node_id']
        ])
print(f"\n=== Summary saved to {summary_csv_path} ===")
print(f"Processed {len(all_max_results)} ODB files")
