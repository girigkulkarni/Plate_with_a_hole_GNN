#%%
import pandas as pd
import numpy as np
import os
#%%
root = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Train_data"
#save_path = r"D:\Agentic_AI\Plate_with_a_hole\Step3_Revised_data\Runs\Train_images"
list_files = os.listdir(root)

results_files = [os.path.join(root, file) for file in list_files if file.endswith("results.csv")]

#%%
for files in results_files:
    df = pd.read_csv(files)
    ######################
    export_1 = df.groupby(['node_id']).agg({
    's11': 'sum',
    's22': 'sum', 
    's33': 'sum',
    's12': 'sum',
    'bc_info':'sum',
    'load_info':'sum',
    'material':'first'
    })
    ##########################
    # Divide by ele_id count per node
    ele_id_counts = df.groupby('node_id')['ele_id'].count()
    export_1['s11'] = export_1['s11'] / ele_id_counts
    export_1['s22'] = export_1['s22'] / ele_id_counts
    export_1['s33'] = export_1['s33'] / ele_id_counts
    export_1['s12'] = export_1['s12'] / ele_id_counts
    export_1['bc_info'] = export_1['bc_info'] / ele_id_counts
    export_1['load_info'] = export_1['load_info'] / ele_id_counts
    export_1['material'] = export_1['material']

    # Add zero columns and rename
    export_1['s13'] = 0
    export_1['s23'] = 0
    #export_1 = export_1.rename(columns={'s11': 'S11', 's22': 'S22', 's33': 'S33'})
    ##########################
    # Calculate von Mises stress (plane stress formula)
    # σ_vm = √(σ₁₁² + σ₂₂² - σ₁₁σ₂₂ + 3σ₁₂²)
    export_1['von_mises'] = np.sqrt(
        export_1['s11']**2 + 
        export_1['s22']**2 - 
        export_1['s11'] * export_1['s22'] + 
        3 * export_1['s12']**2
    )
    
    ##########################
    name = os.path.basename(files).split('.')[0]+'_train_1.csv'
    export_1.to_csv(os.path.join(root, name))

# %%
