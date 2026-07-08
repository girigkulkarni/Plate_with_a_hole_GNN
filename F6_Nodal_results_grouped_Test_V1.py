#%%
import pandas as pd
import numpy as np
import os

# %%
root_1 = r"D:\Agentic_AI\Plate_with_a_hole\Step5_1991_test_cases_plate_model\Test_data"

list_files_1 = os.listdir(root_1)

results_files_1 = [os.path.join(root_1, file) for file in list_files_1 if file.endswith("results.csv")]

#%%
for files_1 in results_files_1:
    print(os.path.basename(files_1))
    ######################
    df_1 = pd.read_csv(files_1)
    ######################
    export_2 = df_1.groupby(['node_id']).agg({
    's11': 'sum',
    's22': 'sum', 
    's33': 'sum',
    's12': 'sum',
    'bc_info':'sum',
    'load_info':'sum',
    'material':'first'
    })
    print(len(export_2))
    ##########################
    # Divide by ele_id count per node
    ele_id_counts_1 = df_1.groupby('node_id')['ele_id'].count()
    print(ele_id_counts_1)
    ##########################
    export_2['s11'] = export_2['s11'] / ele_id_counts_1
    export_2['s22'] = export_2['s22'] / ele_id_counts_1
    export_2['s33'] = export_2['s33'] / ele_id_counts_1
    export_2['s12'] = export_2['s12'] / ele_id_counts_1
    export_2['bc_info'] = export_2['bc_info'] / ele_id_counts_1
    export_2['load_info'] = export_2['load_info'] / ele_id_counts_1
    export_2['material'] = export_2['material']

    # Add zero columns and rename
    export_2['s13'] = 0
    export_2['s23'] = 0
    #export_2 = export_2.rename(columns={'s11': 'S11', 's22': 'S22', 's33': 'S33'})
    ##########################
    # Calculate von Mises stress (plane stress formula)
    # σ_vm = √(σ₁₁² + σ₂₂² - σ₁₁σ₂₂ + 3σ₁₂²)
    export_2['von_mises'] = np.sqrt(
        export_2['s11']**2 + 
        export_2['s22']**2 - 
        export_2['s11'] * export_2['s22'] + 
        3 * export_2['s12']**2
    )
    ##########################
    name_1 = os.path.basename(files_1).split('.')[0]+'_test_1.csv'
    export_2.to_csv(os.path.join(root_1, name_1))

# %%


