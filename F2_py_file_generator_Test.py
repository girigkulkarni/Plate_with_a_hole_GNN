#%%
# DoE of INP creation
import pandas as pd
from string import Template
import os
####################################
df = pd.read_csv("DoE_data/test_set.csv")
#row = df.iloc[0].to_dict()   # only first row
######################################
output_folder = r"py_files/Test"
os.makedirs(output_folder, exist_ok=True)
#######################################
def render_py_template(template_path, output_path, variables):
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()
    ##########################
    ############################
    rendered = Template(template_text).safe_substitute(variables)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

###########################################
for i, row in df.iterrows():
    variables = row.to_dict()
    output_path = os.path.join(output_folder, f"case_{i+1}.py")
    render_py_template(
        "F0_example_template_Base_file.py",
        output_path,
        variables
    )

# %%
