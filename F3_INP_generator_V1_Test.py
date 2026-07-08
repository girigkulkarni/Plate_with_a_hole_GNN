#%%
## INP generator based on PY files
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
################################
folder = "py_files/Test"
py_files = [
     os.path.abspath(os.path.join(folder, f))
    for f in os.listdir(folder)
    if f.endswith(".py")
]
#print(py_files)
##############################
#%%

abaqus_cmd = r"C:\SIMULIA\Commands\abq2025le.bat"   # adjust for your installation
output_dir = "Runs/Test"
os.makedirs(output_dir, exist_ok=True)
failed = []
max_parallel = 4   # start with 2 or 4

def run_script(py_script):
    cmd = [
        abaqus_cmd,
        "cae",
        "noGUI=" + py_script,
        "--",
        output_dir,
    ]
    result = subprocess.run(
        cmd,
        cwd=output_dir,
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    )
    return py_script, result.returncode
########################################
with ThreadPoolExecutor(max_workers=max_parallel) as executor:
    futures = [executor.submit(run_script, py_script) for py_script in py_files]
    ##############
    for future in as_completed(futures):
        try:
            py_script, rc = future.result()
            print(f"Done: {py_script} -> rc={rc}")
        except Exception as e:
            print(f"Failed: {e}")
            failed.append(py_script)

#%%


