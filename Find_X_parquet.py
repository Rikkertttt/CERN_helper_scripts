#Rik van Rhee
# This code checks through a bunch of files to check if they have the "X_pandora" variable

import awkward as ak
import numpy as np
from pathlib import Path

def read_parquet(file_path):
    # Read the Parquet file into an Awkward Array
    data = ak.from_parquet(file_path)
    
    return data

for i in np.arange(0, 999, 1):

    test_file_path = Path(f"/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_uds_CLD_o2_v05_eval_v1/05/pf_tree_10{i}.parquet")

    if test_file_path.exists():
        data = read_parquet(test_file_path)

        if "X_pandora" in ak.fields(data):

            print(f"Found, at {i}")
        
    if i%100==0:
        print(f"Now at {i}")

print("Done")