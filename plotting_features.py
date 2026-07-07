# Rik van Rhee
# This code is able to plot a histogram of a specific feature from a given parquet file. 
# It reads the parquet file, extracts the specified feature, and creates a histogram plot.  

import awkward as ak
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def read_parquet(file_path):
    # Read the Parquet file into an Awkward Array
    data = ak.from_parquet(file_path)
    
    return data


def plot_feature(X_name = "X_track", 
                 feature_number = 0, 
                 filepath = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_uds_CLD_o2_v05_eval_v1/05/pf_tree_10923.parquet",
                 savename = "Default.png",
                 bins = 50,
                 feature_name = None):
    
    """
    Plots a histogram of a specific feature from a given parquet file.
    Args:
        X_name: The name of the feature array in the parquet file (default is "X_track").
        feature_number: The index of the feature to plot (default is 0).
        filepath: Path to the parquet file (default is a specific path).
        savename: Name of the output image file (default is "Default.png").
        bins: Number of bins for the histogram (default is 50).
        feature_name: Optional name for the feature to use in the plot title and xlabel.
    """


    data = read_parquet(filepath)

    features = None

    for i in range(len(data[X_name])):
        feature_data = ak.to_numpy(data[X_name][i])
        
        if features is None:
            features = feature_data[:,feature_number]
        else:
            features = np.concatenate((features, feature_data[:,feature_number]))

    # Basic safety: if no features were found, do nothing
    if features is None or len(features) == 0:
        print(f"No data found for {X_name}[{feature_number}]")
        return

    # Nicely formatted plot
    plt.figure(figsize=(8, 6), dpi=150)
    plt.hist(features, bins=bins, color='C0', alpha=0.75, edgecolor='black')
    plt.ylabel('Counts')

    if feature_name == None:
        plt.title(f'Histogram of {X_name} feature number {feature_number}')
        plt.xlabel(f"{X_name}[{feature_number}]")
    else:
        plt.title(f'Histogram of {X_name} feature: {feature_name}')
        plt.xlabel(f'{feature_name}')


    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(savename, bbox_inches='tight')

    return


def main():


    plot_feature(X_name="X_hit", 
                 feature_number=12, 
                 feature_name=r"$t$",
                 savename="time_test.png",
                 bins=50,
                #  filepath="/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_gun_240_train/parquet/pf_tree_981.parquet",
                 filepath="/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset/05/pf_tree_1.parquet"
                 )
    return

if __name__ == "__main__":
    main()




