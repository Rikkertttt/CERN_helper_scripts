# Rik van Rhee
# This code reads a parquet file and counts the number of particles and antiparticles based on their PDG codes. 
# It then plots the counts in a bar chart.


import awkward as ak
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def read_parquet(file_path):
    # Read the Parquet file into an Awkward Array
    data = ak.from_parquet(file_path)
    
    return data

test_file_path = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_gun_240_train/parquet/pf_tree_981.parquet"

data = read_parquet(test_file_path)

PDGs = None
for i in range(len(data["X_gen"])):
    X_gen = ak.to_numpy(data["X_gen"][i])
    
    if PDGs is None:
        PDGs = X_gen[:,0]
    else:
        PDGs = np.concatenate((PDGs, X_gen[:,0]))


counts = Counter(PDGs)
print(counts)

abs_pdg_to_label = {
    11:   r"$e^{-}$",
    13:   r"$\mu^{-}$",
    22:   r"$\gamma$",
    211:  r"$\pi^{+}$",
    321:  r"$K^{+}$",
    130:  r"$K^{0}_{L}$",
    310:  r"$K^{0}_{S}$",
    2112: r"$n$",
    2212: r"$p$",
    3122: r"$\Lambda$",
    3322: r"$\Xi^{0}$",
}

species_abs_pdgs = sorted({abs(p) for p in counts.keys() if abs(p) in abs_pdg_to_label})

labels = []
particle_counts = []     # PDG > 0
antiparticle_counts = [] # PDG < 0

for apdg in species_abs_pdgs:
    label = abs_pdg_to_label[apdg]
    # particle: +apdg; antiparticle: -apdg
    n_part = counts.get(+apdg, 0)
    n_anti = counts.get(-apdg, 0)

    labels.append(label)
    particle_counts.append(n_part)
    antiparticle_counts.append(n_anti)

x = np.arange(len(labels))

plt.figure(figsize=(8, 4))

# bottom = particle, top = antiparticle
plt.bar(x, particle_counts, color="tab:blue", label="particle (PDG > 0)")
plt.bar(x, antiparticle_counts, bottom=particle_counts, color="tab:orange", label="antiparticle (PDG < 0)")

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Count")
plt.title("Particle / antiparticle counts")
plt.legend()
plt.tight_layout()
plt.savefig("Test.png")