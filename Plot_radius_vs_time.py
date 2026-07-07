import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

CUTOFF = 20 #EVT cutoff

# ── 1. Load the parquet file ──────────────────────────────────────────────────
PATH = "/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_validation/05/pf_tree_3.parquet"


# ── 2. Column indices ──────────────────────────────────────────────────

HIT_COL_X = 6
HIT_COL_Y = 7
HIT_COL_Z = 8
HIT_COL_T = 9

GEN_COL_PDG = 0
GEN_COL_E   = 8


data = ak.from_parquet(PATH)

for EVT in range(min(CUTOFF, len(data["X_gen"]))):

    X_gen    = ak.to_numpy(data["X_gen"][EVT])     # (n_particles, n_gen_features)
    X_hit    = ak.to_numpy(data["X_hit"][EVT])     # (n_hits,      n_hit_features)
    ygen_hit = ak.to_numpy(data["ygen_hit"][EVT])  # (n_hits,)

    for PARTICLE_IDX in range(len(X_gen)):
        if X_gen[PARTICLE_IDX, GEN_COL_E] <= 3.5:
            continue  # Skip low-energy particles

        particle_mask = ygen_hit == PARTICLE_IDX
        hit_x = X_hit[particle_mask, HIT_COL_X]
        hit_y = X_hit[particle_mask, HIT_COL_Y]
        hit_z = X_hit[particle_mask, HIT_COL_Z]
        hit_t = X_hit[particle_mask, HIT_COL_T]

        hit_r = np.sqrt(hit_x**2 + hit_y**2 + hit_z**2)

        # ── 5. Plot radius vs time ──────────────────────────────────────────
        
        print(f"Event {EVT}, Particle {PARTICLE_IDX}: {len(hit_x)} hits, Energy: {X_gen[PARTICLE_IDX, GEN_COL_E]:.2f} GeV")

        plt.figure(figsize=(8, 6))
        plt.scatter(hit_t, hit_r, c=hit_t, cmap='viridis', s=10)
        plt.title(f'Particle {PARTICLE_IDX}, PDG: {X_gen[PARTICLE_IDX, GEN_COL_PDG]}, Hits: {len(hit_x)}, Energy: {X_gen[PARTICLE_IDX, GEN_COL_E]:.2f} GeV')
        plt.xlabel('Time [ns]')
        plt.ylabel('Radius [mm]')
        plt.grid()
        # Save the plot
        output_dir = "Particle_radius_vs_time_plots"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'event_{EVT}_particle_{PARTICLE_IDX}_radius_vs_time.png')
        plt.savefig(output_path)
        plt.close()