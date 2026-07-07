import awkward as ak


GEN_COL_PDG = 0
GEN_COL_E   = 8

high_energy_muons = []

for file_id in range(2, 31):
    print(f"Processing file {file_id}...")
    PATH = f"/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_validation/05/pf_tree_{file_id}.parquet"

    data = ak.from_parquet(PATH)
    N_events = len(data["X_gen"])
    for evt in range(N_events):
        X_gen    = ak.to_numpy(data["X_gen"][evt])     # (n_particles, n_gen_features)
        X_hit    = ak.to_numpy(data["X_hit"][evt])     # (n_hits,      n_hit_features)
        ygen_hit = ak.to_numpy(data["ygen_hit"][evt])  # (n_hits,)

        n_particles = len(X_gen)

        for particle_idx in range(n_particles):
            if X_gen[particle_idx, GEN_COL_PDG] == 13:
                # print(f"Found muon in file {file_id}, event {evt}, particle index {particle_idx}, energy {X_gen[particle_idx, GEN_COL_E]:.2f} GeV")
                if X_gen[particle_idx, GEN_COL_E]  >= 4:
                    print("^^HIGH ENERGY^^")
                    high_energy_muons.append((file_id, evt, particle_idx, X_gen[particle_idx, GEN_COL_E]))

print("\nSummary of high-energy muons (E >= 4 GeV):")
for file_id, evt, particle_idx, energy in high_energy_muons:
    print(f"File {file_id}, Event {evt}, Particle {particle_idx}, Energy: {energy:.2f} GeV")
                    