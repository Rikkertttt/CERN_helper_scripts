import uproot
import awkward as ak
import numpy as np
from collections import defaultdict

# ── 1. Load the file ──────────────────────────────────────────────────────────
# PATH = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_uds_clustering_dataset_3/root_files/pf_tree_99.edm4hep.root"
PATH = "/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_validation/root_files/pf_tree_3.edm4hep.root"

data = uproot.open(PATH)["events"].arrays(library="ak")

# ── 2. Settings ───────────────────────────────────────────────────────────────
EVT = 0

# ── 3. Load MC particle info ──────────────────────────────────────────────────
pdg      = np.array(data["MCParticles.PDG"][EVT])
status   = np.array(data["MCParticles.generatorStatus"][EVT])
sim_stat = np.array(data["MCParticles.simulatorStatus"][EVT])  # set by Geant4
px       = np.array(data["MCParticles.momentum.x"][EVT])
py       = np.array(data["MCParticles.momentum.y"][EVT])
pz       = np.array(data["MCParticles.momentum.z"][EVT])
mass     = np.array(data["MCParticles.mass"][EVT])
mc_time  = np.array(data["MCParticles.time"][EVT])
vtx_x    = np.array(data["MCParticles.vertex.x"][EVT])  # production vertex
vtx_y    = np.array(data["MCParticles.vertex.y"][EVT])
vtx_z    = np.array(data["MCParticles.vertex.z"][EVT])

# Compute momentum magnitude and energy
p_mag  = np.sqrt(px**2 + py**2 + pz**2)
energy = np.sqrt(p_mag**2 + mass**2)

n_particles = len(pdg)

# ── 4. Count how many hits each MC particle has via the link table ────────────
link_to_idx = np.array(data["_CalohitMCTruthLink_to.index"][EVT])

# Count hits per particle index
hits_per_particle = defaultdict(int)
for mc_idx in link_to_idx:
    hits_per_particle[int(mc_idx)] += 1

# ── 5. Print summary table for ALL particles ──────────────────────────────────
print(f"\nAll MCParticles in event {EVT}  ({n_particles} total)")
print(f"{'idx':>4}  {'PDG':>8}  {'genStat':>7}  {'simStat':>7}  "
      f"{'E [GeV]':>8}  {'p [GeV]':>8}  {'vtx_r [mm]':>10}  {'vtx_z [mm]':>10}  {'n_hits':>6}")
print("─" * 90)

for i in range(n_particles):
    vtx_r  = np.sqrt(vtx_x[i]**2 + vtx_y[i]**2)  # transverse radius of production vertex
    n_hits = hits_per_particle.get(i, 0)

    # Highlight particles with no hits
    flag = "  ← NO HITS" if n_hits == 0 else ""

    print(f"{i:>4}  {int(pdg[i]):>8}  {int(status[i]):>7}  {int(sim_stat[i]):>7}  "
          f"{energy[i]:>8.3f}  {p_mag[i]:>8.3f}  {vtx_r:>10.1f}  {vtx_z[i]:>10.1f}  "
          f"{n_hits:>6}{flag}")

# ── 6. Focused summary: only particles with no hits ───────────────────────────
no_hit_mask = np.array([hits_per_particle.get(i, 0) == 0 for i in range(n_particles)])

print(f"\n{'─'*90}")
print(f"Particles with NO hits: {no_hit_mask.sum()} / {n_particles}")
print(f"{'─'*90}")

if no_hit_mask.any():
    print(f"\n{'idx':>4}  {'PDG':>8}  {'genStat':>7}  {'simStat':>7}  "
          f"{'E [GeV]':>8}  {'p [GeV]':>8}  {'vtx_r [mm]':>10}  {'vtx_z [mm]':>10}")
    print("─" * 80)
    for i in np.where(no_hit_mask)[0]:
        vtx_r = np.sqrt(vtx_x[i]**2 + vtx_y[i]**2)
        print(f"{i:>4}  {int(pdg[i]):>8}  {int(status[i]):>7}  {int(sim_stat[i]):>7}  "
              f"{energy[i]:>8.3f}  {p_mag[i]:>8.3f}  {vtx_r:>10.1f}  {vtx_z[i]:>10.1f}")

# ── 7. Statistics broken down by generatorStatus ─────────────────────────────
print(f"\n{'─'*90}")
print("Hit statistics by generatorStatus:")
print(f"{'genStatus':>10}  {'n_particles':>12}  {'n_with_hits':>12}  {'n_no_hits':>10}")
print("─" * 50)
for gs in sorted(set(status)):
    mask_gs       = status == gs
    n_total_gs    = mask_gs.sum()
    n_with_hits   = sum(1 for i in np.where(mask_gs)[0] if hits_per_particle.get(i, 0) > 0)
    n_no_hits_gs  = n_total_gs - n_with_hits
    print(f"{int(gs):>10}  {n_total_gs:>12}  {n_with_hits:>12}  {n_no_hits_gs:>10}")
