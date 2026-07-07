import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── 1. Load the file ──────────────────────────────────────────────────────────
PATH = "/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_validation/root_files/pf_tree_3.edm4hep.root"
# PATH = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_uds_clustering_dataset_3/root_files/pf_tree_99.edm4hep.root"

data = uproot.open(PATH)["events"].arrays(library="ak")

# ── 2. Settings ───────────────────────────────────────────────────────────────
EVT          = 0     # which event to look at
PARTICLE_IDX = 36     # which MC particle to plot (run the diagnostic first!)
N_FRAMES     = 60    # number of animation frames (more = smoother but slower)
OUTPUT_FILE  = "single_particle_hits_3d.gif"

# ── 3. CollectionID → detector mapping (from our earlier diagnostic) ──────────
# Each collectionID maps to: (detector name, x array, y array, z array, time array, colour)
det_map = {
    1265367388: ("ECALBarrel", "royalblue"),
    2257282296: ("ECALEndcap", "cornflowerblue"),
    124629188:  ("HCALBarrel", "tomato"),
    1286131593: ("HCALEndcap", "salmon"),
    1000433565: ("HCALOther",  "gold"),
}

# Position and time arrays per detector
det_arrays = {
    "ECALBarrel": {
        "x": np.array(data["ECALBarrel.position.x"][EVT]),
        "y": np.array(data["ECALBarrel.position.y"][EVT]),
        "z": np.array(data["ECALBarrel.position.z"][EVT]),
        "t": np.array(data["ECALBarrel.time"][EVT]),
    },
    "ECALEndcap": {
        "x": np.array(data["ECALEndcap.position.x"][EVT]),
        "y": np.array(data["ECALEndcap.position.y"][EVT]),
        "z": np.array(data["ECALEndcap.position.z"][EVT]),
        "t": np.array(data["ECALEndcap.time"][EVT]),
    },
    "HCALBarrel": {
        "x": np.array(data["HCALBarrel.position.x"][EVT]),
        "y": np.array(data["HCALBarrel.position.y"][EVT]),
        "z": np.array(data["HCALBarrel.position.z"][EVT]),
        "t": np.array(data["HCALBarrel.time"][EVT]),
    },
    "HCALEndcap": {
        "x": np.array(data["HCALEndcap.position.x"][EVT]),
        "y": np.array(data["HCALEndcap.position.y"][EVT]),
        "z": np.array(data["HCALEndcap.position.z"][EVT]),
        "t": np.array(data["HCALEndcap.time"][EVT]),
    },
    "HCALOther": {
        "x": np.array(data["HCALOther.position.x"][EVT]),
        "y": np.array(data["HCALOther.position.y"][EVT]),
        "z": np.array(data["HCALOther.position.z"][EVT]),
        "t": np.array(data["HCALOther.time"][EVT]),
    },
}

# ── 4. Load the link table and collect hits for our chosen particle ───────────
link_from_idx    = np.array(data["_CalohitMCTruthLink_from.index"][EVT])
link_from_collID = np.array(data["_CalohitMCTruthLink_from.collectionID"][EVT])
link_to_idx      = np.array(data["_CalohitMCTruthLink_to.index"][EVT])

# We will store each hit as (x, y, z, t, det_name, colour)
hits = []  # list of dicts

for i in range(len(link_from_idx)):
    mc_idx   = int(link_to_idx[i])
    reco_idx = int(link_from_idx[i])
    coll_id  = int(link_from_collID[i])

    # Only keep hits linked to our chosen particle
    if mc_idx != PARTICLE_IDX:
        continue

    # Look up which detector this hit belongs to
    if coll_id not in det_map:
        continue

    det_name, colour = det_map[coll_id]
    arr = det_arrays[det_name]

    # Safety check: reco_idx must be within bounds
    if reco_idx >= len(arr["x"]):
        continue

    hits.append({
        "x":      arr["x"][reco_idx],
        "y":      arr["y"][reco_idx],
        "z":      arr["z"][reco_idx],
        "t":      arr["t"][reco_idx],
        "det":    det_name,
        "colour": colour,
    })

print(f"Found {len(hits)} hits for particle idx={PARTICLE_IDX}")
print(f"PDG ID of particle: {data['MCParticles.PDG'][EVT][PARTICLE_IDX]}")

if len(hits) == 0:
    raise RuntimeError("No hits found — check PARTICLE_IDX and collectionID mapping.")

# ── 5. Sort hits by time so the animation plays in time order ─────────────────
hits.sort(key=lambda h: h["t"])

all_x = np.array([h["x"] for h in hits])
all_y = np.array([h["y"] for h in hits])
all_z = np.array([h["z"] for h in hits])
all_t = np.array([h["t"] for h in hits])
all_c = [h["colour"] for h in hits]

t_min, t_max = all_t.min(), all_t.max()

# ── 6. Set up the figure ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), dpi=120)
ax  = fig.add_subplot(111, projection='3d')

# Fix axis limits so the view doesn't jump between frames
pad = 50  # mm padding around hits
ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
ax.set_zlim(all_z.min() - pad, all_z.max() + pad)
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")

# Build a legend from unique detectors seen in the hits
seen_dets = {}
for h in hits:
    if h["det"] not in seen_dets:
        seen_dets[h["det"]] = h["colour"]
for det_name, colour in seen_dets.items():
    ax.scatter([], [], [], c=colour, s=30, label=det_name)
ax.legend(loc="upper left", fontsize=8)

# Time label in the title
time_text = ax.set_title("")

# ── 7. Animation update function ──────────────────────────────────────────────
# Each frame shows all hits up to a certain time threshold (cumulative).
# This gives a "hits appearing over time" effect.

def update(frame):
    ax.cla()  # clear axes each frame to redraw cleanly

    # Restore fixed limits and labels after cla()
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_zlim(all_z.min() - pad, all_z.max() + pad)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    # Time threshold for this frame: linearly interpolated from t_min to t_max
    t_threshold = t_min + (t_max - t_min) * (frame / (N_FRAMES - 1))

    # Mask: only hits up to current time threshold
    mask_past    = all_t <= t_threshold   # hits already shown (full colour)
    mask_current = (all_t > t_threshold - (t_max - t_min) / N_FRAMES) & mask_past  # newest hits (highlighted)

    # Plot past hits (faded)
    if mask_past.any():
        ax.scatter(
            all_x[mask_past], all_y[mask_past], all_z[mask_past],
            c     = [all_c[i] for i in np.where(mask_past)[0]],
            s     = 15,
            alpha = 0.3,
        )

    # Plot newest hits (bright, larger)
    if mask_current.any():
        ax.scatter(
            all_x[mask_current], all_y[mask_current], all_z[mask_current],
            c     = [all_c[i] for i in np.where(mask_current)[0]],
            s     = 60,
            alpha = 1.0,
            edgecolors = "black",
            linewidths = 0.5,
        )

    # Rebuild legend
    for det_name, colour in seen_dets.items():
        ax.scatter([], [], [], c=colour, s=30, label=det_name)
    ax.legend(loc="upper left", fontsize=8)

    ax.set_title(
        f"Particle idx={PARTICLE_IDX} — t ≤ {t_threshold:.3f} ns  "
        f"({mask_past.sum()}/{len(hits)} hits)"
    )

# ── 8. Build and save the animation ───────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    frames   = N_FRAMES,
    interval = 100,   # ms between frames
    repeat   = True,
)

ani.save(OUTPUT_FILE, writer="pillow", fps=15)
print(f"Saved animation to {OUTPUT_FILE}")
plt.show()
