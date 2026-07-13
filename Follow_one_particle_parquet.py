import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# ── Column indices ────────────────────────────────────────────────────────────
HIT_COL_X   = 6
HIT_COL_Y   = 7
HIT_COL_Z   = 8
HIT_COL_T   = 9
HIT_COL_DET = 10

GEN_COL_PDG = 0
GEN_COL_E   = 8

# ── Animation settings ────────────────────────────────────────────────────────
N_FRAMES = 60

# ── Detector colour map ───────────────────────────────────────────────────────
# Maps integer detector ID → (display name, colour).
# Extend this dict to cover all detector IDs present in your data.
DETECTOR_STYLE = {
    1: ("ECAL",    "royalblue"),
    2: ("HCAL",    "tomato"),
    3: ("MUON",    "mediumpurple"),
    4: ("Other",   "mediumseagreen"),
}
DEFAULT_COLOUR = "grey"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_event(filepath: str, evt: int):
    """
    Load a single event from a parquet file.

    Returns:
        X_gen, X_hit, ygen_hit as numpy arrays.
    """
    data = ak.from_parquet(filepath)
    X_gen    = ak.to_numpy(data["X_gen"][evt])
    X_hit    = ak.to_numpy(data["X_hit"][evt])
    ygen_hit = ak.to_numpy(data["ygen_hit"][evt])
    return X_gen, X_hit, ygen_hit


def count_hits_per_particle(ygen_hit: np.ndarray, n_particles: int):
    """
    Count assigned hits per particle, ignoring ygen_hit == -1.

    Returns:
        hit_counts: array of shape (n_particles,)
        valid_mask: boolean mask of assigned hits
    """
    valid_mask = ygen_hit >= 0
    ygen_valid = ygen_hit[valid_mask].astype(int)
    hit_counts = np.bincount(ygen_valid, minlength=n_particles)
    return hit_counts, valid_mask


def det_colours(det_ids: np.ndarray) -> list[str]:
    """Return a list of colours, one per hit, based on detector ID."""
    return [DETECTOR_STYLE.get(int(d), (None, DEFAULT_COLOUR))[1] for d in det_ids]


def det_legend_patches(det_ids: np.ndarray) -> list[mpatches.Patch]:
    """Return legend patches for the detector IDs actually present."""
    patches = []
    for uid in np.unique(det_ids):
        name, colour = DETECTOR_STYLE.get(int(uid), (f"Det {int(uid)}", DEFAULT_COLOUR))
        patches.append(mpatches.Patch(color=colour, label=name))
    return patches


def animate_particle(particle_idx, pdg, energy,
                     hit_x, hit_y, hit_z, hit_t, hit_det,
                     output_path):
    """
    Build a 3D animated GIF showing hits appearing over time for one particle.
    Hits are coloured by detector. hit_* arrays are pre-sorted by time.
    """
    t_min, t_max   = hit_t.min(), hit_t.max()
    pad            = 50  # mm padding around the hit cloud
    all_colours    = det_colours(hit_det)
    legend_patches = det_legend_patches(hit_det)

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()

        ax.set_xlim(hit_x.min() - pad, hit_x.max() + pad)
        ax.set_ylim(hit_y.min() - pad, hit_y.max() + pad)
        ax.set_zlim(hit_z.min() - pad, hit_z.max() + pad)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")

        t_threshold = t_min + (t_max - t_min) * (frame / (N_FRAMES - 1))
        dt_frame    = (t_max - t_min) / N_FRAMES

        mask_past    = hit_t <= t_threshold
        mask_current = mask_past & (hit_t > t_threshold - dt_frame)

        # Past hits — detector colour, faded
        if mask_past.any():
            c_past = [all_colours[i] for i in np.where(mask_past)[0]]
            ax.scatter(
                hit_x[mask_past], hit_y[mask_past], hit_z[mask_past],
                c=c_past, s=15, alpha=0.25,
            )

        # Current hits — detector colour, full opacity + black edge
        if mask_current.any():
            c_curr = [all_colours[i] for i in np.where(mask_current)[0]]
            ax.scatter(
                hit_x[mask_current], hit_y[mask_current], hit_z[mask_current],
                c=c_curr, s=60, alpha=1.0,
                edgecolors='black', linewidths=0.5,
            )

        ax.legend(handles=legend_patches, loc="upper left", fontsize=8, title="Detector")
        ax.set_title(
            f"Particle {particle_idx} (PDG={pdg}, E={energy:.3f} GeV)\n"
            f"t ≤ {t_threshold:.3f} ns  ({mask_past.sum()}/{len(hit_t)} hits)"
        )

    ani = animation.FuncAnimation(
        fig, update,
        frames   = N_FRAMES,
        interval = 100,
        repeat   = True,
    )

    ani.save(output_path, writer="pillow", fps=15)
    plt.close(fig)


def save_static_frame(particle_idx, pdg, energy,
                      hit_x, hit_y, hit_z, hit_t, hit_det,
                      output_path):
    """
    Save a static 3D scatter PNG when all hits share the same timestamp.
    Hits are coloured by detector.
    """
    n_hits = len(hit_t)
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(hit_x, hit_y, hit_z, c=det_colours(hit_det), s=20, alpha=0.7)
    ax.legend(handles=det_legend_patches(hit_det),
              loc="upper left", fontsize=8, title="Detector")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(
        f"Particle {particle_idx} (PDG={pdg}, E={energy:.3f} GeV)\n"
        f"{n_hits} hits at t={hit_t[0]:.4f} ns"
    )
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def printout(file_id, evt, particle_idx, pdg, energy, hit_x, hit_y, hit_z, hit_t, hit_det, output_dir="Printout"):
    '''
    Print out the hit information to a text file
    '''
    out = f"{output_dir}/file_{file_id}_event_{evt}_particle_{particle_idx:03d}.txt"

    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(out, 'w') as f:
        f.write(f"Particle {particle_idx} (PDG={pdg}, E={energy:.3f} GeV)\n")
        f.write(f"{len(hit_t)} hits:\n")
        f.write(f"{'Hit #':<8}{'x [mm]':<12}{'y [mm]':<12}{'z [mm]':<12}{'t [ns]':<12}{'detector ID':<12}\n")
        for i in range(len(hit_t)):
            f.write(f"{i:<8}{hit_x[i]:<12.3f}{hit_y[i]:<12.3f}{hit_z[i]:<12.3f}{hit_t[i]:<12.4f}{hit_det[i]:<12}\n")

        print(f"  Printed hit information to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    FILE_ID = 179
    ONLY_MUON = True
    PRINTOUT = True

    # path       = f"/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_validation/05/pf_tree_{FILE_ID}.parquet"
    # path = f"/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_uds_CLD_o2_v05_eval_v1/05/pf_tree_10{FILE_ID}.parquet"
    path = f"/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_muontimetest/05/pf_tree_2.parquet"
    output_dir = "Parquet_particle_plots"
    printout_dir = "Printout"

    os.makedirs(output_dir, exist_ok=True)


    for evt in [16]:

        print(f"\nProcessing event {evt}...")

        X_gen, X_hit, ygen_hit = load_event(path, evt)
        n_particles = len(X_gen)

        hit_counts, valid_mask = count_hits_per_particle(ygen_hit, n_particles)

        print(f"Event {evt}: {n_particles} particles, {len(X_hit)} total hits")
        print(f"  Unassigned hits (ygen_hit == -1): {(~valid_mask).sum()}")
        print(f"  Assigned hits:                    {valid_mask.sum()}")
        print(f"  Particles with hits: {(hit_counts > 0).sum()} / {n_particles}")
        print(f"  Saving to: {output_dir}/\n")

        for particle_idx in range(n_particles):

            pdg    = int(X_gen[particle_idx, GEN_COL_PDG])
            energy = float(X_gen[particle_idx, GEN_COL_E])

            if energy <= 3.5 and pdg != 13:  # Skip low-energy particles, except muons
                print(f"  Particle {particle_idx:3d}: low energy ({energy:.3f} GeV), skipping")
                continue

            if ONLY_MUON and pdg != 13:
                continue

            if pdg == 13:
                print(f"Found muon!")

            n_hits = hit_counts[particle_idx]

            if n_hits == 0:
                print(f"  Particle {particle_idx:3d}: no hits, skipping")
                continue

            print(f"  Particle {particle_idx:3d} (PDG={pdg}, E={energy:.3f} GeV): "
                  f"{n_hits} hits — animating...", end=" ", flush=True)

            mask    = (ygen_hit == particle_idx)
            hit_x   = X_hit[mask, HIT_COL_X]
            hit_y   = X_hit[mask, HIT_COL_Y]
            hit_z   = X_hit[mask, HIT_COL_Z]
            hit_t   = X_hit[mask, HIT_COL_T]
            hit_det = X_hit[mask, HIT_COL_DET]

            order   = np.argsort(hit_t)
            hit_x   = hit_x[order]
            hit_y   = hit_y[order]
            hit_z   = hit_z[order]
            hit_t   = hit_t[order]
            hit_det = hit_det[order]

            if hit_t.min() == hit_t.max():
                print(f"all hits at same time ({hit_t[0]:.4f} ns), saving static frame")
                out = os.path.join(output_dir, f"file_{FILE_ID}_event_{evt}_particle_{particle_idx:03d}.png")
                save_static_frame(particle_idx, pdg, energy,
                                  hit_x, hit_y, hit_z, hit_t, hit_det, out)
            else:
                out = os.path.join(output_dir, f"file_{FILE_ID}_event_{evt}_particle_{particle_idx:03d}.gif")
                animate_particle(particle_idx, pdg, energy,
                                 hit_x, hit_y, hit_z, hit_t, hit_det, out)

            print(f"saved {out}")

            if PRINTOUT:
                printout(FILE_ID, evt, particle_idx, pdg, energy,
                         hit_x, hit_y, hit_z, hit_t, hit_det, output_dir=os.path.join(output_dir, printout_dir))

    print(f"\nDone. All files saved in '{output_dir}/'")


if __name__ == "__main__":
    main()
