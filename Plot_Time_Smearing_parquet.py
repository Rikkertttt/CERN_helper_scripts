import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import os

# Column indices ────────────────────────────────────────────────────────────
HIT_COL_T  = 9

# Smearing indices
SMEARING = {
    0.1: ("0.1ns", 12),
    1: ("1ns", 13),
    10: ("10ns", 14),
}

HIT_COL_DET = 10

GEN_PDG_COL = 0

DETECTOR_STYLE = {
    1: "ECAL",
    2: "HCAL",
    3: "MUON",
    4: "Other",
}

# Helper functions ───────────────────────────────────────────────────────────

def load_events(filepath: str, n_events: int = None):
    """
    Load multiple events from a parquet file and concatenate hits.
    
    Args:
        filepath: Path to the parquet file.
        n_events: Number of events to load. If None, loads all.
    
    Returns:
        X_hit as a single concatenated numpy array.
    """
    data = ak.from_parquet(filepath)

    # Load all hits and concatenate them into a single array
    X_hit_all = data["X_hit"]  # shape: (n_events, var)
    ygen_hit = data["ygen_hit"]  # shape: (n_events, var)
    X_gen_all = data["X_gen"]  # shape: (n_events, var)

    # If n_events is specified, slice the arrays to only include the first n_events
    if n_events is not None:
        X_hit_all = X_hit_all[:n_events]
        ygen_hit = ygen_hit[:n_events]
        X_gen_all = X_gen_all[:n_events]

    # Concatenate all hits into a single array
    X_hit = np.concatenate([ak.to_numpy(X_hit_all[evt]) for evt in range(len(X_hit_all))])

    # Get the corresponding PDG IDs for the hits
    pdg_per_hit = np.concatenate([
        ak.to_numpy(X_gen_all[evt][ak.to_numpy(ygen_hit[evt])])[:, GEN_PDG_COL]
        for evt in range(len(ygen_hit))
    ])

    return X_hit, pdg_per_hit


def add_stats_annotation(ax, data, x=0.97, y=0.95):
    """
    Adds a mean and RMS text box to the upper-right corner of an axes.

    Args:
        ax:   The matplotlib Axes object.
        data: 1-D array-like of values.
        x, y: Axes-fraction coordinates for the text box anchor.
    """
    mean = np.mean(data)
    rms  = np.sqrt(np.mean(data ** 2))
    textstr = f"Mean: {mean:.4f}\nRMS:  {rms:.4f}"
    ax.text(
        x, y, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='grey'),
    )

def plot_smearing(X_hit, smearing_ns, detector_ids, pdgs = None, pdg_per_hit = None, output_dir = "Testing_pics/smearing_pics", fit_gaussian=False):
    """
    Plot the time smearing distribution for a given event.

    Args:
        X_hit: numpy array of hit data.
        smearing_ns: list of smearing values in nanoseconds.
        detector_ids: list of detector IDs corresponding to hits.
        pdgs: list of particle IDs to include in the plot.
        pdg_per_hit: numpy array of PDG IDs corresponding to hits.
        output_dir: directory to save the plots.
        fit_gaussian: whether to fit a Gaussian to the smearing distribution.
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make sure smearing_ns and detector_ids are lists
    if isinstance(smearing_ns, int):
        smearing_ns = [smearing_ns]
    if isinstance(detector_ids, int):
        detector_ids = [detector_ids]

    for smearing in smearing_ns:
        print(f"Plotting {smearing}ns smearing.")
        # Check if the smearing value is valid
        if smearing not in SMEARING:
            print(f"Smearing value {smearing}ns not recognized. Skipping.")
            continue

        smearing_col = SMEARING[smearing][1] # The smearing index in the X_hit array
        hit_times = X_hit[:, HIT_COL_T]
        smeared_times = X_hit[:, smearing_col]

        fig, axes = plt.subplots(len(detector_ids), 3, figsize=(15, 5 * len(detector_ids)), dpi=150)

        for i in range(len(detector_ids)):
            
            det_id = detector_ids[i]

            # Check if the detector ID is valid
            if det_id not in DETECTOR_STYLE:
                print(f"Detector ID {det_id} not recognized. Skipping.")
                continue

            print(f"Plotting for detector ID {det_id} ({DETECTOR_STYLE[det_id]}).")

            # Create a mask for the current detector ID
            det_mask = X_hit[:, HIT_COL_DET] == det_id
            det_hit_times = hit_times[det_mask]
            det_smeared_times = smeared_times[det_mask]


            # Create a pdg mask if pdgs are specified
            if pdgs is not None:
                if pdg_per_hit is None:
                    raise ValueError("pdg_per_hit must be provided if pdgs are specified.")
                pdg_mask = np.isin(np.abs(pdg_per_hit[det_mask]), np.abs(pdgs))
                det_hit_times = det_hit_times[pdg_mask]
                det_smeared_times = det_smeared_times[pdg_mask]

            axes[i, 0].hist(det_hit_times, bins=50, label='Original Hit Times', color='blue')
            axes[i, 0].set_title(f"Detector: {DETECTOR_STYLE.get(det_id, f'Det {det_id}')}. Original Hit Times")
            axes[i, 0].set_xlabel('Time (ns)')
            axes[i, 0].set_ylabel('Counts')
            add_stats_annotation(axes[i, 0], det_hit_times)

            axes[i, 1].hist(det_smeared_times, bins=50, label=f'Smeared Hit Times ({smearing}ns)', color='orange')
            axes[i, 1].set_title(f"Detector: {DETECTOR_STYLE.get(det_id, f'Det {det_id}')}. Smeared Hit Times")
            axes[i, 1].set_xlabel('Time (ns)')
            axes[i, 1].set_ylabel('Counts')
            add_stats_annotation(axes[i, 1], det_smeared_times)

            axes[i, 2].hist(det_smeared_times - det_hit_times, bins=50, label='Time Smearing', color='green')
            axes[i, 2].set_title(f"Detector: {DETECTOR_STYLE.get(det_id, f'Det {det_id}')}. Time Smearing")
            axes[i, 2].set_xlabel(r'\Delta Time (ns)')
            axes[i, 2].set_ylabel('Counts')
            add_stats_annotation(axes[i, 2], det_smeared_times - det_hit_times)

            if fit_gaussian:
                from scipy.optimize import curve_fit

                def gaussian(x, amp, mean, stddev):
                    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

                delta_times = det_smeared_times - det_hit_times
                counts, bin_edges = np.histogram(delta_times, bins=50)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                try:
                    popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[counts.max(), delta_times.mean(), delta_times.std()])
                    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 100)
                    y_fit = gaussian(x_fit, *popt)
                    axes[i, 2].plot(x_fit, y_fit, color='red', linestyle='--', label='Gaussian Fit')
                    axes[i, 2].legend()
                except Exception as e:
                    print(f"Could not fit Gaussian for detector {det_id}: {e}")
        
        # Set the overall title for the figure
        if pdgs is not None:
            pdg_str = ', '.join(str(pdg) for pdg in pdgs)
            fig.suptitle(f"Time Smearing Analysis for {smearing}ns (PDGs: {pdg_str})", fontsize=16, y=0.98)
        else:
            fig.suptitle(f"Time Smearing Analysis for {smearing}ns", fontsize=16, y=0.98)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Set the output path for saving the figure
        if pdgs is not None:
            pdg_str = '_'.join(str(pdg) for pdg in pdgs)
            output_path = os.path.join(output_dir, f'time_smearing_{smearing}ns_pdgs_{pdg_str}.png')
        else:
            output_path = os.path.join(output_dir, f'time_smearing_{smearing}ns.png')


        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    # Example usage
    filepath = "/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_mumu_clustering_dataset_validation/05/pf_tree_5.parquet"
    n_events = 50  # Number of events to load
    X_hit, pdg_per_hit = load_events(filepath, n_events)
    smearing_ns = [1, 10]  # List of smearing values to plot
    detector_ids = [1, 2, 3]  # List of detector IDs to plot

    pdgs = [11, 13, 22, 211]

    for i in pdgs:
        plot_smearing(X_hit, smearing_ns, detector_ids, pdgs=[i], pdg_per_hit=pdg_per_hit, fit_gaussian=True)