# Rik van Rhee
# Code for plotting the time smearing distribution from a ROOT file.


import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


def open_rootfile(filepath: str, print_keys: bool = False):
    """
    Opens a ROOT file and returns a summary of its contents.

    Args:
        filepath: Path to the .root file.
        print_keys: If True, prints the keys of the ROOT file.

    Returns:
        The opened uproot file object.
    """
    f = uproot.open(filepath)
    print(f"Opened: {filepath}")
    if print_keys:
        print(f"Keys: {f.keys()}")
    return f


def _add_stats_annotation(ax, data, x=0.97, y=0.95):
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


def plot_truth_vs_hit_time(rootfile):
    """
    Plots the distribution of truth time vs hit time from the given ROOT file.

    Args:
        rootfile: The opened uproot file object.
    """
    hit_time       = ak.flatten(rootfile["events"]["ECALBarrel.time"].array())
    hit_truth_time = ak.flatten(rootfile["events"]["ECalBarrelCollectionContributions.time"].array())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axes[0].hist(hit_truth_time, bins=50, color='orange', alpha=0.8)
    axes[0].set_title('Truth Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Counts')
    _add_stats_annotation(axes[0], np.asarray(hit_truth_time))

    axes[1].hist(hit_time, bins=50, color='blue', alpha=0.8)
    axes[1].set_title('Hit Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Counts')
    _add_stats_annotation(axes[1], np.asarray(hit_time))

    fig.tight_layout()
    plt.savefig("hit_vs_truth_time.png", bbox_inches='tight')


def plot_smearing(filename, fit_gaussian=False, which_detector="all", output_file="ecal_time_smearing.png"):
    """
    Plots the time smearing distribution from the given PODIO ROOT file.
    Supports a single detector name, a list of detector names, or "all".

    Args:
        filename:       Path to the .root file.
        fit_gaussian:   Whether to fit a Gaussian to the smearing distribution.
        which_detector: Reco collection name (e.g. "ECALBarrel"), a list of names
                        (e.g. ["ECALBarrel", "HCALBarrel"]), or "all".
        output_file:    The name of the output file.
    """
    from podio import root_io
    from scipy.optimize import curve_fit

    reader = root_io.Reader(filename)

    # Normalise input to a list of detector names (or ["all"])
    if isinstance(which_detector, str):
        detectors = [which_detector]
    else:
        detectors = list(which_detector)

    # Collect matched times per detector: {det_name: {"reco": [], "truth": []}}
    data = {det: {"reco": [], "truth": []} for det in detectors}

    for event in reader.get("events"):
        calo_links = event.get("CalohitMCTruthLink")

        # Build cellID sets once per event, per detector
        cellid_sets = {}
        for det in detectors:
            if det != "all":
                det_collection      = event.get(det)
                cellid_sets[det]    = set(hit.getCellID() for hit in det_collection)

        for link in calo_links:
            reco_hit    = link.getFrom()
            mc_particle = link.getTo()

            try:
                rt = reco_hit.getTime()
                tt = mc_particle.getTime()
            except Exception:
                continue

            cell_id = reco_hit.getCellID()

            for det in detectors:
                if det != "all" and cell_id not in cellid_sets[det]:
                    continue
                data[det]["reco"].append(rt)
                data[det]["truth"].append(tt)

    # --- Plot: (n_detectors, 3) grid ---
    n_det = len(detectors)
    fig, axes = plt.subplots(n_det, 3, figsize=(18, 5 * n_det), dpi=150)

    # Ensure axes is always 2D
    if n_det == 1:
        axes = axes[np.newaxis, :]

    for row, det in enumerate(detectors):
        reco_t  = np.array(data[det]["reco"])
        truth_t = np.array(data[det]["truth"])
        delta_t = reco_t - truth_t

        if len(delta_t) == 0:
            print(f"Warning: no matched hits for detector '{det}', skipping row.")
            continue

        t_min = min(reco_t.min(), truth_t.min())
        t_max = max(reco_t.max(), truth_t.max())

        axes[row, 0].hist(truth_t, bins=100, range=(t_min, t_max), color='orange', alpha=0.8)
        axes[row, 0].set_title(f'Truth Time ({det})')
        axes[row, 0].set_xlabel('Time [ns]')
        axes[row, 0].set_ylabel('Counts')
        _add_stats_annotation(axes[row, 0], truth_t)

        axes[row, 1].hist(reco_t, bins=100, range=(t_min, t_max), color='blue', alpha=0.8)
        axes[row, 1].set_title(f'Reco Hit Time ({det})')
        axes[row, 1].set_xlabel('Time [ns]')
        axes[row, 1].set_ylabel('Counts')
        _add_stats_annotation(axes[row, 1], reco_t)

        counts, bin_edges, _ = axes[row, 2].hist(delta_t, bins=100, color='purple', alpha=0.8)
        axes[row, 2].axvline(0, color='black', linewidth=1, linestyle='--')
        axes[row, 2].set_title(f'Smearing: Reco − Truth Time ({det})')
        axes[row, 2].set_xlabel('Δt [ns]')
        axes[row, 2].set_ylabel('Counts')
        _add_stats_annotation(axes[row, 2], delta_t)

        if fit_gaussian:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            def gaussian(x, amp, mean, stddev):
                return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

            try:
                popt, _ = curve_fit(
                    gaussian, bin_centers, counts,
                    p0=[counts.max(), delta_t.mean(), delta_t.std()]
                )
                mean_fit, stddev_fit = popt[1], popt[2]
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
                axes[row, 2].plot(x_fit, gaussian(x_fit, *popt), color='red', linewidth=2,
                                  label=f'Fit: μ={mean_fit:.3f} ns, σ={abs(stddev_fit):.3f} ns')
                axes[row, 2].legend()
                print(f"[{det}] Gaussian μ: {mean_fit:.4f} ns,  σ: {abs(stddev_fit):.4f} ns")
            except Exception as e:
                print(f"[{det}] Gaussian fit failed: {e}")

        print(f"[{det}] Matched hits: {len(delta_t)},  Mean Δt: {delta_t.mean():.4f} ns,  Std Δt: {delta_t.std():.4f} ns")

    fig.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    PATH = "/afs/cern.ch/work/p/pvanrhee/private/HitPF_datageneration/save/CLD/train/Z_uds_clustering_dataset_validation/root_files/pf_tree_2.edm4hep.root"
    f = open_rootfile(PATH)

    plot_smearing(PATH, which_detector=["ECALBarrel", "HCALBarrel", "ECALEndcap"])