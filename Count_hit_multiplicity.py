#!/usr/bin/env python
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt


def find_events_tree(file):
    if "events" in file:
        return file["events"]
    for key, obj in file.items():
        if hasattr(obj, "keys"):
            print(f"WARNING: using tree '{key}' (no 'events' found)")
            return obj
    raise RuntimeError("No TTree found in file")


def find_calo_hit_link_collection(tree):
    candidates = [
        "CalohitMCTruthLink",
        "CaloHitMCTruthLink",
        "CalorimeterHitMCTruthLink",
    ]
    branches = set(tree.keys())
    for name in candidates:
        # branch can appear as "Name.weight" or "Name/Name.weight"
        if f"{name}.weight" in branches or f"{name}/{name}.weight" in branches:
            print(f"Using calorimeter hit MC truth link collection: {name}")
            return name
    raise RuntimeError(
        f"Could not find a calorimeter-hit truth link collection. "
        f"Tried: {candidates}\n"
        f"Branches containing 'link' or 'Link':\n" +
        "\n".join(f"  {b}" for b in sorted(branches) if "link" in b.lower())
    )


def contributors_per_hit_for_event(from_colid, from_idx, to_idx):
    if len(from_idx) == 0:
        return np.array([], dtype=int)

    hit_pairs = np.stack([np.asarray(from_colid), np.asarray(from_idx)], axis=1)
    _, inv_hit = np.unique(hit_pairs, axis=0, return_inverse=True)

    hit_ids = inv_hit
    gen_ids = np.asarray(to_idx)

    if len(hit_ids) == 0:
        return np.array([], dtype=int)

    # sort by (hit, gen) to find unique (hit, gen) pairs
    order = np.lexsort((gen_ids, hit_ids))
    hit_sorted = hit_ids[order]
    gen_sorted = gen_ids[order]

    same_hit = np.diff(hit_sorted) == 0
    same_gen = np.diff(gen_sorted) == 0
    same_pair_as_prev = same_hit & same_gen
    is_unique_pair = np.concatenate([[True], ~same_pair_as_prev])

    unique_hit_for_pair = hit_sorted[is_unique_pair]
    _, counts = np.unique(unique_hit_for_pair, return_counts=True)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Histogram: number of MC particles contributing to each calorimeter hit"
    )
    parser.add_argument("input_root", help="Input EDM4hep ROOT file")
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Max number of events to process (default: all)"
    )
    args = parser.parse_args()

    with uproot.open(args.input_root) as f:
        tree = find_events_tree(f)
        link_name = find_calo_hit_link_collection(tree)

        from_colid_all = tree[f"_{link_name}_from/_{link_name}_from.collectionID"].array()
        from_idx_all   = tree[f"_{link_name}_from/_{link_name}_from.index"].array()
        to_idx_all     = tree[f"_{link_name}_to/_{link_name}_to.index"].array()

        n_events_file = len(from_colid_all)
        n_events = n_events_file if args.max_events is None else min(n_events_file, args.max_events)
        print(f"Events in file: {n_events_file} — processing: {n_events}")

        all_counts = []
        for ievt in range(n_events):
            contrib = contributors_per_hit_for_event(
                from_colid_all[ievt], from_idx_all[ievt], to_idx_all[ievt]
            )
            all_counts.append(contrib)

    contributors_per_hit = np.concatenate(all_counts)
    if contributors_per_hit.size == 0:
        print("No linked hits found. Nothing to plot.")
        return

    # ---- summary ----
    total = len(contributors_per_hit)
    max_contributors = int(contributors_per_hit.max())
    print(f"\nTotal hits with at least one MC contributor: {total}")
    for n in range(1, min(max_contributors, 10) + 1):
        count = int(np.sum(contributors_per_hit == n))
        print(f"  {n:2d} contributor(s): {count:8d} hits  ({100*count/total:5.2f}%)")
    if max_contributors > 10:
        count = int(np.sum(contributors_per_hit > 10))
        print(f"  >10 contributors: {count:8d} hits  ({100*count/total:5.2f}%)")

    # ---- plot ----
    bins = np.arange(0.5, max_contributors + 1.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"MC particles contributing per calorimeter hit\n"
        f"{args.input_root.split('/')[-1]},  {n_events} events",
        fontsize=12
    )

    for ax, logy in zip(axes, [False, True]):
        ax.hist(contributors_per_hit, bins=bins, color="steelblue", edgecolor="white")
        ax.set_xlabel("Number of contributing MC particles")
        ax.set_ylabel("Number of hits")
        ax.set_xticks(range(1, max_contributors + 1))
        if logy:
            ax.set_yscale("log")
            ax.set_title("Log scale")
        else:
            ax.set_title("Linear scale")

    plt.tight_layout()
    plt.savefig("contributors_per_hit.png", dpi=150)
    print("\nPlot saved to contributors_per_hit.png")
    plt.show()


if __name__ == "__main__":
    main()
