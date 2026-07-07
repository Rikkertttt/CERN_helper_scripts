#!/usr/bin/env python
import sys
import uproot

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_collections.py <file.root>")
        sys.exit(1)

    with uproot.open(sys.argv[1]) as f:
        # Find the events tree
        if "events" in f:
            tree = f["events"]
        else:
            for key, obj in f.items():
                if hasattr(obj, "keys"):
                    tree = obj
                    print(f"Using tree: {key}")
                    break

        branches = sorted(tree.keys())

        # Print all top-level collection names (branches without a dot or slash)
        print("\n=== All branch names containing 'link' or 'Link' or 'Truth' ===")
        for b in branches:
            if any(x in b for x in ["link", "Link", "Truth", "truth"]):
                print(" ", b)

        print("\n=== All top-level collection names (no dot, no slash) ===")
        top_level = sorted(set(
            b.split(".")[0].split("/")[0].lstrip("_")
            for b in branches
        ))
        for name in top_level:
            print(" ", name)

if __name__ == "__main__":
    main()
