import uproot
import sys
import argparse
from pathlib import Path


def list_all(file_path: str, output=sys.stdout):
    """Recursively list all objects in a .root file."""
    print(f"File: {file_path}\n", file=output)
    with uproot.open(file_path) as f:
        _recurse(f, indent=0, output=output)


def _recurse(node, indent: int, output=sys.stdout):
    prefix = "  " * indent

    for key in node.keys(cycle=False):
        obj = node[key]
        class_name = getattr(obj, "classname", type(obj).__name__)

        print(f"{prefix}[{class_name}]  {key}", file=output)

        # If it's a TTree, print its branches
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            _print_branches(obj, indent + 1, output=output)

        # If it's a directory, recurse into it
        elif hasattr(obj, "keys"):
            _recurse(obj, indent + 1, output=output)


def _print_branches(tree, indent: int, output=sys.stdout):
    prefix = "  " * indent
    print(f"{prefix}Branches:", file=output)
    for branch_name in tree.keys():
        branch = tree[branch_name]
        dtype = branch.typename if hasattr(branch, "typename") else "unknown"
        print(f"{prefix}  - {branch_name}  ({dtype})", file=output)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the contents of one or more .root files."
    )
    parser.add_argument("files", nargs="+", help="One or more .root files to inspect")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional: path to a text file to save the output (e.g. output.txt)"
    )
    args = parser.parse_args()

    # Resolve all file paths to absolute paths
    root_files = [str(Path(p).resolve()) for p in args.files]

    # Validate files exist before proceeding
    for path in root_files:
        if not Path(path).exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)

    if args.output:
        out_path = str(Path(args.output).resolve())
        with open(out_path, "w") as txt_file:
            for path in root_files:
                list_all(path, output=txt_file)
                print(file=txt_file)
        print(f"Output saved to: {out_path}")
    else:
        for path in root_files:
            list_all(path)
            print()


if __name__ == "__main__":
    main()
