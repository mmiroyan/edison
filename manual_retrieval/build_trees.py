import os
import json
from tree_utils import generate_leaf_nodes, build_balanced_tree, create_TOC

def process_json(input_file, output_file, branch_factor=3):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        print(f"Skipping {input_file}: Input JSON must be a list of strings.")
        return

    chunks = [chunk for chunk in data if chunk.strip()]
    leaf_nodes = generate_leaf_nodes(chunks)
    hierarchy = build_balanced_tree(leaf_nodes, branch_factor=branch_factor)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(hierarchy, file, indent=4)

    print(f"Processed data saved to {output_file}")

def build_tree_from_chunks(input_file, output_file, tree_dir, branch_factor=3):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        print(f"Skipping {input_file}: Input JSON must be a list of strings.")
        return

    chunks = [chunk for chunk in data if chunk.strip()]
    leaf_nodes = generate_leaf_nodes(chunks)
    hierarchy = build_balanced_tree(leaf_nodes, branch_factor=branch_factor)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(hierarchy, file, indent=4)

    print(f"Processed data saved to {output_file}")

    # Update TOC after processing this file
    create_TOC(tree_dir)
    print("TOC updated!")


def build_tree(chunks, branch_factor=3):
    """
    Accepts a list of string chunks, builds and returns the hierarchy tree as a dict.
    Used when chunks are already in memory (e.g., from blob download).
    """
    if not isinstance(chunks, list):
        raise ValueError("Chunks must be a list of strings.")
    filtered = [chunk for chunk in chunks if chunk.strip()]
    leaf_nodes = generate_leaf_nodes(filtered)
    return build_balanced_tree(leaf_nodes, branch_factor=branch_factor)