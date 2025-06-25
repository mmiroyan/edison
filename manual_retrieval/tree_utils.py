# tree_utils.py
import json
import os

from utils import generate


def get_summary_prompt(text, contents=False):

    # if not contents:
    prompt = (
        "You are an expert at extracting concise summaries and keywords from sections of course assignment documents. "
        "Please extract a summary containing all keywords and key concepts from the given text. "
        "Please be sure to extract the name of the assignments ONLY if it is very obviously present in the chunk. "
        "Please always extract any question numbers along with descriptions of the questions when available. "
        "Always choose to highlight question numbers and descriptions over anything else, when space is limited. "
        "Please structure the output in no more than 6 syntactically accurate English sentences. "
        "Output only the summary without anything else."
    )
    # else:
    #     prompt = (
    #         "You are an expert at extracting concise summaries and keywords from summaries of course assignment documents. "
    #         "Please extract a summary containing any assignment names and their descriptions contained in the text. "
    #         "Individual questions and their contents do NOT need to be extracted. "
    #         "Rather, opt to summarize over the key points of the questions, if any are described in the text. "
    #         # "Please be sure to extract the name of ALL assignments present in the text. "
    #         # "Please always extract any question numbers along with descriptions of the questions when available. "
    #         # "Always choose to highlight  and descriptions over anything else, when space is limited. "
    #         "Please structure the output in syntactically accurate English sentences. "
    #         "Output only the summary without anything else."
    #     )

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]


def extract_summary(text, contents=False):
    try:
        # gpt_output = text
        gpt_output = generate(get_summary_prompt(text, contents), temperature=0.1)
        if not gpt_output.strip():
            print("Warning: Empty summary generated.")
            return "Summary unavailable"
        print(gpt_output)
        return gpt_output.strip()
    except Exception as e:
        print(f"Error generating summary for text: {e}")
        return "Error in summary"


def generate_leaf_nodes(chunks, contents=False):
    """
    For each nonempty chunk, generate a leaf node as a tuple (summary, content).
    """
    return [(extract_summary(chunk, contents), chunk) for chunk in chunks]


def create_TOC(tree_folder):
    table_of_contents = {}

    for filename in sorted(os.listdir(tree_folder)):
        if filename.endswith(".json") and "table_of_contents" not in filename:
            filepath = os.path.join(tree_folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            highest_key = max(data.keys())
            table_of_contents[filename] = highest_key

    with open(f"{tree_folder}/table_of_contents.json", "w", encoding="utf-8") as toc_file:
        json.dump(table_of_contents, toc_file, indent=4)

    print("table_of_contents.json created successfully!")


def build_balanced_tree(nodes, branch_factor=2, contents=False):
    """
    Recursively constructs a balanced tree from a list of nodes.
    Each node is a tuple (summary, content).

    Returns a nested dictionary with the structure:
    { parent_summary: { child_summary: subtree (or leaf content), ... } }
    """
    # Base case: a single node is a leaf.
    if len(nodes) == 1:
        summary, content = nodes[0]
        return {summary: content}

    # If the number of nodes is less than or equal to branch_factor,
    # combine them directly into a parent node.
    if len(nodes) <= branch_factor:
        combined_text = "\n\n".join(node[0] for node in nodes)
        parent_summary = extract_summary(combined_text, contents)
        children_dict = {}
        for summary, content in nodes:
            children_dict[summary] = content
        return {parent_summary: children_dict}

    # Partition nodes into branch_factor groups.
    groups = []
    group_size = len(nodes) // branch_factor
    remainder = len(nodes) % branch_factor
    start_index = 0
    for i in range(branch_factor):
        size = group_size + (1 if i < remainder else 0)
        groups.append(nodes[start_index:start_index + size])
        start_index += size

    child_summaries = []
    children_dict = {}
    # Recursively build a subtree for each group.
    for group in groups:
        subtree = build_balanced_tree(group, branch_factor, contents)
        # Each subtree is a dictionary with a single key.
        key = list(subtree.keys())[0]
        child_summaries.append(key)
        children_dict[key] = subtree[key]

    # Combine all child summaries to form the parent's summary.
    combined_text = "\n\n".join(child_summaries)
    parent_summary = extract_summary(combined_text, contents)
    return {parent_summary: children_dict}