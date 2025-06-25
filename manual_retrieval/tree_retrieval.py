import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient

load_dotenv('./keys.env')

# Azure + OpenAI setup
blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service.get_container_client("ds100-su25")
TREE_PREFIX = "docs_manual/trees/"

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("LLM_ENDPOINT")
)
model = os.getenv("MODEL_NAME")

file_cache = {}
gpt_call_count = 0

def safe_generate(messages, temperature=0.1):
    global gpt_call_count
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            gpt_call_count += 1
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling generate: {e}. Retrying in 1 second...")
            time.sleep(1)

def get_relevant_files(question, toc):
    prompt = (
        f"Given the student question: \"{question}\", and the following table of contents:\n"
        f"{json.dumps(toc, indent=2)}\n"
        "Select the three file names whose content is most likely to answer the question. "
        "Output ONLY a list of exactly three file names, like so: ['hw4.json', 'lab8.json', 'projA1.json'] "
    )
    messages = [{"role": "system", "content": prompt}]
    gpt_output = safe_generate(messages, temperature=0.1)

    # print(gpt_output)
    try:
        file_list = eval(gpt_output)
        if isinstance(file_list, list) and len(file_list) == 3:
            return file_list
        else:
            print("Warning: GPT did not return exactly three file names. Using first three keys from TOC.")
            return list(toc.keys())[:3]
    except Exception as e:
        print(f"Error parsing GPT output: {e}. Falling back to first three keys from TOC.")
        return list(toc.keys())[:3]

def load_blob_json(blob_path):
    blob_client = container_client.get_blob_client(blob_path)
    content = blob_client.download_blob().readall().decode("utf-8")
    return json.loads(content)

def beam_search_across_blobs(question, file_names, beam_width=3, final_doc_count=None):
    if final_doc_count is None:
        final_doc_count = beam_width

    beam = []
    for file_name in file_names:
        blob_path = f"{TREE_PREFIX}{file_name}"
        try:
            if file_name not in file_cache:
                tree_data = load_blob_json(blob_path)
                file_cache[file_name] = tree_data
            else:
                tree_data = file_cache[file_name]

            root_key = list(tree_data.keys())[0]
            root_value = tree_data[root_key]
            beam.append((None, file_name, root_key, root_value))
        except Exception as e:
            print(f"Error loading {blob_path}: {e}")

    if not beam:
        return []

    while True:
        new_beam = []
        expanded = False

        candidate_details = {}
        candidate_display = {}
        leaf_candidates = []

        for score, file_name, node_key, node in beam:
            if isinstance(node, dict):
                for child_key, child_value in node.items():
                    idx = len(candidate_details) + 1
                    candidate_details[idx] = (file_name, child_key, child_value)
                    candidate_display[idx] = child_key
                expanded = True
            else:
                leaf_candidates.append((score, file_name, node_key, node))

        if not expanded or not candidate_details:
            break

        if all(not isinstance(child_value, dict) for (file_name, child_key, child_value) in candidate_details.values()):
            num_to_select = final_doc_count
        else:
            num_to_select = beam_width

        # print(num_to_select)
        example = str(list(range(1, num_to_select+1)))
        # print(example)
        prompt = (
            "You are an expert at relevant document selection. "
            f"Here is a student question:\n\"{question}\"\n"
            f"And here is a list of candidate document keys:\n"
            f"{json.dumps(candidate_display, indent=2)}\n"
            f"Select the top {num_to_select} most relevant document keys for answering the question. "
            f"Output ONLY a list of the keys as natural numbers, like this: {example}"
        )


        messages = [{"role": "system", "content": prompt}]
        time.sleep(1)
        gpt_output = safe_generate(messages)
        # print(gpt_output)
        try:
            selected_keys = eval(gpt_output)
            if not isinstance(selected_keys, list):
                raise ValueError("Invalid GPT output format")
        except Exception as e:
            print(f"Error parsing GPT output: {e}. Falling back to top-{num_to_select}")
            selected_keys = list(candidate_details.keys())[:num_to_select]

        for idx in selected_keys:
            if idx in candidate_details:
                file_name, child_key, child_value = candidate_details[idx]
                new_beam.append((None, file_name, child_key, child_value))
        new_beam.extend(leaf_candidates)
        beam = new_beam

    return [{"File": fname, "Text": text} for (_, fname, _, text) in beam]

def manual_retrieval(question, beam_width=3, final_doc_count=1):
    global gpt_call_count
    gpt_call_count = 0

    try:
        toc = load_blob_json(f"{TREE_PREFIX}table_of_contents.json")
    except Exception as e:
        print(f"Error loading TOC: {e}")
        return [], 0

    selected_files = get_relevant_files(question, toc)

    docs = beam_search_across_blobs(
        question,
        selected_files,
        beam_width=beam_width,
        final_doc_count=final_doc_count
    )

    doc_string = f"Retrieved assignment documents\n==========================================\n{docs}"

    return doc_string, gpt_call_count


if __name__ == "__main__":
    question = "how do i do 1d"
    docs, calls = manual_retrieval(question, beam_width=3)
    print(json.dumps(docs, indent=2))
    # print(f"GPT calls: {calls}")