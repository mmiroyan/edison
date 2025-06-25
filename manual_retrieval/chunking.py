import ast
import json
import os
import re
from dotenv import load_dotenv
from utils import generate

def read_markdown_file(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    return text

def merge_small_chunks(chunks, min_length=128):
    merged_chunks = []
    buffer = ""

    for i in range(len(chunks)):
        current_chunk = chunks[i].strip()

        if buffer:
            current_chunk = buffer + " " + current_chunk
            buffer = ""

        if len(current_chunk) < min_length and i < len(chunks) - 1:
            buffer = current_chunk  # Store for the next iteration
            continue

        if i == len(chunks) - 1 and len(current_chunk) < min_length and merged_chunks:
            merged_chunks[-1] += " " + current_chunk
        else:
            merged_chunks.append(current_chunk)

    return merged_chunks

def split_into_sections(text, type='paragraph', section_length=20, overlap=0):
    if type == 'paragraph':
        paragraphs = text.split('\n\n')
    if type == 'line':
        paragraphs = text.splitlines()
    sections = []
    step = section_length - overlap if section_length > overlap else 1
    for i in range(0, len(paragraphs), step):
        sections.append(paragraphs[i:i + section_length])
    return sections

def get_question_headers_prompt(existing_headers, section_text):
    prompt_text = (
        "You are an expert at identifying question headers in course assignment documents. "
        "Your goal is to contribute to a running list of question headers found in the document. "
        "Please adhere to the following instructions very carefully: "
        "1) A question header is any text that explicitly introduces a question or part of a question. \n"
        "2) The actual question text must be present in order for you to include its question header. \n"
        "3) Any text that does not follow these rules or that you are uncertain about should not be included. \n"
        "4) Longer headers are preferred over shorter ones. \n"
        f"These are the question headers collected so far:\n{existing_headers}\n\n"
        "Please output ONLY a list of the new headers you would like to add to the existing list, like so: ['Q4', 'Part 3', 'Question 5'] "
        "Output ONLY an empty list if no new question headers are found: [] "
    )
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": f"Here is the document section you should extract headers from: \n\n{section_text}"}
    ]

def accumulate_question_headers(sections):
    accumulated_headers = []
    for section in sections:
        section_text = '\n\n'.join(section)
        prompt = get_question_headers_prompt(accumulated_headers, section_text)
        headers_update = generate(prompt, temperature=0.1)
        print("Question headers update:", headers_update)
        try:
            new_headers = ast.literal_eval(headers_update)
            if isinstance(new_headers, list):
                for header in new_headers:
                    if header not in accumulated_headers:
                        accumulated_headers.append(header)
            else:
                print("Unexpected format for headers. Skipping update.")
        except Exception as e:
            print(f"Error parsing headers update: {e}")
    return accumulated_headers

def get_clean_headers_prompt(headers):
    prompt_text = (
        "You are an expert at cleaning up lists of question headers. "
        "The following list contains question headers: \n" + str(headers) + "\n "
        "If any headers in the list clearly stick out as incorrect or not belonging, please remove them. Only do this if you are very sure. "
        "If it appears that a question or question part was possibly 'skipped', please add it to the list in a consistent format. "
        "Please add questions in a manner such that they could actually be 'detected' by some parsing script. "
        "For example, if 'Question 1: Addition' and 'Question 3: Division' are headers, you should add 'Question 2' by itself, as it is the part we know for certain, without guessing what comes after it."                                                                    
        "Additionally, please extend the list to a longer list of potential question headers with the same format. "
        "Do not output anything other than the list."
    )
    return [{"role": "system", "content": prompt_text}]

def clean_question_headers(headers):
    prompt = get_clean_headers_prompt(headers)
    cleaned = generate(prompt, temperature=0.1)
    print(f'GPT Cleaned: {cleaned}')
    try:
        cleaned_headers = ast.literal_eval(cleaned)
        if isinstance(cleaned_headers, list):
            return cleaned_headers
        else:
            print("Unexpected format for cleaned headers. Returning original list.")
            return headers
    except Exception as e:
        print(f"Error parsing cleaned headers: {e}")
        return headers

def split_document_by_headers(text, headers):
    chunks = []
    start_idx = 0
    header_positions = []

    for header in headers:
        matches = [(m.start(), header) for m in re.finditer(re.escape(header), text)]
        header_positions.extend(matches)

    header_positions.sort()

    for pos, header in header_positions:
        if pos > start_idx:
            chunk = text[start_idx:pos].strip()
            if chunk:
                chunks.append(chunk)
        start_idx = pos

    last_chunk = text[start_idx:].strip()
    if last_chunk:
        chunks.append(last_chunk)

    return chunks

def chunk_markdown_file(full_text):
    # print(f"Processing: {input_path}")
    # full_text = read_markdown_file(input_path)
    sections = split_into_sections(full_text, type='line', section_length=32, overlap=16)
    question_headers = accumulate_question_headers(sections)
    print("Accumulated Question Headers:")
    print(question_headers)
    cleaned_headers = clean_question_headers(question_headers)
    print("Cleaned Question Headers:")
    print(cleaned_headers)
    header_split_chunks = split_document_by_headers(full_text, cleaned_headers)
    chunks = merge_small_chunks(header_split_chunks, min_length=128)
    return chunks
    # with open(output_path, "w", encoding="utf-8") as file:
    #     json.dump(chunks, file, indent=4)
    # print(f"Saved header-split chunks to: {output_path}")

def main(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        inpath = os.path.join(input_dir, filename)
        outpath = os.path.join(output_dir, filename.replace(".md", ".json"))
        print(f"Processing: {inpath}")
        full_text = read_markdown_file(inpath)
        sections = split_into_sections(full_text, type='line', section_length=32, overlap=16)
        question_headers = accumulate_question_headers(sections)
        print("Accumulated Question Headers:")
        print(question_headers)
        cleaned_headers = clean_question_headers(question_headers)
        print("Cleaned Question Headers:")
        print(cleaned_headers)
        header_split_chunks = split_document_by_headers(full_text, cleaned_headers)
        chunks = merge_small_chunks(header_split_chunks, min_length=128)
        with open(outpath, "w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4)
        print(f"Saved header-split chunks to: {outpath}")
