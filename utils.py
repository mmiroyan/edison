import os
import re
import ast
import html
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable
from datetime import datetime
from xml.etree import ElementTree as ET

import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def question_ocr(xml: str) -> str:
    """
    Extract text from images embedded in an XML document using Azure's Computer Vision OCR service.

    Args:
        xml (str): An XML string containing image elements with 'src' attribute pointing to the image URL.

    Returns:
        str: A concatenated string of all extracted text from the images in the XML.
    """
    computervision_client = ComputerVisionClient(
        os.getenv('OCR_ENDPOINT'), 
        CognitiveServicesCredentials(os.getenv('OCR_KEY'))
    )
    root = ET.fromstring(xml)
    image_links = [image.get('src') for image in root.iter('image')]
    extracted_text = []

    for img_link in image_links:
        read_response = computervision_client.read(img_link, raw=True)
        operation_id = read_response.headers["Operation-Location"].split("/")[-1]
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                extracted_text.extend(line.text for line in text_result.lines)
    return "\n".join(extracted_text)


def process_question(question_text: str):
    """
    Process a question by removing consent blocks and returning the cleaned text.

    Args:
        question_text (str): The question text to process.

    Returns:
        str: The cleaned question text with Edison-related blocks removed.
    """
    pattern = r'={20,}\s*If you would like to allow the TA to use Edison.*?\[[^]]*]\s*Please write your question above the dashed line\. Thank you!'
    question_text = re.sub(pattern, '', question_text, flags=re.DOTALL)
    question_text = re.sub(r'^\s*edison', '', question_text, flags=re.IGNORECASE)
    return question_text.strip()


def ocr_process_input(thread_title: str, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process input data by extracting context from images and formatting it into structured conversation turns.

    Args:
        thread_title (str): The title of the current thread.
        conversation_history (List[Dict[str, Any]]): A list of previous conversation turns.

    Returns:
        List[Dict[str, Any]]: A list representing the processed conversation turns, including extracted image context.
    """
    processed_conversation = [
        {
            'role': (
                'Student' if turn['user_role'].lower() == 'student'
                else 'Assistant' if turn['user_role'].lower() == 'assistant'
                else 'TA'
            ),
            'text': process_question(turn['text']) if turn['user_role'].lower() == 'student' else turn['text'],
            'image_context': question_ocr(turn['document'])
        }
        for turn in conversation_history
    ]
    processed_conversation[0]['text'] = thread_title + '\n' + processed_conversation[0]['text']
    return processed_conversation


def process_conversation_search(processed_conversation: List[Dict[str, Any]], prompt_summarize: List[Dict[str, Any]]) -> str:
    """
    Process a conversation and return a summary along with the last message's context and text.

    Args:
        processed_conversation (List[Dict[str, Any]]): A list of messages in the conversation.
        prompt_summarize (List[Dict[str, Any]]): A prompt for summarizing the conversation.

    Returns:
        str: A string containing the summarized conversation followed by the context and text of the last message.
    """
    if len(processed_conversation) > 1:
        conversation_summary = generate(prompt=prompt_summarize)
        last_message = processed_conversation[-1]
        return f"{conversation_summary}\n{last_message['image_context']}{last_message['text']}"
    else:
        last_message = processed_conversation[-1]
        return f"{last_message['image_context']}{last_message['text']}"


def generate(prompt: List[Dict[str, str]], temperature: float = 0.7, top_p: float = 0.95) -> str:
    """
    Send a prompt to an API endpoint of an LLM and retrieve a response.

    Args:
        prompt (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.
        temperature (float, optional): The sampling temperature for the model's output. Defaults to 0.7.
        top_p (float, optional): The cumulative probability cutoff for top-p sampling. Defaults to 0.95.

    Returns:
        str: The content of the response message from the API.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv('OPENAI_KEY')
    }
    payload = {
        "messages": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    response = requests.post(os.getenv('LLM_ENDPOINT'), headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def retrieve_qa(conversation: str, top_k: int, confidence_threshold: float = 0.08) -> str:
    """
    Retrieve historical question-answer pairs related to a given conversation using Azure's Question Answering service.

    Args:
        conversation (str): Summary of the conversation of previous turns and the most recent student question.
        top_k (int): The maximum number of top answers to retrieve.
        confidence_threshold (float): The minimum confidence threshold for answers. Defaults to 0.08.

    Returns:
        str: A formatted string containing the top matching question-answer pairs retrieved from the service.
    """
    client = QuestionAnsweringClient(
        os.getenv('QA_ENDPOINT'), 
        AzureKeyCredential(os.getenv('QA_KEY'))
    )
    output = client.get_answers(
        question=conversation[-4999:],  # the limit is 5000 chars
        top=top_k,
        confidence_threshold=confidence_threshold,
        project_name=os.getenv('QA_PROJECT_NAME'),
        deployment_name=os.getenv('QA_DEPLOYMENT_NAME')
    )
    if not output.answers:
        return "None"
    qa_pairs = ""
    for pair in output.answers:
        if pair.questions:
            qa_pairs += f"\n==========================================\nConversation History and Student question: {pair.questions[0]}\nTA's response: {pair.answer}"
    if qa_pairs == "":
        return "None"
    return "Retrieved historical QA" + qa_pairs


def embed_text(text: str, model_name: str) -> List[float]:
    """
    Generate an embedding for a given text using a specified model via Azure OpenAI.

    Args:
        text (str): The input text to generate the embedding for.
        model_name (str): The name of the model to use for generating the embedding.

    Returns:
        List[float]: A list representing the embedding vector for the input text.
    """
    client = AzureOpenAI(
        api_key=os.getenv("OPENAI_KEY"),  
        api_version="2024-02-01",
        azure_endpoint=os.getenv("OPENAI_ENDPOINT")
    )
    response = client.embeddings.create(input=text, model=model_name)
    return response.data[0].embedding


def retrieve_docs_hybrid(text: str, index_name: str, top_k: int, semantic_reranking: bool) -> str:
    """
    Retrieve documents using a hybrid search combining text and vector queries.

    Args:
        text (str): The text query for the search.
        index_name (str): The name of the search index.
        top_k (int): The number of top documents to retrieve.
        semantic_reranking (bool): Whether to use semantic reranking.

    Returns:
        str: The retrieved documents or an empty string if an error occurs.
    """
    try:
        search_client = SearchClient(
            os.getenv("SEARCH_ENDPOINT"), 
            index_name, 
            AzureKeyCredential(os.getenv("SEARCH_KEY"))
        )
        vector_query = VectorizedQuery(
            vector=embed_text(text, model_name=os.getenv("EMBEDDING_MODEL_NAME")),
            k_nearest_neighbors=top_k,
            fields="vector"
        )
        search_params = {
            "search_text": text,
            "vector_queries": [vector_query],
            "select": ["content"],
            "top": top_k
        }
        if semantic_reranking:
            search_params.update({
                "query_type": "semantic",
                "semantic_query": text,
                "semantic_configuration_name": "my-semantic-config"
            })
        results = search_client.search(**search_params)
        retrieved_docs = "Retrieved course documents"
        for retrieved_doc in results:
            retrieved_docs += f"\n==========================================\n{retrieved_doc['content']}"
        return retrieved_docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return ''


def get_file_names_dir(directory_path: str) -> List[str]:
    """
    Retrieve a list of file names from a specified directory within an Azure Blob Storage container.

    Args:
        directory_path (str): The path of the directory within the blob storage container.

    Returns:
        List[str]: A list of file names found in the specified directory.
    """
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    container_client = blob_service_client.get_container_client(os.getenv('AZURE_BLOB_CONTAINER_NAME'))
    blobs_list = container_client.list_blobs(name_starts_with=directory_path)
    return ['/'.join(Path(blob.name).parts[2:]) for blob in blobs_list]


def retrieve_docs_manual(question_category: str, category_mapping: dict, question_subcategory: str, subcategory_mapping: dict, question_info: str, get_prompt: Callable[[List, str], List]) -> tuple:
    """
    Retrieve and return the contents of a specific document from Azure Blob Storage based on a provided question category and information.

    Args:
        question_category (str): The category of the question.
        category_mapping (dict): Mapping of categories to directory paths.
        question_subcategory (str): The subcategory of the question.
        subcategory_mapping (dict): Mapping of subcategories to directory paths.
        question_info (str): The detailed information about the question.
        get_prompt (Callable[[List, str], List]): A function that generates a prompt for selecting a document path.

    Returns:
        tuple: A tuple containing the problem paths list, selected path, and retrieved document content.
    """
    problem_paths_list = 'none'
    if question_category in category_mapping:
        problem_paths_list = get_file_names_dir(f'docs_manual/{category_mapping[question_category]}')
    elif question_subcategory in subcategory_mapping:
        problem_paths_list = get_file_names_dir(f'docs_manual/{subcategory_mapping[question_subcategory]}')

    prompt = get_prompt(paths='\n'.join(problem_paths_list),
                        question_info=re.sub(pattern=r"\n+", repl=" ", string=question_info))
    processed_question = generate(prompt=prompt)
    
    retrieved_docs = 'none'
    try:
        processed_question = ast.literal_eval(processed_question)
        selected_path = processed_question['selected_path']
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        logger.error(f"Processed question: {processed_question}")
        selected_path = 'none'
    
    # If the model selected the assignment dir only, search for setup or index files for general assignment info
    if selected_path != 'none' and (len(selected_path.split('/')) == 1 or selected_path.split('/')[-1] == ''):
        selected_paths = [path for path in problem_paths_list if selected_path in path and ('setup' in path.split('/')[-1].lower() or 'index' in path.split('/')[-1].lower())]
        selected_path = selected_paths[0] if selected_paths else 'none'
            
    if selected_path != 'none':
        try:
            blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
            container_client = blob_service_client.get_container_client(os.getenv('AZURE_BLOB_CONTAINER_NAME'))
            if question_category in category_mapping:
                blob_path = f'docs_manual/{category_mapping[question_category]}/{selected_path}'
            elif question_subcategory in subcategory_mapping:
                blob_path = f'docs_manual/{subcategory_mapping[question_subcategory]}/{selected_path}'
            blob_client = container_client.get_blob_client(blob_path)
            if not blob_client.exists():
                raise FileNotFoundError("Blob does not exist in the container.")
            retrieved_docs = blob_client.download_blob().readall().decode('utf-8')
        except Exception as e:
            retrieved_docs = 'none (error)'
            logger.error(f"Error retrieving manual document: {e} (selected path: {selected_path})")
    return str(problem_paths_list), selected_path, retrieved_docs


def log_local(log_dict: Dict[str, Any], file_path: str) -> None:
    """
    Save a log entry by combining input and output dictionaries and appending it to a file.

    Args:
        log_dict (Dict[str, Any]): A dictionary containing data to be logged.
        file_path (str): The file path where the log entry should be appended.
    """
    log_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'a') as f:
        json.dump(log_dict, f)
        f.write('\n')


def log_blob(log_dict: Dict[str, Any], blob_name: str) -> None:
    """
    Save a log entry to an Azure Blob Storage append blob.

    Args:
        log_dict (Dict[str, Any]): The dictionary containing data to be logged.
        blob_name (str): The name of the blob file where the log entry will be saved.
    """
    log_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    container_client = blob_service_client.get_container_client(os.getenv('AZURE_BLOB_CONTAINER_NAME'))
    if not container_client.exists():
        container_client.create_container()
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(json.dumps(log_dict) + '\n', blob_type="AppendBlob", overwrite=False)


def xml_to_markdown(xml_text: str) -> str:
    """
    Convert XML text to Markdown format.

    Args:
        xml_text (str): The input XML text.

    Returns:
        str: The converted Markdown text.
    """
    xml_text = html.unescape(xml_text)
    root = ET.fromstring(xml_text)    
    md_text = element_to_markdown(root).strip()
    return re.sub(r'\n{3,}', '\n\n', md_text)
    

def element_to_markdown(element: ET.Element, list_type: str = None, list_index: int = 0, depth: int = 0) -> str:
    """
    Recursively convert an XML element to Markdown format.

    Args:
        element (ET.Element): The XML element to convert.
        list_type (str, optional): The type of list ('bullet' or 'number'). Defaults to None.
        list_index (int, optional): The current index in a numbered list. Defaults to 0.
        depth (int, optional): The current depth in nested lists. Defaults to 0.

    Returns:
        str: The converted Markdown text for the element and its children.
    """
    text = ""
    if element.tag == 'document':
        text = ''.join(element_to_markdown(child, depth=depth) for child in element)
    elif element.tag == 'paragraph':
        text = element.text or ''
        text += ''.join(element_to_markdown(child, depth=depth) for child in element)
        text += element.tail or ''
        text += '\n\n'
    elif element.tag == 'list':
        list_type = element.get('style', 'bullet')
        text = ''.join(element_to_markdown(child, list_type, i, depth + 1) for i, child in enumerate(element, start=1))
        text += '\n'
    elif element.tag == 'list-item':
        indent = '  ' * depth
        prefix = f"{list_index}. " if list_type == 'number' else "- "
        text = f"{indent}{prefix}"
        text += ''.join(element_to_markdown(child, list_type, list_index, depth).strip() for child in element)
        text += '\n'
    elif element.tag == 'bold':
        text = f"**{element.text or ''}**{element.tail or ''}"
    elif element.tag == 'code':
        text = f"`{element.text or ''}`{element.tail or ''}"
    elif element.tag == 'pre':
        text = f"\n```\n{element.text.strip() if element.text else ''}\n```\n"
    else:
        text = element.text or ''
        text += ''.join(element_to_markdown(child, depth=depth) for child in element)
        text += element.tail or ''
    return text


def process_markdown(text: str) -> str:
    """
    Process Markdown text by escaping special characters.

    Args:
        text (str): The input Markdown text.

    Returns:
        str: The processed Markdown text with escaped special characters.
    """
    return re.sub(r'[<>&"\']', lambda m: {
        '<': '&lt;',
        '>': '&gt;',
        '&': '&amp;',
        '"': '&quot;',
        "'": '&apos;'
    }[m.group()], text)


def get_edstem_token(course: str) -> str:
    """
    Get the EdStem API token for a given course.

    Args:
        course (str): The course identifier.

    Returns:
        str: The EdStem API token for the specified course.
    """
    course_tokens = {
        'ds100': 'DS100_EDSTEM_KEY',
        'ds100-sp25': 'DS100_EDSTEM_KEY',
        'ds8': 'DS8_EDSTEM_KEY',
        'cs61a': 'CS61A_EDSTEM_KEY'
    }
    return os.getenv(course_tokens.get(course, ''))


def delete_comment(course: str, id: str) -> None:
    """
    Delete a comment from EdStem for a given course.

    Args:
        course (str): The course identifier.
        comment_id (str): The ID of the comment to delete.
    """
    url = f"https://us.edstem.org/api/comments/{id}"
    headers = {
        'Authorization': f'Bearer {get_edstem_token(course)}',
        'Content-Type': 'application/json'
    }
    response = requests.delete(url, headers=headers)
    response.raise_for_status()


def reply_to_ed(course: str, id: str, text: str, post_answer: bool, private: bool) -> None:
    """
    Reply to a thread on EdStem for a given course.

    Args:
        course (str): The course identifier.
        thread_id (str): The ID of the thread to reply to.
        text (str): The content of the reply.
        post_answer (bool): Whether to post as an answer or a comment.
        private (bool): Whether the reply should be private.
    """
    url = f"https://us.edstem.org/api/{'threads' if post_answer else 'comments'}/{id}/comments"
    payload = {
        "comment": {
            "type": "answer" if post_answer else "comment",
            "content": f"<document version=\"2.0\"><paragraph>{process_markdown(text)}</paragraph></document>",
            "is_private": private,
        }
    }
    headers = {
        'Authorization': f'Bearer {get_edstem_token(course)}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
