import os
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from bs4 import BeautifulSoup
import requests
import re
# import pdb

def extract_text(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text()
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        cleaned_text = '\n'.join(non_empty_lines)
        write_to_local_file(cleaned_text)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching the URL: {e}")


def write_to_local_file(text):
    filename = "insurance_products.txt"
    try:
        # Open the file in read mode to check if it's empty
        with open(filename, 'r') as file:
            # Check if the file is empty
            if file.read().strip() == '':
                # If it's empty, open the file in write mode to overwrite its content
                with open(filename, 'w') as write_file:
                    write_file.write(text)
            else:
                # If it's not empty, open the file in append mode to add content to it
                with open(filename, 'a') as append_file:
                    append_file.write(text)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")


#Loading repositories as documents
def load_info(path):
    loader = TextLoader(path)
    documents = loader.load()
    return documents


#Creating text chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                      chunk_size = 2000,
                                                                      chunk_overlap = 200)

    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks


#loading embeddings model
def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings


def validate_url(input):
    regex_pattern = r'^(https?|ftp):\/\/[^\s\/$.?#].[^\s]*$'
    if re.match(regex_pattern, input):
        return True
    else:
        return False

def clear_previous_data():
    file_path = "insurance_products.txt"
    with open(file_path, 'w') as file:
        file.truncate()