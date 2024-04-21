import os
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from bs4 import BeautifulSoup
import requests

def extract_text(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text()
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    cleaned_text = '\n'.join(non_empty_lines)
    write_to_local_file(cleaned_text)


def write_to_local_file(text):
    with open("insurance_products.txt", "w") as f:
        f.write(text)


#Loading repositories as documents
def load_info(path):
    loader = TextLoader(path)
    documents = loader.load()
    print(documents)
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
