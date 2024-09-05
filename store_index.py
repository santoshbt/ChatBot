from src.helper import load_info, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

documents = load_info("insurance_products.txt")
text_chunks = text_splitter(documents)
embeddings = load_embedding()


#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
