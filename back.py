
import certifi
import os
import httpx
import requests
import urllib3
import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import ssl
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


# =========================
# Load ENV Variables
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================
# SSL FIX (Corporate Network)
# =========================

ssl._create_default_https_context = ssl.create_default_context(
    cafile=certifi.where()
)

requests.packages.urllib3.disable_warnings()

session = requests.Session()
session.verify = False
requests.get = session.get

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"


# Shared HTTP Client
client = httpx.Client(verify=False)


# =========================
# 1️⃣ Load PDF
# =========================

pdf_files = ["medical_guidelines.pdf"]

def load_pdfs(files):
    documents = []

    for file in files:
        reader = PdfReader(file)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                documents.append(Document(page_content=text))

    return documents


documents = load_pdfs(pdf_files)

print("PDF Loaded. Pages:", len(documents))


# =========================
# 2️⃣ Split Documents
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(documents)

print("Total Chunks:", len(split_docs))


# =========================
# 3️⃣ Create Embeddings
# =========================
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
    http_client=client
)
vectorstore = FAISS.from_documents(split_docs, embeddings)

vectorstore.save_local("medical_vector_db")

print("Vector DB Created")


# =========================
# 4️⃣ Load LLM
# =========================

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.4,
    http_client=client
)

response = llm.invoke("Hello")

print("LLM Test:", response.content)


# =========================
# 5️⃣ Create RAG Chain
# =========================

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)


# =========================
# 6️⃣ Chat Loop
# =========================

chat_history = []

while True:

    question = input("Ask Medical Question: ")

    if question.lower() == "exit":
        break

    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    answer = result["answer"]

    print("\nMedical Assistant:", answer)

    chat_history.append((question, answer))
