import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# -------------------------------
# 1️⃣ Load PDF Documents
# -------------------------------
pdf_files=r"C:\Users\GenAICHNKPRUSR17\Desktop\final\data\medical.pdf"
def load_pdfs(pdf_files):
    documents = []

    for file in pdf_files:
        reader = PdfReader(file)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                documents.append(Document(page_content=text))

    return documents


documents = load_pdfs(pdf_files)

# -------------------------------
# 2️⃣ Split Documents
# -------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(documents)

# -------------------------------
# 3️⃣ Create Embeddings
# -------------------------------

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(split_docs, embeddings)

# Save vectors
vectorstore.save_local("medical_vector_db")

# -------------------------------
# 4️⃣ Load LLM
# -------------------------------

llm = ChatOpenAI(
   base_url="https://genailab.tcs.in",
    model="gpt-4o",
    openai_api_key="sk-vXJdsONxDGBIgkN7HR9dhA",
    temperature=0.7,
    http_client=client

)

# -------------------------------
# 5️⃣ Create RAG Chain
# -------------------------------

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# -------------------------------
# 6️⃣ Chat Loop
# -------------------------------

chat_history = []

while True:

    question = input("\nAsk medical question: ")

    if question.lower() == "exit":
        break

    result = qa_chain({
        "question": question,
        "chat_history": chat_history
    })

    answer = result["answer"]

    print("\nMedical Assistant:", answer)

    chat_history.append((question, answer))