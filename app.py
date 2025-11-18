#!/usr/bin/env python3
"""
RAG-Based Multi-Source Document Q&A System with Streamlit UI
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import sqlite3
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def _format_docs(docs) -> str:
    """Formats retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_data_and_create_retriever(uploaded_file=None):
    """Load data from all sources and create a retriever."""
    all_docs = []

    # 1. Load from uploaded PDF or default PDF
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        all_docs.extend(loader.load())
    else:
        pdf_path = Path("./pdfs")
        if pdf_path.exists():
            pdf_files = list(pdf_path.glob("*.pdf"))
            if pdf_files:
                for pdf_file in pdf_files:
                    try:
                        loader = PyPDFLoader(str(pdf_file))
                        all_docs.extend(loader.load())
                    except Exception as e:
                        logger.error(f"Error loading {pdf_file.name}: {e}")
    
    # 2. Load from Arxiv
    try:
        arxiv_loader = ArxivLoader(query="1706.03762", load_max_docs=1) # "Attention Is All You Need"
        all_docs.extend(arxiv_loader.load())
    except Exception as e:
        logger.error(f"Error loading from Arxiv: {e}")

    # 3. Load from SQLite DB
    try:
        conn = sqlite3.connect('sample.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, job, salary FROM employees")
        rows = cursor.fetchall()
        for row in rows:
            content = f"Employee Name: {row[0]}, Job: {row[1]}, Salary: ${row[2]}"
            doc = Document(page_content=content, metadata={"source": "sqlite"})
            all_docs.append(doc)
        conn.close()
    except Exception as e:
        logger.error(f"Error loading from SQLite: {e}")

    if not all_docs:
        return None

    # Split and embed all documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(all_docs)
    
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vector_store.as_retriever()

# --- Streamlit App ---

st.title("🔎 LangChain - Multi-Source RAG")
"""
This app is a RAG-based Q&A system that answers questions based on data from multiple sources.
You can upload a PDF, or it will use the default data sources:
- A default PDF document (`./pdfs/CSS Color Attributes.pdf`)
- A scientific paper from Arxiv ("Attention Is All You Need")
- A local SQLite database with employee information
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = os.getenv("GROQ_API_KEY")

uploaded_file = st.sidebar.file_uploader("Upload a PDF (optional)", type="pdf")

# Load data and create retriever
with st.spinner("Loading data..."):
    retriever = load_data_and_create_retriever(uploaded_file)

if not retriever:
    st.error("Failed to load data. Please check your data sources and configuration.")
    st.stop()

st.success("Data loaded successfully!")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I can answer questions about the loaded documents. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Chat input
if prompt := st.chat_input(placeholder="Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b", streaming=True)
    
    # Create the RAG chain
    rag_prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
{context}

Question: {question}""" )

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        response = rag_chain.invoke(prompt)
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
