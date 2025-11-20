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

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- HELPER FUNCTIONS ---

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
        st.sidebar.success(f"Successfully loaded '{uploaded_file.name}'!")
    else:
        pdf_path = Path("./pdfs")
        if pdf_path.exists() and any(pdf_path.glob("*.pdf")):
            for pdf_file in pdf_path.glob("*.pdf"):
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    all_docs.extend(loader.load())
                    st.sidebar.info(f"Loaded default PDF: {pdf_file.name}")
                except Exception as e:
                    st.sidebar.error(f"Error loading {pdf_file.name}: {e}")
    
    # 2. Load from Arxiv
    try:
        arxiv_loader = ArxivLoader(query="1706.03762", load_max_docs=1) # "Attention Is All You Need"
        all_docs.extend(arxiv_loader.load())
        st.sidebar.info("Loaded paper from Arxiv: 'Attention Is All You Need'")
    except Exception as e:
        st.sidebar.error(f"Arxiv load error: {e}")

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
        st.sidebar.info("Loaded data from SQLite database.")
    except Exception as e:
        st.sidebar.error(f"SQLite load error: {e}")

    if not all_docs:
        st.warning("No documents were loaded. The chatbot may not have any information to answer questions.")
        return None

    # Split and embed documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(all_docs)
    
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vector_store.as_retriever()

def get_rag_chain():
    """Creates and returns the RAG chain."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
        st.stop()
        
    llm = ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768", streaming=True)
    
    rag_prompt = ChatPromptTemplate.from_template("""**Answer the question based *only* on the following context.****Answer the question based *only* on the following context.**
    
    **Context:**
    {context}
    
    **Question:**
    {question}
    """
    )
    
    retriever = st.session_state.retriever
    
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def home_page():
    """The main page of the Streamlit app."""
    st.header("📚 Multi-Source RAG Q&A")
    st.markdown("""
    Welcome! This app uses Retrieval-Augmented Generation (RAG) to answer your questions.
    It pulls information from multiple sources:
    - A PDF you upload or a default one
    - The famous 'Attention Is All You Need' paper from Arxiv
    - A local SQLite database of employee records
    
    Ask a question below and the AI will synthesize an answer from the available data.
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you today?"}
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            rag_chain = get_rag_chain()
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def source_code_page():
    """A page to display the app's source code."""
    st.header("📄 Source Code")
    st.markdown("Here's the Python code for this Streamlit application:")
    
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")

# --- MAIN APP ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Multi-Source RAG", page_icon="🔎", layout="wide")
    
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📄 View Source Code"])
    st.sidebar.markdown("---")
    
    # Data loading section
    st.sidebar.header("Data Sources")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF", 
        type="pdf",
        help="Upload a custom PDF to include its content as a data source."
    )
    
    with st.sidebar.expander("Loading Status", expanded=True):
        with st.spinner("Connecting to data sources..."):
            st.session_state.retriever = load_data_and_create_retriever(uploaded_file)
            
    if st.session_state.retriever is None:
        st.error("Failed to load data. The chatbot will be unavailable. Please check configurations.")
        st.stop()
    
    st.sidebar.success("Data sources loaded successfully!")
    st.sidebar.markdown("---")
    st.sidebar.info("Built with ❤️ by an AI assistant.")

    if page == "🏠 Home":
        home_page()
    elif page == "📄 View Source Code":
        source_code_page()

if __name__ == "__main__":
    main()