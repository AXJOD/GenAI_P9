# Multi-Source RAG Q&A System

## Overview

This project is a multi-source Retrieval-Augmented Generation (RAG) question-answering system with a web-based user interface built with Streamlit. The application can ingest data from multiple, varied sources and allow users to ask questions about the combined knowledge base.

## Features

- **Multi-Source Data Ingestion:** The system retrieves information from several sources simultaneously:
  - A user-uploaded PDF file.
  - A default PDF file if none is uploaded.
  - A scientific paper from Arxiv.
  - A local SQLite database.
- **Web-Based UI:** A simple and interactive user interface built with Streamlit.
- **RAG Pipeline:** Utilizes the RAG technique to provide context-aware answers from the loaded documents.
- **Powered by LangChain and Groq:** Uses the LangChain framework for building the RAG pipeline and the Groq API for fast LLM inference.

## Data Sources

The application is configured to load data from the following sources:

1.  **PDF Document:** You can upload your own PDF via the sidebar. If no file is uploaded, it defaults to using the PDF located in the `./pdfs` directory.
2.  **Arxiv Paper:** It loads the famous "Attention Is All You Need" paper (1706.03762) from Arxiv.
3.  **SQLite Database:** It reads data from a local SQLite database named `sample.db`, which contains sample employee information.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+
- `pip` for package management

### Installation

1.  **Clone the repository (if applicable) or download the project files.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    - Create a file named `.env` in the root of the project.
    - Add your Groq API key to the `.env` file:
      ```
      GROQ_API_KEY="your_groq_api_key_here"
      ```
    - You can also set this as a system environment variable.

5.  **Create the SQLite database:**
    - Run the `create_db.py` script to generate the `sample.db` file with sample data:
      ```bash
      python create_db.py
      ```

### Running the Application

1.  **Ensure your virtual environment is activated.**

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  The application will load the default data sources (default PDF, Arxiv, SQLite) when it starts.
2.  Optionally, you can upload your own PDF file using the file uploader in the sidebar. The system will then re-process the data to include your uploaded PDF.
3.  Once the data is loaded, you can ask questions in the chat input box at the bottom of the page.
4.  The system will retrieve relevant information from the combined data sources and generate an answer.
