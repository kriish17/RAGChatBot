# RAG Chatbot with Groq

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** that allows you to upload a PDF document and interactively ask questions about its content. The system leverages **vector embeddings**, **FAISS** for retrieval, and **Groq LLM** for generating responses.

## Features

- Upload PDF documents and split them into manageable chunks.
- Generate embeddings using **HuggingFace Transformers**.
- Store embeddings in **FAISS vector store** for fast retrieval.
- Ask questions about your PDF and get accurate answers.
- Maintains **chat history** during your session.
- Fully implemented with **Streamlit** for an interactive UI.
- Free-to-use

## Tech Stack

- **Python 3.9+**
- **Streamlit** – UI interface
- **LangChain Community** – Document loading, text splitting, and FAISS integration
- **LangChain Groq** – Chat LLM
- **HuggingFace Embeddings** – Sentence transformer embeddings
- **FAISS** – Fast vector search
