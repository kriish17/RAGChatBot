import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# ========================
# Page Config
# ========================
st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")

# ========================
# Dark Mode & Custom CSS
# ========================
st.markdown("""
<style>
    body, .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2c2c3c);
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-bubble-user {
        background-color: #4e9af1;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 75%;
        word-wrap: break-word;
        align-self: flex-end;
        font-size: 15px;
    }
    .chat-bubble-bot {
        background-color: #3c3c4a;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 75%;
        word-wrap: break-word;
        align-self: flex-start;
        font-size: 15px;
    }
    .pdf-name {
        background-color: #6c63ff;
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .sidebar .stHeader {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 RAG Chatbot with Groq + PDF Upload")

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.header("📌 How to Use")
    st.markdown("""
    1. Upload your **PDF document**.
    2. Wait for processing.
    3. Ask questions in the chat below.
    """)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun = False  # No deprecated rerun
        st.experimental_rerun = None

# ========================
# API Key Check
# ========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ Please set the `GROQ_API_KEY` environment variable.")
    st.stop()

# ========================
# File Uploader
# ========================
uploaded_file = st.file_uploader("📂 Upload a PDF document", type=["pdf"])

if uploaded_file:
    st.markdown(f"<div class='pdf-name'>📄 {uploaded_file.name}</div>", unsafe_allow_html=True)

    with st.spinner("📄 Processing your PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and split PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Initialize Groq LLM
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

        st.success("✅ PDF processed successfully! Start chatting below.")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat input
        user_input = st.text_input("💬 Ask something about the PDF:")

        if user_input:
            # Add user message
            st.session_state.chat_history.append(("user", user_input))

            # Retrieve relevant docs
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Answer based only on the context below:\n\n{context}\n\nQuestion: {user_input}"

            # Generate bot reply
            with st.spinner("💭 Thinking..."):
                response = llm.invoke(prompt)
                bot_reply = response.content
                st.session_state.chat_history.append(("bot", bot_reply))

        # Display chat history
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"<div class='chat-bubble-user'>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-bot'>{message}</div>", unsafe_allow_html=True)
else:
    st.info("👆 Upload a PDF to start chatting.")

