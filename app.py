import os
import streamlit as st
from dotenv import load_dotenv
import time
from fpdf import FPDF
import google.generativeai as genai

# Import our utility functions
from utils.pdf_loader import save_uploaded_file, load_pdf_documents
from utils.chunking import split_documents
from utils.embeddings import create_vector_db, load_vector_db
from utils.retriever import retrieve_relevant_chunks
from utils.gemini_handler import generate_answer

# Load environment variables (.env)
load_dotenv()

api_key = os.getenv("AIzaSyDwjt08DB6PkGdJV5_IZOjmVphrBTFy6rY")

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


genai.configure(api_key=AIzaSyDwjt08DB6PkGdJV5_IZOjmVphrBTFy6rY)
st.write(api_key)
# Custom CSS for modern styling
st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .source-chunk {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    [data-theme="dark"] .source-chunk {
        background-color: #1e1e1e;
        border-left: 4px solid #90caf9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False


# ── PDF export (only called on demand) ────────────────────────────────────────
def create_chat_pdf() -> str:
    """Generate a PDF of the current chat history and return its file path."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Study Buddy RAG - Chat History", ln=True, align="C")
    pdf.ln(6)

    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Study Buddy"
        # Safely encode to latin-1; replace unknown chars with '?'
        text = msg["content"].encode("latin-1", errors="replace").decode("latin-1")

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{role}:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 7, text)
        pdf.ln(4)

    pdf_path = "chat_history.pdf"
    pdf.output(pdf_path)
    return pdf_path


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3145/3145765.png", width=100)
    st.title("📚 Document Settings")
    st.write("Upload your textbooks or notes to start studying!")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    process_btn = st.button("Process Documents 🚀", type="primary", use_container_width=True)

    if process_btn:
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Extracting text and building knowledge base..."):
                try:
                    # Save uploads to disk
                    file_paths = [save_uploaded_file(f) for f in uploaded_files]

                    # 1. Load PDFs
                    docs = load_pdf_documents(file_paths)
                    st.info(f"Loaded {len(docs)} pages.")

                    # 2. Chunk
                    chunks = split_documents(docs)
                    st.info(f"Created {len(chunks)} text chunks.")

                    # 3. Build vector DB
                    vector_db = create_vector_db(chunks)
                    st.session_state.vector_db = vector_db
                    st.session_state.is_processed = True
                    st.success("✅ Knowledge base built! Ask your questions below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building knowledge base: {e}")

    st.divider()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat 🗑️", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        # Only generate & offer PDF when there are messages
        if st.session_state.messages:
            try:
                pdf_path = create_chat_pdf()
                with open(pdf_path, "rb") as fh:
                    st.download_button(
                        label="Export PDF 📥",
                        data=fh,
                        file_name="StudyBuddy_Chat.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"PDF export failed: {e}")
        else:
            st.button("Export PDF 📥", disabled=True, use_container_width=True)


# ── MAIN INTERFACE ─────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>Study Buddy RAG 🤖</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Your personal AI tutor. Ask questions strictly from your uploaded materials.</p>",
    unsafe_allow_html=True,
)

# Auto-load existing FAISS index from a previous session
if not st.session_state.is_processed and os.path.exists("vectorstore/faiss_index"):
    try:
        st.session_state.vector_db = load_vector_db()
        st.session_state.is_processed = True
        st.sidebar.success("✅ Loaded existing knowledge base from previous session.")
    except Exception as e:
        st.sidebar.warning(f"Could not load existing knowledge base: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 View Retrieved Context"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(
                        f"<div class='source-chunk'><b>Chunk {idx + 1}:</b><br>{doc.page_content}</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append & show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & show assistant response
    with st.chat_message("assistant"):
        if not st.session_state.is_processed or st.session_state.vector_db is None:
            st.warning("Please upload and process a document first from the sidebar.")
            # Append a placeholder so history stays consistent
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ No knowledge base loaded. Please upload a PDF first.",
                "sources": [],
            })
        else:
            with st.spinner("Thinking..."):
                try:
                    retrieved_docs = retrieve_relevant_chunks(st.session_state.vector_db, prompt)
                    answer = generate_answer(prompt, retrieved_docs)
                except Exception as e:
                    answer = f"An error occurred while generating the answer: {e}"
                    retrieved_docs = []

            # Typing animation
            message_placeholder = st.empty()
            full_response = ""
            for word in answer.split():
                full_response += word + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.04)
            message_placeholder.markdown(full_response.strip())

            # Determine whether to show sources
            no_answer_phrase = "The answer is not available in the uploaded documents."
            has_sources = bool(retrieved_docs) and answer != no_answer_phrase

            if has_sources:
                with st.expander("📄 View Retrieved Context"):
                    for idx, doc in enumerate(retrieved_docs):
                        st.markdown(
                            f"<div class='source-chunk'><b>Chunk {idx + 1}:</b><br>{doc.page_content}</div>",
                            unsafe_allow_html=True,
                        )

            # Persist to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response.strip(),
                "sources": retrieved_docs if has_sources else [],
            })