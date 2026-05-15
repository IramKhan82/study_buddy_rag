import streamlit as st
import os
import tempfile
from pathlib import Path

from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .chunk-box {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #3c4043;
    }
    .answer-box {
        background: #e8f0fe;
        border-radius: 8px;
        padding: 1.2rem;
        font-size: 1rem;
        color: #1a1a2e;
        line-height: 1.7;
    }
    .out-of-scope {
        background: #fce8e6;
        border-left: 4px solid #d93025;
        border-radius: 4px;
        padding: 1rem;
        color: #c5221f;
    }
    .status-ok  { color: #1e8e3e; font-weight: 600; }
    .status-err { color: #d93025; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📚 Study Buddy RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your textbook or notes — ask anything from it!</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF (textbook / notes)",
        type=["pdf"],
        help="Handwritten notes converted to PDF also work.",
    )

    st.divider()
    st.subheader("🔧 Chunking Settings")
    chunk_size    = st.slider("Chunk size (chars)", 300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 300, 100, 25)

    st.divider()
    top_k = st.slider("Top-K chunks to retrieve", 1, 10, 4)

    st.divider()
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    # ── Process button ────────────────────────────────────────────────────────
    process_btn = st.button("🚀 Process Document", use_container_width=True, type="primary")

# ── Document Processing ───────────────────────────────────────────────────────
if process_btn:
    if not gemini_key:
        st.sidebar.error("Please enter your Gemini API key.")
    elif not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    else:
        with st.spinner("Processing document — building vector store…"):
            try:
                # Save upload to a temp file
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                engine = RAGEngine(
                    gemini_api_key=gemini_key,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                n_chunks = engine.ingest(tmp_path)
                os.unlink(tmp_path)

                st.session_state.rag       = engine
                st.session_state.doc_ready = True
                st.session_state.history   = []

                st.sidebar.markdown(
                    f'<span class="status-ok">✅ Ready — {n_chunks} chunks indexed</span>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.sidebar.markdown(
                    f'<span class="status-err">❌ Error: {e}</span>',
                    unsafe_allow_html=True,
                )

# ── Main Q&A Area ─────────────────────────────────────────────────────────────
if not st.session_state.doc_ready:
    st.info("👈  Enter your Gemini API key, upload a PDF, then click **Process Document** to begin.")
else:
    st.success(f"Document loaded. Ask me anything from it!")

    query = st.chat_input("Ask a question about your document…")

    # Render history
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            if item["out_of_scope"]:
                st.markdown(
                    f'<div class="out-of-scope">⚠️ {item["answer"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="answer-box">{item["answer"]}</div>',
                    unsafe_allow_html=True,
                )
            if item.get("chunks") and not item["out_of_scope"]:
                with st.expander("📎 Retrieved source chunks"):
                    for i, chunk in enumerate(item["chunks"], 1):
                        st.markdown(
                            f'<div class="chunk-box"><b>Chunk {i} '
                            f'(page {chunk.get("page","?")})</b><br>{chunk["text"]}</div>',
                            unsafe_allow_html=True,
                        )

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state.rag.query(query, top_k=top_k)

            if result["out_of_scope"]:
                st.markdown(
                    f'<div class="out-of-scope">⚠️ {result["answer"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("📎 Retrieved source chunks"):
                    for i, chunk in enumerate(result["chunks"], 1):
                        st.markdown(
                            f'<div class="chunk-box"><b>Chunk {i} '
                            f'(page {chunk.get("page","?")})</b><br>{chunk["text"]}</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.history.append({
            "question":    query,
            "answer":      result["answer"],
            "out_of_scope": result["out_of_scope"],
            "chunks":      result.get("chunks", []),
        })
