"""
Chat With Your PDF  -  Streamlit + FAISS + free local models (no API key needed)

Install:
    pip install streamlit pypdf langchain-text-splitters faiss-cpu sentence-transformers transformers torch

Run:
    streamlit run app.py

First run downloads two small models (about 1.2 GB total), so you need internet once.
"""
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------- 1. Page config ----------------
st.set_page_config(page_title="Chat With Your PDF", page_icon="📄", layout="wide")
st.title("📄 Chat With Your PDF")
st.caption("Upload a PDF and ask questions about its content.")

SAMPLE_QUESTIONS = [
    "What is this document about?",
    "List the key points.",
    "What are the main steps mentioned?",
    "Which tools are mentioned?",
    "Summarize this document.",
]

# ---------------- 2. Models (loaded once, cached) ----------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading language model (first run downloads it)...")
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")


# ---------------- 3. PDF text extraction ----------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages), len(reader.pages)


# ---------------- 4. Vector database ----------------
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(text)
    emb = load_embeddings().encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))
    return index, chunks


# ---------------- 5. Question answering ----------------
def answer_question(question, vectorstore):
    index, chunks = vectorstore
    qv = load_embeddings().encode([question], normalize_embeddings=True)
    _, ids = index.search(np.asarray(qv, dtype="float32"), min(4, len(chunks)))
    context = "\n\n".join(chunks[i] for i in ids[0] if i >= 0)[:3500]
    prompt = (
        "Answer the question in detail using only the context below. "
        'If the answer is not in the context, say "I could not find that in the document."\n\n'
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    llm = load_llm()
    out = llm(prompt, max_new_tokens=256, num_beams=3, no_repeat_ngram_size=3, truncation=True)
    return out[0]["generated_text"].strip()


# ---------------- Session state ----------------
for key, default in (("messages", []), ("pending", None), ("doc_key", None),
                     ("vectorstore", None), ("info", {})):
    if key not in st.session_state:
        st.session_state[key] = default


def clear_chat():
    st.session_state.messages = []


def ask_sample(q):
    st.session_state.pending = q


# ---------------- Layout ----------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("⬆️ Upload PDF")
    uploaded_file = st.file_uploader("Upload PDF File", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None:
        key = (uploaded_file.name, uploaded_file.size)
        if st.session_state.doc_key != key:  # new file -> (re)build index
            with st.spinner("Reading PDF and building search index..."):
                text, n_pages = extract_text_from_pdf(uploaded_file)
                if not text.strip():
                    st.error("No readable text found (the PDF may be scanned images).")
                    st.stop()
                st.session_state.vectorstore = build_vectorstore(text)
                st.session_state.info = {
                    "name": uploaded_file.name,
                    "size": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                    "pages": n_pages,
                    "chars": len(text),
                }
                st.session_state.doc_key = key
                st.session_state.messages = []
        st.success("PDF uploaded successfully!")

        info = st.session_state.info
        st.subheader("📑 Document Info")
        st.write(f"**File Name:** {info['name']}")
        st.write(f"**File Size:** {info['size']}")
        st.write(f"**Total Pages:** {info['pages']}")
        st.write(f"**Total Characters:** {info['chars']:,}")

        st.subheader("💡 Sample Questions")
        for i, q in enumerate(SAMPLE_QUESTIONS):
            st.button(q, key=f"sample_{i}", on_click=ask_sample, args=(q,), use_container_width=True)
    else:
        st.session_state.doc_key = None
        st.session_state.vectorstore = None
        st.info("Upload a PDF to get started.")

typed = st.chat_input("Ask a question about the PDF...", disabled=st.session_state.vectorstore is None)

with right:
    head_l, head_r = st.columns([4, 1])
    head_l.subheader("💬 Chat")
    head_r.button("🗑️ Clear Chat", on_click=clear_chat)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question and st.session_state.vectorstore is not None:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                answer = answer_question(question, st.session_state.vectorstore)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
