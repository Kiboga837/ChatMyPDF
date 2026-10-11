"""
Chat With Your PDF  -  Streamlit + Google Gemini (free API key)

The WHOLE PDF is given to the AI on every question, so it can answer like a normal
chatbot (summaries, comparisons, details from any page, tables, even scanned pages).

Local run:
    pip install streamlit pypdf google-genai
    python -m streamlit run app.py

Get a free key at https://aistudio.google.com/apikey
On Streamlit Cloud: Manage app > Settings > Secrets, add:
    GEMINI_API_KEY = "your-key-here"
"""
import io

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="Chat With Your PDF", page_icon="📄", layout="wide")
st.title("📄 Chat With Your PDF")
st.caption("Upload a PDF and chat with it. The AI reads the whole document.")

MODELS = ["gemini-3.8-flash"]  # tried first; other available models are discovered automatically
MAX_PDF_MB = 18
SAMPLE_QUESTIONS = [
    "Summarize this document.",
    "What are the key points?",
    "List the main steps or recommendations.",
    "What tools, names or figures are mentioned?",
    "What are the conclusions?",
]
SYSTEM = (
    "You are a helpful assistant that answers questions about the PDF document the user "
    "uploaded. Read the entire document before answering. Give clear, complete, well-organised "
    "answers and mention page numbers when helpful. Base your answers on the document; if the "
    "answer is not in it, say so plainly. You may summarise, explain, compare, extract tables "
    "and do calculations using the document's content."
)


def get_secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def discover_models(client):
    """Ask Google which Flash models this key can use right now (names change over time)."""
    try:
        found = []
        for m in client.models.list():
            n = m.name.replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if "generateContent" in acts and "flash" in n and not any(
                    x in n for x in ("image", "tts", "live", "audio", "native", "embedding")):
                found.append(n)
        full = sorted([n for n in found if "lite" not in n], reverse=True)
        lite = sorted([n for n in found if "lite" in n], reverse=True)
        return full + lite
    except Exception:
        return []


def ask_gemini(api_key, pdf_bytes, history, question, model_override=""):
    client = genai.Client(api_key=api_key)
    pdf = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    contents = [
        types.Content(role="user", parts=[pdf, types.Part(text="Here is the document we will discuss.")]),
        types.Content(role="model", parts=[types.Part(text="I have read the whole document. What would you like to know?")]),
    ]
    for m in history[-12:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    config = types.GenerateContentConfig(system_instruction=SYSTEM)

    models = []
    for n in ([model_override] if model_override else []) + MODELS + discover_models(client):
        if n and n not in models:
            models.append(n)
    errors = []
    for model in models[:6]:
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            if resp.text:
                return resp.text
            errors.append(f"{model}: empty answer")
        except Exception as e:  # try the next model
            errors.append(f"{model}: {e}")
    if any("429" in e or "RESOURCE_EXHAUSTED" in e for e in errors):
        raise RuntimeError("The free daily/minute quota for this API key has been reached. "
                           "Please wait a while and try again, or use a different key.")
    if any("API key" in e or "API_KEY" in e or "401" in e or "403" in e for e in errors):
        raise RuntimeError("The API key was rejected. Check that it is correct.")
    raise RuntimeError(errors[-1] if errors else "Unknown error")


for key, default in (("messages", []), ("pending", None), ("doc_key", None), ("pdf_bytes", None), ("info", {})):
    if key not in st.session_state:
        st.session_state[key] = default


def clear_chat():
    st.session_state.messages = []


def ask_sample(q):
    st.session_state.pending = q


left, right = st.columns([1, 2], gap="large")

with left:
    secret = get_secret_key()
    api_key = secret
    if not secret:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Free key from https://aistudio.google.com/apikey")
        st.caption("Get a free key at aistudio.google.com/apikey")
    with st.expander("Advanced"):
        model_override = st.text_input("Model name (optional)", value="")

    st.subheader("⬆️ Upload PDF")
    uploaded = st.file_uploader("Upload PDF File", type="pdf", label_visibility="collapsed")

    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        if st.session_state.doc_key != key:
            data = uploaded.getvalue()
            if len(data) > MAX_PDF_MB * 1024 * 1024:
                st.error(f"PDF is larger than {MAX_PDF_MB} MB. Please upload a smaller file.")
                st.stop()
            try:
                reader = PdfReader(io.BytesIO(data))
                pages = len(reader.pages)
                chars = sum(len(p.extract_text() or "") for p in reader.pages)
            except Exception:
                pages, chars = "?", 0
            st.session_state.pdf_bytes = data
            st.session_state.info = {"name": uploaded.name, "size": f"{uploaded.size / (1024 * 1024):.2f} MB",
                                     "pages": pages, "chars": chars}
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
        st.session_state.pdf_bytes = None
        st.info("Upload a PDF to get started.")

ready = st.session_state.pdf_bytes is not None and bool(api_key)
typed = st.chat_input("Ask anything about the PDF...", disabled=not ready)

with right:
    hl, hr = st.columns([4, 1])
    hl.subheader("💬 Chat")
    hr.button("🗑️ Clear Chat", on_click=clear_chat)
    if st.session_state.pdf_bytes is not None and not api_key:
        st.warning("Enter your Gemini API key on the left to start chatting.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question and ready:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Reading the document..."):
                try:
                    answer = ask_gemini(api_key, st.session_state.pdf_bytes,
                                        st.session_state.messages, question, model_override.strip())
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(str(e))
