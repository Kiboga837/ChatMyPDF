"""
Chat With Your Documents  -  PDF, Word and Excel/CSV
Streamlit + Google Gemini (free API key)

* PDFs and Word files are read completely by the AI on every question.
* Excel/CSV files are analysed with real calculations (EDA, pivot tables, Excel formulas, charts).
* Nothing is saved: documents and chats disappear when the page is refreshed.

Files needed in the same folder: app.py, data_tools.py, requirements.txt
Secret (Streamlit Cloud: Manage app > Settings > Secrets):
    GEMINI_API_KEY = "your-gemini-key"
"""
import io

import streamlit as st
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from google import genai
from google.genai import types
from pypdf import PdfReader

from data_tools import Workspace

st.set_page_config(page_title="Chat With Your Documents", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    .stAppDeployButton, #MainMenu, footer {display: none !important; visibility: hidden !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS = ["gemini-3.8-flash"]  # tried first; other available models are discovered automatically
MAX_PDF_MB = 18
MAX_DOCX_CHARS = 400_000
GENERAL_SAMPLES = ["Summarize the document(s).", "What are the key points?", "What are the main conclusions or recommendations?"]
SHEET_SAMPLES = [
    "Give me a full exploratory data analysis (EDA).",
    "Summarize the data and classify the variables.",
    "Which columns are strongly correlated? Show a chart.",
    "Build a pivot table of the main numeric column by the main category.",
    "Total of the main numeric column with 16% added, using an Excel formula.",
]
SYSTEM = """You are a friendly, expert assistant that chats about documents the user uploaded: PDFs, Word files and Excel/CSV spreadsheets.

DOCUMENTS (PDF / Word): read them completely before answering. Give clear, complete, well-organised answers, mention the document name and page or section when helpful. Base answers on the documents; if something is not in them, say so plainly. You may summarise, explain, compare and extract tables.

SPREADSHEETS (Excel / CSV): you have analysis tools. Follow these rules:
- Never calculate numbers yourself. Use the tools for every figure (totals, averages, counts, percentages, pivots, statistics).
- Sheets follow Excel layout: row 1 = headers, data starts in row 2, columns are lettered A, B, C... (see the column list in the data description). Use dataset names exactly as given.
- For any Excel-style question (arithmetic, SUM, AVERAGE, COUNTIF, SUMIFS, IF, VLOOKUP, INDEX/MATCH, XLOOKUP, ROUND ...) call evaluate_excel_formula with a real Excel formula. In your answer show: the formula(s) used, the result, and a one-line explanation. For row-by-row formulas use add_calculated_column (formula written for row 2).
- Pivot tables: use pivot_table, and also tell the user how to build the same pivot in Excel (the tool returns the steps).
- Exploratory data analysis: use eda_report (or describe_dataset, correlations, detect_outliers, value_counts). Interpret the findings in plain language: data quality problems, missing values, skew, outliers, strong correlations and suggested next steps.
- Classification of variables: use classify_variables and present the result as a table, then explain it.
- Charts: use make_chart. Charts and the full result tables appear below your answer automatically.
- Put the key results in markdown tables inside your answer (the chat is saved as text). Keep each table to about 15 rows and say that the full table can be downloaded.
- If a tool returns an error, fix the call and try again. If a function is not supported (for example SUBTOTAL) use an equivalent formula and say so.
Reply in the language the user writes in. Be concise but complete."""


# ----------------------------------------------------------------------------- secrets
def secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ----------------------------------------------------------------------------- document reading
def docx_text(data):
    d = Document(io.BytesIO(data))
    lines = []
    for child in d.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, d)
            txt = p.text.strip()
            if txt:
                lines.append(("## " if p.style is not None and p.style.name.startswith("Heading") else "") + txt)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, d).rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(lines)[:MAX_DOCX_CHARS]


def read_document(f, ws):
    """Turn an uploaded file into a small dict. Nothing is written to disk."""
    data = f.getvalue()
    name = f.name
    low = name.lower()
    doc = {"name": name, "size": len(data)}
    if low.endswith(".pdf"):
        doc.update(kind="pdf", data=data, info=f"PDF, {len(PdfReader(io.BytesIO(data)).pages)} pages")
    elif low.endswith(".docx"):
        txt = docx_text(data)
        doc.update(kind="docx", text=txt, info=f"Word, {len(txt.split()):,} words")
    else:
        keys = ws.add_file(name, data)
        shape = ", ".join(f"{k.split(' / ')[-1] or 'data'}: {len(ws.frames[k]):,} rows x {ws.frames[k].shape[1]} cols" for k in keys)
        doc.update(kind="sheet", info=f"Spreadsheet ({shape})")
    return doc


# ----------------------------------------------------------------------------- AI
def document_parts(docs, ws):
    parts = []
    for d in docs.values():
        if d["kind"] == "pdf":
            parts.append(types.Part(text=f'[PDF document: "{d["name"]}"]'))
            parts.append(types.Part.from_bytes(data=d["data"], mime_type="application/pdf"))
        elif d["kind"] == "docx":
            parts.append(types.Part(text=f'[Word document: "{d["name"]}"]\n{d["text"]}'))
    if ws.frames:
        parts.append(types.Part(text="[SPREADSHEET DATA - analyse it with your tools]\n" + ws.profile()[:30000]))
    return parts


def discover_models(client):
    try:
        found = []
        for m in client.models.list():
            n = m.name.replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if "generateContent" in acts and "flash" in n and not any(
                    x in n for x in ("image", "tts", "live", "audio", "native", "embedding")):
                found.append(n)
        return sorted([n for n in found if "lite" not in n], reverse=True) + sorted([n for n in found if "lite" in n], reverse=True)
    except Exception:
        return []


def ask_gemini(api_key, docs, ws, history, question, model_override=""):
    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(role="user", parts=document_parts(docs, ws) + [types.Part(text="Here are the documents we will discuss.")]),
        types.Content(role="model", parts=[types.Part(text="I have read everything. What would you like to know?")]),
    ]
    for m in history[-12:]:
        contents.append(types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    cfg = {"system_instruction": SYSTEM}
    if ws.frames:
        cfg["tools"] = [ws.describe_dataset, ws.classify_variables, ws.eda_report, ws.correlations, ws.value_counts,
                        ws.detect_outliers, ws.filter_rows, ws.pivot_table, ws.make_chart,
                        ws.evaluate_excel_formula, ws.add_calculated_column]
        cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(maximum_remote_calls=14)
    config = types.GenerateContentConfig(**cfg)

    models = []
    for n in ([model_override] if model_override else []) + MODELS + discover_models(client):
        if n and n not in models:
            models.append(n)
    errors = []
    for model in models[:6]:
        try:
            ws.reset_outputs()
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            if resp.text:
                return resp.text
            errors.append(f"{model}: empty answer")
        except Exception as e:
            errors.append(f"{model}: {e}")
    if any("429" in e or "RESOURCE_EXHAUSTED" in e for e in errors):
        raise RuntimeError("The free daily/minute quota for this API key has been reached. "
                           "Please wait a while and try again, or use a different key.")
    if any("API key" in e or "API_KEY" in e or "401" in e or "403" in e for e in errors):
        raise RuntimeError("The API key was rejected. Check that it is correct.")
    raise RuntimeError(errors[-1] if errors else "Unknown error")


# ----------------------------------------------------------------------------- state
for key, default in (("messages", []), ("pending", None), ("docs", {}), ("ws", None), ("export", None)):
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.ws is None:
    st.session_state.ws = Workspace()
ws = st.session_state.ws


def clear_chat():
    st.session_state.messages = []
    ws.reset_outputs()
    st.session_state.export = None


def ask_sample(q):
    st.session_state.pending = q


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🔒 Privacy")
    st.caption("Nothing is saved. Your documents and chat disappear when you refresh or close the page.")
    api_key = secret("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password", help="Free key from https://aistudio.google.com/apikey")
        st.caption("Get a free key at aistudio.google.com/apikey")
    with st.expander("Advanced"):
        model_override = st.text_input("Model name (optional)", value="")

# ----------------------------------------------------------------------------- main page
st.title("📄 Chat With Your Documents")
st.caption("Upload PDF, Word or Excel/CSV files and chat with them. The AI reads everything.")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("⬆️ Upload documents")
    uploaded = st.file_uploader("Upload", type=["pdf", "docx", "xlsx", "xlsm", "csv"], accept_multiple_files=True,
                                label_visibility="collapsed")
    current = {f.name: f for f in uploaded}
    docs = st.session_state.docs
    for name in list(docs):                       # removed or replaced files
        if name not in current or docs[name]["size"] != current[name].size:
            if docs[name]["kind"] == "sheet":
                ws.remove_file(name)
            docs.pop(name)
    for name, f in current.items():               # new files
        if name not in docs:
            try:
                docs[name] = read_document(f, ws)
            except Exception as e:
                st.error(f"Could not read {name}: {e}")
    pdf_mb = sum(d["size"] for d in docs.values() if d["kind"] == "pdf") / (1024 * 1024)
    if pdf_mb > MAX_PDF_MB:
        st.error(f"The PDFs add up to {pdf_mb:.1f} MB (limit {MAX_PDF_MB} MB). Remove some.")
        st.stop()

    if docs:
        st.success(f"{len(docs)} document(s) ready")
        st.subheader("📑 Documents")
        for d in docs.values():
            icon = {"pdf": "📕", "docx": "📘", "sheet": "📗"}[d["kind"]]
            st.write(f"{icon} **{d['name']}**  \n{d['size'] / (1024 * 1024):.2f} MB · {d['info']}")
        st.subheader("💡 Sample questions")
        samples = GENERAL_SAMPLES * any(d["kind"] != "sheet" for d in docs.values()) + SHEET_SAMPLES * bool(ws.frames)
        for i, q in enumerate(samples):
            st.button(q, key=f"sample_{i}", on_click=ask_sample, args=(q,), use_container_width=True)
    else:
        st.info("Upload one or more files to get started.")

ready = bool(docs) and bool(api_key)
typed = st.chat_input("Ask anything about your documents...", disabled=not ready)

with right:
    hl, hr = st.columns([3, 2])
    hl.subheader("💬 Chat")
    hr.button("🗑️ Clear chat", on_click=clear_chat)
    if docs and not api_key:
        st.warning("Enter your Gemini API key in the sidebar to start chatting.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question and ready:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Reading and analysing..."):
                try:
                    answer = ask_gemini(api_key, docs, ws, st.session_state.messages, question, model_override.strip())
                    st.markdown(answer)
                    st.session_state.messages += [{"role": "user", "content": question},
                                                  {"role": "assistant", "content": answer}]
                    st.session_state.export = ws.export_workbook() if (ws.tables or ws.calc) else None
                except Exception as e:
                    st.error(str(e))
    elif question and not ready:
        st.warning("Upload your document(s) first (and add the API key if asked).")

    if ws.figures or ws.tables:
        st.divider()
        st.subheader("📎 Results from the latest answer")
        for title, fig in ws.figures:
            st.pyplot(fig)
        if ws.tables:
            with st.expander(f"Full result tables ({len(ws.tables)})"):
                for title, tdf in ws.tables:
                    st.caption(title)
                    st.dataframe(tdf, use_container_width=True)
    if st.session_state.export:
        st.download_button("⬇️ Download results & updated data (.xlsx)", st.session_state.export,
                           file_name="analysis_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
