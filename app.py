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
import datetime
import io
import re
import time

import streamlit as st
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from google import genai
from google.genai import types
from pypdf import PdfReader

import report
from cleaning import DEFAULTS
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
MISS_NUM = {"Keep blank": "keep", "Fill with mean": "mean", "Fill with median": "median", "Fill with 0": "zero", "Delete those rows": "drop_rows"}
MISS_TXT = {"Keep blank": "keep", "Fill with most common value": "mode", "Fill with 'Unknown'": "Unknown", "Delete those rows": "drop_rows"}
OUTLIERS = {"Keep": "keep", "Cap to normal range (IQR rule)": "cap", "Delete rows with outliers": "remove"}
HEADERS = {"Tidy spaces only": "clean", "snake_case": "snake_case", "Title Case": "Title Case", "UPPER": "UPPER"}
CASES = {"Keep as is": "keep", "Title Case": "Title Case", "lower": "lower", "UPPER": "UPPER"}
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SYSTEM = """You are a friendly, expert assistant that chats about documents the user uploaded: PDFs, Word files and Excel/CSV spreadsheets.

DOCUMENTS (PDF / Word): read them completely before answering. Give clear, complete, well-organised answers, mention the document name and page or section when helpful. Base answers on the documents; if something is not in them, say so plainly. You may summarise, explain, compare and extract tables.

SPREADSHEETS (Excel / CSV): you have analysis tools. Follow these rules:
- Never calculate numbers yourself. Use the tools for every figure (totals, averages, counts, percentages, pivots, statistics).
- The data description already contains an AUTOMATIC DIGEST with exact summary statistics, variable classification, missing values, outliers, correlations and top categories. Use it directly for summary, EDA and classification questions WITHOUT calling tools (call eda_report only if charts are wanted).
- Use as FEW tool calls as possible: request all the tools you need together in one step, and never repeat a call.
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
REPORT_SYSTEM = """You write clear, professional analytical reports. Use ONLY the facts and numbers in the material provided; never invent or estimate numbers. Write in markdown with exactly these sections, each starting with '## ': Executive Summary, Key Findings (bullet points with the actual numbers), Insights and Interpretation, Recommendations and Next Steps, Limitations and Data Quality Notes. Do not write a title, and do not repeat full tables (refer to them as 'see the Analysis results section'). If data cleaning was applied, mention what was done in Limitations and Data Quality Notes."""


def document_parts(docs, ws, sheets=True):
    parts = []
    for d in docs.values():
        if d["kind"] == "pdf":
            parts.append(types.Part(text=f'[PDF document: "{d["name"]}"]'))
            parts.append(types.Part.from_bytes(data=d["data"], mime_type="application/pdf"))
        elif d["kind"] == "docx":
            parts.append(types.Part(text=f'[Word document: "{d["name"]}"]\n{d["text"]}'))
    if sheets and ws.frames:
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


def generate(api_key, contents, config, model_override="", ws=None):
    """Call Gemini, trying several models and waiting once on a short per-minute limit."""
    client = genai.Client(api_key=api_key)
    models = []
    for n in ([model_override] if model_override else []) + MODELS + discover_models(client):
        if n and n not in models:
            models.append(n)
    errors = []
    for model in models[:6]:
        for attempt in (1, 2):
            try:
                if ws is not None:
                    ws.reset_outputs()
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                if resp.text:
                    return resp.text
                errors.append(f"{model}: empty answer")
                break
            except Exception as e:
                msg = str(e)
                errors.append(f"{model}: {msg}")
                quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                m = re.search(r"retry\D{0,15}(\d+(?:\.\d+)?)\s*s", msg, re.I)
                if quota and attempt == 1 and m and float(m.group(1)) <= 25 and "perday" not in msg.lower().replace(" ", ""):
                    time.sleep(float(m.group(1)) + 1)  # short per-minute limit: wait once and retry
                    continue
                break
    if any("perday" in e.lower().replace(" ", "") for e in errors):
        raise RuntimeError("Today's free Gemini quota is used up. It resets once a day (usually mid-morning Nairobi time). "
                           "Try again later, or create a new API key in a NEW project at aistudio.google.com.")
    if any("429" in e or "RESOURCE_EXHAUSTED" in e for e in errors):
        raise RuntimeError("Gemini's free per-minute limit was hit. Wait about a minute and ask again. "
                           "Large PDFs and long analyses use the limit faster.")
    if any("API key" in e or "API_KEY" in e or "401" in e or "403" in e for e in errors):
        raise RuntimeError("The API key was rejected. Check that it is correct.")
    raise RuntimeError(errors[-1] if errors else "Unknown error")


def ask_gemini(api_key, docs, ws, history, question, model_override=""):
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
        cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(maximum_remote_calls=6)
    return generate(api_key, contents, types.GenerateContentConfig(**cfg), model_override, ws)


def write_report_text(api_key, docs, ws, messages, audience, length, focus, model_override=""):
    words = {"Short": "about 250 words", "Standard": "about 500 words", "Detailed": "about 900 words"}[length.split()[0]]
    brief = f"Audience: {audience}. Length: {words}." + (f" Emphasise: {focus.strip()}" if focus.strip() else "")
    parts = document_parts(docs, ws, sheets=False) + [
        types.Part(text=brief + "\n\nMATERIAL (exact facts and numbers computed by code):\n" + report.context_text(ws, messages))]
    contents = [types.Content(role="user", parts=parts)]
    return generate(api_key, contents, types.GenerateContentConfig(system_instruction=REPORT_SYSTEM), model_override)


# ----------------------------------------------------------------------------- state
for key, default in (("messages", []), ("pending", None), ("docs", {}), ("ws", None), ("export", None),
                     ("scan", None), ("clean_msg", None), ("clean_xlsx", None), ("report", None)):
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.ws is None:
    st.session_state.ws = Workspace()
ws = st.session_state.ws


def clear_chat():
    st.session_state.messages = []
    ws.reset_outputs()
    ws.reset_history()
    st.session_state.export = None
    st.session_state.report = None


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
        if ws.frames:
            with st.expander("🧹 Clean data (do this before analysis)"):
                st.caption("Cleaning runs inside the app and uses no AI request.")
                ds = st.selectbox("Spreadsheet", list(ws.frames), key="clean_ds")
                if st.button("🔍 Scan data quality", key="scan_btn", width="stretch"):
                    st.session_state.scan = (ds, *ws.quality(ds))
                scan = st.session_state.scan
                if scan and scan[0] == ds:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", f"{scan[1]['rows']:,}")
                    c2.metric("Duplicates", scan[1]["duplicate_rows"])
                    c3.metric("Missing cells", f"{scan[1]['missing_cells']:,}")
                    st.caption(f"{scan[1]['columns_with_issues']} of {scan[1]['columns']} columns have issues.")
                    st.dataframe(scan[2][["column", "missing", "missing_%", "issues"]], width="stretch", hide_index=True)
                with st.form("clean_form"):
                    st.markdown("**Choose what to clean**")
                    o_trim = st.checkbox("Remove extra spaces in text", True)
                    o_ph = st.checkbox("Turn N/A, null, -, ? into blanks", True)
                    o_num = st.checkbox("Convert numbers stored as text (e.g. KES 1,200)", True)
                    o_date = st.checkbox("Convert dates stored as text", True)
                    o_day = st.checkbox("Dates are day-first (03/04/2025 = 3 April)", True)
                    o_dup = st.checkbox("Remove duplicate rows", True)
                    o_empty = st.checkbox("Remove empty rows and columns", True)
                    o_head = st.selectbox("Column names", list(HEADERS))
                    o_case = st.selectbox("Text in category columns", list(CASES), help="Title Case can change acronyms such as KRA to Kra.")
                    o_mn = st.selectbox("Missing numbers", list(MISS_NUM))
                    o_mt = st.selectbox("Missing text", list(MISS_TXT))
                    o_thr = st.slider("Drop columns with more than this % missing (0 = off)", 0, 100, 0)
                    o_out = st.selectbox("Outliers", list(OUTLIERS))
                    go = st.form_submit_button("🧹 Clean now", width="stretch")
                if go:
                    opts = dict(trim_text=o_trim, placeholders_to_blank=o_ph, fix_numbers=o_num, fix_dates=o_date, dayfirst=o_day,
                                drop_duplicates=o_dup, drop_empty=o_empty, header_style=HEADERS[o_head], case_style=CASES[o_case],
                                missing_numeric=MISS_NUM[o_mn], missing_text=MISS_TXT[o_mt], drop_col_missing_pct=o_thr,
                                outliers=OUTLIERS[o_out])
                    st.session_state.clean_msg = (ds, ws.apply_cleaning(ds, opts))
                    st.session_state.scan = None
                    st.session_state.clean_xlsx = ws.export_workbook(include_tables=False)
                msg = st.session_state.clean_msg
                if msg and msg[0] == ds:
                    st.success("Cleaning done. Your next questions use the cleaned data.")
                    st.markdown("\n".join(f"- {line}" for line in msg[1]))
                    if st.session_state.messages:
                        st.warning("Earlier answers in this chat used the old data. Click **Clear chat** to start fresh.")
                if ds in ws.originals:
                    st.caption("Preview of the cleaned data")
                    st.dataframe(ws.frames[ds].head(10), width="stretch", hide_index=True)
                    b1, b2 = st.columns(2)
                    if b1.button("↩️ Reset to original", key="reset_clean", width="stretch"):
                        ws.reset_cleaning(ds)
                        st.session_state.clean_msg = None
                        st.session_state.clean_xlsx = None
                        st.rerun()
                    if st.session_state.clean_xlsx:
                        b2.download_button("⬇️ Cleaned .xlsx", st.session_state.clean_xlsx, file_name="cleaned_data.xlsx",
                                           mime=XLSX_MIME, width="stretch")
        st.subheader("💡 Sample questions")
        samples = GENERAL_SAMPLES * any(d["kind"] != "sheet" for d in docs.values()) + SHEET_SAMPLES * bool(ws.frames)
        for i, q in enumerate(samples):
            st.button(q, key=f"sample_{i}", on_click=ask_sample, args=(q,), width="stretch")
        with st.expander("📝 Report (after analysis)"):
            if not (ws.hist_tables or ws.hist_figs or st.session_state.messages):
                st.caption("Tip: ask your questions first, then generate the report so it contains the results.")
            r_title = st.text_input("Report title", value="Data Analysis Report", key="r_title")
            r_aud = st.selectbox("Audience", ["Management / executives", "Technical team", "Client / external", "General reader"], key="r_aud")
            r_len = st.selectbox("Length", ["Short (about 1 page)", "Standard", "Detailed"], index=1, key="r_len")
            r_focus = st.text_area("Anything to emphasise? (optional)", key="r_focus", height=68)
            r_ai = st.checkbox("Include AI-written summary, findings and recommendations (uses 1 AI request)", True, key="r_ai")
            k1, k2 = st.columns(2)
            r_tab = k1.checkbox("Result tables", True, key="r_tab")
            r_cha = k2.checkbox("Charts", True, key="r_cha")
            r_for = k1.checkbox("Excel formulas", True, key="r_for")
            r_ovr = k2.checkbox("Data overview", True, key="r_ovr")
            if st.button("📄 Generate report", key="r_go", width="stretch", type="primary"):
                ai_md, warn = "", ""
                if r_ai and not api_key:
                    warn = "No API key, so the AI-written part was skipped."
                elif r_ai:
                    with st.spinner("Writing the report..."):
                        try:
                            ai_md = write_report_text(api_key, docs, ws, st.session_state.messages, r_aud, r_len, r_focus,
                                                      model_override.strip())
                        except Exception as e:
                            warn = f"The AI-written part was skipped: {e}"
                subtitle = f"Prepared {datetime.date.today():%d %B %Y} · Files: {', '.join(docs)}"
                blocks = report.build_blocks(ws, r_title, subtitle, ai_md, r_ovr, r_tab, r_cha, r_for)
                st.session_state.report = {"docx": report.to_docx(blocks), "pdf": report.to_pdf(blocks), "warn": warn,
                                           "name": re.sub(r"[^A-Za-z0-9_-]+", "_", r_title)[:40] or "report"}
            rep = st.session_state.report
            if rep:
                if rep["warn"]:
                    st.warning(rep["warn"])
                st.success("Report ready")
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Word (.docx)", rep["docx"], file_name=f"{rep['name']}.docx", width="stretch",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                d2.download_button("⬇️ PDF", rep["pdf"], file_name=f"{rep['name']}.pdf", mime="application/pdf",
                                   width="stretch")
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
                    st.dataframe(tdf, width="stretch")
    if st.session_state.export:
        st.download_button("⬇️ Download results & updated data (.xlsx)", st.session_state.export,
                           file_name="analysis_results.xlsx",
                           mime=XLSX_MIME)
