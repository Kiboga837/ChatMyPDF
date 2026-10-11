"""
Safe data-analysis tools + a real Excel-formula engine.

The AI model never runs free-form code. It calls the small, fixed set of tools below
(EDA, variable classification, pivot tables, charts, Excel formulas ...). Every number
therefore comes from real computation on the uploaded sheet.

Convention used everywhere: row 1 of a sheet holds the column headers, data starts in
row 2, and columns are lettered A, B, C ... exactly as in Excel.
"""
import functools
from typing import Optional
import io
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cleaning  # noqa: E402
import formulas  # noqa: E402
import schedula as sh  # noqa: E402

EPOCH = pd.Timestamp("1899-12-30")
AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std"}
ORDINAL_WORDS = {"low", "medium", "high", "small", "large", "poor", "fair", "good", "excellent", "bad",
                 "average", "none", "mild", "moderate", "severe", "very low", "very high", "very good",
                 "strongly disagree", "disagree", "neutral", "agree", "strongly agree"}


# ----------------------------------------------------------------------------- helpers
def col_letter(i):
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def _py(x, depth=0):
    """Convert numpy / formulas objects to plain Python values for display or JSON."""
    if x is sh.EMPTY or x is None:
        return ""
    if depth > 4:
        return str(x)
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return _py(x.reshape(-1)[0], depth + 1)
        arr = x[:50, :20] if x.ndim == 2 else x[:50]
        return [_py(v, depth + 1) for v in arr.tolist()] if arr.ndim == 1 else \
            [[_py(v, depth + 1) for v in row] for row in arr.tolist()]
    if isinstance(x, dict):
        return {str(k): _py(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v, depth + 1) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 10)
    if isinstance(x, str):
        return x
    return str(x)  # Excel errors such as #DIV/0!, dates, etc.


def _cell(v):
    try:
        if pd.isna(v):
            return sh.EMPTY
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return (v - EPOCH).total_seconds() / 86400
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def shift_formula(fx, offset):
    """Fill-down: move relative row numbers in an Excel formula by `offset` rows."""
    pat = re.compile(r"(?<![A-Za-z0-9_.])(\$?[A-Z]{1,3})(\$?)(\d+)(?![A-Za-z0-9_(!])")
    out = []
    for i, part in enumerate(re.split(r'("[^"]*")', fx)):
        if i % 2 == 1:  # text inside quotes is left alone
            out.append(part)
        else:
            out.append(pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) if m.group(2) else int(m.group(3)) + offset}", part))
    return "".join(out)


def _csv(df, max_rows=40, index=False):
    note = f"\n... ({len(df) - max_rows} more rows not shown)" if len(df) > max_rows else ""
    d = df.head(max_rows).copy()
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].round(4)
    return d.to_csv(index=index) + note


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return _py(fn(*a, **k))
        except Exception as e:  # report to the model so it can correct itself
            return {"error": f"{type(e).__name__}: {e}"}
    return wrapper


def classify_series(s):
    """Return (variable type, measurement scale, suggested use) for one column."""
    n = s.dropna()
    total, nun = len(n), n.nunique()
    name = str(s.name).lower()
    if total == 0:
        return "empty", "n/a", "drop"
    if pd.api.types.is_bool_dtype(s):
        return "binary (boolean)", "nominal", "group by / filter"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", "interval", "time trends, group by month/year"
    if pd.api.types.is_numeric_dtype(s):
        if nun == 2:
            return "binary (numeric 0/1)", "nominal", "group by / filter"
        is_int = bool((n % 1 == 0).all())
        if is_int and total > 5 and nun == total and (name.endswith("id") or name.endswith("no") or "code" in name or "number" in name or nun == total):
            return "identifier (unique key)", "nominal", "do not aggregate"
        if is_int and nun <= 10:
            return "numeric discrete (few levels; may be ordinal/categorical code)", "ordinal/ratio", "group by or aggregate"
        if is_int:
            return "numeric discrete (count-like)", "ratio", "aggregate (sum/mean)"
        return "numeric continuous", "ratio", "aggregate (sum/mean), histogram, correlation"
    vals = n.astype(str)
    if not vals.str.fullmatch(r"[\d.,\-\s]+").all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(vals.head(100), errors="coerce")
        if parsed.notna().mean() > 0.9:
            return "datetime (stored as text)", "interval", "convert to date first"
    if vals.str.len().mean() > 40:
        return "free text", "n/a", "text analysis only"
    if nun == total and total > 20:
        return "identifier / unique text", "nominal", "do not aggregate"
    if nun <= 2:
        return "binary (categorical)", "nominal", "group by / filter"
    if nun >= 3 and set(v.lower() for v in vals.unique()) <= ORDINAL_WORDS:
        return "categorical ordinal", "ordinal", "group by (keep order)"
    if nun <= 25 or nun / total <= 0.05:
        return "categorical nominal", "nominal", "group by / pivot rows / pie or bar chart"
    return "categorical high-cardinality", "nominal", "group by with care (many groups)"


# ----------------------------------------------------------------------------- workspace
class Workspace:
    """Holds the uploaded spreadsheets for one chat session and exposes the AI tools."""

    def __init__(self):
        self.frames = {}   # dataset name -> DataFrame
        self.files = {}    # file name -> [dataset names]
        self.calc = {}     # dataset -> {column: excel formula written for row 2}
        self.tables = []   # (title, DataFrame) shown to the user / exported
        self.figures = []  # (title, matplotlib figure)
        self._cols = {}    # cache of converted columns
        self.originals = {}      # dataset -> copy before cleaning (for reset)
        self.clean_log = {}      # dataset -> cleaning log lines
        self.hist_tables = []    # every result table of the session (for the report)
        self.hist_figs = []      # every chart of the session as PNG bytes (for the report)
        self.formulas_used = []  # (dataset, formula, result) (for the report)

    # ---- loading ------------------------------------------------------------------
    def add_file(self, name, data):
        keys = []
        if name.lower().endswith(".csv"):
            for enc in ("utf-8", "latin-1"):
                try:
                    sheets = {"": pd.read_csv(io.BytesIO(data), encoding=enc)}
                    break
                except UnicodeDecodeError:
                    continue
        else:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        for sname, df in sheets.items():
            df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            df.columns = [str(c) for c in df.columns]
            key = f"{name} / {sname}" if sname else name
            self.frames[key] = df
            keys.append(key)
        self.files[name] = keys
        return keys

    def remove_file(self, name):
        for k in self.files.pop(name, []):
            self.frames.pop(k, None)
            self.calc.pop(k, None)
            self.originals.pop(k, None)
            self.clean_log.pop(k, None)
        self._cols.clear()

    def profile(self):
        """Compact description given to the AI so it knows what data exists."""
        parts = []
        for key, df in self.frames.items():
            cols = ", ".join(f"{col_letter(i)}={c} [{str(df[c].dtype)}]" for i, c in enumerate(df.columns[:60]))
            parts.append(f'DATASET "{key}": {len(df)} data rows (rows 2-{len(df) + 1}), {df.shape[1]} columns.\n'
                         f"Columns (Excel letter=header): {cols}\nFirst rows:\n{_csv(df, 4)}\n"
                         f"AUTOMATIC DIGEST (exact numbers computed by code):\n{self.digest(df)}")
        return "\n".join(parts)


    def digest(self, df):
        """Exact, code-computed overview that is handed to the AI up front (saves AI calls)."""
        out = [f"Rows: {len(df)}, columns: {df.shape[1]}, duplicate rows: {int(df.duplicated().sum())}, "
               f"missing cells: {int(df.isna().sum().sum())}"]
        cls = self._class_table(df.iloc[:, :40])
        out.append("Variable classification:\n" + _csv(cls.drop(columns=["examples", "suggested_use"]), 40))
        num = df.select_dtypes("number")
        if not num.empty:
            d = num.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
            out.append("Numeric summary:\n" + _csv(d.iloc[:30], 30, index=True))
            outl = {}
            for c in num.columns[:30]:
                x = num[c].dropna()
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                k = int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())
                if k:
                    outl[c] = k
            out.append(f"IQR outlier counts: {outl or 'none'}")
            if num.shape[1] >= 2:
                cm = num.iloc[:, :20].corr()
                pairs = sorted(((a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:]
                                if pd.notna(cm.loc[a, b])), key=lambda p: -abs(p[2]))[:6]
                out.append("Strongest correlations: " + "; ".join(f"{a} vs {b} = {r:.2f}" for a, b, r in pairs))
        cats = []
        for c in df.columns[:40]:
            if c not in num.columns and 1 < df[c].nunique() <= 25:
                vc = df[c].value_counts().head(5)
                cats.append(f"{c}: " + ", ".join(f"{k} ({v})" for k, v in vc.items()))
        if cats:
            out.append("Top categories:\n" + "\n".join(cats))
        return "\n".join(out)[:9000]

    # ---- internal lookups ---------------------------------------------------------
    def _resolve(self, dataset):
        if not self.frames:
            raise ValueError("No spreadsheet data is loaded.")
        d = (dataset or "").strip().lower()
        if d in {k.lower() for k in self.frames}:
            return next(k for k in self.frames if k.lower() == d)
        by_sheet = [k for k in self.frames if d and k.split(" / ")[-1].lower() == d]
        if len(by_sheet) == 1:
            return by_sheet[0]
        hits = [k for k in self.frames if d and (d in k.lower() or k.lower() in d)]
        if len(hits) == 1:
            return hits[0]
        if len(self.frames) == 1:
            return next(iter(self.frames))
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(self.frames)}")

    def _df(self, dataset):
        key = self._resolve(dataset)
        return key, self.frames[key]

    def _column(self, df, name):
        if name in df.columns:
            return name
        low = {c.lower(): c for c in df.columns}
        if str(name).lower() in low:
            return low[str(name).lower()]
        if re.fullmatch(r"[A-Za-z]{1,3}", str(name)) and col_index(str(name).upper()) < df.shape[1]:
            return df.columns[col_index(str(name).upper())]
        raise ValueError(f"Column '{name}' not found. Columns: {list(df.columns)}")

    def _col_values(self, key, ci):
        """Excel view of one column: [header, row2, row3, ...]."""
        ck = (key, ci)
        if ck not in self._cols:
            df = self.frames[key]
            self._cols[ck] = [df.columns[ci]] + [_cell(v) for v in df.iloc[:, ci].tolist()]
        return self._cols[ck]

    def _ref_value(self, key, name):
        """Turn a reference such as B2, B2:B9, A:A or Sheet2!A1:C9 into a value / array."""
        sheet, _, ref = name.rpartition("!") if "!" in name else ("", "", name)
        sheet = sheet.strip("'")
        if sheet:
            file = next((f for f, ks in self.files.items() if key in ks), None)
            cands = [k for k in self.files.get(file, []) if k.lower().endswith(f" / {sheet.lower()}")]
            if not cands:
                raise ValueError(f"Sheet '{sheet}' not found.")
            key = cands[0]
        m = re.fullmatch(r"\$?([A-Z]{1,3})?\$?(\d+)?(?::\$?([A-Z]{1,3})?\$?(\d+)?)?", ref.upper())
        if not m or not (m.group(1) or m.group(3)):
            raise ValueError(f"Unsupported reference '{name}'.")
        c1, r1, c2, r2 = m.groups()
        n_rows = len(self.frames[key]) + 1
        n_cols = self.frames[key].shape[1]
        single = ":" not in ref
        c2 = c2 or c1
        r1 = int(r1) if r1 else 1
        r2 = int(r2) if r2 else (r1 if single else n_rows)
        if single:
            r2 = r1
        ci1, ci2 = col_index(c1), col_index(c2)

        def val(ci, r):
            if ci >= n_cols or r > n_rows or r < 1:
                return sh.EMPTY
            return self._col_values(key, ci)[r - 1]

        if single:
            return val(ci1, r1)
        arr = np.empty((r2 - r1 + 1, ci2 - ci1 + 1), dtype=object)
        for j, ci in enumerate(range(ci1, ci2 + 1)):
            for i, r in enumerate(range(r1, r2 + 1)):
                arr[i, j] = val(ci, r)
        return arr

    def _compile(self, formula):
        fx = formula.strip()
        if not fx.startswith("="):
            fx = "=" + fx
        return fx, formulas.Parser().ast(fx)[1].compile()

    # ---- display registry ---------------------------------------------------------
    def reset_outputs(self):
        for _, fig in self.figures:
            plt.close(fig)
        self.tables, self.figures = [], []

    def _table(self, title, df, index=False):
        t = df.reset_index() if index else df
        self.tables.append((title, t))
        self.hist_tables = (self.hist_tables + [(title, t.head(300).copy())])[-30:]

    def _figure(self, title, fig):
        self.figures.append((title, fig))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        self.hist_figs = (self.hist_figs + [(title, buf.getvalue())])[-20:]

    def reset_history(self):
        self.hist_tables, self.hist_figs, self.formulas_used = [], [], []

    # ---- cleaning -----------------------------------------------------------------
    def quality(self, key):
        return cleaning.quality_report(self.frames[key])

    def apply_cleaning(self, key, opts):
        if key not in self.originals:
            self.originals[key] = self.frames[key].copy()
        new, log = cleaning.clean_dataframe(self.frames[key], opts)
        self.frames[key] = new
        self.calc.pop(key, None)
        self._cols = {k: v for k, v in self._cols.items() if k[0] != key}
        self.clean_log.setdefault(key, []).extend(log + [""])
        return log

    def reset_cleaning(self, key):
        if key in self.originals:
            self.frames[key] = self.originals.pop(key)
        self.clean_log.pop(key, None)
        self.calc.pop(key, None)
        self._cols = {k: v for k, v in self._cols.items() if k[0] != key}

    # ---- export -------------------------------------------------------------------
    def export_workbook(self, include_tables=True):
        """Excel file with every dataset (calculated columns as live formulas) and result tables."""
        bio = io.BytesIO()
        used = set()

        def sheet_name(txt):
            base = re.sub(r"[\[\]:*?/\\]", "", txt.split(" / ")[-1] or txt)[:28] or "Sheet"
            name, i = base, 2
            while name.lower() in used:
                name, i = f"{base[:25]}_{i}", i + 1
            used.add(name.lower())
            return name

        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            names = {}
            for key, df in self.frames.items():
                names[key] = sheet_name(key)
                df.to_excel(xw, sheet_name=names[key], index=False)
            for i, (title, tdf) in enumerate(self.tables if include_tables else [], 1):
                nm = re.sub(r"[\[\]:*?/\\]", "", f"R{i} {title}")[:31]
                used.add(nm.lower())
                tdf.to_excel(xw, sheet_name=nm, index=False)
            for key, cols in self.calc.items():
                ws = xw.book[names[key]]
                df = self.frames[key]
                for col, fx in cols.items():
                    if col in df.columns:
                        ci = list(df.columns).index(col) + 1
                        for r in range(2, len(df) + 2):
                            ws.cell(r, ci).value = shift_formula(fx, r - 2)
        return bio.getvalue()

    # ================================================================== AI TOOLS ====
    @_safe
    def describe_dataset(self, dataset: str) -> dict:
        """Data summary of one sheet: size, duplicates, missing values, numeric statistics and categorical summaries."""
        key, df = self._df(dataset)
        miss = df.isna().sum()
        miss_df = pd.DataFrame({"column": df.columns, "missing": miss.values,
                                "missing_pct": (miss.values / max(len(df), 1) * 100).round(2)})
        out = {"dataset": key, "rows": len(df), "columns": df.shape[1],
               "duplicate_rows": int(df.duplicated().sum()),
               "missing_values": _csv(miss_df[miss_df.missing > 0]) or "none"}
        num = df.select_dtypes("number")
        if not num.empty:
            d = num.describe().T
            d["missing"] = num.isna().sum()
            out["numeric_summary"] = _csv(d, index=True)
            self._table(f"Numeric summary - {key}", d, index=True)
        cat = df.select_dtypes(exclude="number")
        if not cat.empty:
            rows = [{"column": c, "unique": int(cat[c].nunique()), "most_common": str(cat[c].mode().iloc[0]) if cat[c].notna().any() else "",
                     "top_freq": int(cat[c].value_counts().iloc[0]) if cat[c].notna().any() else 0} for c in cat.columns]
            cdf = pd.DataFrame(rows)
            out["categorical_summary"] = _csv(cdf)
            self._table(f"Categorical summary - {key}", cdf)
        self._table(f"Missing values - {key}", miss_df)
        return out

    def _class_table(self, df):
        rows = []
        for i, c in enumerate(df.columns):
            vtype, scale, use = classify_series(df[c])
            ex = df[c].dropna().astype(str).unique()[:3]
            rows.append({"excel_col": col_letter(i), "column": c, "pandas_dtype": str(df[c].dtype), "variable_type": vtype,
                         "measurement_scale": scale, "unique_values": int(df[c].nunique()),
                         "missing": int(df[c].isna().sum()), "examples": ", ".join(ex), "suggested_use": use})
        return pd.DataFrame(rows)

    @_safe
    def classify_variables(self, dataset: str) -> dict:
        """Classify every column: numeric continuous/discrete, categorical nominal/ordinal, binary, datetime, identifier, free text; with measurement scale and suggested use."""
        key, df = self._df(dataset)
        t = self._class_table(df)
        self._table(f"Variable classification - {key}", t)
        return {"dataset": key, "classification": _csv(t, 80)}

    @_safe
    def correlations(self, dataset: str, method: str = "pearson") -> dict:
        """Correlation matrix of the numeric columns (method: pearson, spearman or kendall) and the strongest pairs."""
        key, df = self._df(dataset)
        num = df.select_dtypes("number")
        if num.shape[1] < 2:
            return {"error": "Fewer than two numeric columns."}
        cm = num.corr(method=method)
        pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:] if pd.notna(cm.loc[a, b])]
        pairs.sort(key=lambda p: -abs(p[2]))
        self._table(f"Correlation matrix ({method}) - {key}", cm.round(3), index=True)
        return {"strongest_pairs": [f"{a} vs {b}: {r:.3f}" for a, b, r in pairs[:10]], "matrix": _csv(cm, 30, index=True)}

    @_safe
    def value_counts(self, dataset: str, column: str, top_n: int = 15) -> dict:
        """Frequency table (count and percent) for one column."""
        key, df = self._df(dataset)
        c = self._column(df, column)
        vc = df[c].value_counts(dropna=False).head(top_n)
        t = pd.DataFrame({c: vc.index.astype(str), "count": vc.values, "percent": (vc.values / len(df) * 100).round(2)})
        self._table(f"Value counts - {c}", t)
        return {"column": c, "unique_values": int(df[c].nunique()), "table": _csv(t)}

    @_safe
    def detect_outliers(self, dataset: str, column: str) -> dict:
        """Find outliers in one numeric column using the IQR rule (1.5 x IQR) and z-scores (|z| > 3)."""
        key, df = self._df(dataset)
        c = self._column(df, column)
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = s[(s < lo) | (s > hi)]
        z = ((s - s.mean()) / s.std(ddof=0)).abs() if s.std(ddof=0) else s * 0
        self._table(f"Outliers - {c}", pd.DataFrame({"row_in_excel": out.index + 2, c: out.values}).head(500))
        return {"column": c, "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lo, "upper_bound": hi,
                "iqr_outliers": int(len(out)), "zscore_outliers": int((z > 3).sum()),
                "examples": [f"row {i + 2}: {v}" for i, v in out.head(10).items()]}

    @_safe
    def filter_rows(self, dataset: str, conditions: list[str], sort_by: str = "", ascending: bool = True, limit: int = 20) -> dict:
        """Filter and sort rows. Each condition is 'column|operator|value' with operator one of ==, !=, >, >=, <, <=, contains, startswith, endswith, isnull, notnull (all conditions must hold)."""
        key, df = self._df(dataset)
        mask = pd.Series(True, index=df.index)
        for cond in conditions:
            col, op, *rest = [p.strip() for p in cond.split("|")]
            c = self._column(df, col)
            val = rest[0] if rest else ""
            s = df[c]
            if op in ("isnull", "notnull"):
                m = s.isna() if op == "isnull" else s.notna()
            elif op in ("contains", "startswith", "endswith"):
                st = s.astype(str).str.lower()
                v = val.lower()
                m = {"contains": st.str.contains(v, regex=False), "startswith": st.str.startswith(v), "endswith": st.str.endswith(v)}[op]
            else:
                if pd.api.types.is_numeric_dtype(s):
                    v = float(val)
                elif pd.api.types.is_datetime64_any_dtype(s):
                    v = pd.to_datetime(val)
                else:
                    v, s = val, s.astype(str)
                m = {"==": s == v, "=": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
            mask &= m.fillna(False)
        res = df[mask]
        if sort_by:
            res = res.sort_values(self._column(df, sort_by), ascending=ascending)
        self._table(f"Filtered rows - {key}", res.head(1000))
        return {"matching_rows": int(len(res)), "of_total": len(df), "preview": _csv(res, limit)}

    @_safe
    def pivot_table(self, dataset: str, rows: list[str], values: list[str], aggfunc: str = "sum",
                    columns: Optional[list[str]] = None, show_totals: bool = True, top_n: int = 0) -> dict:
        """Excel-style pivot table. rows = row fields, columns = optional column fields, values = fields to aggregate, aggfunc = sum, mean, median, min, max, count, nunique or std. top_n > 0 keeps only the largest top_n row groups."""
        key, df = self._df(dataset)
        agg = aggfunc.lower()
        if agg not in AGGS:
            return {"error": f"aggfunc must be one of {sorted(AGGS)}"}
        r = [self._column(df, c) for c in rows]
        v = [self._column(df, c) for c in values]
        c = [self._column(df, x) for x in (columns or [])]
        pv = pd.pivot_table(df, index=r, columns=c or None, values=v, aggfunc=agg, margins=show_totals, margins_name="Grand Total", observed=True)
        if isinstance(pv.columns, pd.MultiIndex):
            pv.columns = [" | ".join(str(x) for x in col if str(x) != "") for col in pv.columns]
        if top_n and len(v) >= 1:
            body = pv.drop(index="Grand Total", errors="ignore") if show_totals else pv
            order = body.iloc[:, 0].sort_values(ascending=False).head(top_n).index
            pv = pd.concat([body.loc[order], pv.loc[["Grand Total"]]]) if show_totals else body.loc[order]
        flat = pv.reset_index()
        self._table(f"Pivot table - {', '.join(v)} by {', '.join(r)}", flat)
        steps = (f"Insert > PivotTable > pick the sheet; Rows = {', '.join(r)}; "
                 + (f"Columns = {', '.join(c)}; " if c else "")
                 + f"Values = {agg.title()} of {', '.join(v)}.")
        return {"pivot": _csv(flat, 40), "rows_in_pivot": len(flat), "excel_steps": steps}

    @_safe
    def make_chart(self, dataset: str, kind: str, x: str, y: str = "", aggfunc: str = "sum", title: str = "", top_n: int = 15) -> dict:
        """Draw a chart for the user. kind = histogram, bar, line, scatter, box or pie. x = main column; y = optional value column (bar/line aggregate y by x with aggfunc; scatter and box need y). Without y, bar and pie show counts."""
        key, df = self._df(dataset)
        xc = self._column(df, x)
        yc = self._column(df, y) if y else None
        kind = kind.lower()
        fig, ax = plt.subplots(figsize=(7, 4))
        if kind == "histogram":
            ax.hist(pd.to_numeric(df[xc], errors="coerce").dropna(), bins=20, color="#e8590c", edgecolor="white")
            ax.set_xlabel(xc), ax.set_ylabel("Frequency")
        elif kind in ("bar", "pie", "line"):
            if yc:
                data = df.groupby(xc, observed=True)[yc].agg(aggfunc.lower())
                label = f"{aggfunc.title()} of {yc}"
            else:
                data = df[xc].value_counts()
                label = "Count"
            data = data.sort_index() if kind == "line" else data.sort_values(ascending=False).head(top_n)
            if kind == "pie":
                ax.pie(data.values, labels=data.index.astype(str), autopct="%1.1f%%")
            elif kind == "bar":
                ax.bar(data.index.astype(str), data.values, color="#e8590c")
                ax.set_xlabel(xc), ax.set_ylabel(label)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.plot(data.index.astype(str), data.values, marker="o", color="#e8590c")
                ax.set_xlabel(xc), ax.set_ylabel(label)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        elif kind == "scatter":
            ax.scatter(pd.to_numeric(df[xc], errors="coerce"), pd.to_numeric(df[yc], errors="coerce"), s=14, alpha=0.6, color="#e8590c")
            ax.set_xlabel(xc), ax.set_ylabel(yc)
        elif kind == "box":
            if yc:
                groups = [g[yc].dropna().values for _, g in df.groupby(xc, observed=True)][:top_n]
                ax.boxplot(groups, labels=[str(k) for k, _ in df.groupby(xc, observed=True)][:top_n])
                ax.set_xlabel(xc), ax.set_ylabel(yc)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.boxplot(pd.to_numeric(df[xc], errors="coerce").dropna())
                ax.set_ylabel(xc)
        else:
            plt.close(fig)
            return {"error": "kind must be histogram, bar, line, scatter, box or pie"}
        ax.set_title(title or f"{kind.title()} of {xc}" + (f" by {yc}" if yc else ""))
        fig.tight_layout()
        self._figure(title or kind, fig)
        return {"status": "Chart displayed to the user below the answer."}

    @_safe
    def eda_report(self, dataset: str) -> dict:
        """Full exploratory data analysis of one sheet: summary, missing values, duplicates, variable classification, correlations, outliers, plus histograms, correlation heatmap and category charts."""
        key, df = self._df(dataset)
        out = {"summary": self.describe_dataset(key), "variables": self.classify_variables(key)["classification"]}
        num = df.select_dtypes("number")
        if num.shape[1] >= 2:
            out["correlations"] = self.correlations(key)
        outl = {}
        for c in num.columns[:15]:
            s = num[c].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            n_out = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
            if n_out:
                outl[c] = n_out
        out["iqr_outlier_counts"] = outl or "none found"
        if not num.empty:
            cols = list(num.columns[:6])
            ncol = min(3, len(cols))
            nrow = -(-len(cols) // ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
            for ax, c in zip(axes.ravel(), cols):
                ax.hist(num[c].dropna(), bins=20, color="#e8590c", edgecolor="white")
                ax.set_title(c, fontsize=9)
            for ax in axes.ravel()[len(cols):]:
                ax.axis("off")
            fig.suptitle("Distributions of numeric columns")
            fig.tight_layout()
            self._figure("Distributions", fig)
        if num.shape[1] >= 2:
            cm = num.iloc[:, :12].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cm))), ax.set_xticklabels(cm.columns, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(cm))), ax.set_yticklabels(cm.columns, fontsize=8)
            fig.colorbar(im), ax.set_title("Correlation heatmap")
            fig.tight_layout()
            self._figure("Correlation heatmap", fig)
        cats = [c for c in df.columns if c not in num.columns and 1 < df[c].nunique() <= 25][:3]
        if cats:
            fig, axes = plt.subplots(1, len(cats), figsize=(4 * len(cats), 3.5), squeeze=False)
            for ax, c in zip(axes[0], cats):
                vc = df[c].value_counts().head(10)
                ax.bar(vc.index.astype(str), vc.values, color="#e8590c")
                ax.set_title(c, fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
            fig.tight_layout()
            self._figure("Top categories", fig)
        return out

    @_safe
    def evaluate_excel_formula(self, dataset: str, formula: str) -> dict:
        """Evaluate a real Excel formula on a sheet and return its result. Row 1 holds headers, data starts at row 2, columns are lettered A, B, C... Whole-column references (D:D) and ranges (D2:D100) work (SUBTOTAL and a few exotic functions are not supported), as do functions such as SUM, AVERAGE, COUNTIF, SUMIFS, AVERAGEIF, IF, VLOOKUP, INDEX/MATCH, XLOOKUP, ROUND, MEDIAN, STDEV, SUMPRODUCT, TEXT, PMT, NPV and plain arithmetic like =SUM(D:D)*1.16."""
        key, _ = self._df(dataset)
        fx, func = self._compile(formula)
        args = [self._ref_value(key, name) for name in func.inputs]
        res = _py(func(*args))
        self.formulas_used.append((key, fx, str(res)[:120]))
        return {"formula": fx, "result": res}

    @_safe
    def add_calculated_column(self, dataset: str, new_column: str, excel_formula: str) -> dict:
        """Add a new column computed row by row, like typing an Excel formula in row 2 and filling it down. Write the formula for row 2, e.g. =D2*E2*1.16 or =IF(F2>1000,"High","Low"). The column stays available for later questions and is exported with live formulas."""
        key, df = self._df(dataset)
        fx, func = self._compile(excel_formula)
        up = fx.upper()
        locked = {re.sub(r"\$", "", t) for t in re.findall(r"(?<![A-Za-z0-9_.])\$?[A-Z]{1,3}\$\d+", up)}

        def shifted(name, off):
            sheet, bang, ref = name.rpartition("!")
            parts = []
            for ep in ref.split(":"):
                m = re.fullmatch(r"([A-Z]{1,3})(\d+)", ep)
                parts.append(ep if not m or ep in locked else f"{m.group(1)}{int(m.group(2)) + off}")
            return f"{sheet}{bang}{':'.join(parts)}"

        names = list(func.inputs)
        if len(df) > 50000:
            return {"error": "Sheet too large for row-by-row formulas (limit 50,000 rows)."}
        vals = []
        for i in range(len(df)):
            res = func(*[self._ref_value(key, shifted(n, i)) for n in names])
            vals.append(_py(res))
        s = pd.Series(vals, index=df.index)
        try:
            s = pd.to_numeric(s)
        except (ValueError, TypeError):
            pass
        df[new_column] = s
        self._cols = {k: v for k, v in self._cols.items() if k[0] != key}
        self.calc.setdefault(key, {})[new_column] = fx
        self.formulas_used.append((key, fx, f"new column '{new_column}' (filled down)"))
        info = {"dataset": key, "new_column": new_column, "excel_column": col_letter(df.shape[1] - 1),
                "formula_row2": fx, "preview": _csv(df[[new_column]], 8)}
        if pd.api.types.is_numeric_dtype(s):
            info["sum"], info["mean"] = float(s.sum()), float(s.mean())
        return info
