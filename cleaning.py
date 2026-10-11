"""
Local data cleaning for spreadsheets (no AI calls, so it costs no quota).

quality_report(df)           -> (summary, per-column issues table)   scan before cleaning
clean_dataframe(df, options) -> (cleaned DataFrame, list of log lines)
"""
import re
import warnings

import numpy as np
import pandas as pd

PLACEHOLDERS = {"n/a", "na", "n.a.", "n.a", "null", "none", "nil", "-", "--", "---", "?", "#n/a", "nan",
                "not available", "missing"}
CURRENCY = re.compile(r"(?i)(kes|ksh[s.]*|usd|us\$|[$€£])")
SKIP_NUMERIC_NAMES = ("phone", "mobile", "tel", "zip", "postal", "account", "policy", "national", "passport", "reg")
DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}")

DEFAULTS = dict(
    header_style="clean",        # clean (trim spaces) | snake_case | Title Case | UPPER
    trim_text=True,              # remove extra spaces in text cells
    placeholders_to_blank=True,  # N/A, null, -, ? ... -> empty
    fix_numbers=True,            # "KES 1,200" / "12%" / "(500)" stored as text -> numbers
    fix_dates=True,              # date-looking text -> real dates
    dayfirst=True,               # 03/04/2025 means 3 April (Kenya style)
    drop_empty=True,             # fully empty rows and columns
    drop_duplicates=True,
    case_style="keep",           # keep | Title Case | lower | UPPER  (short-category text columns only)
    drop_col_missing_pct=0,      # drop columns with more than this % missing (0 = off)
    missing_numeric="keep",      # keep | mean | median | zero | drop_rows
    missing_text="keep",         # keep | mode | Unknown | drop_rows
    outliers="keep",             # keep | cap | remove   (IQR rule)
)


# ----------------------------------------------------------------------------- helpers
def _is_text(s):
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _to_number(s):
    t = s.astype("string").str.strip()
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    t = t.str.replace(CURRENCY, "", regex=True).str.replace(",", "", regex=False)
    t = t.str.replace(r"\s+", "", regex=True).str.rstrip("%")
    return pd.to_numeric(t, errors="coerce")


def _number_candidate(name, s):
    """Return the converted series if this text column is really numbers, else None."""
    if any(k in str(name).lower() for k in SKIP_NUMERIC_NAMES):
        return None
    nn = s.dropna().astype(str).str.strip()
    nn = nn[nn != ""]
    if nn.empty or nn.str.match(r"^0\d").any():
        return None
    conv = _to_number(s)
    ok = conv.notna().sum()
    if ok / max(len(nn), 1) >= 0.9 and ok > 0:
        return conv
    return None


def _date_candidate(s, dayfirst):
    nn = s.dropna().astype(str).str.strip()
    nn = nn[nn != ""]
    if nn.empty or nn.str.contains(DATE_LIKE).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    return conv if conv.notna().sum() / max(len(nn), 1) >= 0.9 else None


def _clean_text(s):
    return s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def _iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def _is_key(s):
    n = s.dropna()
    return bool(len(n) > 5 and pd.api.types.is_numeric_dtype(s) and (n % 1 == 0).all() and n.nunique() == len(n))


def _tidy_header(name, style):
    n = re.sub(r"\s+", " ", str(name)).strip()
    if style == "snake_case":
        n = re.sub(r"[^0-9a-zA-Z]+", "_", n).strip("_").lower()
    elif style == "Title Case":
        n = n.title()
    elif style == "UPPER":
        n = n.upper()
    return n or "column"


# ----------------------------------------------------------------------------- scan
def quality_report(df):
    rows = []
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        item = {"column": c, "type": str(s.dtype), "missing": miss, "missing_%": round(miss / max(len(df), 1) * 100, 1),
                "placeholders": 0, "extra_spaces": 0, "numbers_as_text": "", "dates_as_text": "", "case_variants": 0, "outliers": 0}
        if _is_text(s):
            t = s.dropna().astype(str)
            item["placeholders"] = int(t.str.strip().str.lower().isin(PLACEHOLDERS | {""}).sum())
            item["extra_spaces"] = int((t != t.str.replace(r"\s+", " ", regex=True).str.strip()).sum())
            if _number_candidate(c, s) is not None:
                item["numbers_as_text"] = "yes"
            elif _date_candidate(s, True) is not None:
                item["dates_as_text"] = "yes"
            else:
                norm = t.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
                g = pd.DataFrame({"raw": t.str.replace(r"\s+", " ", regex=True).str.strip(), "n": norm}).drop_duplicates()
                item["case_variants"] = int((g.groupby("n").size() > 1).sum())
        elif pd.api.types.is_numeric_dtype(s) and s.nunique() > 10 and not _is_key(s):
            lo, hi = _iqr_bounds(s.dropna())
            item["outliers"] = int(((s < lo) | (s > hi)).sum())
        issues = []
        if miss: issues.append("missing values")
        if item["placeholders"]: issues.append("placeholders")
        if item["extra_spaces"]: issues.append("extra spaces")
        if item["numbers_as_text"]: issues.append("numbers stored as text")
        if item["dates_as_text"]: issues.append("dates stored as text")
        if item["case_variants"]: issues.append("inconsistent spelling/case")
        if item["outliers"]: issues.append("outliers")
        item["issues"] = ", ".join(issues) or "ok"
        rows.append(item)
    issues_df = pd.DataFrame(rows)
    summary = {"rows": len(df), "columns": df.shape[1], "duplicate_rows": int(df.duplicated().sum()),
               "empty_rows": int(df.isna().all(axis=1).sum()), "missing_cells": int(df.isna().sum().sum()),
               "columns_with_issues": int((issues_df["issues"] != "ok").sum())}
    return summary, issues_df


# ----------------------------------------------------------------------------- clean
def clean_dataframe(df, opts=None):
    o = {**DEFAULTS, **(opts or {})}
    df = df.copy()
    log = []
    r0, c0 = df.shape

    # headers
    new = [_tidy_header(c, o["header_style"]) for c in df.columns]
    seen = {}
    for i, n in enumerate(new):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            new[i] = f"{n}_{seen[n]}"
    changed = sum(1 for a, b in zip(df.columns, new) if str(a) != b)
    df.columns = new
    if changed:
        log.append(f"Tidied {changed} column header(s).")

    if o["drop_empty"]:
        n_r = int(df.isna().all(axis=1).sum()); n_c = int(df.isna().all(axis=0).sum())
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if n_r or n_c:
            log.append(f"Removed {n_r} empty row(s) and {n_c} empty column(s).")

    text_cols = [c for c in df.columns if _is_text(df[c])]
    if o["trim_text"] and text_cols:
        cells = 0
        for c in text_cols:
            before = df[c]
            after = _clean_text(before).where(before.notna())
            cells += int((before.notna() & (before.astype(str) != after.astype(str))).sum())
            df[c] = after
        if cells:
            log.append(f"Removed extra spaces in {cells} text cell(s).")
    if o["placeholders_to_blank"] and text_cols:
        cells = 0
        for c in text_cols:
            m = df[c].astype(str).str.strip().str.lower().isin(PLACEHOLDERS | {""}) & df[c].notna()
            cells += int(m.sum())
            df.loc[m, c] = pd.NA
        if cells:
            log.append(f"Turned {cells} placeholder value(s) (N/A, null, -, ? ...) into blanks.")

    if o["fix_numbers"]:
        for c in [c for c in df.columns if _is_text(df[c])]:
            conv = _number_candidate(c, df[c])
            if conv is not None:
                lost = int(df[c].notna().sum() - conv.notna().sum())
                df[c] = conv
                log.append(f"Column '{c}': converted text to numbers" + (f" ({lost} unreadable value(s) blanked)." if lost else "."))
    if o["fix_dates"]:
        for c in [c for c in df.columns if _is_text(df[c])]:
            conv = _date_candidate(df[c], o["dayfirst"])
            if conv is not None:
                lost = int(df[c].notna().sum() - conv.notna().sum())
                df[c] = conv
                log.append(f"Column '{c}': converted text to dates" + (f" ({lost} unreadable value(s) blanked)." if lost else "."))

    if o["case_style"] != "keep":
        done = []
        for c in [c for c in df.columns if _is_text(df[c])]:
            if 1 < df[c].nunique() <= 50:
                s = df[c].astype("string")
                df[c] = {"Title Case": s.str.title(), "lower": s.str.lower(), "UPPER": s.str.upper()}[o["case_style"]]
                done.append(c)
        if done:
            log.append(f"Applied {o['case_style']} to category column(s): {', '.join(done)}.")

    if o["drop_duplicates"]:
        n = int(df.duplicated().sum())
        if n:
            df = df.drop_duplicates()
            log.append(f"Removed {n} duplicate row(s).")

    if o["drop_col_missing_pct"]:
        pct = df.isna().mean() * 100
        drop = [c for c in df.columns if pct[c] > o["drop_col_missing_pct"]]
        if drop:
            df = df.drop(columns=drop)
            log.append(f"Dropped column(s) with more than {o['drop_col_missing_pct']}% missing: {', '.join(drop)}.")

    # missing values
    for c in list(df.columns):
        s = df[c]
        n = int(s.isna().sum())
        if not n:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            how = o["missing_numeric"]
            if how == "drop_rows":
                df = df[df[c].notna()]
                log.append(f"Column '{c}': dropped {n} row(s) with missing values.")
            elif how in ("mean", "median", "zero") and not _is_key(s):
                val = {"mean": s.mean(), "median": s.median(), "zero": 0}[how]
                df[c] = s.fillna(val)
                log.append(f"Column '{c}': filled {n} missing value(s) with the {how if how != 'zero' else 'value 0'}"
                           + (f" ({val:,.4g})." if how != "zero" else "."))
        elif _is_text(s):
            how = o["missing_text"]
            if how == "drop_rows":
                df = df[df[c].notna()]
                log.append(f"Column '{c}': dropped {n} row(s) with missing values.")
            elif how in ("mode", "Unknown"):
                val = (s.mode().iloc[0] if s.notna().any() else "Unknown") if how == "mode" else "Unknown"
                df[c] = s.fillna(val)
                log.append(f"Column '{c}': filled {n} blank(s) with '{val}'.")

    if o["outliers"] != "keep":
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and df[c].nunique() > 10 and not _is_key(df[c])]
        if o["outliers"] == "cap":
            for c in num:
                lo, hi = _iqr_bounds(df[c].dropna())
                n = int(((df[c] < lo) | (df[c] > hi)).sum())
                if n:
                    df[c] = df[c].clip(lo, hi)
                    log.append(f"Column '{c}': capped {n} outlier(s) to [{lo:,.4g}, {hi:,.4g}].")
        else:
            mask = pd.Series(False, index=df.index)
            for c in num:
                lo, hi = _iqr_bounds(df[c].dropna())
                mask |= (df[c] < lo) | (df[c] > hi)
            if int(mask.sum()):
                log.append(f"Removed {int(mask.sum())} row(s) containing outliers (IQR rule).")
                df = df[~mask]

    df = df.reset_index(drop=True)
    for c in df.columns:  # plain numpy dtypes keep every later tool simple
        col = df[c]
        if pd.api.types.is_extension_array_dtype(col.dtype) and pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            col = col.astype("float64")
        elif isinstance(col.dtype, pd.StringDtype):
            col = col.astype(object).where(col.notna(), np.nan)
        if pd.api.types.is_float_dtype(col) and col.notna().all() and len(col) and (col % 1 == 0).all():
            col = col.astype("int64")
        df[c] = col
    log.append(f"Result: {r0:,} rows x {c0} columns  ->  {len(df):,} rows x {df.shape[1]} columns.")
    return df, log
