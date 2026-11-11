"""
Data Processing and Ingestion Module for Online Retail II Dataset
Handles:
- ZIP, Excel, and CSV loading (with sheet concatenation and caching).
- Strict transaction validation and cleaning.
- Detailed audit reporting of rows removed at each filtering stage.
"""

import io
import os
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["Invoice", "InvoiceDate", "Quantity", "Price", "Customer ID"]
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def read_excel_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Read Online Retail II Excel workbook and concatenate both years."""
    book = pd.ExcelFile(io.BytesIO(raw_bytes))
    preferred_sheets = ["Year 2009-2010", "Year 2010-2011"]
    selected_sheets = [s for s in preferred_sheets if s in book.sheet_names]
    if not selected_sheets:
        selected_sheets = list(book.sheet_names)
    if not selected_sheets:
        raise ValueError("No worksheets found in the Excel workbook.")
    
    frames = [pd.read_excel(book, sheet_name=s) for s in selected_sheets]
    return pd.concat(frames, ignore_index=True)


def load_dataset_from_source(source_type: str, uploaded_file=None, local_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset from user upload, packaged local file, or remote UCI download.
    """
    if source_type == "local":
        candidates = [
            local_path,
            "data/online_retail_II.zip",
            "../data/online_retail_II.zip",
            "data/online_retail_II.xlsx",
            "../data/online_retail_II.xlsx",
            "data/online_retail_II.csv",
        ]
        chosen = next((c for c in candidates if c and os.path.exists(c)), None)
        if not chosen:
            raise FileNotFoundError("Local dataset file not found in ./data directory.")
            
        if chosen.endswith(".zip"):
            with zipfile.ZipFile(chosen) as zf:
                xlsx_name = next((n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv"))), None)
                if not xlsx_name:
                    raise FileNotFoundError("No supported spreadsheet found in ZIP.")
                content = zf.read(xlsx_name)
                if xlsx_name.lower().endswith(".csv"):
                    return pd.read_csv(io.BytesIO(content), low_memory=False)
                return read_excel_from_bytes(content)
        elif chosen.endswith((".xlsx", ".xls")):
            with open(chosen, "rb") as f:
                return read_excel_from_bytes(f.read())
        elif chosen.endswith(".csv"):
            return pd.read_csv(chosen, low_memory=False)
            
    elif source_type == "upload":
        if uploaded_file is None:
            raise ValueError("No file uploaded. Please select a file to upload.")
            
        name = uploaded_file.name.lower()
        content = uploaded_file.getvalue()
        
        if name.endswith((".xlsx", ".xls")):
            return read_excel_from_bytes(content)
        elif name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content), low_memory=False)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv"))]
                if not names:
                    raise ValueError("The ZIP archive contains no Excel or CSV files.")
                preferred = next((n for n in names if "online_retail_ii" in n.lower()), names[0])
                file_bytes = zf.read(preferred)
                if preferred.lower().endswith(".csv"):
                    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
                return read_excel_from_bytes(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {name}. Upload .zip, .xlsx, .xls, or .csv.")
            
    elif source_type == "download":
        import requests
        resp = requests.get(UCI_ZIP_URL, timeout=(20, 180))
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xlsx_name = next((n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))), None)
            if not xlsx_name:
                raise FileNotFoundError("The downloaded UCI archive contains no Excel workbook.")
            return read_excel_from_bytes(zf.read(xlsx_name))
            
    raise ValueError(f"Unknown source type: {source_type}")


def clean_and_audit_transactions(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw transactions and generates a comprehensive, auditable filter report.
    
    Filters applied:
    1. Strip column header whitespace and check required columns.
    2. Missing Customer ID removal.
    3. Cancellation invoice removal (Invoice starting with 'C' or 'c').
    4. Quantity <= 0 removal.
    5. Price <= 0 removal.
    6. Invalid InvoiceDate removal.
    7. Calculate TotalPrice = Quantity * Price.
    """
    df = raw_df.copy()
    initial_rows = len(df)
    
    # 1. Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Expected Online Retail II schema: {', '.join(REQUIRED_COLUMNS)}."
        )
        
    # Standardize data types
    df["Invoice"] = df["Invoice"].fillna("").astype(str).str.strip()
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    
    # Audit row removal breakdown
    null_customer_count = int(df["Customer ID"].isna().sum())
    cancellation_count = int((df["Invoice"].str.startswith("C", na=False) | df["Invoice"].str.startswith("c", na=False)).sum())
    non_positive_qty_count = int((df["Quantity"] <= 0).sum())
    non_positive_price_count = int((df["Price"] <= 0).sum())
    null_date_count = int(df["InvoiceDate"].isna().sum())
    
    # Apply combined filter
    valid_mask = (
        df["Customer ID"].notna() &
        (~df["Invoice"].str.startswith("C", na=False)) &
        (~df["Invoice"].str.startswith("c", na=False)) &
        (df["Quantity"] > 0) &
        (df["Price"] > 0) &
        df["InvoiceDate"].notna() &
        (df["Invoice"].ne(""))
    )
    
    clean_df = df[valid_mask].copy()
    clean_df["Customer ID"] = clean_df["Customer ID"].astype("int64")
    clean_df["TotalPrice"] = clean_df["Quantity"] * clean_df["Price"]
    
    final_transactions = len(clean_df)
    if final_transactions == 0:
        raise ValueError("No valid positive, non-cancelled transactions remain after cleaning.")
        
    unique_customers = int(clean_df["Customer ID"].nunique())
    
    audit_report = {
        "initial_rows": initial_rows,
        "null_customer_ids": null_customer_count,
        "cancellations": cancellation_count,
        "non_positive_quantity": non_positive_qty_count,
        "non_positive_price": non_positive_price_count,
        "null_invoice_dates": null_date_count,
        "clean_transactions": final_transactions,
        "unique_customers": unique_customers,
        "retention_rate_pct": round((final_transactions / max(1, initial_rows)) * 100, 2),
        "total_gross_revenue": round(float(clean_df["TotalPrice"].sum()), 2),
        "date_range_start": str(clean_df["InvoiceDate"].min().date()),
        "date_range_end": str(clean_df["InvoiceDate"].max().date()),
    }
    
    return clean_df, audit_report
