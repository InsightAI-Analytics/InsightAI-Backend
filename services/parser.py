"""
File parsing service.
Accepts raw bytes + filename, returns a pandas DataFrame and metadata.
"""

from __future__ import annotations
import io
import re
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class ParseError(Exception):
    """Raised when a file cannot be parsed."""


def _extension(filename: str) -> str:
    lower = filename.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return ""


def parse_file(filename: str, content: bytes) -> pd.DataFrame:
    """
    Parse CSV or Excel file bytes into a DataFrame.

    Raises ParseError on unsupported type, empty file, or parse failure.
    """
    if not content:
        raise ParseError(f"File '{filename}' is empty.")

    ext = _extension(filename)
    if not ext:
        raise ParseError(
            f"Unsupported file type for '{filename}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext == ".csv":
            df = _parse_csv(content)
        else:
            df = _parse_excel(content)
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"Failed to parse '{filename}': {exc}") from exc

    if df.empty:
        raise ParseError(f"File '{filename}' contains no data rows.")

    if len(df.columns) < 1:
        raise ParseError(f"File '{filename}' contains no columns.")

    return df


def _parse_csv(content: bytes) -> pd.DataFrame:
    """Try multiple encodings and delimiters."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    delimiters = [",", ";", "\t", "|"]

    for encoding in encodings:
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue

        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    delimiter=delimiter,
                    on_bad_lines="skip",
                )
                # Accept if we got at least one column (even header-only)
                if len(df.columns) >= 1:
                    return df
            except Exception:
                continue

    raise ParseError("Could not decode or parse CSV file.")


def _parse_excel(content: bytes) -> pd.DataFrame:
    """Parse first sheet of XLSX/XLS."""
    buf = io.BytesIO(content)
    df = pd.read_excel(buf, engine="openpyxl", sheet_name=0)
    return df


def get_file_metadata(df: pd.DataFrame, original_columns: dict[str, str]) -> dict[str, Any]:
    """
    Build a metadata dict describing the DataFrame.
    original_columns: {normalized_name -> original_name}
    """
    column_info = []
    for col in df.columns:
        column_info.append(
            {
                "name": col,
                "original_name": original_columns.get(col, col),
                "dtype": str(df[col].dtype),
            }
        )

    # Safe sample (up to 5 rows), convert to JSON-serializable types
    sample_df = df.head(5).copy()
    sample = _safe_records(sample_df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": column_info,
        "sample": sample,
    }


def _safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to JSON-safe dicts."""
    records = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[col] = None
            elif hasattr(val, "item"):
                rec[col] = val.item()
            else:
                rec[col] = val
        records.append(rec)
    return records
