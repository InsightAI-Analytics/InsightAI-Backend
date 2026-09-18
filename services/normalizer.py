"""
Data normalization service.
- Normalizes column names (lowercase, snake_case)
- Detects and converts date columns
- Cleans numeric columns (strips currency symbols, commas)
- Preserves original column name mapping
"""

from __future__ import annotations
import re
from typing import Any

import pandas as pd


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Normalize a DataFrame's column names and types.

    Returns:
        (normalized_df, original_columns_map)
        where original_columns_map is {normalized_name: original_name}
    """
    df = df.copy()

    # 1. Normalize column names
    original_columns: dict[str, str] = {}
    new_columns: list[str] = []
    seen: dict[str, int] = {}

    for col in df.columns:
        normalized = _normalize_column_name(str(col))
        # Handle duplicates
        if normalized in seen:
            seen[normalized] += 1
            normalized = f"{normalized}_{seen[normalized]}"
        else:
            seen[normalized] = 0
        original_columns[normalized] = str(col)
        new_columns.append(normalized)

    df.columns = new_columns

    # 2. Drop fully empty rows and columns
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]  # drop Excel index cols

    # 3. Type inference
    for col in df.columns:
        df[col] = _infer_column_type(df[col])

    df = df.reset_index(drop=True)
    return df, original_columns


def _normalize_column_name(name: str) -> str:
    """Convert a column name to snake_case."""
    # Strip leading/trailing whitespace
    name = name.strip()
    # Replace common separators with underscore
    name = re.sub(r"[\s\-\.\/\\]+", "_", name)
    # Remove non-alphanumeric chars (except underscore)
    name = re.sub(r"[^\w]", "", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Lowercase
    name = name.lower()
    # Strip leading/trailing underscores
    name = name.strip("_")
    # Ensure doesn't start with a digit
    if name and name[0].isdigit():
        name = "col_" + name
    return name or "unnamed"


def _infer_column_type(series: pd.Series) -> pd.Series:
    """Try to convert a series to a more specific dtype."""
    # Already numeric
    if pd.api.types.is_numeric_dtype(series):
        return series

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # Work on non-null string values
    sample = series.dropna().astype(str)
    if sample.empty:
        return series

    # Try numeric (strip currency symbols, commas, percent)
    cleaned = sample.str.replace(r"[₹$€£,%\s]", "", regex=True)
    try:
        numeric_vals = pd.to_numeric(cleaned, errors="raise")
        result = pd.to_numeric(
            series.astype(str).str.replace(r"[₹$€£,%\s]", "", regex=True),
            errors="coerce",
        )
        # Only convert if majority are numeric
        if result.notna().sum() / max(len(series), 1) >= 0.7:
            return result
    except (ValueError, TypeError):
        pass

    # Try datetime
    try:
        dt = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if dt.notna().sum() / max(len(series), 1) >= 0.7:
            return dt
    except Exception:
        pass

    return series


def get_schema_description(
    name: str, df: pd.DataFrame, original_columns: dict[str, str]
) -> dict[str, Any]:
    """
    Build a schema dict suitable for the LLM prompt.
    Includes column names, types, and sample values.
    """
    columns = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(5).tolist()
        sample_vals = [_safe_val(v) for v in sample_vals]
        columns.append(
            {
                "name": col,
                "original": original_columns.get(col, col),
                "dtype": str(df[col].dtype),
                "sample": sample_vals,
            }
        )
    return {"dataset": name, "rows": len(df), "columns": columns}


def _safe_val(val: Any) -> Any:
    """Make a value JSON-serializable."""
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return None
    if hasattr(val, "item"):
        return val.item()
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val
