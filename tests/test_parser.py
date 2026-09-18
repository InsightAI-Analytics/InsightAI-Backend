"""
Unit tests for file parsing service.
"""

import io
import pytest
import pandas as pd

from services.parser import parse_file, ParseError


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_CSV = b"name,revenue,month\nProduct A,1000,Jan\nProduct B,2000,Feb\n"
VALID_CSV_SEMICOLON = b"name;revenue;month\nProduct A;1000;Jan\nProduct B;2000;Feb\n"
EMPTY_CSV = b""
NO_DATA_CSV = b"name,revenue,month\n"
INVALID_FILE = b"\x00\x01\x02\x03\x04\xff\xfe"  # Binary garbage


# ── CSV Tests ─────────────────────────────────────────────────────────────────

def test_parse_valid_csv():
    df = parse_file("test.csv", VALID_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns


def test_parse_csv_semicolon_delimiter():
    df = parse_file("test.csv", VALID_CSV_SEMICOLON)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_parse_empty_file():
    with pytest.raises(ParseError, match="empty"):
        parse_file("test.csv", EMPTY_CSV)


def test_parse_csv_no_data_rows():
    with pytest.raises(ParseError, match="no data"):
        parse_file("test.csv", NO_DATA_CSV)


def test_parse_unsupported_extension():
    with pytest.raises(ParseError, match="Unsupported file type"):
        parse_file("test.txt", VALID_CSV)


def test_parse_json_extension():
    with pytest.raises(ParseError, match="Unsupported file type"):
        parse_file("data.json", b'{"key": "value"}')


# ── Excel Tests ───────────────────────────────────────────────────────────────

def test_parse_valid_xlsx():
    """Create an in-memory XLSX and test parsing."""
    buf = io.BytesIO()
    df_orig = pd.DataFrame({"product": ["A", "B"], "sales": [100, 200]})
    df_orig.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    content = buf.read()

    df = parse_file("test.xlsx", content)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "product" in df.columns
    assert "sales" in df.columns


def test_parse_xlsx_with_multiple_columns():
    """XLSX with many column types."""
    buf = io.BytesIO()
    df_orig = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "Amount": [100.5, 200.0, 150.75],
            "Category": ["A", "B", "A"],
        }
    )
    df_orig.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    df = parse_file("test.xlsx", buf.read())
    assert len(df) == 3
    assert len(df.columns) == 3


# ── File Naming ───────────────────────────────────────────────────────────────

def test_parse_case_insensitive_extension():
    """CSV with uppercase extension should still parse."""
    df = parse_file("TEST.CSV", VALID_CSV)
    assert len(df) == 2


def test_parse_csv_with_unicode():
    """CSV with UTF-8 characters."""
    content = "名前,売上\nProduct A,1000\n".encode("utf-8")
    df = parse_file("data.csv", content)
    assert len(df) == 1
