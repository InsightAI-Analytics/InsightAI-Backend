"""
Unit tests for the analytics engine.
"""

import pytest
import pandas as pd

from models.schemas import QueryIntent
from services.analytics import run_analytics, AnalyticsError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dataset(name: str, df: pd.DataFrame) -> dict:
    return {"name": name, "df": df, "original_columns": {col: col for col in df.columns}}


@pytest.fixture
def sales_dataset():
    df = pd.DataFrame(
        {
            "product": ["Widget", "Gadget", "Widget", "Gadget", "Widget"],
            "region": ["North", "North", "South", "South", "North"],
            "revenue": [1000.0, 1500.0, 800.0, 1200.0, 900.0],
            "units": [10, 15, 8, 12, 9],
            "date": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-10", "2024-02-15", "2024-03-01"]),
        }
    )
    return make_dataset("sales.csv", df)


@pytest.fixture
def orders_dataset():
    df = pd.DataFrame(
        {
            "category": ["Electronics", "Clothing", "Electronics", "Food"],
            "amount": [500.0, 200.0, 750.0, 100.0],
            "orders": [5, 8, 6, 12],
        }
    )
    return make_dataset("orders.csv", df)


# ── Sum Tests ──────────────────────────────────────────────────────────────────

def test_sum_total(sales_dataset):
    intent = QueryIntent(operation="sum", metric="revenue")
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == pytest.approx(5400.0)


def test_sum_grouped(sales_dataset):
    intent = QueryIntent(operation="sum", metric="revenue", group_by="product")
    result = run_analytics([sales_dataset], intent)
    data = result["result"]
    assert "Widget" in data
    assert "Gadget" in data
    assert data["Widget"] == pytest.approx(2700.0)
    assert data["Gadget"] == pytest.approx(2700.0)


# ── Average Tests ─────────────────────────────────────────────────────────────

def test_average_total(sales_dataset):
    intent = QueryIntent(operation="average", metric="revenue")
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == pytest.approx(1080.0)


def test_average_grouped(sales_dataset):
    intent = QueryIntent(operation="average", metric="revenue", group_by="region")
    result = run_analytics([sales_dataset], intent)
    data = result["result"]
    assert "North" in data
    assert "South" in data


# ── Count Tests ───────────────────────────────────────────────────────────────

def test_count_total(sales_dataset):
    intent = QueryIntent(operation="count")
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == 5


def test_count_grouped(sales_dataset):
    intent = QueryIntent(operation="count", group_by="product")
    result = run_analytics([sales_dataset], intent)
    data = result["result"]
    assert data["Widget"] == 3
    assert data["Gadget"] == 2


# ── Min / Max Tests ───────────────────────────────────────────────────────────

def test_max(sales_dataset):
    intent = QueryIntent(operation="max", metric="revenue")
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == pytest.approx(1500.0)


def test_min(sales_dataset):
    intent = QueryIntent(operation="min", metric="revenue")
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == pytest.approx(800.0)


# ── Filter Tests ───────────────────────────────────────────────────────────────

def test_filter_by_value(sales_dataset):
    intent = QueryIntent(
        operation="sum",
        metric="revenue",
        filters={"product": "Widget"},
    )
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == pytest.approx(2700.0)


def test_filter_numeric_op(sales_dataset):
    intent = QueryIntent(
        operation="count",
        filters={"revenue": {"op": ">", "value": 1000}},
    )
    result = run_analytics([sales_dataset], intent)
    assert result["result"] == 2  # 1500 and 1200


# ── Trend Tests ───────────────────────────────────────────────────────────────

def test_trend_by_month(sales_dataset):
    intent = QueryIntent(
        operation="trend",
        metric="revenue",
        date_column="date",
        chart_required=True,
        chart_type="line",
    )
    result = run_analytics([sales_dataset], intent)
    data = result["result"]
    assert isinstance(data, dict)
    assert len(data) == 3  # Jan, Feb, Mar
    # Chart should be populated
    assert result["chart_data"] is not None
    assert result["chart_data"].chart_type == "line"


# ── Cross-File Tests ──────────────────────────────────────────────────────────

def test_cross_file_count(sales_dataset, orders_dataset):
    intent = QueryIntent(operation="count")
    result = run_analytics([sales_dataset, orders_dataset], intent)
    assert result["result"] == 9  # 5 + 4


# ── Error Tests ───────────────────────────────────────────────────────────────

def test_no_datasets():
    intent = QueryIntent(operation="sum", metric="revenue")
    with pytest.raises(AnalyticsError, match="No datasets found"):
        run_analytics([], intent)


def test_filter_returns_empty(sales_dataset):
    intent = QueryIntent(
        operation="sum",
        metric="revenue",
        filters={"product": "NonExistentProduct"},
    )
    with pytest.raises(AnalyticsError, match="No data matched"):
        run_analytics([sales_dataset], intent)


# ── Chart Tests ────────────────────────────────────────────────────────────────

def test_chart_generated_when_requested(sales_dataset):
    intent = QueryIntent(
        operation="sum",
        metric="revenue",
        group_by="product",
        chart_required=True,
        chart_type="bar",
    )
    result = run_analytics([sales_dataset], intent)
    assert result["chart_data"] is not None
    chart = result["chart_data"]
    assert chart.chart_type == "bar"
    assert len(chart.labels) == 2
    assert len(chart.datasets) == 1


def test_no_chart_when_not_requested(sales_dataset):
    intent = QueryIntent(operation="sum", metric="revenue", chart_required=False)
    result = run_analytics([sales_dataset], intent)
    assert result["chart_data"] is None
