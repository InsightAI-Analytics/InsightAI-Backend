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


# ── Relational Join Tests ──────────────────────────────────────────────────────

def test_cross_file_relational_join():
    """Test relational join across orders and products on product_id."""
    orders_df = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "product_id": ["P1", "P2", "P1", "P3"],
        "quantity": [2, 1, 4, 3],
    })
    products_df = pd.DataFrame({
        "product_id": ["P1", "P2", "P3"],
        "product_name": ["Laptop", "Mouse", "Keyboard"],
        "price": [1000.0, 25.0, 75.0],
    })

    ds_orders = make_dataset("orders.csv", orders_df)
    ds_products = make_dataset("products.csv", products_df)

    # Calculate total quantity grouped by product_name from products.csv
    intent = QueryIntent(
        operation="sum",
        metric="quantity",
        group_by="product_name",
        join_keys=["product_id"],
    )
    result = run_analytics([ds_orders, ds_products], intent)
    assert "result" in result
    res_dict = result["result"]
    assert res_dict["Laptop"] == 6  # 2 + 4
    assert res_dict["Mouse"] == 1
    assert res_dict["Keyboard"] == 3


def test_cross_file_auto_detect_join_key():
    """Test automatic detection of common identifier key column."""
    customers_df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "region": ["West", "East"],
    })
    sales_df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "spend": [150.0, 250.0, 300.0],
    })

    ds_customers = make_dataset("customers.csv", customers_df)
    ds_sales = make_dataset("sales.csv", sales_df)

    intent = QueryIntent(
        operation="sum",
        metric="spend",
        group_by="region",
    )
    result = run_analytics([ds_customers, ds_sales], intent)
    res_dict = result["result"]
    assert res_dict["West"] == 400.0
    assert res_dict["East"] == 300.0


# ── Delta Solutioning Tests ───────────────────────────────────────────────────

def test_delta_solutioning_grouped_breakdown(sales_dataset):
    from services.delta_solutioning import generate_delta_solution
    from services.response_builder import build_response

    intent = QueryIntent(operation="sum", metric="revenue", group_by="product")
    analytics_result = run_analytics([sales_dataset], intent)
    response = build_response("What is revenue by product?", intent, analytics_result)

    assert response.delta_solution is not None
    assert len(response.delta_solution.insights) > 0
    assert len(response.delta_solution.recommendations) > 0
    assert len(response.delta_solution.suggested_questions) > 0

    # Ensure recommendations and questions contain meaningful text
    rec = response.delta_solution.recommendations[0]
    assert len(rec) > 10
    q = response.delta_solution.suggested_questions[0]
    assert len(q) > 5


def test_delta_solutioning_trend():
    from services.delta_solutioning import generate_delta_solution

    intent = QueryIntent(operation="trend", metric="revenue", date_column="date")
    analytics_result = {
        "result": {"2024-01": 1000.0, "2024-02": 1500.0, "2024-03": 1200.0},
        "metric": "revenue",
    }
    sol = generate_delta_solution(intent, analytics_result, "Show trend of revenue")
    assert sol is not None
    assert any("Recent Period Change" in i.metric for i in sol.insights)
    assert len(sol.recommendations) >= 2
    assert len(sol.suggested_questions) >= 2

