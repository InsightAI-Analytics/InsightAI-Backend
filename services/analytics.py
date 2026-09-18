"""
Analytics engine — pure Pandas calculations.
Takes DataFrames + QueryIntent, returns results and chart data.
No LLM calls here.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import pandas as pd
import numpy as np

from models.schemas import QueryIntent, ChartData, ChartDataset

logger = logging.getLogger(__name__)


class AnalyticsError(Exception):
    """Raised when analytics cannot produce a result."""


def run_analytics(
    datasets: list[dict[str, Any]],
    intent: QueryIntent,
) -> dict[str, Any]:
    """
    Execute the analytical intent against the provided datasets.

    Args:
        datasets: List of session DatasetRecord dicts (with 'name', 'df', 'original_columns').
        intent: Parsed QueryIntent from LLM.

    Returns:
        Dict with keys: result (scalar/dict/list), chart_data (ChartData | None)
    """
    # Select relevant datasets
    selected = _select_datasets(datasets, intent.datasets)
    if not selected:
        raise AnalyticsError("No datasets found for this query.")

    # Combine datasets if multiple
    df, source_name = _combine_datasets(selected)

    # Apply filters
    df = _apply_filters(df, intent.filters)

    if df.empty:
        raise AnalyticsError(
            "No data matched the specified filters. "
            "Try broadening your question or checking the filter values."
        )

    operation = (intent.operation or "list").lower()

    # Route to the appropriate operation
    if operation == "sum":
        return _op_aggregate(df, intent, "sum")
    elif operation == "average":
        return _op_aggregate(df, intent, "mean")
    elif operation == "count":
        return _op_count(df, intent)
    elif operation == "min":
        return _op_aggregate(df, intent, "min")
    elif operation == "max":
        return _op_aggregate(df, intent, "max")
    elif operation == "comparison":
        return _op_comparison(df, intent)
    elif operation == "trend":
        return _op_trend(df, intent)
    elif operation == "list":
        return _op_list(df, intent)
    else:
        # Fallback: try aggregate if we have a metric
        if intent.metric:
            return _op_aggregate(df, intent, "sum")
        return _op_list(df, intent)


# ── Dataset selection & combination ─────────────────────────────────────────

def _select_datasets(
    datasets: list[dict[str, Any]],
    names: Optional[list[str]],
) -> list[dict[str, Any]]:
    if not names:
        return datasets
    name_set = {n.lower() for n in names}
    return [d for d in datasets if d["name"].lower() in name_set] or datasets


def _combine_datasets(datasets: list[dict[str, Any]]) -> tuple[pd.DataFrame, str]:
    """
    Combine multiple DataFrames with a source column.
    If single dataset, returns as-is with source column added.
    """
    if len(datasets) == 1:
        df = datasets[0]["df"].copy()
        df["_source"] = datasets[0]["name"]
        return df, datasets[0]["name"]

    frames = []
    for d in datasets:
        frame = d["df"].copy()
        frame["_source"] = d["name"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, "all datasets"


# ── Filters ──────────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply filter conditions from the intent."""
    if not filters:
        return df

    mask = pd.Series([True] * len(df), index=df.index)

    for col, condition in filters.items():
        # Find column (case-insensitive)
        actual_col = _find_column(df, col)
        if actual_col is None:
            logger.warning("Filter column '%s' not found in DataFrame, skipping.", col)
            continue

        if isinstance(condition, dict):
            op = condition.get("op", "==")
            val = condition.get("value")
            try:
                val = float(val) if _is_numeric_col(df[actual_col]) else val
            except (ValueError, TypeError):
                pass
            mask &= _apply_op(df[actual_col], op, val)
        else:
            # Simple equality — case-insensitive for strings
            col_series = df[actual_col]
            if pd.api.types.is_object_dtype(col_series):
                mask &= col_series.str.lower() == str(condition).lower()
            else:
                try:
                    mask &= col_series == type(col_series.iloc[0])(condition)
                except Exception:
                    mask &= col_series.astype(str).str.lower() == str(condition).lower()

    return df[mask].copy()


def _apply_op(series: pd.Series, op: str, value: Any) -> pd.Series:
    ops = {
        "==": series == value,
        "!=": series != value,
        ">": series > value,
        ">=": series >= value,
        "<": series < value,
        "<=": series <= value,
        "contains": series.astype(str).str.lower().str.contains(str(value).lower(), na=False),
    }
    return ops.get(op, series == value)


# ── Operations ───────────────────────────────────────────────────────────────

def _op_aggregate(
    df: pd.DataFrame, intent: QueryIntent, func: str
) -> dict[str, Any]:
    """Sum / average / min / max with optional group_by."""
    metric_col = _resolve_metric(df, intent.metric)

    if intent.group_by:
        group_col = _find_column(df, intent.group_by)
        if group_col is None:
            raise AnalyticsError(
                f"Column '{intent.group_by}' not found. "
                f"Available: {list(df.columns)}"
            )
        grouped = df.groupby(group_col)[metric_col].agg(func).reset_index()
        grouped.columns = [group_col, "value"]
        grouped = grouped.dropna(subset=["value"])

        # Sort
        sort_col = "value"
        ascending = intent.sort_order == "asc"
        grouped = grouped.sort_values(sort_col, ascending=ascending)

        if intent.limit:
            grouped = grouped.head(intent.limit)

        labels = grouped[group_col].astype(str).tolist()
        values = _safe_floats(grouped["value"].tolist())
        result_dict = dict(zip(labels, values))

        chart = None
        if intent.chart_required:
            chart = _make_chart(
                intent.chart_type,
                labels,
                [{"label": _pretty_name(metric_col), "data": values}],
                x_label=_pretty_name(group_col),
                y_label=_pretty_name(metric_col),
            )

        return {"result": result_dict, "chart_data": chart, "metric": metric_col}
    else:
        # Scalar aggregate
        agg_fn = getattr(df[metric_col], func, None)
        if agg_fn is None:
            raise AnalyticsError(f"Operation '{func}' not supported.")
        value = agg_fn()
        value = _safe_float(value)
        return {"result": value, "chart_data": None, "metric": metric_col}


def _op_count(df: pd.DataFrame, intent: QueryIntent) -> dict[str, Any]:
    """Count rows, optionally grouped."""
    if intent.group_by:
        group_col = _find_column(df, intent.group_by)
        if group_col is None:
            return {"result": len(df), "chart_data": None, "metric": "count"}

        grouped = df.groupby(group_col).size().reset_index(name="count")
        grouped = grouped.sort_values("count", ascending=(intent.sort_order == "asc"))
        if intent.limit:
            grouped = grouped.head(intent.limit)

        labels = grouped[group_col].astype(str).tolist()
        values = grouped["count"].tolist()
        result_dict = dict(zip(labels, values))

        chart = None
        if intent.chart_required:
            chart = _make_chart(
                intent.chart_type,
                labels,
                [{"label": "Count", "data": values}],
                x_label=_pretty_name(group_col),
                y_label="Count",
            )

        return {"result": result_dict, "chart_data": chart, "metric": "count"}
    else:
        return {"result": len(df), "chart_data": None, "metric": "count"}


def _op_comparison(df: pd.DataFrame, intent: QueryIntent) -> dict[str, Any]:
    """Compare metric values across groups."""
    return _op_aggregate(df, intent, "sum")


def _op_trend(df: pd.DataFrame, intent: QueryIntent) -> dict[str, Any]:
    """Time-based trend analysis."""
    date_col = _find_column(df, intent.date_column) if intent.date_column else _detect_date_column(df)
    if date_col is None:
        raise AnalyticsError(
            "No date column found for trend analysis. "
            "Please ensure your dataset has a date column."
        )

    metric_col = _resolve_metric(df, intent.metric)

    # Ensure date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        raise AnalyticsError("Could not parse dates in the date column.")

    # Group by month (or year if sparse)
    df["_period"] = df[date_col].dt.to_period("M").astype(str)
    grouped = df.groupby("_period")[metric_col].sum().reset_index()
    grouped = grouped.sort_values("_period")

    labels = grouped["_period"].tolist()
    values = _safe_floats(grouped[metric_col].tolist())
    result_dict = dict(zip(labels, values))

    chart = _make_chart(
        "line",
        labels,
        [{"label": _pretty_name(metric_col), "data": values}],
        x_label="Period",
        y_label=_pretty_name(metric_col),
    )

    return {"result": result_dict, "chart_data": chart, "metric": metric_col}


def _op_list(df: pd.DataFrame, intent: QueryIntent) -> dict[str, Any]:
    """Return a list/sample of rows."""
    limit = intent.limit or 10
    sample = df.head(limit)

    result = []
    for _, row in sample.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if col.startswith("_"):
                continue
            rec[col] = _safe_val(val)
        result.append(rec)

    return {"result": result, "chart_data": None, "metric": None}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    """Find a column by exact or partial match (case-insensitive)."""
    if not name:
        return None
    name_lower = name.lower()
    # Exact match
    for col in df.columns:
        if col.lower() == name_lower:
            return col
    # Partial match
    for col in df.columns:
        if name_lower in col.lower() or col.lower() in name_lower:
            return col
    return None


def _resolve_metric(df: pd.DataFrame, metric: Optional[str]) -> str:
    """Find the metric column or auto-detect the first numeric column."""
    if metric:
        col = _find_column(df, metric)
        if col:
            return col

    # Auto-detect: first numeric column (excluding internal cols)
    for col in df.columns:
        if col.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            logger.info("Auto-detected metric column: %s", col)
            return col

    raise AnalyticsError(
        "Could not find a numeric column to aggregate. "
        "Please specify which column you want to analyze."
    )


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect a date column."""
    for col in df.columns:
        if col.startswith("_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # Try column names with date-like keywords
    date_keywords = ["date", "time", "month", "year", "day", "period"]
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            return col
    return None


def _is_numeric_col(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _safe_float(val: Any) -> Any:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def _safe_floats(vals: list[Any]) -> list[Any]:
    return [_safe_float(v) for v in vals]


def _safe_val(val: Any) -> Any:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "item"):
        return val.item()
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


def _pretty_name(col: Optional[str]) -> str:
    if not col:
        return ""
    return col.replace("_", " ").title()


def _make_chart(
    chart_type: str,
    labels: list[str],
    datasets: list[dict[str, Any]],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> ChartData:
    return ChartData(
        chart_type=chart_type,
        labels=labels,
        datasets=[
            ChartDataset(label=d["label"], data=d["data"]) for d in datasets
        ],
        x_label=x_label,
        y_label=y_label,
    )
