"""
Pydantic schemas for request/response models.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── File Upload ──────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    original_name: str
    dtype: str


class DatasetInfo(BaseModel):
    name: str
    rows: int
    columns: int
    column_info: list[ColumnInfo]
    sample: list[dict[str, Any]] = Field(default_factory=list)


class UploadResponse(BaseModel):
    session_id: str
    uploaded: list[DatasetInfo]
    errors: list[dict[str, str]] = Field(default_factory=list)


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question: str


class ChartDataset(BaseModel):
    label: str
    data: list[float]


class ChartData(BaseModel):
    chart_type: str  # "bar" | "line" | "pie"
    labels: list[str]
    datasets: list[ChartDataset]
    x_label: Optional[str] = None
    y_label: Optional[str] = None


class DeltaInsight(BaseModel):
    metric: str
    change_percent: Optional[float] = None
    direction: Optional[str] = None  # "up" | "down" | "neutral"
    summary: str


class DeltaSolution(BaseModel):
    insights: list[DeltaInsight] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    answer: str
    metrics: Optional[dict[str, Any]] = None
    chart: Optional[ChartData] = None
    delta_solution: Optional[DeltaSolution] = None
    error: Optional[str] = None


# ── Internal LLM Intent ───────────────────────────────────────────────────────

class QueryIntent(BaseModel):
    operation: str  # sum|average|count|min|max|comparison|trend|list
    metric: Optional[str] = None          # column to aggregate
    filters: dict[str, Any] = Field(default_factory=dict)
    group_by: Optional[str] = None
    date_column: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    limit: Optional[int] = None
    chart_required: bool = False
    chart_type: str = "bar"               # bar|line|pie
    datasets: Optional[list[str]] = None  # which datasets to use (None = all)
    join_keys: Optional[list[str]] = None # optional specific join columns across datasets
    explanation: str = ""                 # human-readable explanation from LLM
