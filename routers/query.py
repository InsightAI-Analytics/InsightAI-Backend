"""
Query router — handles natural language questions.
"""

from __future__ import annotations
import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException

from models.schemas import QueryRequest, QueryResponse
from services.llm_service import extract_intent
from services.analytics import run_analytics, AnalyticsError
from services.normalizer import get_schema_description
from services.response_builder import build_response
import session_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_data(body: QueryRequest) -> QueryResponse:
    """
    Accept a natural language question, extract intent via LLM,
    run Pandas analytics, and return an answer + optional chart.
    """
    session_id = body.session_id
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    datasets = session_store.get_datasets(session_id)
    if not datasets:
        return QueryResponse(
            question=question,
            answer="No data uploaded yet. Please upload one or more CSV or Excel files first.",
            error="no_data",
        )

    # Build schemas for the LLM prompt
    schemas = [
        get_schema_description(d["name"], d["df"], d["original_columns"])
        for d in datasets
    ]

    # Step 1: LLM extracts structured intent
    try:
        intent = extract_intent(question, schemas)
        logger.info("Intent extracted: operation=%s metric=%s group_by=%s",
                    intent.operation, intent.metric, intent.group_by)
    except ValueError as exc:
        logger.error("LLM intent extraction failed: %s", exc)
        return QueryResponse(
            question=question,
            answer=(
                "I had trouble understanding your question. "
                "Please try rephrasing it more specifically, for example: "
                "'What is the total revenue?' or 'Show sales by product.'"
            ),
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected LLM error")
        return QueryResponse(
            question=question,
            answer="An error occurred while interpreting your question. Please try again.",
            error=str(exc),
        )

    # Step 2: Pandas analytics
    try:
        analytics_result = run_analytics(datasets, intent)
    except AnalyticsError as exc:
        logger.warning("Analytics error: %s", exc)
        return QueryResponse(
            question=question,
            answer=str(exc),
            error="analytics_error",
        )
    except Exception as exc:
        logger.exception("Unexpected analytics error")
        return QueryResponse(
            question=question,
            answer="An error occurred while analyzing your data. Please try again.",
            error=str(exc),
        )

    # Step 3: Build response
    return build_response(question, intent, analytics_result)
