"""
LLM service — converts a natural language question into a structured QueryIntent.

The LLM is ONLY used for intent extraction.
All numerical calculations happen in analytics.py.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import settings
from models.schemas import QueryIntent

logger = logging.getLogger(__name__)


def extract_intent(
    question: str,
    schemas: list[dict[str, Any]],
) -> QueryIntent:
    """
    Call the LLM to extract a structured analytical intent from the question.

    Args:
        question: The user's natural-language question.
        schemas: List of dataset schemas (from normalizer.get_schema_description).

    Returns:
        A QueryIntent object.

    Raises:
        ValueError if no API key is configured or the LLM returns unparseable output.
    """
    schema_text = _format_schemas(schemas)
    system_prompt = _build_system_prompt(schema_text)

    # 1. Prefer Gemini API if GEMINI_API_KEY is provided
    if settings.gemini_api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.gemini_api_key)
            logger.info("Sending intent extraction request to Gemini (%s)", settings.gemini_model)

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            raw = response.text or ""
            return _parse_intent(raw)
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise ValueError(f"Gemini API error: {exc}") from exc

    # 2. Fallback to OpenAI if OPENAI_API_KEY is provided
    if settings.openai_api_key:
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        logger.info("Sending intent extraction request to OpenAI for question: %s", question)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        logger.debug("LLM raw response: %s", raw)
        return _parse_intent(raw)

    raise ValueError(
        "LLM API key not configured. Set GEMINI_API_KEY in your backend/.env file."
    )


def _format_schemas(schemas: list[dict[str, Any]]) -> str:
    """Format dataset schemas into a readable string for the prompt."""
    lines = []
    for schema in schemas:
        lines.append(f"\nDataset: {schema['dataset']} ({schema['rows']} rows)")
        for col in schema["columns"]:
            sample_str = ", ".join(str(v) for v in col["sample"][:3])
            lines.append(
                f"  - {col['name']} (type: {col['dtype']}, samples: [{sample_str}])"
            )
    return "\n".join(lines)


def _build_system_prompt(schema_text: str) -> str:
    return f"""You are a data analysis assistant. Your ONLY job is to understand the user's question
and return a structured JSON intent. You must NOT calculate any values yourself.

Available datasets and their columns:
{schema_text}

Return a JSON object with EXACTLY these fields:
{{
  "operation": "<one of: sum|average|count|min|max|comparison|trend|list>",
  "metric": "<column name to aggregate, or null>",
  "filters": {{}},
  "group_by": "<column name to group by, or null>",
  "date_column": "<date column name if needed, or null>",
  "sort_by": "<column to sort results by, or null>",
  "sort_order": "<asc|desc>",
  "limit": <integer or null>,
  "chart_required": <true|false>,
  "chart_type": "<bar|line|pie>",
  "datasets": <list of dataset names to use, or null for all>,
  "explanation": "<one-sentence explanation of what you understood>"
}}

Rules:
1. Use ONLY column names that exist in the datasets above (normalized snake_case names).
2. For "filters", use format: {{"column_name": "value"}} or {{"column_name": {{"op": ">", "value": 100}}}}.
3. For "trend" questions (e.g., "show trend by month"), set date_column to the date column and group_by to null.
4. For "comparison" questions, use group_by with the relevant dimension column.
5. If chart makes sense: bar for comparisons/rankings, line for trends over time, pie for proportions.
6. "datasets": null means use all datasets. Otherwise list exact dataset names.
7. Return ONLY the JSON object, no other text.
"""


def _parse_intent(raw: str) -> QueryIntent:
    """Parse LLM JSON output into a QueryIntent."""
    # Extract JSON if wrapped in markdown code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw: {raw}") from exc

    # Ensure required fields have defaults
    data.setdefault("operation", "list")
    data.setdefault("metric", None)
    data.setdefault("filters", {})
    data.setdefault("group_by", None)
    data.setdefault("date_column", None)
    data.setdefault("sort_by", None)
    data.setdefault("sort_order", "desc")
    data.setdefault("limit", None)
    data.setdefault("chart_required", False)
    data.setdefault("chart_type", "bar")
    data.setdefault("datasets", None)
    data.setdefault("explanation", "")

    return QueryIntent(**data)
