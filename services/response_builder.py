"""
Response builder — converts raw analytics results into human-readable answers.
"""

from __future__ import annotations
from typing import Any, Optional

from models.schemas import QueryIntent, ChartData, QueryResponse


def build_response(
    question: str,
    intent: QueryIntent,
    analytics_result: dict[str, Any],
) -> QueryResponse:
    """
    Build the final QueryResponse from analytics results.
    """
    result = analytics_result.get("result")
    chart_data: Optional[ChartData] = analytics_result.get("chart_data")
    metric = analytics_result.get("metric")

    answer = _compose_answer(question, intent, result, metric)
    metrics = _compose_metrics(result, metric, intent)

    return QueryResponse(
        question=question,
        answer=answer,
        metrics=metrics,
        chart=chart_data,
    )


def _compose_answer(
    question: str,
    intent: QueryIntent,
    result: Any,
    metric: Optional[str],
) -> str:
    """Generate a natural-language answer."""
    operation = intent.operation.lower()
    pretty_metric = _pretty(metric) if metric else "the data"

    if isinstance(result, (int, float)) and result is not None:
        formatted = _format_number(result)
        op_words = {
            "sum": f"The total {pretty_metric} is **{formatted}**.",
            "average": f"The average {pretty_metric} is **{formatted}**.",
            "count": f"There are **{formatted}** records.",
            "min": f"The minimum {pretty_metric} is **{formatted}**.",
            "max": f"The maximum {pretty_metric} is **{formatted}**.",
        }
        return op_words.get(operation, f"The result is **{formatted}**.")

    elif isinstance(result, dict) and result:
        items = sorted(result.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)

        if operation in ("comparison", "sum", "average", "max", "min"):
            top_key, top_val = items[0]
            top_val_str = _format_number(top_val) if isinstance(top_val, (int, float)) else str(top_val)

            if intent.group_by:
                group_pretty = _pretty(intent.group_by)
                lines = [
                    f"Here is the breakdown of **{pretty_metric}** by **{group_pretty}**:",
                ]
                for k, v in items[:10]:
                    v_str = _format_number(v) if isinstance(v, (int, float)) else str(v)
                    lines.append(f"• **{k}**: {v_str}")

                if len(items) > 10:
                    lines.append(f"*… and {len(items) - 10} more.*")
                return "\n".join(lines)

        if operation == "trend":
            first_k, first_v = items[-1] if items else ("—", 0)
            last_k, last_v = items[0] if items else ("—", 0)
            lines = [f"**{pretty_metric}** trend over time:"]
            for k, v in sorted(result.items()):
                v_str = _format_number(v) if isinstance(v, (int, float)) else str(v)
                lines.append(f"• {k}: {v_str}")
            return "\n".join(lines)

        # Generic dict
        lines = []
        for k, v in list(result.items())[:15]:
            v_str = _format_number(v) if isinstance(v, (int, float)) else str(v)
            lines.append(f"• **{k}**: {v_str}")
        return "\n".join(lines)

    elif isinstance(result, list) and result:
        return f"Here are the top {min(len(result), 10)} records from your data."

    return "I could not find a specific answer. Please try rephrasing your question."


def _compose_metrics(
    result: Any,
    metric: Optional[str],
    intent: QueryIntent,
) -> Optional[dict[str, Any]]:
    """Build a metrics dict for the frontend to display as a table."""
    if isinstance(result, (int, float)):
        return {
            "operation": intent.operation,
            "metric": metric,
            "value": result,
            "formatted": _format_number(result),
        }
    if isinstance(result, dict):
        return {
            "operation": intent.operation,
            "metric": metric,
            "group_by": intent.group_by,
            "values": result,
            "total_groups": len(result),
        }
    if isinstance(result, list):
        return {
            "operation": "list",
            "total_rows": len(result),
            "data": result,
        }
    return None


def _pretty(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.replace("_", " ").title()


def _format_number(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if f == int(f) and abs(f) < 1e15:
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(val)
