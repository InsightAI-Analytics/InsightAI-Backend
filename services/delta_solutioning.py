"""
Delta Solutioning engine.
Calculates variances, period-over-period changes, actionable recommendations,
and smart follow-up questions on top of analytical results.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

from models.schemas import DeltaInsight, DeltaSolution, QueryIntent

logger = logging.getLogger(__name__)


def generate_delta_solution(
    intent: QueryIntent,
    analytics_result: dict[str, Any],
    question: str,
) -> Optional[DeltaSolution]:
    """
    Generate delta insights, prescriptive solutions, and suggested questions
    based on the computed analytics result.
    """
    result = analytics_result.get("result")
    metric = analytics_result.get("metric") or intent.metric or "value"
    operation = (intent.operation or "").lower()

    insights: list[DeltaInsight] = []
    recommendations: list[str] = []
    suggested_questions: list[str] = []

    pretty_metric = metric.replace("_", " ").title()
    group_col = intent.group_by
    pretty_group = group_col.replace("_", " ").title() if group_col else ""

    # Case 1: Trend / Time-series data (dict with ordered time keys -> float)
    if operation == "trend" and isinstance(result, dict) and len(result) >= 2:
        items = list(result.items())
        # Sort items by date string key
        items.sort(key=lambda x: str(x[0]))
        periods = [str(k) for k, _ in items]
        values = [float(v) for _, v in items if isinstance(v, (int, float))]

        if len(values) >= 2:
            first_val = values[0]
            last_val = values[-1]
            prev_val = values[-2]

            # Latest period-over-period change
            if prev_val != 0:
                recent_pct = ((last_val - prev_val) / abs(prev_val)) * 100.0
                direction = "up" if recent_pct > 0.5 else ("down" if recent_pct < -0.5 else "neutral")
                insights.append(
                    DeltaInsight(
                        metric=f"Recent Period Change ({periods[-2]} -> {periods[-1]})",
                        change_percent=round(recent_pct, 1),
                        direction=direction,
                        summary=f"{pretty_metric} changed by {recent_pct:+.1f}% in the most recent period ({periods[-1]}).",
                    )
                )

            # Overall period change
            if first_val != 0:
                overall_pct = ((last_val - first_val) / abs(first_val)) * 100.0
                dir_overall = "up" if overall_pct > 0.5 else ("down" if overall_pct < -0.5 else "neutral")
                insights.append(
                    DeltaInsight(
                        metric=f"Overall Trend ({periods[0]} -> {periods[-1]})",
                        change_percent=round(overall_pct, 1),
                        direction=dir_overall,
                        summary=f"Net change across recorded periods is {overall_pct:+.1f}%.",
                    )
                )

            # Find peak & valley
            max_val = max(values)
            min_val = min(values)
            peak_period = periods[values.index(max_val)]
            trough_period = periods[values.index(min_val)]

            if peak_period != trough_period:
                insights.append(
                    DeltaInsight(
                        metric="Peak vs Trough",
                        change_percent=round(((max_val - min_val) / abs(min_val) * 100.0), 1) if min_val != 0 else None,
                        direction="up",
                        summary=f"Peak reached in {peak_period} ({max_val:,.2f}), low point observed in {trough_period} ({min_val:,.2f}).",
                    )
                )

            # Solutions / Recommendations
            if last_val < prev_val:
                recommendations.append(
                    f"Investigate the drop in {pretty_metric} during {periods[-1]} to identify underperforming segments or external factors."
                )
                recommendations.append(
                    f"Audit historical drivers that produced peak performance in {peak_period} to replicate successful strategies."
                )
            else:
                recommendations.append(
                    f"Capitalize on positive momentum in {periods[-1]} by expanding resources toward high-growth drivers."
                )
                recommendations.append(
                    f"Monitor consistency against the peak record set in {peak_period}."
                )

            # Suggested follow-ups
            suggested_questions.append(f"What caused the change in {pretty_metric} during {periods[-1]}?")
            suggested_questions.append(f"Compare {pretty_metric} between {peak_period} and {trough_period}")
            suggested_questions.append(f"Show average {pretty_metric} across all periods")

    # Case 2: Grouped breakdown / Comparison (dict with categories -> numeric values)
    elif isinstance(result, dict) and len(result) > 1:
        numeric_items = [
            (str(k), float(v)) for k, v in result.items() if isinstance(v, (int, float))
        ]
        if numeric_items:
            numeric_items.sort(key=lambda x: x[1], reverse=True)
            top_k, top_v = numeric_items[0]
            bottom_k, bottom_v = numeric_items[-1]
            total_v = sum(v for _, v in numeric_items)
            avg_v = total_v / len(numeric_items)

            # Share of top performer
            if total_v > 0:
                top_share = (top_v / total_v) * 100.0
                insights.append(
                    DeltaInsight(
                        metric=f"Top Contributor: {top_k}",
                        change_percent=round(top_share, 1),
                        direction="up",
                        summary=f"'{top_k}' accounts for {top_share:.1f}% of overall {pretty_metric} ({top_v:,.2f}).",
                    )
                )

            # Delta gap between leader and laggard
            if bottom_v != 0:
                spread_pct = ((top_v - bottom_v) / abs(bottom_v)) * 100.0
                insights.append(
                    DeltaInsight(
                        metric="Leader vs Laggard Spread",
                        change_percent=round(spread_pct, 1),
                        direction="up",
                        summary=f"'{top_k}' outperforms '{bottom_k}' by {spread_pct:+.1f}%.",
                    )
                )

            # Performance relative to mean
            above_avg = [k for k, v in numeric_items if v > avg_v]
            below_avg = [k for k, v in numeric_items if v < avg_v]

            insights.append(
                DeltaInsight(
                    metric="Distribution Balance",
                    direction="neutral",
                    summary=f"{len(above_avg)} segment(s) perform above the average ({avg_v:,.2f}), while {len(below_avg)} trail below.",
                )
            )

            # Recommendations
            recommendations.append(
                f"Double down on strategies in '{top_k}' while conducting a gap analysis on lower performing segment '{bottom_k}'."
            )
            if len(below_avg) > len(above_avg):
                recommendations.append(
                    f"Concentration risk: High dependence on top segments. Diversify focus toward {', '.join(below_avg[:2])} to stabilize {pretty_metric}."
                )
            else:
                recommendations.append(
                    f"Establish benchmark playbooks from '{top_k}' to elevate underperforming categories."
                )

            # Suggested follow-ups
            if pretty_group:
                suggested_questions.append(f"Show details for top {pretty_group} '{top_k}'")
                suggested_questions.append(f"Why is {pretty_group} '{bottom_k}' underperforming in {pretty_metric}?")
                suggested_questions.append(f"What is the average {pretty_metric} across all {pretty_group}s?")
            else:
                suggested_questions.append(f"Break down {pretty_metric} by another category")
                suggested_questions.append(f"Filter data where {pretty_metric} is greater than {avg_v:.0f}")

    # Case 3: Single scalar aggregate (e.g. Total sum or Average)
    elif isinstance(result, (int, float)):
        val = float(result)
        insights.append(
            DeltaInsight(
                metric=f"Baseline {operation.title()} Measurement",
                direction="neutral",
                summary=f"Calculated aggregate {pretty_metric} stands at {val:,.2f}.",
            )
        )

        recommendations.append(
            f"Segment this {pretty_metric} by categorical dimensions (e.g., region, product, category) to discover hidden performance drivers."
        )
        recommendations.append(
            f"Track this baseline against target thresholds or previous quarters to establish growth variance."
        )

        suggested_questions.append(f"Break down {pretty_metric} by category")
        suggested_questions.append(f"Show trend of {pretty_metric} over time")
        suggested_questions.append(f"Which items contribute most to {pretty_metric}?")

    if not insights and not recommendations:
        return None

    return DeltaSolution(
        insights=insights[:3],
        recommendations=recommendations[:3],
        suggested_questions=suggested_questions[:3],
    )
