"""Adaptive routing policy (shared by pipeline + Colab inference trace)."""

from __future__ import annotations

from dataclasses import dataclass

from .generate import is_trivial_query
from .query_analyzer import AnalyzerOutput


@dataclass(frozen=True)
class AdaptiveRoutingDecision:
    skip_retrieval: bool
    policy: str
    retrieve_mode: str  # current | historical | semantic | "" when skip


def resolve_adaptive_routing(
    analysis: AnalyzerOutput,
    query_text: str,
    conf_threshold: float,
) -> AdaptiveRoutingDecision:
    """Map analyzer output to policy name and retrieve_mode (matches AdaptiveTemporalRAGPipeline)."""
    if analysis.needs_retrieval == 0 and is_trivial_query(query_text):
        return AdaptiveRoutingDecision(True, "no_retrieval", "")

    mode = analysis.temporal_class
    soft = analysis.needs_retrieval == 0
    policy = "soft_retrieve_timeless" if soft else mode
    if soft or analysis.confidence < conf_threshold or mode == "timeless":
        if not soft and analysis.confidence < conf_threshold:
            policy = "low_conf_semantic"
        elif mode == "timeless" and not soft:
            policy = "timeless_semantic"
        retrieve_mode = "semantic"
    elif mode == "current":
        retrieve_mode = "current"
    elif mode == "historical":
        retrieve_mode = "historical"
        policy = "historical_adaptive_window"
    else:
        retrieve_mode = "semantic"
    return AdaptiveRoutingDecision(False, policy, retrieve_mode)
