"""Modular search interfaces and shared types."""

from .modular import (
    Candidate,
    CandidateBudget,
    CandidateSource,
    FacetBuilder,
    FacetPayload,
    ProgressivePage,
    ProgressiveSessionStore,
    Reranker,
    ScoredResult,
    SearchRequest,
)

__all__ = [
    "Candidate",
    "CandidateBudget",
    "CandidateSource",
    "FacetBuilder",
    "FacetPayload",
    "ProgressivePage",
    "ProgressiveSessionStore",
    "Reranker",
    "ScoredResult",
    "SearchRequest",
]
