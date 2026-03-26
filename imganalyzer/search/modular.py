"""Search modular architecture contracts.

These interfaces define boundaries between retrieval, filtering, ranking,
facet building, and progressive session management so search upgrades can be
introduced in controlled, independently testable steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchRequest:
    query: str = ""
    mode: str = "hybrid"
    semantic_weight: float = 0.5
    intent: str = "general"
    similar_to_image_id: int | None = None
    must_terms: list[str] = field(default_factory=list)
    should_terms: list[str] = field(default_factory=list)
    rank_preference: str = "relevance"
    debug_search: bool = False
    facet_request: bool = False
    face: str | None = None
    faces: list[str] = field(default_factory=list)
    face_match: str = "all"
    country: str | None = None
    recurring_month_day: str | None = None
    time_of_day: str | None = None
    sort_by: str = "relevance"
    expanded_terms: list[str] = field(default_factory=list)
    camera: str | None = None
    lens: str | None = None
    location: str | None = None
    aesthetic_min: float | None = None
    aesthetic_max: float | None = None
    sharpness_min: float | None = None
    sharpness_max: float | None = None
    noise_max: float | None = None
    iso_min: int | None = None
    iso_max: int | None = None
    faces_min: int | None = None
    faces_max: int | None = None
    date_from: str | None = None
    date_to: str | None = None
    has_people: bool | None = None
    limit: int = 200
    offset: int = 0


@dataclass(frozen=True)
class CandidateBudget:
    limit: int


@dataclass(frozen=True)
class Candidate:
    image_id: int
    file_path: str
    score: float
    source: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredResult:
    image_id: int
    final_score: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FacetPayload:
    buckets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressivePage:
    results: list[dict[str, Any]]
    next_cursor: str | None
    is_complete: bool
    progress: dict[str, Any]


class CandidateSource(Protocol):
    def name(self) -> str:
        ...

    def collect(
        self,
        request: SearchRequest,
        budget: CandidateBudget,
        context: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        ...


class Reranker(Protocol):
    def profile(self) -> str:
        ...

    def score(
        self,
        candidates: list[Candidate],
        request: SearchRequest,
        features: dict[int, dict[str, Any]] | None = None,
    ) -> list[ScoredResult]:
        ...


class FacetBuilder(Protocol):
    def build(
        self,
        results: list[dict[str, Any]],
        request: SearchRequest,
    ) -> FacetPayload:
        ...


class ProgressiveSessionStore(Protocol):
    def start(self, request: SearchRequest) -> str:
        ...

    def next(self, search_id: str, cursor: str | None, limit: int) -> ProgressivePage:
        ...

    def cancel(self, search_id: str) -> None:
        ...
