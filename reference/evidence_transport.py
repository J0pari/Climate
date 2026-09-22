"""Explicit cross-model evidence transport without evidence-class promotion.

The bridge represented here compares a declared transferable object across two
model classes.  A successful comparison is only a bridge result: it does not
change either source's model class, domain of validity, or scientific evidence
class, and it cannot manufacture evidence from a model class that was not
actually observed or executed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


MODEL_CLASSES = {
    "idealized_model",
    "learned_emulator",
    "parent_gcm",
    "model_ensemble",
    "reanalysis",
    "observation",
}
TRANSFERABLE_OBJECT_KINDS = {"intervention_response"}


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    model_class: str
    domain_of_validity: str
    object_kind: str
    unit: str
    coordinate: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if self.model_class not in MODEL_CLASSES:
            raise ValueError(f"unsupported model_class: {self.model_class!r}")
        if not self.domain_of_validity.strip():
            raise ValueError("domain_of_validity must be non-empty")
        if self.object_kind not in TRANSFERABLE_OBJECT_KINDS:
            raise ValueError(f"unsupported transferable object: {self.object_kind!r}")
        if not self.unit.strip():
            raise ValueError("unit must be explicit")
        if len(self.coordinate) < 2 or len(self.coordinate) != len(self.values):
            raise ValueError("coordinate and values must be matching vectors of length >= 2")
        if not all(math.isfinite(value) for value in (*self.coordinate, *self.values)):
            raise ValueError("coordinate and values must be finite")
        if any(b <= a for a, b in zip(self.coordinate, self.coordinate[1:])):
            raise ValueError("coordinate must be strictly increasing")


@dataclass(frozen=True)
class BridgeSpec:
    bridge_id: str
    semantic_version: str
    source_id: str
    target_id: str
    transferable_object: str
    comparison_metric: str
    max_normalized_rmse: float
    bridge_assumption: str

    def __post_init__(self) -> None:
        if not self.bridge_id.strip() or not self.semantic_version.strip():
            raise ValueError("bridge identity and semantic version are required")
        if self.source_id == self.target_id:
            raise ValueError("bridge endpoints must be distinct sources")
        if self.transferable_object not in TRANSFERABLE_OBJECT_KINDS:
            raise ValueError("bridge transferable object is unsupported")
        if self.comparison_metric != "normalized_rmse":
            raise ValueError("only normalized_rmse is supported")
        if not math.isfinite(self.max_normalized_rmse) or self.max_normalized_rmse < 0.0:
            raise ValueError("max_normalized_rmse must be finite and non-negative")
        if not self.bridge_assumption.strip():
            raise ValueError("bridge_assumption must be explicit")


@dataclass(frozen=True)
class BridgeResult:
    bridge_id: str
    source: EvidenceSource
    target: EvidenceSource
    normalized_rmse: float
    compatible: bool
    interpretation: str = "bridge_compatibility_only_no_evidence_promotion"


def _normalized_rmse(source: EvidenceSource, target: EvidenceSource) -> float:
    if source.object_kind != target.object_kind:
        raise ValueError("bridge endpoints must expose the same transferable object")
    if source.unit != target.unit:
        raise ValueError("bridge endpoints must use the same transferable-object unit")
    if source.coordinate != target.coordinate:
        raise ValueError("bridge endpoints must share the declared comparison coordinate")
    squared_error = [
        (left - right) ** 2 for left, right in zip(source.values, target.values)
    ]
    rmse = math.sqrt(sum(squared_error) / len(squared_error))
    scale = max(max(abs(value) for value in source.values), 1.0e-15)
    return rmse / scale


def compare_bridge(
    sources: Iterable[EvidenceSource],
    bridge: BridgeSpec,
) -> BridgeResult:
    source_items = tuple(sources)
    source_map = {item.source_id: item for item in source_items}
    if len(source_map) != len(source_items):
        raise ValueError("source_id values must be unique")
    if bridge.source_id not in source_map or bridge.target_id not in source_map:
        raise ValueError("bridge endpoints must resolve to declared evidence sources")
    source = source_map[bridge.source_id]
    target = source_map[bridge.target_id]
    if source.object_kind != bridge.transferable_object or target.object_kind != bridge.transferable_object:
        raise ValueError("bridge object does not match endpoint transferable objects")
    value = _normalized_rmse(source, target)
    return BridgeResult(
        bridge_id=bridge.bridge_id,
        source=source,
        target=target,
        normalized_rmse=value,
        compatible=value <= bridge.max_normalized_rmse,
    )


def require_direct_evidence_class(
    sources: Iterable[EvidenceSource],
    requested_model_class: str,
) -> tuple[EvidenceSource, ...]:
    """Require actual source evidence for a model class; bridges never synthesize it."""
    if requested_model_class not in MODEL_CLASSES:
        raise ValueError(f"unsupported requested model class: {requested_model_class!r}")
    matched = tuple(item for item in sources if item.model_class == requested_model_class)
    if not matched:
        present = sorted({item.model_class for item in sources})
        raise ValueError(
            "cross-model promotion rejected: requested direct evidence class "
            f"{requested_model_class!r} is absent; present classes={present}"
        )
    return matched


def synthesized_source_scope(sources: Iterable[EvidenceSource]) -> tuple[dict[str, str], ...]:
    """Return source provenance without collapsing class or domain distinctions."""
    ordered = sorted(sources, key=lambda item: item.source_id)
    return tuple(
        {
            "source_id": item.source_id,
            "model_class": item.model_class,
            "domain_of_validity": item.domain_of_validity,
        }
        for item in ordered
    )
