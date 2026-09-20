"""Provider-neutral, content-addressed station federation substrate.

Global station scale is represented as immutable catalog-shard and observation-
partition references. The manifest never requires station-by-time materialization.
Provider aliases, resolved location history, revisions, and tombstones remain
explicit; cross-provider identity is never inferred from geographic proximity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import hashlib
import json
import math
import re
from typing import Iterable, Sequence

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BINDING_METHODS = {"root", "provider_crosswalk"}
_PARTITION_KINDS = {"data", "tombstone"}


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
    return value


def _day(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date or null") from exc


def canonical_json_digest(value) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class ProviderAlias:
    source_id: str
    provider_station_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "provider_station_id",
            _nonempty(self.provider_station_id, "provider_station_id"),
        )


def canonical_station_id(root_alias: ProviderAlias) -> str:
    encoded = (
        root_alias.source_id + "\0" + root_alias.provider_station_id
    ).encode("utf-8")
    return "station.v1." + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AliasBinding:
    alias: ProviderAlias
    binding_method: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.binding_method not in _BINDING_METHODS:
            raise ValueError(
                f"binding_method must be one of {sorted(_BINDING_METHODS)}"
            )
        _digest(self.evidence_digest, "evidence_digest")


@dataclass(frozen=True)
class CrossProviderAliasEvidence:
    root_alias: ProviderAlias
    alias: ProviderAlias
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.root_alias.source_id == self.alias.source_id:
            raise ValueError(
                "cross-provider alias evidence must bind different provider namespaces"
            )
        if self.root_alias == self.alias:
            raise ValueError("cross-provider alias cannot equal the root alias")
        _digest(self.evidence_digest, "evidence_digest")


@dataclass(frozen=True)
class StationLocationEpoch:
    latitude_deg: float
    longitude_deg: float
    elevation_m: float | None
    valid_from: str | None
    valid_to: str | None
    source_alias: ProviderAlias
    evidence_digest: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude_deg) or not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be finite and within [-90, 90]")
        if not math.isfinite(self.longitude_deg) or not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg must be finite and within [-180, 180]")
        if self.elevation_m is not None and not math.isfinite(self.elevation_m):
            raise ValueError("elevation_m must be finite when present")
        start = _day(self.valid_from, "valid_from")
        stop = _day(self.valid_to, "valid_to")
        if start is not None and stop is not None and start >= stop:
            raise ValueError("location epoch requires valid_from < valid_to")
        _digest(self.evidence_digest, "evidence_digest")

    def contains(self, value: str) -> bool:
        target = _day(value, "date")
        assert target is not None
        start = _day(self.valid_from, "valid_from")
        stop = _day(self.valid_to, "valid_to")
        return (start is None or target >= start) and (stop is None or target < stop)


@dataclass(frozen=True)
class FederatedStation:
    canonical_station_id: str
    aliases: tuple[AliasBinding, ...]
    location_history: tuple[StationLocationEpoch, ...]
    variable_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.canonical_station_id, "canonical_station_id")
        aliases = tuple(self.aliases)
        if not aliases:
            raise ValueError("station requires at least one provider alias")
        alias_values = [binding.alias for binding in aliases]
        if len(set(alias_values)) != len(alias_values):
            raise ValueError("provider aliases must be unique within a station")
        roots = [binding for binding in aliases if binding.binding_method == "root"]
        if len(roots) != 1:
            raise ValueError("station requires exactly one root alias")
        expected = canonical_station_id(roots[0].alias)
        if self.canonical_station_id != expected:
            raise ValueError(
                "canonical_station_id must be derived from the immutable root alias"
            )

        variables = tuple(self.variable_ids)
        if any(not isinstance(item, str) or not item.strip() for item in variables):
            raise ValueError("variable_ids must contain non-empty strings")
        if len(set(variables)) != len(variables):
            raise ValueError("variable_ids must be unique")

        locations = tuple(self.location_history)
        if not locations:
            raise ValueError("station requires resolved location history")
        alias_set = set(alias_values)
        for epoch in locations:
            if epoch.source_alias not in alias_set:
                raise ValueError("location source_alias is not bound to this station")
        ordered = sorted(
            locations,
            key=lambda item: (
                _day(item.valid_from, "valid_from") or date.min,
                _day(item.valid_to, "valid_to") or date.max,
            ),
        )
        previous_stop: date | None = None
        for epoch in ordered:
            start = _day(epoch.valid_from, "valid_from") or date.min
            stop = _day(epoch.valid_to, "valid_to") or date.max
            if previous_stop is not None and start < previous_stop:
                raise ValueError(
                    "resolved location epochs must not overlap; preserve conflicting "
                    "provider locations upstream until an explicit resolution exists"
                )
            previous_stop = stop

    def location_at(self, value: str) -> StationLocationEpoch | None:
        matches = [epoch for epoch in self.location_history if epoch.contains(value)]
        if len(matches) > 1:
            raise ValueError("resolved station location history overlaps")
        return matches[0] if matches else None


def apply_cross_provider_alias_evidence(
    stations: Sequence[FederatedStation],
    evidence: Sequence[CrossProviderAliasEvidence],
) -> tuple[FederatedStation, ...]:
    """Apply explicit cross-provider identity evidence without changing root ids.

    No coordinate, name, or proximity heuristic participates in this operation.
    Existing bindings are idempotent only when they name the same canonical
    station, binding method, and evidence artifact.
    """
    records = tuple(stations)
    root_owner: dict[ProviderAlias, int] = {}
    alias_owner: dict[ProviderAlias, tuple[int, AliasBinding]] = {}
    for index, station in enumerate(records):
        for binding in station.aliases:
            previous = alias_owner.get(binding.alias)
            if previous is not None and previous[0] != index:
                raise ValueError(
                    "provider alias is already bound to multiple canonical stations"
                )
            alias_owner[binding.alias] = (index, binding)
            if binding.binding_method == "root":
                if binding.alias in root_owner and root_owner[binding.alias] != index:
                    raise ValueError(
                        "root provider alias resolves to multiple canonical stations"
                    )
                root_owner[binding.alias] = index

    additions: dict[int, list[AliasBinding]] = {}
    requested_aliases: dict[ProviderAlias, int] = {}
    for item in evidence:
        owner = root_owner.get(item.root_alias)
        if owner is None:
            raise ValueError(
                f"crosswalk root alias does not resolve: {item.root_alias!r}"
            )
        requested_owner = requested_aliases.get(item.alias)
        if requested_owner is not None and requested_owner != owner:
            raise ValueError(
                "one crosswalk alias cannot bind multiple canonical stations"
            )
        requested_aliases[item.alias] = owner

        existing = alias_owner.get(item.alias)
        expected = AliasBinding(
            item.alias,
            "provider_crosswalk",
            item.evidence_digest,
        )
        if existing is not None:
            existing_owner, binding = existing
            if existing_owner != owner:
                raise ValueError(
                    "crosswalk alias is already owned by another canonical station"
                )
            if binding != expected:
                raise ValueError(
                    "crosswalk alias already exists with different binding evidence"
                )
            continue
        additions.setdefault(owner, []).append(expected)
        alias_owner[item.alias] = (owner, expected)

    updated = list(records)
    for index, new_bindings in additions.items():
        station = records[index]
        updated[index] = replace(
            station,
            aliases=tuple((*station.aliases, *new_bindings)),
        )
    return tuple(updated)


def remove_cross_provider_alias_evidence(
    stations: Sequence[FederatedStation],
    evidence_digest: str,
) -> tuple[FederatedStation, ...]:
    """Reverse only crosswalk aliases introduced by one evidence artifact."""
    _digest(evidence_digest, "evidence_digest")
    updated: list[FederatedStation] = []
    for station in stations:
        removable = {
            binding.alias
            for binding in station.aliases
            if (
                binding.binding_method == "provider_crosswalk"
                and binding.evidence_digest == evidence_digest
            )
        }
        if not removable:
            updated.append(station)
            continue
        if any(
            epoch.source_alias in removable
            for epoch in station.location_history
        ):
            raise ValueError(
                "cannot remove crosswalk evidence while location history depends "
                "on one of its provider aliases"
            )
        aliases = tuple(
            binding
            for binding in station.aliases
            if binding.alias not in removable
        )
        updated.append(replace(station, aliases=aliases))
    return tuple(updated)


@dataclass(frozen=True)
class StationCatalogShard:
    shard_id: str
    spatial_partition: str
    stations: tuple[FederatedStation, ...]

    def __post_init__(self) -> None:
        _nonempty(self.shard_id, "shard_id")
        _nonempty(self.spatial_partition, "spatial_partition")
        stations = tuple(self.stations)
        if not stations:
            raise ValueError("catalog shard must contain at least one station")
        ids = [station.canonical_station_id for station in stations]
        if len(set(ids)) != len(ids):
            raise ValueError("canonical station ids must be unique within a shard")
        aliases = [
            binding.alias
            for station in stations
            for binding in station.aliases
        ]
        if len(set(aliases)) != len(aliases):
            raise ValueError(
                "a provider alias cannot resolve to multiple canonical stations "
                "inside one catalog shard"
            )

    def digest(self) -> str:
        return canonical_json_digest(self)

    def active_records(self, at_date: str) -> tuple[FederatedStation, ...]:
        return tuple(
            station
            for station in sorted(
                self.stations, key=lambda item: item.canonical_station_id
            )
            if station.location_at(at_date) is not None
        )

    def to_station_catalog(self, at_date: str):
        """Project one bounded shard to the canonical topology substrate."""
        import numpy as np
        from src.station_sheaf import StationCatalog

        records = self.active_records(at_date)
        if not records:
            raise ValueError("catalog shard has no active station at requested date")
        locations = [station.location_at(at_date) for station in records]
        assert all(location is not None for location in locations)
        return StationCatalog(
            station_ids=tuple(station.canonical_station_id for station in records),
            latitude_deg=np.asarray(
                [location.latitude_deg for location in locations], dtype=np.float64
            ),
            longitude_deg=np.asarray(
                [location.longitude_deg for location in locations], dtype=np.float64
            ),
        )

    def station_variable_ids(self, at_date: str) -> tuple[tuple[str, ...], ...]:
        return tuple(station.variable_ids for station in self.active_records(at_date))


@dataclass(frozen=True)
class CatalogShardRef:
    shard_id: str
    spatial_partition: str
    digest: str
    station_count: int
    provider_source_ids: tuple[str, ...]
    variable_ids: tuple[str, ...]
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.shard_id, "shard_id")
        _nonempty(self.spatial_partition, "spatial_partition")
        _digest(self.digest, "digest")
        if self.station_count < 0:
            raise ValueError("station_count must be non-negative")
        if not self.provider_source_ids or any(
            not item.strip() for item in self.provider_source_ids
        ):
            raise ValueError("provider_source_ids must be non-empty")
        if len(set(self.provider_source_ids)) != len(self.provider_source_ids):
            raise ValueError("provider_source_ids must be unique")
        if len(set(self.variable_ids)) != len(self.variable_ids):
            raise ValueError("variable_ids must be unique")
        for value in self.supersedes:
            _digest(value, "supersedes digest")


@dataclass(frozen=True)
class ObservationPartitionRef:
    source_id: str
    spatial_partition: str
    time_start: str
    time_end: str
    variable_ids: tuple[str, ...]
    digest: str
    source_revision: str
    media_type: str
    row_count: int
    byte_count: int
    kind: str = "data"
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.source_id, "source_id")
        _nonempty(self.spatial_partition, "spatial_partition")
        start = _day(self.time_start, "time_start")
        stop = _day(self.time_end, "time_end")
        assert start is not None and stop is not None
        if start > stop:
            raise ValueError("time_start must not be after time_end")
        if not self.variable_ids or any(not item.strip() for item in self.variable_ids):
            raise ValueError("variable_ids must be non-empty")
        if len(set(self.variable_ids)) != len(self.variable_ids):
            raise ValueError("variable_ids must be unique")
        _digest(self.digest, "digest")
        _nonempty(self.source_revision, "source_revision")
        _nonempty(self.media_type, "media_type")
        if self.kind not in _PARTITION_KINDS:
            raise ValueError(f"kind must be one of {sorted(_PARTITION_KINDS)}")
        if self.row_count < 0 or self.byte_count < 0:
            raise ValueError("row_count and byte_count must be non-negative")
        if self.kind == "tombstone" and self.row_count != 0:
            raise ValueError("tombstone partitions must have row_count=0")
        for value in self.supersedes:
            _digest(value, "supersedes digest")

    @property
    def logical_key(self) -> tuple:
        return (
            self.source_id,
            self.spatial_partition,
            self.time_start,
            self.time_end,
            tuple(self.variable_ids),
        )


def _active_revisions(items: Sequence, *, key) -> tuple:
    by_digest = {}
    for item in items:
        if item.digest in by_digest and by_digest[item.digest] != item:
            raise ValueError(f"digest {item.digest} names conflicting metadata")
        by_digest[item.digest] = item
    for item in items:
        for predecessor in item.supersedes:
            if predecessor not in by_digest:
                raise ValueError(
                    f"revision {item.digest} supersedes unknown digest {predecessor}"
                )
            if key(by_digest[predecessor]) != key(item):
                raise ValueError("revision may supersede only the same logical partition")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(digest: str) -> None:
        if digest in visited:
            return
        if digest in visiting:
            raise ValueError("revision graph contains a cycle")
        visiting.add(digest)
        for predecessor in by_digest[digest].supersedes:
            visit(predecessor)
        visiting.remove(digest)
        visited.add(digest)

    for digest in by_digest:
        visit(digest)

    superseded = {
        predecessor
        for item in by_digest.values()
        for predecessor in item.supersedes
    }
    leaves = [item for digest, item in by_digest.items() if digest not in superseded]
    by_key: dict[tuple, list] = {}
    for item in leaves:
        by_key.setdefault(key(item), []).append(item)
    conflicts = [logical for logical, values in by_key.items() if len(values) != 1]
    if conflicts:
        raise ValueError(
            "logical partitions have multiple active revisions; new provider "
            "revisions must explicitly supersede the prior digest"
        )
    return tuple(sorted(leaves, key=lambda item: (key(item), item.digest)))


@dataclass(frozen=True)
class StationFederationManifest:
    manifest_id: str
    semantic_version: str
    provider_registry_digest: str
    catalog_shards: tuple[CatalogShardRef, ...]
    observation_partitions: tuple[ObservationPartitionRef, ...]

    def __post_init__(self) -> None:
        _nonempty(self.manifest_id, "manifest_id")
        _nonempty(self.semantic_version, "semantic_version")
        _digest(self.provider_registry_digest, "provider_registry_digest")
        _active_revisions(self.catalog_shards, key=lambda item: item.shard_id)
        _active_revisions(
            self.observation_partitions, key=lambda item: item.logical_key
        )

    def digest(self) -> str:
        return canonical_json_digest(self)

    def active_catalog_shards(self) -> tuple[CatalogShardRef, ...]:
        return _active_revisions(
            self.catalog_shards, key=lambda item: item.shard_id
        )

    def active_observation_partitions(
        self, *, include_tombstones: bool = False
    ) -> tuple[ObservationPartitionRef, ...]:
        active = _active_revisions(
            self.observation_partitions, key=lambda item: item.logical_key
        )
        if include_tombstones:
            return active
        return tuple(item for item in active if item.kind == "data")

    def with_observation_partitions(
        self, additions: Iterable[ObservationPartitionRef]
    ) -> "StationFederationManifest":
        by_digest = {item.digest: item for item in self.observation_partitions}
        for item in additions:
            existing = by_digest.get(item.digest)
            if existing is not None and existing != item:
                raise ValueError(f"digest {item.digest} names conflicting metadata")
            by_digest[item.digest] = item
        return StationFederationManifest(
            manifest_id=self.manifest_id,
            semantic_version=self.semantic_version,
            provider_registry_digest=self.provider_registry_digest,
            catalog_shards=self.catalog_shards,
            observation_partitions=tuple(by_digest.values()),
        )

    def coverage_manifest(self) -> dict:
        catalogs = self.active_catalog_shards()
        observations = self.active_observation_partitions()
        providers = sorted({
            source_id
            for shard in catalogs
            for source_id in shard.provider_source_ids
        } | {item.source_id for item in observations})
        variables = sorted({
            variable
            for shard in catalogs
            for variable in shard.variable_ids
        } | {
            variable
            for item in observations
            for variable in item.variable_ids
        })
        return {
            "schema": "station-federation-coverage/v1",
            "manifest_id": self.manifest_id,
            "manifest_digest": self.digest(),
            "provider_source_ids": providers,
            "catalog_shard_count": len(catalogs),
            "catalog_record_count": sum(item.station_count for item in catalogs),
            "observation_partition_count": len(observations),
            "variables": variables,
            "time_start": min((item.time_start for item in observations), default=None),
            "time_end": max((item.time_end for item in observations), default=None),
            "spatial_partitions": sorted({
                item.spatial_partition for item in (*catalogs, *observations)
            }),
            "catalog_availability": [
                {
                    "shard_id": item.shard_id,
                    "spatial_partition": item.spatial_partition,
                    "digest": item.digest,
                    "station_count": item.station_count,
                    "provider_source_ids": list(item.provider_source_ids),
                    "variable_ids": list(item.variable_ids),
                }
                for item in catalogs
            ],
            "observation_availability": [
                {
                    "source_id": item.source_id,
                    "spatial_partition": item.spatial_partition,
                    "time_start": item.time_start,
                    "time_end": item.time_end,
                    "variable_ids": list(item.variable_ids),
                    "digest": item.digest,
                    "source_revision": item.source_revision,
                    "row_count": item.row_count,
                }
                for item in observations
            ],
        }
