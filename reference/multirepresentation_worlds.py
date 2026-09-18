#!/usr/bin/env python3
"""Deterministic ground-truth worlds for multirepresentation falsification.

This module defines benchmark data, not a structure-selection algorithm. It
contains worlds in which the correct relationship is common-manifold, product,
fibered, quotient/noninjective, stratified, nuisance-dominated, or null. Some
relationships are deliberately not identifiable from static paired samples, so
a valid downstream method may need to abstain rather than force a geometry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "fixtures" / "multirepresentation" / "structural-worlds-v1.json"
)


@dataclass(frozen=True)
class StructuralWorld:
    world_id: str
    relationship: str
    static_identifiability: str
    view_a: np.ndarray
    view_b: np.ndarray
    targets: dict[str, np.ndarray]
    ground_truth: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.world_id or not self.relationship or not self.static_identifiability:
            raise ValueError("world identity and semantics must be non-empty")
        a = np.asarray(self.view_a, dtype=float)
        b = np.asarray(self.view_b, dtype=float)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
            raise ValueError("world views must be matched two-dimensional arrays")
        if a.shape[0] == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("world views must be non-empty and finite")
        targets: dict[str, np.ndarray] = {}
        for name, values in self.targets.items():
            value = np.asarray(values)
            if value.shape[0] != a.shape[0]:
                raise ValueError(f"target {name} does not match sample count")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f"target {name} contains non-finite values")
            copy = np.array(value, copy=True)
            copy.setflags(write=False)
            targets[name] = copy
        a = np.array(a, copy=True)
        b = np.array(b, copy=True)
        a.setflags(write=False)
        b.setflags(write=False)
        object.__setattr__(self, "view_a", a)
        object.__setattr__(self, "view_b", b)
        object.__setattr__(self, "targets", targets)


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported structural-world fixture schema")
    return payload


def _latent(rng: np.random.Generator, count: int, bounds: tuple[float, float]) -> np.ndarray:
    return rng.uniform(bounds[0], bounds[1], size=count)


def _world(
    spec: dict[str, Any],
    *,
    count: int,
    bounds: tuple[float, float],
    nuisance_scale: float,
    boundary: float,
) -> StructuralWorld:
    rng = np.random.default_rng(int(spec["seed"]))
    world_id = str(spec["world_id"])
    ground_truth = {
        "relationship": spec["relationship"],
        "static_identifiability": spec["static_identifiability"],
        "shared_dimension": int(spec["shared_dimension"]),
        "private_dimensions": [int(value) for value in spec["private_dimensions"]],
    }

    if world_id == "shared_manifold":
        shared = _latent(rng, count, bounds)
        view_a = np.column_stack((shared, shared**2))
        view_b = np.column_stack((np.sin(shared), np.cos(shared)))
        targets = {"shared": shared}

    elif world_id == "product":
        private_a = _latent(rng, count, bounds)
        private_b = _latent(rng, count, bounds)
        view_a = np.column_stack((private_a, private_a**2))
        view_b = np.column_stack((private_b, private_b**2))
        targets = {"private_a": private_a, "private_b": private_b}
        ground_truth["paired_sample_semantics"] = "same_joint_product_realization"
        ground_truth["static_dependence_can_prove_product_semantics"] = False

    elif world_id == "fibered":
        base = _latent(rng, count, bounds)
        fiber_a = _latent(rng, count, bounds)
        fiber_b = _latent(rng, count, bounds)
        view_a = np.column_stack((base, fiber_a))
        view_b = np.column_stack((np.sin(base), np.cos(base), fiber_b))
        targets = {"shared_base": base, "fiber_a": fiber_a, "fiber_b": fiber_b}

    elif world_id == "quotient_noninjective":
        signed = _latent(rng, count, bounds)
        private_a = _latent(rng, count, bounds)
        private_b = _latent(rng, count, bounds)
        quotient = signed**2
        view_a = np.column_stack((signed, private_a))
        view_b = np.column_stack((quotient, private_b))
        targets = {
            "signed_coordinate": signed,
            "shared_quotient": quotient,
            "private_a": private_a,
            "private_b": private_b,
        }
        ground_truth["lost_in_view_b"] = "sign_of_signed_coordinate"

    elif world_id == "stratified_regime":
        shared = _latent(rng, count, bounds)
        private_a = _latent(rng, count, bounds)
        private_b = _latent(rng, count, bounds)
        regime = shared >= boundary
        branch_x = np.where(regime, 2.0 * shared + 1.0, shared)
        branch_y = np.where(regime, -shared, shared**2)
        view_a = np.column_stack((shared, private_a))
        view_b = np.column_stack((branch_x, branch_y + 0.05 * private_b, private_b))
        targets = {
            "shared": shared,
            "regime": regime.astype(np.int8),
            "private_a": private_a,
            "private_b": private_b,
        }
        ground_truth["stratum_count"] = 2
        ground_truth["boundary"] = boundary

    elif world_id == "nuisance_dominated":
        shared = _latent(rng, count, bounds)
        nuisance_a = rng.normal(0.0, nuisance_scale, size=(count, 2))
        nuisance_b = rng.normal(0.0, nuisance_scale, size=(count, 2))
        view_a = np.column_stack((shared, nuisance_a))
        view_b = np.column_stack((np.sin(shared), nuisance_b))
        targets = {
            "shared": shared,
            "nuisance_a_0": nuisance_a[:, 0],
            "nuisance_a_1": nuisance_a[:, 1],
            "nuisance_b_0": nuisance_b[:, 0],
            "nuisance_b_1": nuisance_b[:, 1],
        }
        ground_truth["nuisance_scale"] = nuisance_scale

    elif world_id == "independent_null":
        private_a = rng.normal(0.0, 1.0, size=(count, 2))
        private_b = rng.normal(0.0, 1.0, size=(count, 2))
        view_a = private_a
        view_b = private_b
        targets = {
            "private_a_0": private_a[:, 0],
            "private_a_1": private_a[:, 1],
            "private_b_0": private_b[:, 0],
            "private_b_1": private_b[:, 1],
        }
        ground_truth["paired_sample_semantics"] = "no_cross_view_identity"
        ground_truth["method_should_invent_shared_structure"] = False

    else:
        raise ValueError(f"unknown structural world {world_id!r}")

    return StructuralWorld(
        world_id=world_id,
        relationship=str(spec["relationship"]),
        static_identifiability=str(spec["static_identifiability"]),
        view_a=view_a,
        view_b=view_b,
        targets=targets,
        ground_truth=ground_truth,
    )


def generate_worlds(fixture: dict[str, Any]) -> dict[str, StructuralWorld]:
    count = int(fixture["sample_count"])
    policy = fixture["generation_policy"]
    minimum = int(policy["minimum_samples"])
    if count < minimum:
        raise ValueError(f"structural-world suite requires at least {minimum} samples")
    bounds_raw = policy["latent_bounds"]
    if (
        not isinstance(bounds_raw, list)
        or len(bounds_raw) != 2
        or not float(bounds_raw[0]) < float(bounds_raw[1])
    ):
        raise ValueError("latent_bounds must contain increasing finite endpoints")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1]))
    if not np.isfinite(bounds).all():
        raise ValueError("latent_bounds must be finite")
    nuisance_scale = float(policy["nuisance_scale"])
    boundary = float(policy["stratified_boundary"])
    if not np.isfinite(nuisance_scale) or nuisance_scale <= 0.0 or not np.isfinite(boundary):
        raise ValueError("generation policy must be finite and nuisance_scale positive")

    specs = fixture.get("worlds")
    if not isinstance(specs, list) or not specs:
        raise ValueError("structural-world fixture must declare worlds")
    worlds: dict[str, StructuralWorld] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("world specification must be an object")
        world = _world(
            spec,
            count=count,
            bounds=bounds,
            nuisance_scale=nuisance_scale,
            boundary=boundary,
        )
        if world.world_id in worlds:
            raise ValueError(f"duplicate structural world {world.world_id}")
        worlds[world.world_id] = world
    return worlds


def summarize_worlds(fixture: dict[str, Any]) -> dict[str, Any]:
    worlds = generate_worlds(fixture)
    return {
        "fixture_id": fixture["fixture_id"],
        "sample_count": int(fixture["sample_count"]),
        "worlds": {
            name: {
                "relationship": world.relationship,
                "static_identifiability": world.static_identifiability,
                "view_a_shape": list(world.view_a.shape),
                "view_b_shape": list(world.view_b.shape),
                "ground_truth": world.ground_truth,
            }
            for name, world in worlds.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(summarize_worlds(load_fixture()), indent=2, sort_keys=True))
