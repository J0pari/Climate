#!/usr/bin/env python3
"""Ground-truth worlds for multirepresentation falsification.

This module is the sole repository authority for how each declared structural
world maps latent roles into paired views and evaluation targets. Static and
temporal samplers share the same renderer so adding time ordering cannot create
a second implementation of shared, product, fibered, quotient, stratified,
nuisance-dominated, or null semantics.

Evaluation and dynamics-policy fixtures may bind to the exact structural-world
fixture bytes, but they do not restate per-world truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "fixtures" / "multirepresentation" / "structural-worlds-v1.json"
)
DEFAULT_DYNAMICS_FIXTURE = (
    ROOT
    / "fixtures"
    / "multirepresentation"
    / "structural-world-dynamics-v1.json"
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


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def bind_world_fixture_authority(
    world_fixture: dict[str, Any],
    policy_fixture: dict[str, Any],
    *,
    world_fixture_path: Path = DEFAULT_FIXTURE,
) -> None:
    if world_fixture.get("fixture_id") != policy_fixture.get("world_fixture_id"):
        raise ValueError("policy resolved a different structural-world fixture identity")
    observed_blob = git_blob_sha(world_fixture_path)
    expected_blob = policy_fixture.get("world_fixture_git_blob_sha")
    if observed_blob != expected_blob:
        raise ValueError(
            "authoritative structural-world fixture bytes changed without "
            "a new policy binding"
        )


def world_set_digest(worlds: dict[str, StructuralWorld]) -> str:
    digest = hashlib.sha256()
    for world_id in sorted(worlds):
        world = worlds[world_id]
        digest.update(world_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(world.view_a, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(world.view_b, dtype="<f8").tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def load_dynamics_fixture(
    path: Path = DEFAULT_DYNAMICS_FIXTURE,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported structural-world dynamics fixture schema")
    steps = payload.get("trajectory_steps")
    burn = payload.get("burn_in_steps")
    dt = float(payload.get("dt"))
    discovery_offset = payload.get("discovery_seed_offset")
    confirmation_offset = payload.get("confirmation_seed_offset")
    if not isinstance(steps, int) or steps < 128:
        raise ValueError("trajectory_steps must be an integer >= 128")
    if not isinstance(burn, int) or burn < 0:
        raise ValueError("burn_in_steps must be a nonnegative integer")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(discovery_offset, int) or discovery_offset <= 0:
        raise ValueError("discovery_seed_offset must be a positive integer")
    if not isinstance(confirmation_offset, int) or confirmation_offset <= 0:
        raise ValueError("confirmation_seed_offset must be a positive integer")
    if discovery_offset == confirmation_offset:
        raise ValueError("discovery and confirmation seed offsets must differ")

    processes = payload.get("role_processes")
    if not isinstance(processes, dict) or set(processes) != {
        "shared",
        "private",
        "nuisance",
    }:
        raise ValueError(
            "role_processes must declare shared, private, and nuisance policies"
        )
    for role, policy in processes.items():
        rho = float(policy["rho"])
        stationary_std = float(policy["stationary_std"])
        if not math.isfinite(rho) or abs(rho) >= 1.0:
            raise ValueError(f"{role} rho must have magnitude < 1")
        if not math.isfinite(stationary_std) or stationary_std <= 0.0:
            raise ValueError(f"{role} stationary_std must be finite and positive")
    return payload


def _latent(
    rng: np.random.Generator,
    count: int,
    bounds: tuple[float, float],
) -> np.ndarray:
    return rng.uniform(bounds[0], bounds[1], size=count)


def _static_latents(
    world_id: str,
    rng: np.random.Generator,
    *,
    count: int,
    bounds: tuple[float, float],
    nuisance_scale: float,
) -> dict[str, np.ndarray]:
    if world_id == "shared_manifold":
        return {"shared": _latent(rng, count, bounds)}
    if world_id == "product":
        return {
            "private_a": _latent(rng, count, bounds),
            "private_b": _latent(rng, count, bounds),
        }
    if world_id == "fibered":
        return {
            "base": _latent(rng, count, bounds),
            "fiber_a": _latent(rng, count, bounds),
            "fiber_b": _latent(rng, count, bounds),
        }
    if world_id == "quotient_noninjective":
        return {
            "signed": _latent(rng, count, bounds),
            "private_a": _latent(rng, count, bounds),
            "private_b": _latent(rng, count, bounds),
        }
    if world_id == "stratified_regime":
        return {
            "shared": _latent(rng, count, bounds),
            "private_a": _latent(rng, count, bounds),
            "private_b": _latent(rng, count, bounds),
        }
    if world_id == "nuisance_dominated":
        shared = _latent(rng, count, bounds)
        nuisance_a = rng.normal(0.0, nuisance_scale, size=(count, 2))
        nuisance_b = rng.normal(0.0, nuisance_scale, size=(count, 2))
        return {
            "shared": shared,
            "nuisance_a_0": nuisance_a[:, 0],
            "nuisance_a_1": nuisance_a[:, 1],
            "nuisance_b_0": nuisance_b[:, 0],
            "nuisance_b_1": nuisance_b[:, 1],
        }
    if world_id == "independent_null":
        private_a = rng.normal(0.0, 1.0, size=(count, 2))
        private_b = rng.normal(0.0, 1.0, size=(count, 2))
        return {
            "private_a_0": private_a[:, 0],
            "private_a_1": private_a[:, 1],
            "private_b_0": private_b[:, 0],
            "private_b_1": private_b[:, 1],
        }
    raise ValueError(f"unknown structural world {world_id!r}")


def _render_world(
    spec: dict[str, Any],
    latents: dict[str, np.ndarray],
    *,
    nuisance_scale: float,
    boundary: float,
) -> StructuralWorld:
    world_id = str(spec["world_id"])
    shared_evaluation_target_name: str | None = None
    ground_truth = {
        "relationship": spec["relationship"],
        "static_identifiability": spec["static_identifiability"],
        "shared_dimension": int(spec["shared_dimension"]),
        "private_dimensions": [int(value) for value in spec["private_dimensions"]],
    }

    if world_id == "shared_manifold":
        shared = latents["shared"]
        view_a = np.column_stack((shared, shared**2))
        view_b = np.column_stack((np.sin(shared), np.cos(shared)))
        targets = {"shared": shared}
        shared_evaluation_target_name = "shared"

    elif world_id == "product":
        private_a = latents["private_a"]
        private_b = latents["private_b"]
        view_a = np.column_stack((private_a, private_a**2))
        view_b = np.column_stack((private_b, private_b**2))
        targets = {"private_a": private_a, "private_b": private_b}
        ground_truth["paired_sample_semantics"] = "same_joint_product_realization"
        ground_truth["static_dependence_can_prove_product_semantics"] = False

    elif world_id == "fibered":
        base = latents["base"]
        fiber_a = latents["fiber_a"]
        fiber_b = latents["fiber_b"]
        view_a = np.column_stack((base, fiber_a))
        view_b = np.column_stack((np.sin(base), np.cos(base), fiber_b))
        targets = {"shared_base": base, "fiber_a": fiber_a, "fiber_b": fiber_b}
        shared_evaluation_target_name = "shared_base"

    elif world_id == "quotient_noninjective":
        signed = latents["signed"]
        private_a = latents["private_a"]
        private_b = latents["private_b"]
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
        shared_evaluation_target_name = "shared_quotient"

    elif world_id == "stratified_regime":
        shared = latents["shared"]
        private_a = latents["private_a"]
        private_b = latents["private_b"]
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
        shared_evaluation_target_name = "shared"

    elif world_id == "nuisance_dominated":
        shared = latents["shared"]
        nuisance_a = np.column_stack(
            [latents["nuisance_a_0"], latents["nuisance_a_1"]]
        )
        nuisance_b = np.column_stack(
            [latents["nuisance_b_0"], latents["nuisance_b_1"]]
        )
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
        shared_evaluation_target_name = "shared"

    elif world_id == "independent_null":
        private_a = np.column_stack(
            [latents["private_a_0"], latents["private_a_1"]]
        )
        private_b = np.column_stack(
            [latents["private_b_0"], latents["private_b_1"]]
        )
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

    ground_truth["shared_evaluation_target_name"] = shared_evaluation_target_name
    return StructuralWorld(
        world_id=world_id,
        relationship=str(spec["relationship"]),
        static_identifiability=str(spec["static_identifiability"]),
        view_a=view_a,
        view_b=view_b,
        targets=targets,
        ground_truth=ground_truth,
    )


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
    latents = _static_latents(
        world_id,
        rng,
        count=count,
        bounds=bounds,
        nuisance_scale=nuisance_scale,
    )
    return _render_world(
        spec,
        latents,
        nuisance_scale=nuisance_scale,
        boundary=boundary,
    )


def _validated_generation_policy(
    fixture: dict[str, Any],
) -> tuple[int, tuple[float, float], float, float, list[dict[str, Any]]]:
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
    if (
        not np.isfinite(nuisance_scale)
        or nuisance_scale <= 0.0
        or not np.isfinite(boundary)
    ):
        raise ValueError(
            "generation policy must be finite and nuisance_scale positive"
        )
    specs = fixture.get("worlds")
    if not isinstance(specs, list) or not specs:
        raise ValueError("structural-world fixture must declare worlds")
    if not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("world specification must be an object")
    return count, bounds, nuisance_scale, boundary, specs


def generate_worlds(
    fixture: dict[str, Any], *, seed_offset: int = 0
) -> dict[str, StructuralWorld]:
    if not isinstance(seed_offset, int) or seed_offset < 0:
        raise ValueError("seed_offset must be a nonnegative integer")
    count, bounds, nuisance_scale, boundary, specs = _validated_generation_policy(
        fixture
    )
    worlds: dict[str, StructuralWorld] = {}
    for spec in specs:
        resolved_spec = dict(spec)
        resolved_spec["seed"] = int(spec["seed"]) + seed_offset
        world = _world(
            resolved_spec,
            count=count,
            bounds=bounds,
            nuisance_scale=nuisance_scale,
            boundary=boundary,
        )
        if world.world_id in worlds:
            raise ValueError(f"duplicate structural world {world.world_id}")
        worlds[world.world_id] = world
    return worlds


def _ar1_series(
    rng: np.random.Generator,
    *,
    count: int,
    burn_in: int,
    rho: float,
    stationary_std: float,
) -> np.ndarray:
    total = count + burn_in
    values = np.empty(total, dtype=float)
    values[0] = float(rng.normal(0.0, stationary_std))
    innovation_std = stationary_std * math.sqrt(1.0 - rho * rho)
    for index in range(1, total):
        values[index] = (
            rho * values[index - 1]
            + innovation_std * float(rng.normal())
        )
    retained = values[burn_in:]
    if retained.shape != (count,) or not np.isfinite(retained).all():
        raise RuntimeError("AR(1) role process produced invalid trajectory")
    return retained


def _dynamic_role_map(world_id: str) -> dict[str, str]:
    if world_id == "shared_manifold":
        return {"shared": "shared"}
    if world_id == "product":
        return {"private_a": "private", "private_b": "private"}
    if world_id == "fibered":
        return {
            "base": "shared",
            "fiber_a": "private",
            "fiber_b": "private",
        }
    if world_id == "quotient_noninjective":
        return {
            "signed": "shared",
            "private_a": "private",
            "private_b": "private",
        }
    if world_id == "stratified_regime":
        return {
            "shared": "shared",
            "private_a": "private",
            "private_b": "private",
        }
    if world_id == "nuisance_dominated":
        return {
            "shared": "shared",
            "nuisance_a_0": "nuisance",
            "nuisance_a_1": "nuisance",
            "nuisance_b_0": "nuisance",
            "nuisance_b_1": "nuisance",
        }
    if world_id == "independent_null":
        return {
            "private_a_0": "private",
            "private_a_1": "private",
            "private_b_0": "private",
            "private_b_1": "private",
        }
    raise ValueError(f"unknown structural world {world_id!r}")


def generate_world_trajectories(
    world_fixture: dict[str, Any],
    dynamics_fixture: dict[str, Any],
    *,
    confirmation: bool = False,
    world_fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, StructuralWorld]:
    bind_world_fixture_authority(
        world_fixture,
        dynamics_fixture,
        world_fixture_path=world_fixture_path,
    )
    _, _, nuisance_scale, boundary, specs = _validated_generation_policy(
        world_fixture
    )
    count = int(dynamics_fixture["trajectory_steps"])
    burn_in = int(dynamics_fixture["burn_in_steps"])
    offset_name = (
        "confirmation_seed_offset" if confirmation else "discovery_seed_offset"
    )
    seed_offset = int(dynamics_fixture[offset_name])
    processes = dynamics_fixture["role_processes"]

    worlds: dict[str, StructuralWorld] = {}
    for spec in specs:
        world_id = str(spec["world_id"])
        rng = np.random.default_rng(int(spec["seed"]) + seed_offset)
        latents: dict[str, np.ndarray] = {}
        for latent_name, role in _dynamic_role_map(world_id).items():
            policy = processes[role]
            latents[latent_name] = _ar1_series(
                rng,
                count=count,
                burn_in=burn_in,
                rho=float(policy["rho"]),
                stationary_std=float(policy["stationary_std"]),
            )
        world = _render_world(
            spec,
            latents,
            nuisance_scale=nuisance_scale,
            boundary=boundary,
        )
        if world_id in worlds:
            raise ValueError(f"duplicate structural world {world_id}")
        worlds[world_id] = world
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
