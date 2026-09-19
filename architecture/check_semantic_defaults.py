#!/usr/bin/env python3
"""Reject implicit semantic substitution in audited Climate source.

This is the binding source-level guard for the repository's no-implicit-
substitution rule. Required scientific inputs and execution identities must be
explicit. Missing or unavailable semantics may stay missing or fail closed, but
must not be replaced by an alternate dataset, source, calibration, configuration,
method, solver, backend, precision, resource class, preprocessing policy, or
other semantic identity through a language convenience.

Focused immutable *_reference_values authorities remain valid defaults for the
narrow conventional model parameters that explicitly name those authorities.
Lexical marker scans live in source_gates.py only as defense in depth; spelling a
path "fallback" is neither necessary nor sufficient for a finding here.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from architecture.source_surface import AUDITED_ROLES, iter_source_files

RUST_DEFAULT_IMPL = re.compile(r"\bimpl\s+Default\s+for\s+([A-Za-z_][A-Za-z0-9_]*)")
RUST_DERIVE_DEFAULT = re.compile(r"#\[derive\([^\]]*\bDefault\b[^\]]*\)\]")
RUST_ZERO_ARG_NEW = re.compile(r"\bpub\s+(?:const\s+)?fn\s+new\s*\(\s*\)")
RUST_OPTION_COALESCE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?:unwrap_or|unwrap_or_else|unwrap_or_default|or_else|or)\s*\("
)

FORTRAN_PUBLIC_TYPE = re.compile(
    r"^\s*type\s*,\s*public\s*::\s*([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
FORTRAN_END_TYPE = re.compile(r"^\s*end\s+type\b", re.IGNORECASE)
FORTRAN_COMPONENT_DEFAULT = re.compile(
    r"::\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^!,\n]+)",
    re.IGNORECASE,
)
FORTRAN_REFERENCE_SYMBOL = re.compile(
    r"\bparameter\s*,\s*public\s*::\s*([A-Z][A-Z0-9_]*)\s*=",
    re.IGNORECASE,
)
SEMANTIC_INPUT_TYPE = re.compile(
    r"(?:boundary|parameters|configuration|config|policy|options|settings|"
    r"observation|forcing|calibration|input)",
    re.IGNORECASE,
)

SEMANTIC_NAMES = frozenset({
    "backend", "backend_id", "implementation", "implementation_id",
    "implementation_build", "method", "method_id", "precision",
    "resource_class", "dataset", "datasets", "dataset_id", "dataset_ref",
    "source_id", "source_ref", "provider", "provider_id", "configuration",
    "configuration_id", "configuration_ref", "calibration", "calibration_id",
    "calibration_ref", "forcing", "forcing_id", "forcing_ref", "observation",
    "observations", "observation_id", "observation_ref", "boundary_condition",
    "solver", "solver_id", "model", "model_id", "model_ref", "preprocessing",
    "quality_control_policy", "execution_policy", "data_policy",
    "numerical_policy", "network_policy", "determinism", "determinism_class",
})
SEMANTIC_PREFIXES = (
    "requested_", "resolved_", "candidate_", "baseline_", "selected_",
    "required_", "provided_", "supplied_", "input_", "alternate_",
)
SEMANTIC_SUFFIXES = ("_id", "_ref", "_digest", "_policy")


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _is_semantic_name(name: str) -> bool:
    normalized = name.strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in SEMANTIC_PREFIXES:
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                normalized = normalized[len(prefix):]
                changed = True
                break
    return normalized in SEMANTIC_NAMES or normalized.endswith(SEMANTIC_SUFFIXES)


def _reference_symbols(root: Path) -> set[str]:
    symbols: set[str] = set()
    for path in sorted((root / "src" / "fortran").glob("*_reference_values.f90")):
        text = path.read_text(encoding="utf-8")
        symbols.update(
            match.group(1).upper()
            for match in FORTRAN_REFERENCE_SYMBOL.finditer(text)
        )
    return symbols


def _check_rust(relative_path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in RUST_DEFAULT_IMPL.finditer(text):
        findings.append(Finding(
            "semantic_substitution.rust_default_impl",
            relative_path,
            f"audited type {match.group(1)!r} implements Default; semantic construction must be explicit",
        ))
    if RUST_DERIVE_DEFAULT.search(text):
        findings.append(Finding(
            "semantic_substitution.rust_derive_default",
            relative_path,
            "audited scientific types must not derive Default",
        ))
    if RUST_ZERO_ARG_NEW.search(text):
        findings.append(Finding(
            "semantic_substitution.rust_zero_arg_constructor",
            relative_path,
            "zero-argument public new() can materialize omitted scientific semantics; require explicit inputs",
        ))
    for match in RUST_OPTION_COALESCE.finditer(text):
        if _is_semantic_name(match.group(1)):
            findings.append(Finding(
                "semantic_substitution.rust_option_coalescing",
                relative_path,
                (
                    f"semantic identity/input {match.group(1)!r} uses Option "
                    "coalescing; preserve absence or require an explicitly "
                    "requested alternative identity"
                ),
            ))
    return findings


def _check_fortran(
    relative_path: str,
    text: str,
    reference_symbols: set[str],
) -> list[Finding]:
    findings: list[Finding] = []
    active_type: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        type_match = FORTRAN_PUBLIC_TYPE.match(line)
        if type_match:
            active_type = type_match.group(1)
            continue
        if FORTRAN_END_TYPE.match(line):
            active_type = None
            continue
        if active_type is None or not SEMANTIC_INPUT_TYPE.search(active_type):
            continue

        default_match = FORTRAN_COMPONENT_DEFAULT.search(line)
        if default_match is None:
            continue

        field = default_match.group(1)
        rhs = default_match.group(2).strip()
        if "parameters" in active_type.lower() and rhs.upper() in reference_symbols:
            continue

        findings.append(Finding(
            "semantic_substitution.fortran_input_default",
            relative_path,
            (
                f"line {line_number}: public semantic input type "
                f"{active_type!r} defaults {field!r} to {rhs!r}; require an "
                "explicit caller choice or a named *_reference_values authority"
            ),
        ))
    return findings


def _python_expr_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return node.slice.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"get", "pop", "setdefault"} and node.args:
            key = node.args[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value
    return None


def _is_semantic_expr(node: ast.AST | None) -> bool:
    name = _python_expr_name(node)
    return name is not None and _is_semantic_name(name)


def _is_none(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_false(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _target_names(target: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, ast.Attribute):
        names.add(target.attr)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            names.update(_target_names(item))
    return names


def _semantic_assignments(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets.append(node.target)
            elif isinstance(node, ast.NamedExpr):
                targets.append(node.target)
            for target in targets:
                for name in _target_names(target):
                    if _is_semantic_name(name):
                        names.add(name)
    return names


def _allowed_failure_log(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    func = statement.value.func
    if not isinstance(func, ast.Attribute):
        return False
    root = func.value
    return (
        isinstance(root, ast.Name)
        and root.id in {"logger", "logging", "warnings"}
        and func.attr in {
            "debug", "info", "warning", "error", "exception", "critical", "warn",
        }
    )


def _fail_closed_block(statements: list[ast.stmt]) -> bool:
    return (
        bool(statements)
        and isinstance(statements[-1], ast.Raise)
        and all(
            isinstance(statement, ast.Raise) or _allowed_failure_log(statement)
            for statement in statements
        )
    )


def _caught_exception_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return {"<bare>"}
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for item in node.elts:
            names.update(_caught_exception_names(item))
        return names
    return set()


def _try_contains_import(node: ast.Try) -> bool:
    return any(
        isinstance(item, (ast.Import, ast.ImportFrom))
        for statement in node.body
        for item in ast.walk(statement)
    )


def _handler_catches_import_failure(handler: ast.ExceptHandler) -> bool:
    return bool(_caught_exception_names(handler.type).intersection({
        "<bare>", "ImportError", "ModuleNotFoundError", "Exception", "BaseException",
    }))


def _handler_has_semantic_substitution(handler: ast.ExceptHandler) -> bool:
    if _semantic_assignments(handler.body):
        return True
    return any(
        isinstance(item, (ast.Import, ast.ImportFrom))
        for statement in handler.body
        for item in ast.walk(statement)
    )


def _semantic_unavailable_branch(node: ast.If) -> list[ast.stmt]:
    test = node.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if _is_semantic_expr(test.operand):
            return node.body

    if not isinstance(test, ast.Compare):
        return []
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return []

    left = test.left
    right = test.comparators[0]
    if not _is_semantic_expr(left):
        return []
    if not (_is_none(right) or _is_false(right)):
        return []

    op = test.ops[0]
    if isinstance(op, (ast.Is, ast.Eq)):
        return node.body
    if isinstance(op, (ast.IsNot, ast.NotEq)):
        return node.orelse
    return []


def _explicit_unavailable_return(value: ast.AST | None) -> bool:
    if value is None or _is_none(value):
        return True
    if isinstance(value, ast.Call):
        name = _python_expr_name(value.func)
        if name and any(
            word in name.lower()
            for word in ("unavailable", "missing", "error", "failure", "failed")
        ):
            return True
    if isinstance(value, ast.Dict):
        pairs = {
            key.value: val.value
            for key, val in zip(value.keys, value.values)
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(val, ast.Constant)
            and isinstance(val.value, str)
        }
        return str(pairs.get("status", "")).lower() in {
            "unavailable", "missing", "failed", "error", "ineligible",
        }
    return False


def _block_has_success_return(statements: list[ast.stmt]) -> bool:
    return any(
        isinstance(node, ast.Return) and not _explicit_unavailable_return(node.value)
        for statement in statements
        for node in ast.walk(statement)
    )


def _check_python(relative_path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as error:
        return [Finding(
            "semantic_substitution.python_parse_error",
            relative_path,
            f"cannot structurally inspect Python source: {error.msg} at line {error.lineno}",
        )]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = (
                [None] * (len(positional) - len(node.args.defaults))
                + list(node.args.defaults)
            )
            for argument, default in zip(positional, defaults):
                if (
                    default is not None
                    and not _is_none(default)
                    and _is_semantic_name(argument.arg)
                ):
                    findings.append(Finding(
                        "semantic_substitution.python_parameter_default",
                        relative_path,
                        (
                            f"line {node.lineno}: semantic parameter "
                            f"{argument.arg!r} has a non-None default; callers "
                            "must select that semantic explicitly"
                        ),
                    ))
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if (
                    default is not None
                    and not _is_none(default)
                    and _is_semantic_name(argument.arg)
                ):
                    findings.append(Finding(
                        "semantic_substitution.python_parameter_default",
                        relative_path,
                        (
                            f"line {node.lineno}: semantic parameter "
                            f"{argument.arg!r} has a non-None default; callers "
                            "must select that semantic explicitly"
                        ),
                    ))

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"get", "pop", "setdefault"} and node.args:
                key = node.args[0]
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_semantic_name(key.value)
                ):
                    default: ast.AST | None = None
                    has_default = False
                    if node.func.attr == "get" and len(node.args) >= 2:
                        default, has_default = node.args[1], True
                    elif node.func.attr == "pop" and len(node.args) >= 2:
                        default, has_default = node.args[1], True
                    elif node.func.attr == "setdefault":
                        default = (
                            node.args[1] if len(node.args) >= 2 else ast.Constant(None)
                        )
                        has_default = True
                    if has_default and not _is_none(default):
                        findings.append(Finding(
                            "semantic_substitution.python_mapping_default",
                            relative_path,
                            (
                                f"line {node.lineno}: semantic key {key.value!r} "
                                f"is materialized through {node.func.attr}() "
                                "defaulting; preserve absence or reject it explicitly"
                            ),
                        ))

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
        ):
            key = node.args[1]
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and _is_semantic_name(key.value)
                and not _is_none(node.args[2])
            ):
                findings.append(Finding(
                    "semantic_substitution.python_attribute_default",
                    relative_path,
                    (
                        f"line {node.lineno}: semantic attribute {key.value!r} "
                        "uses a getattr default; preserve absence or reject it explicitly"
                    ),
                ))

        if (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and len(node.values) >= 2
            and _is_semantic_expr(node.values[0])
        ):
            name = _python_expr_name(node.values[0]) or "<semantic>"
            findings.append(Finding(
                "semantic_substitution.python_boolean_coalescing",
                relative_path,
                (
                    f"line {node.lineno}: semantic identity/input {name!r} is "
                    "selected with boolean coalescing; an alternative must have "
                    "its own explicit request/identity"
                ),
            ))

        if isinstance(node, ast.Try):
            import_try = _try_contains_import(node)
            for handler in node.handlers:
                if (
                    import_try
                    and _handler_catches_import_failure(handler)
                    and not _fail_closed_block(handler.body)
                ):
                    findings.append(Finding(
                        "semantic_substitution.python_import_recovery",
                        relative_path,
                        (
                            f"line {handler.lineno}: import failure must fail "
                            "closed; the handler may log and raise but must not "
                            "select another implementation"
                        ),
                    ))
                    continue
                if (
                    not _fail_closed_block(handler.body)
                    and _handler_has_semantic_substitution(handler)
                ):
                    findings.append(Finding(
                        "semantic_substitution.python_exception_recovery",
                        relative_path,
                        (
                            f"line {handler.lineno}: exception recovery mutates "
                            "or imports semantic identity; preserve the failure "
                            "instead of selecting a replacement"
                        ),
                    ))

        if isinstance(node, ast.If):
            branch = _semantic_unavailable_branch(node)
            if branch and not _fail_closed_block(branch):
                assigned = _semantic_assignments(branch)
                if assigned:
                    findings.append(Finding(
                        "semantic_substitution.python_availability_rewrite",
                        relative_path,
                        (
                            f"line {node.lineno}: missing semantic branch assigns "
                            f"identity/input(s) {sorted(assigned)!r}; fail closed "
                            "or require a separately declared alternative"
                        ),
                    ))
                elif _block_has_success_return(branch):
                    findings.append(Finding(
                        "semantic_substitution.python_missing_success",
                        relative_path,
                        (
                            f"line {node.lineno}: missing semantic branch returns "
                            "a successful value instead of preserving unavailability"
                        ),
                    ))

    return findings


def check(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    reference_symbols = _reference_symbols(root)
    findings: list[Finding] = []

    for source in iter_source_files(root, roles=AUDITED_ROLES):
        text = source.path.read_text(encoding="utf-8")
        if source.language == "rust":
            findings.extend(_check_rust(source.relative_path, text))
        elif source.language == "fortran":
            findings.extend(
                _check_fortran(source.relative_path, text, reference_symbols)
            )
        elif source.language == "python":
            findings.extend(_check_python(source.relative_path, text))
    return sorted(findings, key=lambda item: (item.path, item.code, item.message))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="reject implicit semantic substitution"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = check(args.root)
    if args.json:
        print(json.dumps(
            {"ok": not findings, "findings": [asdict(item) for item in findings]},
            indent=2,
            sort_keys=True,
        ))
    elif findings:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
    else:
        print(
            "semantic substitution: audited inputs and identities are explicit "
            "and fail closed"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
