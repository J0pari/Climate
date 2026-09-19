from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_semantic_defaults


class SemanticSubstitutionTests(unittest.TestCase):
    def _root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "src" / "fortran").mkdir(parents=True)
        (root / "reference").mkdir()
        return temporary, root

    def test_rust_default_construction_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.rs").write_text(
                "#[derive(Default)]\npub struct ObservationConfig { value: f64 }\n"
                "impl Default for OtherConfig { fn default() -> Self { todo!() } }\n"
                "pub fn new() -> ObservationConfig { todo!() }\n",
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.rust_derive_default", codes)
            self.assertIn("semantic_substitution.rust_default_impl", codes)
            self.assertIn("semantic_substitution.rust_zero_arg_constructor", codes)

    def test_rust_semantic_option_coalescing_is_rejected_without_fallback_word(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.rs").write_text(
                'fn choose(requested_backend: Option<&str>) -> &str {\n'
                '    requested_backend.unwrap_or("cpu")\n'
                '}\n',
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.rust_option_coalescing", codes)

    def test_rust_mathematical_zero_default_is_not_semantic_substitution(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "good.rs").write_text(
                "fn coefficient(value: Option<f64>) -> f64 { value.unwrap_or(0.0) }\n",
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_fortran_boundary_default_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "fortran" / "bad.f90").write_text(
                "module bad\n"
                "  type, public :: diffusion_boundary\n"
                "    integer :: kind = 0\n"
                "  end type diffusion_boundary\n"
                "end module bad\n",
                encoding="utf-8",
            )
            findings = check_semantic_defaults.check(root)
            self.assertEqual(
                [item.code for item in findings],
                ["semantic_substitution.fortran_input_default"],
            )

    def test_fortran_parameter_default_from_reference_authority_is_allowed(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "fortran" / "thermodynamic_reference_values.f90").write_text(
                "module refs\n"
                "  real, parameter, public :: GAS_CONSTANT = 287.05\n"
                "end module refs\n",
                encoding="utf-8",
            )
            (root / "src" / "fortran" / "good.f90").write_text(
                "module good\n"
                "  type, public :: thermo_parameters\n"
                "    real :: gas_constant = GAS_CONSTANT\n"
                "  end type thermo_parameters\n"
                "end module good\n",
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_diagnostic_storage_initializers_are_not_semantic_inputs(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "fortran" / "good.f90").write_text(
                "module good\n"
                "  type, public :: solve_diagnostics\n"
                "    real :: residual = 0.0\n"
                "  end type solve_diagnostics\n"
                "end module good\n",
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_python_mapping_default_for_semantic_identity_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.py").write_text(
                'source_id = payload.get("source_id", "")\n',
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_mapping_default", codes)

    def test_python_direct_semantic_identity_access_is_allowed(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "good.py").write_text(
                'source_id = payload["source_id"]\n',
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_python_boolean_semantic_coalescing_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.py").write_text(
                'backend = requested_backend or "cpu"\n',
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_boolean_coalescing", codes)

    def test_python_semantic_parameter_default_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.py").write_text(
                'def execute(*, backend="cpu"):\n    return backend\n',
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_parameter_default", codes)

    def test_python_none_semantic_parameter_keeps_absence_explicit(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "good.py").write_text(
                'def execute(*, backend=None):\n'
                '    if backend is None:\n'
                '        raise RuntimeError("backend required")\n'
                '    return backend\n',
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_python_import_substitution_is_rejected_without_fallback_word(self):
        temporary, root = self._root()
        with temporary:
            (root / "reference" / "bad.py").write_text(
                "try:\n"
                "    import accelerated_backend\n"
                "except ImportError:\n"
                "    import cpu_backend\n"
                "    backend = cpu_backend\n",
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_import_recovery", codes)

    def test_python_import_failure_that_logs_and_raises_is_allowed(self):
        temporary, root = self._root()
        with temporary:
            (root / "reference" / "good.py").write_text(
                "try:\n"
                "    import required_backend\n"
                "except ImportError as exc:\n"
                "    logger.error('required backend unavailable')\n"
                "    raise RuntimeError('required backend unavailable') from exc\n",
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_python_exception_semantic_rewrite_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.py").write_text(
                "try:\n"
                "    backend = load_requested_backend()\n"
                "except RuntimeError:\n"
                "    backend = load_cpu_backend()\n",
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_exception_recovery", codes)

    def test_python_unavailable_semantic_rewrite_is_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.py").write_text(
                "if requested_backend is None:\n"
                '    requested_backend = "cpu"\n',
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_availability_rewrite", codes)

    def test_python_missing_observation_cannot_return_plausible_value(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.py").write_text(
                "def project(observation=None):\n"
                "    if observation is None:\n"
                "        return 273.15\n"
                "    return observation\n",
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_substitution.python_missing_success", codes)

    def test_python_explicit_unavailable_result_is_allowed(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "good.py").write_text(
                "def project(observation=None):\n"
                "    if observation is None:\n"
                '        return {"status": "unavailable"}\n'
                "    return observation\n",
                encoding="utf-8",
            )
            self.assertEqual(check_semantic_defaults.check(root), [])

    def test_repository_audited_source_has_no_implicit_semantic_substitution(self):
        self.assertEqual(check_semantic_defaults.check(), [])


if __name__ == "__main__":
    unittest.main()
