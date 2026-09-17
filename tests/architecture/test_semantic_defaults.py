from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_semantic_defaults


class SemanticDefaultTests(unittest.TestCase):
    def _root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "src" / "fortran").mkdir(parents=True)
        return temporary, root

    def test_rust_default_and_fallback_constructors_are_rejected(self):
        temporary, root = self._root()
        with temporary:
            (root / "src" / "bad.rs").write_text(
                "#[derive(Default)]\npub struct ObservationConfig { value: f64 }\n"
                "impl Default for OtherConfig { fn default() -> Self { todo!() } }\n"
                "pub fn legacy_reference() -> ObservationConfig { todo!() }\n"
                "pub fn new() -> ObservationConfig { todo!() }\n",
                encoding="utf-8",
            )
            codes = {item.code for item in check_semantic_defaults.check(root)}
            self.assertIn("semantic_default.rust_derive_default", codes)
            self.assertIn("semantic_default.rust_default_impl", codes)
            self.assertIn("semantic_default.rust_fallback_constructor", codes)
            self.assertIn("semantic_default.rust_zero_arg_constructor", codes)

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
                ["semantic_default.fortran_input_component"],
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

    def test_repository_canonical_source_has_no_implicit_semantic_defaults(self):
        self.assertEqual(check_semantic_defaults.check(), [])


if __name__ == "__main__":
    unittest.main()
