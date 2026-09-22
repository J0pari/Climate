cmake_minimum_required(VERSION 3.18)

project(
    ClimateGeometricFramework
    VERSION 0.1.0
    DESCRIPTION "Portable verification and research kernels for Climate"
    LANGUAGES Fortran
)

option(ENABLE_LAPACK_BACKEND "Build the production-oriented LAPACK tridiagonal backend" OFF)
option(ENABLE_TESTING "Build and run canonical portable tests" ON)

# Keep generated module files outside the source tree.
set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/fortran-modules")
file(MAKE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}")

add_library(climate_spectral_reference STATIC
    src/fortran/spectral_reference.f90
)
target_include_directories(climate_spectral_reference PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_library(climate_time_integration STATIC
    src/fortran/time_integration.f90
)
target_include_directories(climate_time_integration PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_library(climate_linear_implicit STATIC
    src/fortran/linear_implicit.f90
)
target_include_directories(climate_linear_implicit PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Focused immutable reference-value namespaces. They are not runtime config
# objects; domain kernels import only the conventional defaults they need.
add_library(climate_thermodynamic_reference_values STATIC
    src/fortran/thermodynamic_reference_values.f90
)
target_include_directories(climate_thermodynamic_reference_values PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_library(climate_geophysical_reference_values STATIC
    src/fortran/geophysical_reference_values.f90
)
target_include_directories(climate_geophysical_reference_values PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Exact constant-f Coriolis flow for the primitive-equation sign convention.
# This is a domain-specific closed-form operator, not a generic solver stack.
add_library(climate_coriolis_rotation STATIC
    src/fortran/coriolis_rotation.f90
)
target_include_directories(climate_coriolis_rotation PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Dry ideal-gas thermodynamic identities with explicit SI parameters and
# fail-closed domain validation. Moist and phase-change physics stay separate.
add_library(climate_dry_thermodynamics STATIC
    src/fortran/dry_thermodynamics.f90
)
target_include_directories(climate_dry_thermodynamics PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_dry_thermodynamics PUBLIC
    climate_thermodynamic_reference_values
    climate_geophysical_reference_values
)

# Shared pressure-coordinate domain semantics. This target owns only interface
# pressure validity and orientation, not dynamics, geometry, or time stepping.
add_library(climate_pressure_coordinate_grid STATIC
    src/fortran/pressure_coordinate_grid.f90
)
target_include_directories(climate_pressure_coordinate_grid PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Dimensionless sigma-coordinate mapping into the canonical pressure-interface
# domain. Layer mass and hydrostatics remain owned by pressure-coordinate kernels.
add_library(climate_sigma_coordinate STATIC
    src/fortran/sigma_coordinate.f90
)
target_include_directories(climate_sigma_coordinate PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_sigma_coordinate PUBLIC
    climate_pressure_coordinate_grid
)

# Hybrid pressure-coordinate mapping p(k)=A(k)+B(k)*ps from caller-owned,
# explicitly identified coefficient sets. No coefficient table lives here.
add_library(climate_hybrid_pressure_coordinate STATIC
    src/fortran/hybrid_pressure_coordinate.f90
)
target_include_directories(climate_hybrid_pressure_coordinate PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_hybrid_pressure_coordinate PUBLIC
    climate_pressure_coordinate_grid
)

# Hydrostatic mass/geopotential relationships on a declared pressure-coordinate
# column. Pressure velocity and geometric vertical velocity remain separate.
add_library(climate_pressure_coordinate_hydrostatics STATIC
    src/fortran/pressure_coordinate_hydrostatics.f90
)
target_include_directories(climate_pressure_coordinate_hydrostatics PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_pressure_coordinate_hydrostatics PUBLIC
    climate_pressure_coordinate_grid
    climate_thermodynamic_reference_values
    climate_geophysical_reference_values
)

# Pressure-coordinate continuity integration for omega from layer pressure-mean
# horizontal divergence and an explicit lower-boundary omega. No geometric
# vertical-velocity conversion or boundary closure policy is implied.
add_library(climate_pressure_coordinate_continuity STATIC
    src/fortran/pressure_coordinate_continuity.f90
)
target_include_directories(climate_pressure_coordinate_continuity PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_pressure_coordinate_continuity PUBLIC
    climate_pressure_coordinate_grid
)

# Climate-semantic adapter from caller-oriented horizontal face carrier-mass
# flux to pressure-layer mass and pressure-mean horizontal divergence. Mesh
# topology and velocity-to-flux construction remain caller-owned.
add_library(climate_pressure_flux_coupling STATIC
    src/fortran/pressure_flux_coupling.f90
)
target_include_directories(climate_pressure_flux_coupling PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_pressure_flux_coupling PUBLIC
    climate_pressure_coordinate_grid
)

# Pressure-coordinate kinetic/enthalpy/geopotential exchange identity. Spatial
# derivatives, EOS construction, and time integration remain caller-owned.
add_library(climate_pressure_energy_exchange STATIC
    src/fortran/pressure_energy_exchange.f90
)
target_include_directories(climate_pressure_energy_exchange PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Exact ideal-gas algebra for a dry-air/water-vapor mixture. Saturation-vapor-
# pressure and condensed-phase models remain separate scientific providers.
add_library(climate_moist_vapor_algebra STATIC
    src/fortran/moist_vapor_algebra.f90
)
target_include_directories(climate_moist_vapor_algebra PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_moist_vapor_algebra PUBLIC
    climate_thermodynamic_reference_values
)

# Published phase-explicit equilibrium vapor pressure. Phase selection remains
# caller-owned; this target intentionally has no automatic liquid/ice switch.
add_library(climate_saturation_vapor_pressure STATIC
    src/fortran/saturation_vapor_pressure.f90
)
target_include_directories(climate_saturation_vapor_pressure PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Thin semantic composition of the phase provider and exact vapor algebra.
# It introduces no new saturation or moist-air formula.
add_library(climate_saturation_moisture STATIC
    src/fortran/saturation_moisture.f90
)
target_include_directories(climate_saturation_moisture PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
target_link_libraries(climate_saturation_moisture PUBLIC
    climate_saturation_vapor_pressure
    climate_moist_vapor_algebra
)

# Coordinate-independent extensive flux-form mass and tracer update. Geometry,
# face-state reconstruction, and limiter choice remain caller-owned.
add_library(climate_conservative_transport STATIC
    src/fortran/conservative_transport.f90
)
target_include_directories(climate_conservative_transport PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Climate-specific geometric-height finite-volume diffusion operator. Solver
# choice is deliberately separate from operator construction.
add_library(climate_vertical_diffusion STATIC
    src/fortran/vertical_diffusion.f90
)
target_include_directories(climate_vertical_diffusion PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

if(ENABLE_LAPACK_BACKEND)
    find_package(LAPACK REQUIRED)

    add_library(climate_lapack_tridiagonal STATIC
        src/fortran/lapack_tridiagonal.f90
    )
    target_include_directories(climate_lapack_tridiagonal PUBLIC
        "${CMAKE_Fortran_MODULE_DIRECTORY}"
    )
    target_link_libraries(climate_lapack_tridiagonal PUBLIC LAPACK::LAPACK)
endif()

if(ENABLE_TESTING)
    enable_testing()

    # CTest owns the canonical witness group. Explicit repository verification
    # consumes the label instead of maintaining a second list of test names.
    function(climate_add_canonical_test test_name target_name)
        add_test(NAME ${test_name} COMMAND ${target_name})
        set_tests_properties(${test_name} PROPERTIES LABELS "canonical")
    endfunction()

    add_executable(test_spectral_reference
        tests/fortran/test_spectral_reference.f90
    )
    target_link_libraries(test_spectral_reference PRIVATE climate_spectral_reference)
    climate_add_canonical_test(spectral_reference test_spectral_reference)

    add_executable(test_time_integration
        tests/fortran/test_time_integration.f90
    )
    target_link_libraries(test_time_integration PRIVATE climate_time_integration)
    climate_add_canonical_test(time_integration test_time_integration)

    add_executable(test_linear_implicit
        tests/fortran/test_linear_implicit.f90
    )
    target_link_libraries(test_linear_implicit PRIVATE climate_linear_implicit)
    climate_add_canonical_test(linear_implicit test_linear_implicit)

    add_executable(test_coriolis_rotation
        tests/fortran/test_coriolis_rotation.f90
    )
    target_link_libraries(test_coriolis_rotation PRIVATE climate_coriolis_rotation)
    climate_add_canonical_test(coriolis_rotation test_coriolis_rotation)

    add_executable(test_dry_thermodynamics
        tests/fortran/test_dry_thermodynamics.f90
    )
    target_link_libraries(test_dry_thermodynamics PRIVATE climate_dry_thermodynamics)
    climate_add_canonical_test(dry_thermodynamics test_dry_thermodynamics)

    add_executable(test_pressure_coordinate_grid
        tests/fortran/test_pressure_coordinate_grid.f90
    )
    target_link_libraries(test_pressure_coordinate_grid PRIVATE
        climate_pressure_coordinate_grid
    )
    climate_add_canonical_test(pressure_coordinate_grid test_pressure_coordinate_grid)

    add_executable(test_sigma_coordinate
        tests/fortran/test_sigma_coordinate.f90
    )
    target_link_libraries(test_sigma_coordinate PRIVATE
        climate_sigma_coordinate
        climate_pressure_coordinate_hydrostatics
    )
    climate_add_canonical_test(sigma_coordinate test_sigma_coordinate)

    add_executable(test_hybrid_pressure_coordinate
        tests/fortran/test_hybrid_pressure_coordinate.f90
    )
    target_link_libraries(test_hybrid_pressure_coordinate PRIVATE
        climate_hybrid_pressure_coordinate
        climate_sigma_coordinate
        climate_pressure_coordinate_hydrostatics
    )
    climate_add_canonical_test(
        hybrid_pressure_coordinate
        test_hybrid_pressure_coordinate
    )

    add_executable(test_pressure_coordinate_hydrostatics
        tests/fortran/test_pressure_coordinate_hydrostatics.f90
    )
    target_link_libraries(test_pressure_coordinate_hydrostatics PRIVATE
        climate_pressure_coordinate_hydrostatics
        climate_moist_vapor_algebra
    )
    climate_add_canonical_test(
        pressure_coordinate_hydrostatics
        test_pressure_coordinate_hydrostatics
    )

    add_executable(test_pressure_coordinate_continuity
        tests/fortran/test_pressure_coordinate_continuity.f90
    )
    target_link_libraries(test_pressure_coordinate_continuity PRIVATE
        climate_pressure_coordinate_continuity
    )
    climate_add_canonical_test(
        pressure_coordinate_continuity
        test_pressure_coordinate_continuity
    )

    add_executable(test_pressure_energy_exchange
        tests/fortran/test_pressure_energy_exchange.f90
    )
    target_link_libraries(test_pressure_energy_exchange PRIVATE
        climate_pressure_energy_exchange
    )
    climate_add_canonical_test(pressure_energy_exchange test_pressure_energy_exchange)

    add_executable(test_moist_vapor_algebra
        tests/fortran/test_moist_vapor_algebra.f90
    )
    target_link_libraries(test_moist_vapor_algebra PRIVATE
        climate_moist_vapor_algebra
        climate_dry_thermodynamics
    )
    climate_add_canonical_test(moist_vapor_algebra test_moist_vapor_algebra)

    add_executable(test_pressure_thermo_composition
        tests/fortran/test_pressure_thermo_composition.f90
    )
    target_link_libraries(test_pressure_thermo_composition PRIVATE
        climate_thermodynamic_reference_values
        climate_geophysical_reference_values
        climate_dry_thermodynamics
        climate_moist_vapor_algebra
        climate_pressure_coordinate_hydrostatics
    )
    climate_add_canonical_test(
        pressure_thermo_composition
        test_pressure_thermo_composition
    )

    add_executable(test_saturation_vapor_pressure
        tests/fortran/test_saturation_vapor_pressure.f90
    )
    target_link_libraries(test_saturation_vapor_pressure PRIVATE
        climate_saturation_vapor_pressure
        climate_moist_vapor_algebra
    )
    climate_add_canonical_test(
        saturation_vapor_pressure
        test_saturation_vapor_pressure
    )

    add_executable(test_saturation_moisture
        tests/fortran/test_saturation_moisture.f90
    )
    target_link_libraries(test_saturation_moisture PRIVATE
        climate_saturation_moisture
        climate_saturation_vapor_pressure
        climate_moist_vapor_algebra
    )
    climate_add_canonical_test(saturation_moisture test_saturation_moisture)

    add_executable(test_conservative_transport
        tests/fortran/test_conservative_transport.f90
    )
    target_link_libraries(test_conservative_transport PRIVATE climate_conservative_transport)
    climate_add_canonical_test(conservative_transport test_conservative_transport)

    add_executable(test_pressure_flux_coupling
        tests/fortran/test_pressure_flux_coupling.f90
    )
    target_link_libraries(test_pressure_flux_coupling PRIVATE
        climate_pressure_flux_coupling
        climate_geophysical_reference_values
        climate_conservative_transport
        climate_pressure_coordinate_continuity
        climate_pressure_energy_exchange
    )
    climate_add_canonical_test(pressure_flux_coupling test_pressure_flux_coupling)

    add_executable(test_vertical_diffusion
        tests/fortran/test_vertical_diffusion.f90
    )
    target_link_libraries(test_vertical_diffusion PRIVATE climate_vertical_diffusion)
    climate_add_canonical_test(vertical_diffusion test_vertical_diffusion)

    if(ENABLE_LAPACK_BACKEND)
        add_executable(test_lapack_tridiagonal
            tests/fortran/test_lapack_tridiagonal.f90
        )
        target_link_libraries(test_lapack_tridiagonal PRIVATE
            climate_lapack_tridiagonal
            climate_linear_implicit
        )
        climate_add_canonical_test(lapack_tridiagonal test_lapack_tridiagonal)

        add_executable(test_vertical_diffusion_lapack
            tests/fortran/test_vertical_diffusion_lapack.f90
        )
        target_link_libraries(test_vertical_diffusion_lapack PRIVATE
            climate_vertical_diffusion
            climate_lapack_tridiagonal
        )
        climate_add_canonical_test(
            vertical_diffusion_lapack
            test_vertical_diffusion_lapack
        )
    endif()
endif()

message(STATUS "")
message(STATUS "Climate portable configuration:")
message(STATUS "  Canonical spectral reference: ON")
message(STATUS "  Canonical time integration:   ON")
message(STATUS "  Canonical linear reference:   ON")
message(STATUS "  Thermodynamic references:     ON")
message(STATUS "  Geophysical references:       ON")
message(STATUS "  Exact Coriolis rotation:      ON")
message(STATUS "  Dry thermodynamics:           ON")
message(STATUS "  Pressure coordinate contract: ON")
message(STATUS "  Pressure hydrostatics:        ON")
message(STATUS "  Pressure continuity/omega:    ON")
message(STATUS "  Pressure flux coupling:       ON")
message(STATUS "  Pressure energy exchange:     ON")
message(STATUS "  Moist vapor algebra:          ON")
message(STATUS "  Saturation vapor pressure:    ON")
message(STATUS "  Saturation moisture:          ON")
message(STATUS "  Conservative transport:       ON")
message(STATUS "  Vertical diffusion operator:  ON")
message(STATUS "  LAPACK tridiagonal backend:   ${ENABLE_LAPACK_BACKEND}")
message(STATUS "  Canonical tests:              ${ENABLE_TESTING}")
message(STATUS "")
