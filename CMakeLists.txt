cmake_minimum_required(VERSION 3.18)

project(
    ClimateGeometricFramework
    VERSION 0.1.0
    DESCRIPTION "Portable verification and research kernels for Climate"
    LANGUAGES Fortran
)

option(ENABLE_LEGACY_FORTRAN_PROBES "Compile selected legacy Fortran sources without running them" ON)
option(ENABLE_LAPACK_BACKEND "Build the production-oriented LAPACK tridiagonal backend" OFF)
option(ENABLE_MPI "Compile MPI-dependent legacy source probes" OFF)
option(ENABLE_CUDA "Compile the experimental CUDA source probe" OFF)
option(ENABLE_TESTING "Build and run canonical portable tests" ON)
option(ENABLE_BENCHMARKS "Enable CMake-managed benchmarks when a real target exists" OFF)
option(ENABLE_EXPERIMENTAL_OPTIMIZER "Attempt to wire the experimental C++ optimizer" OFF)

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

# Shared pressure-coordinate domain semantics. This target owns only interface
# pressure validity and orientation, not dynamics, geometry, or time stepping.
add_library(climate_pressure_coordinate_grid STATIC
    src/fortran/pressure_coordinate_grid.f90
)
target_include_directories(climate_pressure_coordinate_grid PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
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

    # CTest owns the canonical witness group. CI consumes the label instead of
    # maintaining a second list of test names.
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

if(ENABLE_LEGACY_FORTRAN_PROBES)
    add_executable(climate_physics_compile_probe climate_physics_core.f90)
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)
    add_executable(climate_oscillation_compile_probe climate_oscillation_monitor.f90)
    target_link_libraries(climate_oscillation_compile_probe PRIVATE MPI::MPI_Fortran)
endif()

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "ENABLE_CUDA=ON requested, but no CUDA compiler is available. "
            "CUDA absence must remain an explicit capability gate.")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(climate_cuda_compile_probe OBJECT climate_curvature_compute.cu)
    set_target_properties(climate_cuda_compile_probe PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

if(ENABLE_EXPERIMENTAL_OPTIMIZER)
    message(FATAL_ERROR
        "ENABLE_EXPERIMENTAL_OPTIMIZER=ON is not yet supported by a complete "
        "dependency contract; climate_optimizer.cpp remains inventory-only.")
endif()

if(ENABLE_BENCHMARKS)
    message(FATAL_ERROR
        "ENABLE_BENCHMARKS=ON requested, but no CMake benchmark target is defined yet.")
endif()

message(STATUS "")
message(STATUS "Climate portable configuration:")
message(STATUS "  Canonical spectral reference: ON")
message(STATUS "  Canonical time integration:   ON")
message(STATUS "  Canonical linear reference:   ON")
message(STATUS "  Exact Coriolis rotation:      ON")
message(STATUS "  Dry thermodynamics:           ON")
message(STATUS "  Pressure coordinate contract: ON")
message(STATUS "  Pressure hydrostatics:        ON")
message(STATUS "  Pressure continuity/omega:    ON")
message(STATUS "  Pressure energy exchange:     ON")
message(STATUS "  Moist vapor algebra:          ON")
message(STATUS "  Saturation vapor pressure:    ON")
message(STATUS "  Saturation moisture:          ON")
message(STATUS "  Conservative transport:       ON")
message(STATUS "  Vertical diffusion operator:  ON")
message(STATUS "  LAPACK tridiagonal backend:   ${ENABLE_LAPACK_BACKEND}")
message(STATUS "  Canonical tests:              ${ENABLE_TESTING}")
message(STATUS "  Legacy Fortran probes:        ${ENABLE_LEGACY_FORTRAN_PROBES}")
message(STATUS "  MPI probe:                    ${ENABLE_MPI}")
message(STATUS "  CUDA probe:                   ${ENABLE_CUDA}")
message(STATUS "  CMake benchmarks:             ${ENABLE_BENCHMARKS}")
message(STATUS "  C++ optimizer target:         ${ENABLE_EXPERIMENTAL_OPTIMIZER}")
message(STATUS "")
