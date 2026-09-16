cmake_minimum_required(VERSION 3.18)

project(
    ClimateGeometricFramework
    VERSION 0.1.0
    DESCRIPTION "Portable verification and research kernels for Climate"
    LANGUAGES Fortran
)

option(ENABLE_LEGACY_FORTRAN_PROBES "Compile selected legacy Fortran sources without running them" ON)
option(ENABLE_MPI "Compile MPI-dependent legacy source probes" OFF)
option(ENABLE_CUDA "Compile the experimental CUDA source probe" OFF)
option(ENABLE_TESTING "Build and run canonical portable tests" ON)
option(ENABLE_BENCHMARKS "Enable CMake-managed benchmarks when a real target exists" OFF)
option(ENABLE_EXPERIMENTAL_OPTIMIZER "Attempt to wire the experimental C++ optimizer" OFF)

# Keep generated module files outside the source tree.
set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/fortran-modules")
file(MAKE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}")

# Canonical portable numerical code. Unlike the legacy omnibus spectral source,
# this target contains only implemented primitives with executable witnesses.
add_library(climate_spectral_reference STATIC
    src/fortran/spectral_reference.f90
)
target_include_directories(climate_spectral_reference PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Canonical deterministic explicit time integration. This is intentionally a
# small ODE primitive; adaptive, split, semi-implicit, and climate-specific
# stepping remain separate obligations rather than being implied here.
add_library(climate_time_integration STATIC
    src/fortran/time_integration.f90
)
target_include_directories(climate_time_integration PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# Canonical linear implicit primitives for tridiagonal operators. The operator,
# boundary semantics, and pivot acceptance policy remain explicit caller inputs.
add_library(climate_linear_implicit STATIC
    src/fortran/linear_implicit.f90
)
target_include_directories(climate_linear_implicit PUBLIC
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

if(ENABLE_TESTING)
    enable_testing()
    add_executable(test_spectral_reference
        tests/fortran/test_spectral_reference.f90
    )
    target_link_libraries(test_spectral_reference PRIVATE climate_spectral_reference)
    add_test(NAME spectral_reference COMMAND test_spectral_reference)

    add_executable(test_time_integration
        tests/fortran/test_time_integration.f90
    )
    target_link_libraries(test_time_integration PRIVATE climate_time_integration)
    add_test(NAME time_integration COMMAND test_time_integration)

    add_executable(test_linear_implicit
        tests/fortran/test_linear_implicit.f90
    )
    target_link_libraries(test_linear_implicit PRIVATE climate_linear_implicit)
    add_test(NAME linear_implicit COMMAND test_linear_implicit)
endif()

if(ENABLE_LEGACY_FORTRAN_PROBES)
    # This file currently contains both a module and an embedded test program.
    # Compile/link it to expose source defects, but do not register or run it.
    add_executable(climate_physics_compile_probe climate_physics_core.f90)
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)

    # The source contains a placeholder program with extremely large allocations.
    # It is compile-only here. CMake deliberately does not register or run it.
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
message(STATUS "  Canonical linear implicit:    ON")
message(STATUS "  Canonical tests:              ${ENABLE_TESTING}")
message(STATUS "  Legacy Fortran probes:        ${ENABLE_LEGACY_FORTRAN_PROBES}")
message(STATUS "  MPI probe:                    ${ENABLE_MPI}")
message(STATUS "  CUDA probe:                   ${ENABLE_CUDA}")
message(STATUS "  CMake benchmarks:             ${ENABLE_BENCHMARKS}")
message(STATUS "  C++ optimizer target:         ${ENABLE_EXPERIMENTAL_OPTIMIZER}")
message(STATUS "")
