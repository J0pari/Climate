cmake_minimum_required(VERSION 3.18)

# The portable CMake surface is a compile-verification harness, not a claim that
# the repository is an integrated production model.  Keep accelerator and MPI
# requirements opt-in so a CPU checkout can establish what is actually portable.
project(
    ClimateGeometricFramework
    VERSION 0.1.0
    DESCRIPTION "Compile-verification harness for experimental climate sources"
    LANGUAGES Fortran
)

option(ENABLE_PORTABLE_SOURCE_PROBES "Compile portable Fortran source probes" ON)
option(ENABLE_MPI "Compile MPI-dependent source probes" OFF)
option(ENABLE_CUDA "Compile the experimental CUDA source probe" OFF)
option(ENABLE_TESTING "Enable CMake-managed tests when a real test target exists" OFF)
option(ENABLE_BENCHMARKS "Enable CMake-managed benchmarks when a real target exists" OFF)
option(ENABLE_EXPERIMENTAL_OPTIMIZER "Attempt to wire the experimental C++ optimizer" OFF)

# Keep generated module files outside the source tree.
set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/fortran-modules")
file(MAKE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}")

if(ENABLE_PORTABLE_SOURCE_PROBES)
    # climate_physics_core.f90 currently contains both the module and an embedded
    # test program.  Build it as an executable compile probe, but never register
    # it as a CTest test: compilation is evidence of syntax/link coherence only,
    # not scientific validation or safe runtime behavior.
    add_executable(climate_physics_compile_probe climate_physics_core.f90)

    # This file is a module-only source and can be compiled without MPI/CUDA.
    add_library(climate_spectral_compile_probe STATIC climate_spectral_analysis.f90)
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS Fortran)

    # The source contains a placeholder program with extremely large allocations.
    # It is compile-only here.  CMake deliberately does not register or run it.
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

    # Compile the CUDA translation unit only when explicitly requested.  This
    # target does not make a GPU-validation claim; additional headers/libraries
    # (for example NCCL/cuDNN/MPI/CUTLASS) may still be required and should fail
    # visibly rather than falling back silently.
    add_library(climate_cuda_compile_probe OBJECT climate_curvature_compute.cu)
    set_target_properties(climate_cuda_compile_probe PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

if(ENABLE_EXPERIMENTAL_OPTIMIZER)
    # climate_optimizer.cpp is not a portable C++ core.  It directly includes a
    # large CUDA/MPI/numerical-optimization stack (CUDA, NCCL, cuDNN, MPI, Ipopt,
    # SNOPT, GSL, NLopt, NetCDF/HDF5, TBB, ...).  Until those dependencies and
    # fallback identities are represented explicitly, constructing a target here
    # would create a false build contract.
    message(FATAL_ERROR
        "ENABLE_EXPERIMENTAL_OPTIMIZER=ON is not yet supported by a complete "
        "dependency contract; climate_optimizer.cpp remains inventory-only.")
endif()

if(ENABLE_TESTING)
    message(FATAL_ERROR
        "ENABLE_TESTING=ON requested, but no CMake test target is defined yet. "
        "Architecture/reference tests are executed by their own authoritative harnesses.")
endif()

if(ENABLE_BENCHMARKS)
    message(FATAL_ERROR
        "ENABLE_BENCHMARKS=ON requested, but no CMake benchmark target is defined yet.")
endif()

message(STATUS "")
message(STATUS "Climate compile-verification configuration:")
message(STATUS "  Portable Fortran probes: ${ENABLE_PORTABLE_SOURCE_PROBES}")
message(STATUS "  MPI probe:              ${ENABLE_MPI}")
message(STATUS "  CUDA probe:             ${ENABLE_CUDA}")
message(STATUS "  CMake tests:            ${ENABLE_TESTING}")
message(STATUS "  CMake benchmarks:       ${ENABLE_BENCHMARKS}")
message(STATUS "  C++ optimizer target:   ${ENABLE_EXPERIMENTAL_OPTIMIZER}")
message(STATUS "")
