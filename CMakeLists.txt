cmake_minimum_required(VERSION 3.18)
project(ClimateGeometricFramework 
    VERSION 0.1.0
    DESCRIPTION "Geometric framework for climate modeling with GPU acceleration"
    LANGUAGES CXX CUDA Fortran)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Fortran flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fcheck=all -fbacktrace")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_MPI "Enable MPI support" OFF)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_TESTING "Enable testing" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" OFF)
option(USE_DOUBLE_PRECISION "Use double precision floating point" ON)

# Find packages
find_package(Threads REQUIRED)

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# MPI
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    add_definitions(-DUSE_MPI)
endif()

# CUDA
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        
        # Set CUDA architectures (adjust for your GPU)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
        endif()
        
        # CUDA libraries
        set(CUDA_LIBRARIES
            CUDA::cudart
            CUDA::cublas
            CUDA::cufft
            CUDA::curand
            CUDA::cusparse
            CUDA::cusolver
        )
        
        # Check for NCCL (may not be available on Windows)
        find_package(NCCL QUIET)
        if(NCCL_FOUND)
            list(APPEND CUDA_LIBRARIES nccl)
            add_definitions(-DHAVE_NCCL)
        else()
            message(STATUS "NCCL not found, using stub implementation")
        endif()
        
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "CUDA requested but not available")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# External libraries
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, will download")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

# NetCDF (optional)
find_package(NetCDF QUIET)
if(NetCDF_FOUND)
    include_directories(${NetCDF_INCLUDE_DIRS})
    add_definitions(-DHAVE_NETCDF)
endif()

# HDF5 (optional)
find_package(HDF5 COMPONENTS C CXX QUIET)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(-DHAVE_HDF5)
endif()

# Enzyme for automatic differentiation (optional)
find_package(Enzyme QUIET)
if(Enzyme_FOUND)
    add_definitions(-DHAVE_ENZYME)
    message(STATUS "Found Enzyme for automatic differentiation")
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/CORE
    ${PROJECT_BINARY_DIR}
)

# Source files
set(CLIMATE_CORE_SOURCES
    CORE/climate_optimizer.cpp
    CORE/climate_physics_core.f90
    CORE/climate_spectral_analysis.f90
    CORE/climate_oscillation_monitor.f90
)

set(CLIMATE_CUDA_SOURCES
    CORE/climate_curvature_compute.cu
)

# Create main library
add_library(climate_core ${CLIMATE_CORE_SOURCES})
target_link_libraries(climate_core 
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(Eigen3_FOUND)
    target_link_libraries(climate_core Eigen3::Eigen)
else()
    target_link_libraries(climate_core eigen)
endif()

if(ENABLE_MPI)
    target_link_libraries(climate_core ${MPI_CXX_LIBRARIES})
endif()

if(NetCDF_FOUND)
    target_link_libraries(climate_core ${NetCDF_LIBRARIES})
endif()

if(HDF5_FOUND)
    target_link_libraries(climate_core ${HDF5_LIBRARIES})
endif()

# CUDA library
if(ENABLE_CUDA)
    add_library(climate_cuda ${CLIMATE_CUDA_SOURCES})
    target_link_libraries(climate_cuda ${CUDA_LIBRARIES})
    set_target_properties(climate_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Link CUDA library to main library
    target_link_libraries(climate_core climate_cuda)
endif()

# Configure precision
if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
    if(ENABLE_CUDA)
        target_compile_definitions(climate_cuda PRIVATE USE_DOUBLE_PRECISION)
    endif()
else()
    add_definitions(-DUSE_SINGLE_PRECISION)
    if(ENABLE_CUDA)
        target_compile_definitions(climate_cuda PRIVATE USE_SINGLE_PRECISION)
    endif()
endif()

# Create configuration header
configure_file(
    "${PROJECT_SOURCE_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/climate_config.h"
)

# Create config.h.in template
file(WRITE "${PROJECT_SOURCE_DIR}/config.h.in"
"#ifndef CLIMATE_CONFIG_H
#define CLIMATE_CONFIG_H

#cmakedefine USE_MPI
#cmakedefine USE_CUDA
#cmakedefine HAVE_NCCL
#cmakedefine HAVE_NETCDF
#cmakedefine HAVE_HDF5
#cmakedefine HAVE_ENZYME
#cmakedefine USE_DOUBLE_PRECISION
#cmakedefine USE_SINGLE_PRECISION

#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

#define VERSION_MAJOR ${PROJECT_VERSION_MAJOR}
#define VERSION_MINOR ${PROJECT_VERSION_MINOR}
#define VERSION_PATCH ${PROJECT_VERSION_PATCH}

#endif // CLIMATE_CONFIG_H
")

# Executables
add_executable(climate_optimizer_test CORE/test_optimizer.cpp)
target_link_libraries(climate_optimizer_test climate_core)

if(ENABLE_CUDA)
    add_executable(climate_cuda_test CORE/test_cuda.cu)
    target_link_libraries(climate_cuda_test climate_cuda ${CUDA_LIBRARIES})
endif()

# Testing
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS climate_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(ENABLE_CUDA)
    install(TARGETS climate_cuda
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

install(FILES
    "${PROJECT_BINARY_DIR}/climate_config.h"
    DESTINATION include
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ClimateGeometricFrameworkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Print configuration summary
message(STATUS "")
message(STATUS "Climate Geometric Framework Configuration Summary:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler:    ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Fortran compiler: ${CMAKE_Fortran_COMPILER}")
message(STATUS "")
message(STATUS "  Options:")
message(STATUS "    MPI:            ${ENABLE_MPI}")
message(STATUS "    OpenMP:         ${ENABLE_OPENMP}")
message(STATUS "    CUDA:           ${ENABLE_CUDA}")
if(ENABLE_CUDA)
message(STATUS "    CUDA archs:     ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "    NCCL:           ${NCCL_FOUND}")
endif()
message(STATUS "    NetCDF:         ${NetCDF_FOUND}")
message(STATUS "    HDF5:           ${HDF5_FOUND}")
message(STATUS "    Enzyme:         ${Enzyme_FOUND}")
message(STATUS "    Testing:        ${ENABLE_TESTING}")
message(STATUS "    Benchmarks:     ${ENABLE_BENCHMARKS}")
message(STATUS "    Precision:      ${USE_DOUBLE_PRECISION}")
message(STATUS "")