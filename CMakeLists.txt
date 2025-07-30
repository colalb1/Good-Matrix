cmake_minimum_required(VERSION 3.27)

project(gms
  VERSION 0.1.0
  DESCRIPTION "General Matrix Solver (dense/sparse/banded) with auto-routing and Python bindings"
  LANGUAGES CXX)

# ------------------------------
# Options
# ------------------------------
option(GMS_BUILD_PYTHON    "Build Python extension module via pybind11" ON)
option(GMS_BUILD_TESTS     "Build unit tests"                            OFF)
option(GMS_USE_OPENMP      "Enable OpenMP for parallel loops"            ON)
option(GMS_USE_BLAS        "Link BLAS/LAPACK for dense paths"            ON)
option(GMS_WARNINGS_AS_ERRORS "Treat warnings as errors"                 OFF)

# ------------------------------
# C++ standard and compiler flags
# ------------------------------
# Prefer C++23 for portability. If you truly need pre-standard C++2c, see check below.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common warnings
if (MSVC)
  add_compile_options(/W4 /permissive-)
  if (GMS_WARNINGS_AS_ERRORS)
    add_compile_options(/WX)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if (GMS_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
  endif()
endif()

# Try enabling experimental "C++2c" if available and desired.
option(GMS_TRY_CXX2C "Attempt to compile with -std=c++2c if supported" OFF)
if (GMS_TRY_CXX2C AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++2c" HAS_CXX2C)
  if (HAS_CXX2C)
    message(STATUS "Enabling experimental -std=c++2c")
    add_compile_options(-std=c++2c)
  endif()
endif()

# Build type default
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Position-independent code for shared libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------
# Dependencies (optional)
# ------------------------------
find_package(Threads REQUIRED)

# OpenMP (optional)
if (GMS_USE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling")
  else()
    message(STATUS "OpenMP not found: continuing without")
    set(GMS_USE_OPENMP OFF)
  endif()
endif()

# BLAS/LAPACK (optional, for dense backends)
if (GMS_USE_BLAS)
  find_package(BLAS)
  find_package(LAPACK)
  if (BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "BLAS/LAPACK found: enabling dense accelerated paths")
  else()
    message(STATUS "BLAS/LAPACK not found: continuing without")
    set(GMS_USE_BLAS OFF)
  endif()
endif()

# pybind11 (for Python module) via FetchContent for self-contained builds
if (GMS_BUILD_PYTHON)
  include(FetchContent)
  set(PYBIND11_NEWPYTHON ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ------------------------------
# Library target (gms)
# ------------------------------
# List your sources; keep headers in include/gms
set(GMS_SOURCES
  src/analyzer.cpp
  src/router.cpp
  src/dense.cpp
  src/banded.cpp
  src/tridiag.cpp
  src/csr.cpp
  src/spmv.cpp
  src/cg.cpp
  src/gmres.cpp
  src/precond.cpp
  src/ilu0.cpp
  src/mixed_precision.cpp
)

add_library(gms ${GMS_SOURCES})
add_library(gms::gms ALIAS gms)

target_include_directories(gms
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(gms PUBLIC cxx_std_23)
target_link_libraries(gms PUBLIC Threads::Threads)

if (GMS_USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(gms PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(gms PUBLIC GMS_WITH_OPENMP=1)
endif()

if (GMS_USE_BLAS AND BLAS_FOUND AND LAPACK_FOUND)
  target_link_libraries(gms PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  target_compile_definitions(gms PUBLIC GMS_WITH_BLAS=1 GMS_WITH_LAPACK=1)
endif()

# Stricter warnings for this target only
if (MSVC)
  target_compile_options(gms PRIVATE /Zc:__cplusplus)
else()
  target_compile_options(gms PRIVATE -fno-omit-frame-pointer)
endif()

# ------------------------------
# Python extension (pybind11 module)
# ------------------------------
if (GMS_BUILD_PYTHON)
  # The module source should implement PYBIND11_MODULE(gms, m) { ... }
  pybind11_add_module(gms_python MODULE src/pybind_module.cpp)
  # Name the module "gms" when imported in Python
  set_target_properties(gms_python PROPERTIES OUTPUT_NAME "gms")
  target_link_libraries(gms_python PRIVATE gms)
  target_compile_features(gms_python PRIVATE cxx_std_23)

  if (GMS_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(gms_python PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if (GMS_USE_BLAS AND BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(gms_python PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  endif()
endif()

# ------------------------------
# Te
