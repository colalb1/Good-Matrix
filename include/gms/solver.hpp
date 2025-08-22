#pragma once

#include "router.hpp"
#include "analyzer.hpp"
#include "dense.hpp"
#include "banded.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>

namespace gms {

/**
 * @brief Structure to hold information about the solve process.
 */
struct SolverReport {
  SolverMethod method;
  Preconditioner preconditioner;
  Precision precision;
  double residual_norm;
  double relative_residual;
  std::size_t iterations;
  double setup_time_ms;
  double solve_time_ms;
  std::string rationale; // Explanation of solver selection
  bool success;          // Whether the solve was successful

  // Constructor with defaults
  SolverReport()
      : method(SolverMethod::LU), preconditioner(Preconditioner::NONE),
        precision(Precision::FLOAT64), residual_norm(0.0),
        relative_residual(0.0), iterations(0), setup_time_ms(0.0),
        solve_time_ms(0.0), rationale(""), success(false) {}
};

/**
 * @brief Solves a linear system Ax = b using the most efficient method.
 *
 * @tparam T Floating point type (float, double)
 * @param A Pointer to matrix data (row-major)
 * @param b Pointer to right-hand side vector
 * @param x Pointer to solution vector (output)
 * @param n Number of rows/columns in A (for square matrices)
 * @param m Number of rows in A (for non-square matrices, default = n)
 * @param is_sparse Whether A is in sparse format
 * @param strategy Solver strategy ("auto", "direct", "iterative")
 * @return SolverReport containing information about the solve process
 */
template <typename T>
SolverReport solve_system(const T *A, const T *b, T *x, std::size_t n,
                          std::size_t m = 0, bool is_sparse = false,
                          const std::string &strategy = "auto") {
  using namespace std::chrono;
  SolverReport report;

  // Start setup timer
  auto setup_start = high_resolution_clock::now();

  // Default m to n
  if (m == 0) {
    m = n;
  }

  // Analyze features
  MatrixFeatures features;
  features.n = n;
  features.m = m;
  
  // Proper matrix analysis
  if (!is_sparse) {
    // Calculate density
    features.nnz = 0;
    const T eps = std::numeric_limits<T>::epsilon();

    for (std::size_t i = 0; i < m * n; ++i) {
      if (std::abs(A[i]) > eps) {
        features.nnz++;
      }
    }
    features.density = static_cast<double>(features.nnz) / (m * n);
    
    // Check if matrix is square
    if (m == n) {
      // Check symmetry
      features.is_symmetric = is_symmetric(A, n, static_cast<T>(eps * 10));
      
      // Check SPD (only if symmetric)
      if (features.is_symmetric) {
        features.is_spd = is_spd(A, n, n, static_cast<T>(eps * 100));
      } else {
        features.is_spd = false;
      }
      
      // Calculate bandwidth
      features.bandwidth = bandwidth(A, n, BandType::MAX, static_cast<T>(eps * 10));
      
      // Check diagonal dominance
      features.is_diagonally_dominant = is_diagonally_dominant(A, n, false);
      
      // Estimate condition number
      features.condition_estimate = condition_estimate<T>(A, n, 5);
      
      // Check rank deficiency
      std::vector<T> A_copy(A, A + n * n);
      std::size_t rank = rank_estimate_cpqr(A_copy.data(), n, static_cast<T>(eps * 1000));
      features.is_rank_deficient = (rank < n);
    } else {
      // Non-square matrix
      features.is_symmetric = false;
      features.is_spd = false;
      features.bandwidth = std::min(m, n) - 1;
      features.is_diagonally_dominant = false;
      features.condition_estimate = 1000.0; // Default for non-square
      features.is_rank_deficient = false;
      features.is_least_squares = (m > n);
    }
  } else {
    // For sparse matrices, we'll use some default values
    features.nnz = static_cast<std::size_t>(0.01 * m * n);
    features.density = 0.01;
    features.is_symmetric = false;
    features.is_spd = false;
    features.condition_estimate = 1000.0;
    features.bandwidth = n - 1;
    features.is_diagonally_dominant = false;
    features.is_rank_deficient = false;
    features.is_least_squares = (m > n);
  }

  // Determine solver goal based on strategy
  SolverGoal goal = SolverGoal::ACCURACY;
  if (strategy == "speed") {
    goal = SolverGoal::SPEED;
  }

  // Route to appropriate solver
  SolverConfig config = route(features, goal);

  // Copy config to report
  report.method = config.method;
  report.preconditioner = config.precond;
  report.precision = config.precision;
  report.rationale = config.rationale;

  // End setup timer
  auto setup_end = high_resolution_clock::now();
  report.setup_time_ms =
      duration_cast<microseconds>(setup_end - setup_start).count() / 1000.0;

  // Start solve timer
  auto solve_start = high_resolution_clock::now();

  // Implement actual solvers based on the router's decision
  report.success = false;
  
  // Temp buffers
  std::vector<T> A_copy;
  std::vector<T> workspace;
  std::vector<std::size_t> pivots;
  
  // Solve based on the selected method
  switch (report.method) {
    case SolverMethod::LLT: {
      if (m == n) {
        A_copy.assign(A, A + n * n);
        report.success = cholesky_solve_inplace(A_copy.data(), n, n, b, x);
      }
      break;
    }
    case SolverMethod::LDLT: {
      if (m == n) {
        A_copy.assign(A, A + n * n);
        std::vector<T> d(n);
        report.success = ldlt_solve_inplace(A_copy.data(), n, n, b, x, d.data());
      }
      break;
    }
    case SolverMethod::LU: {
      if (m == n) {
        A_copy.assign(A, A + n * n);
        pivots.resize(n);
        report.success = lu_solve_inplace(A_copy.data(), b, x, pivots.data(), n, n);
      }
      break;
    }
    case SolverMethod::QR: {
      if (m >= n) {
        A_copy.assign(A, A + m * n);
        std::vector<T> tau(n);
        report.success = qr_solve_inplace(A_copy.data(), b, x, tau.data(), m, n, n);
      }
      break;
    }
    case SolverMethod::BANDED_CHOL: {
      // Not fully implemented yet
      // In a real implementation, we would extract the banded structure and use banded_cholesky_solve
      if (m == n) {
        A_copy.assign(A, A + n * n);
        report.success = cholesky_solve_inplace(A_copy.data(), n, n, b, x);
      }
      break;
    }
    case SolverMethod::TRIDIAG: {
      // Not fully implemented yet
      // Would extract the tridiagonal structure and use tridiag_solve
      if (m == n) {
        A_copy.assign(A, A + n * n);
        report.success = lu_solve_inplace(A_copy.data(), b, x, pivots.data(), n, n);
      }
      break;
    }
    case SolverMethod::CG: {
      // Not fully implemented yet
      // In a real implementation, we would use the CG solver
      if (m == n) {
        A_copy.assign(A, A + n * n);
        report.success = cholesky_solve_inplace(A_copy.data(), n, n, b, x);
      }
      break;
    }
    case SolverMethod::GMRES: {
      // Not fully implemented yet
      // In a real implementation, we would use the GMRES solver
      if (m == n) {
        A_copy.assign(A, A + n * n);
        pivots.resize(n);
        report.success = lu_solve_inplace(A_copy.data(), b, x, pivots.data(), n, n);
      }
      break;
    }
    default:
      // Fallback to LU for unknown methods
      if (m == n) {
        A_copy.assign(A, A + n * n);
        pivots.resize(n);
        report.success = lu_solve_inplace(A_copy.data(), b, x, pivots.data(), n, n);
      }
      break;
  }
  
  // Calculate residual if solve was successful
  if (report.success) {
    T residual_norm_sq = 0;
    T b_norm_sq = 0;
    
    for (std::size_t i = 0; i < m; ++i) {
      T row_sum = 0;
      for (std::size_t j = 0; j < n; ++j) {
        row_sum += A[i * n + j] * x[j];
      }
      T residual = row_sum - b[i];
      residual_norm_sq += residual * residual;
      b_norm_sq += b[i] * b[i];
    }
    
    report.residual_norm = std::sqrt(residual_norm_sq);
    if (b_norm_sq > std::numeric_limits<T>::epsilon()) {
      report.relative_residual = report.residual_norm / std::sqrt(b_norm_sq);
    } else {
      report.relative_residual = report.residual_norm;
    }
  }

  // End solve timer
  auto solve_end = high_resolution_clock::now();
  report.solve_time_ms =
      duration_cast<microseconds>(solve_end - solve_start).count() / 1000.0;

  return report;
}

} // namespace gms
