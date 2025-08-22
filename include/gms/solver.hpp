#pragma once

#include "router.hpp"
#include <chrono>
#include <string>
#include <vector>

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

  // TODO: Implement proper matrix analysis
  // For now, we'll use some default values
  features.density = is_sparse ? 0.01 : 1.0;
  features.is_symmetric = false;
  features.is_spd = false;
  features.condition_estimate = 100.0;
  features.bandwidth = n - 1;
  features.is_diagonally_dominant = false;
  features.is_rank_deficient = false;
  features.is_least_squares = (m > n);

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

  // TODO: Implement actual solvers
  // For now, we'll just set success to true
  report.success = true;

  // End solve timer
  auto solve_end = high_resolution_clock::now();
  report.solve_time_ms =
      duration_cast<microseconds>(solve_end - solve_start).count() / 1000.0;

  return report;
}

} // namespace gms
