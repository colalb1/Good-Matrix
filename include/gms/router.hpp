#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gms {

/**
 * @brief Available solver methods.
 */
enum class SolverMethod {
  LLT,         // Cholesky for SPD
  LDLT,        // LDL^T for symmetric
  LU,          // LU with partial pivoting
  QR,          // QR for least squares problems
  BANDED_CHOL, // Banded Cholesky for SPD banded
  TRIDIAG,     // Thomas for tridiagonal
  CG,          // CG for sparse SPD
  GMRES        // GMRES for sparse
};

/**
 * @brief Aavailable preconditioners.
 */
enum class Preconditioner { NONE, JACOBI, ILU0 };

/**
 * @brief Available precision modes.
 */
enum class Precision { FLOAT32, FLOAT64, MIXED };

/**
 * @brief Solver priorities.
 */
enum class SolverGoal { ACCURACY, SPEED };

/**
 * @brief Structure to hold matrix features detected by analyzers.
 */
struct MatrixFeatures {
  std::size_t n;   // Matrix dimension (n x n)
  std::size_t m;   // Number rows (for non-square)
  std::size_t nnz; // Number of non-zero elements
  double density;  // Fraction of non-zero elements
  bool is_symmetric;
  bool is_spd;
  double condition_estimate;
  std::size_t bandwidth;
  bool is_diagonally_dominant;
  bool is_rank_deficient;
  bool is_least_squares;
};

/**
 * @brief Structure to hold solver configuration.
 */
struct SolverConfig {
  SolverMethod method; // Solver
  Preconditioner precond;
  Precision precision;
  double tol;
  std::size_t max_iter;
  std::string rationale; // Explanation for the chosen configuration
};

/**
 * @brief Routes to the most efficient solver based on matrix features.
 * @param features Matrix features detected by the analyzer.
 * @param goal Solver goal.
 * @return SolverConfig with the chosen method, preconditioner, precision, and
 * rationale.
 */
SolverConfig route(const MatrixFeatures &features,
                   SolverGoal goal = SolverGoal::ACCURACY);

} // namespace gms
