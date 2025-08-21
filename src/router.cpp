#include "../include/gms/router.hpp"
#include <cmath>
#include <sstream>

namespace gms {

// Heuristic thresholds
constexpr double DENSITY_THRESHOLD = 0.1; // Dense vs sparse
constexpr double BANDWIDTH_RATIO_THRESHOLD = 0.01;
constexpr double CONDITION_THRESHOLD_MILD = 1e3;
constexpr double CONDITION_THRESHOLD_SEVERE = 1e6;
constexpr double MIXED_PRECISION_THRESHOLD = 1e4;
constexpr double VERY_SPARSE_THRESHOLD = 0.001;

SolverConfig route(const MatrixFeatures &features, SolverGoal goal) {
  SolverConfig config;
  std::ostringstream rationale;

  // Defaults
  config.tol = 1e-8;
  config.max_iter = 1000;
  config.precision = Precision::FLOAT64;

  // Check if this is a least squares problem
  if (features.is_least_squares || features.is_rank_deficient) {
    config.method = SolverMethod::QR;
    config.precond = Preconditioner::NONE;

    rationale << "Chose QR: ";

    if (features.is_least_squares) {
      rationale << "Least squares problem (m=" << features.m
                << ", n=" << features.n << "). ";
    }

    if (features.is_rank_deficient) {
      rationale << "Rank-deficient matrix (rank < " << features.n << "). ";
    }
  }
  // Check if matrix is tridiagonal (bandwidth = 1)
  else if (features.bandwidth == 1) {
    config.method = SolverMethod::TRIDIAG;
    config.precond = Preconditioner::NONE;

    rationale << "Chose Tridiagonal solver: Bandwidth=1, specialized Thomas "
                 "algorithm is optimal. ";
  }
  // Check if matrix is banded with very small bandwidth relative to size
  // Only use banded solvers for matrices with small bandwidth and not extremely
  // sparse
  else if (features.bandwidth < features.n * BANDWIDTH_RATIO_THRESHOLD &&
           features.density < DENSITY_THRESHOLD &&
           features.density > VERY_SPARSE_THRESHOLD) {
    if (features.is_spd) {
      config.method = SolverMethod::BANDED_CHOL;
      config.precond = Preconditioner::NONE;

      rationale << "Chose Banded Cholesky: SPD=true, bandwidth="
                << features.bandwidth << " (< "
                << features.n * BANDWIDTH_RATIO_THRESHOLD
                << "), density=" << features.density << ". ";
    } else {
      // For non-SPD banded matrices, fall back to LU
      config.method = SolverMethod::LU;
      config.precond = Preconditioner::NONE;

      rationale << "Chose Dense LU: Banded but not SPD, bandwidth="
                << features.bandwidth
                << ". No specialized banded LU solver available. ";
    }
  }
  // Dense matrices (density > threshold)
  else if (features.density > DENSITY_THRESHOLD) {
    if (features.is_spd) {
      config.method = SolverMethod::LLT;
      config.precond = Preconditioner::NONE;

      rationale << "Chose Dense LLT: SPD=true, density=" << features.density
                << " (> " << DENSITY_THRESHOLD << "). ";
    } else if (features.is_symmetric) {
      config.method = SolverMethod::LDLT;
      config.precond = Preconditioner::NONE;

      rationale << "Chose Dense LDLT: Symmetric=true but not SPD, density="
                << features.density << " (> " << DENSITY_THRESHOLD << "). ";
    } else {
      config.method = SolverMethod::LU;
      config.precond = Preconditioner::NONE;

      rationale << "Chose Dense LU: General matrix, density="
                << features.density << " (> " << DENSITY_THRESHOLD << "). ";
    }
  }
  // Sparse matrices
  else {
    // For sparse matrices, prefer iterative methods
    // This is especially true for very sparse matrices (density <
    // VERY_SPARSE_THRESHOLD)
    if (features.is_spd) {
      config.method = SolverMethod::CG;

      // Choose preconditioner based on multiple matrix properties
      if (features.condition_estimate > CONDITION_THRESHOLD_SEVERE) {
        // For severely ill-conditioned matrices, ILU(0) has stronger preconditioning
        config.precond = Preconditioner::ILU0;

        rationale << "Chose Sparse CG+ILU(0): SPD=true, density="
                  << features.density << ", κ≈" << features.condition_estimate
                  << " (severe ill-conditioning). ";
      } else if (features.condition_estimate > CONDITION_THRESHOLD_MILD) {
        // For moderately ill-conditioned matrices
        if (features.is_diagonally_dominant) {
          // Diagonal dominance makes Jacobi more effective
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse CG+Jacobi: SPD=true, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (moderate ill-conditioning, diagonally dominant). ";
        } else {
          // Non-diagonally dominant matrices benefit from ILU(0)
          config.precond = Preconditioner::ILU0;

          rationale
              << "Chose Sparse CG+ILU(0): SPD=true, density="
              << features.density << ", κ≈" << features.condition_estimate
              << " (moderate ill-conditioning, not diagonally dominant). ";
        }
      } else {
        // For well-conditioned matrices
        if (features.density < VERY_SPARSE_THRESHOLD) {
          // Very sparse matrices benefit from Jacobi due to low setup cost
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse CG+Jacobi: SPD=true, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (well-conditioned, very sparse). ";
        } else if (features.is_diagonally_dominant) {
          // Diagonal dominance makes Jacobi very effective
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse CG+Jacobi: SPD=true, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (well-conditioned, diagonally dominant). ";
        } else {
          // For other well-conditioned matrices, no preconditioner may be sufficient
          config.precond = Preconditioner::NONE;

          rationale << "Chose Sparse CG: SPD=true, density=" << features.density
                    << ", κ≈" << features.condition_estimate
                    << " (well-conditioned). ";
        }
      }
    } else {
      config.method = SolverMethod::GMRES;

      // Choose preconditioner based on multiple matrix properties
      if (features.condition_estimate > CONDITION_THRESHOLD_SEVERE) {
        config.precond = Preconditioner::ILU0;

        rationale << "Chose Sparse GMRES+ILU(0): General matrix, density="
                  << features.density << ", κ≈" << features.condition_estimate
                  << " (severe ill-conditioning). ";
      } else if (features.condition_estimate > CONDITION_THRESHOLD_MILD) {
        // For moderately ill-conditioned matrices
        if (features.is_diagonally_dominant) {
          // Diagonal dominance makes Jacobi more effective
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse GMRES+Jacobi: General matrix, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (moderate ill-conditioning, diagonally dominant). ";
        } else {
          // Non-symmetric, non-diagonally dominant matrices benefit from ILU(0)
          config.precond = Preconditioner::ILU0;

          rationale
              << "Chose Sparse GMRES+ILU(0): General matrix, density="
              << features.density << ", κ≈" << features.condition_estimate
              << " (moderate ill-conditioning, not diagonally dominant). ";
        }
      } else {
        // For well-conditioned matrices
        if (features.density < VERY_SPARSE_THRESHOLD) {
          // Very sparse matrices benefit from Jacobi due to its lower setup cost
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse GMRES+Jacobi: General matrix, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (well-conditioned, very sparse). ";
        } else if (features.is_diagonally_dominant) {
          // Diagonal dominance makes Jacobi very effective
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse GMRES+Jacobi: General matrix, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (well-conditioned, diagonally dominant). ";
        } else {
          // For other well-conditioned matrices, GMRES often needs some preconditioning
          config.precond = Preconditioner::JACOBI;

          rationale << "Chose Sparse GMRES+Jacobi: General matrix, density="
                    << features.density << ", κ≈" << features.condition_estimate
                    << " (well-conditioned, general case). ";
        }
      }
    }
  }

  // Mixed precision decision
  // Enable if condition number is below threshold and not rank-deficient
  if (!features.is_rank_deficient &&
      features.condition_estimate < MIXED_PRECISION_THRESHOLD &&
      goal == SolverGoal::SPEED) {
    config.precision = Precision::MIXED;

    rationale << "Using mixed precision: κ≈" << features.condition_estimate
              << " < " << MIXED_PRECISION_THRESHOLD
              << " and speed prioritized. ";
  } else {

    config.precision = Precision::FLOAT64;

    if (features.is_rank_deficient) {
      rationale << "Using double precision: Rank-deficient problems require "
                   "higher precision. ";
    } else if (features.condition_estimate >= MIXED_PRECISION_THRESHOLD) {
      rationale << "Using double precision: κ≈" << features.condition_estimate
                << " >= " << MIXED_PRECISION_THRESHOLD
                << " (ill-conditioned). ";
    } else if (goal == SolverGoal::ACCURACY) {
      rationale << "Using double precision: Accuracy prioritized over speed. ";
    }
  }

  config.rationale = rationale.str();

  return config;
}

} // namespace gms
