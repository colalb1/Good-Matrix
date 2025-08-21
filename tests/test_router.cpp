#include "../include/gms/router.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace gms;

std::string solver_method_to_string(SolverMethod method) {
  switch (method) {
  case SolverMethod::LLT:
    return "LLT";
  case SolverMethod::LDLT:
    return "LDLT";
  case SolverMethod::LU:
    return "LU";
  case SolverMethod::QR:
    return "QR";
  case SolverMethod::BANDED_CHOL:
    return "BANDED_CHOL";
  case SolverMethod::TRIDIAG:
    return "TRIDIAG";
  case SolverMethod::CG:
    return "CG";
  case SolverMethod::GMRES:
    return "GMRES";
  default:
    return "UNKNOWN";
  }
}

std::string preconditioner_to_string(Preconditioner precond) {
  switch (precond) {
  case Preconditioner::NONE:
    return "NONE";
  case Preconditioner::JACOBI:
    return "JACOBI";
  case Preconditioner::ILU0:
    return "ILU0";
  default:
    return "UNKNOWN";
  }
}

std::string precision_to_string(Precision precision) {
  switch (precision) {
  case Precision::FLOAT32:
    return "FLOAT32";
  case Precision::FLOAT64:
    return "FLOAT64";
  case Precision::MIXED:
    return "MIXED";
  default:
    return "UNKNOWN";
  }
}

void print_solver_config(const SolverConfig &config) {
  std::cout << "Method: " << solver_method_to_string(config.method) << "\n";
  std::cout << "Preconditioner: " << preconditioner_to_string(config.precond)
            << "\n";
  std::cout << "Precision: " << precision_to_string(config.precision) << "\n";
  std::cout << "Tolerance: " << config.tol << "\n";
  std::cout << "Max Iterations: " << config.max_iter << "\n";
  std::cout << "Rationale: " << config.rationale << "\n";
  std::cout << "---------------------------------------------------\n";
}

void test_dense_spd() {
  std::cout << "\n=== Test Case 1: Dense SPD Matrix ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 1000 * 1000;
  features.density = 1.0;
  features.is_symmetric = true;
  features.is_spd = true;
  features.condition_estimate = 100.0;
  features.bandwidth = 999;
  features.is_diagonally_dominant = true;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: LLT
  if (config.method == SolverMethod::LLT) {
    std::cout << "✅ Test Passed: Correctly chose LLT for dense SPD matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected LLT, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_dense_symmetric_non_spd() {
  std::cout << "\n=== Test Case 2: Dense Symmetric Non-SPD Matrix ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 1000 * 1000;
  features.density = 1.0;
  features.is_symmetric = true;
  features.is_spd = false;
  features.condition_estimate = 1000.0;
  features.bandwidth = 999;
  features.is_diagonally_dominant = false;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: LDLT
  if (config.method == SolverMethod::LDLT) {
    std::cout << "✅ Test Passed: Correctly chose LDLT for dense symmetric "
                 "non-SPD matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected LDLT, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_dense_general() {
  std::cout << "\n=== Test Case 3: Dense General Matrix ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 1000 * 1000;
  features.density = 1.0;
  features.is_symmetric = false;
  features.is_spd = false;
  features.condition_estimate = 1000.0;
  features.bandwidth = 999;
  features.is_diagonally_dominant = false;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: LU
  if (config.method == SolverMethod::LU) {
    std::cout
        << "✅ Test Passed: Correctly chose LU for dense general matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected LU, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_sparse_spd() {
  std::cout << "\n=== Test Case 4: Sparse SPD Matrix ===\n";

  MatrixFeatures features;
  features.n = 10000;
  features.m = 10000;
  features.nnz = 50000;
  features.density = 0.0005;
  features.is_symmetric = true;
  features.is_spd = true;
  features.condition_estimate = 1000.0;
  features.bandwidth = 100;
  features.is_diagonally_dominant = true;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: CG with Jacobi preconditioner
  if (config.method == SolverMethod::CG &&
      config.precond == Preconditioner::JACOBI) {
    std::cout
        << "✅ Test Passed: Correctly chose CG+Jacobi for sparse SPD matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected CG+Jacobi, got "
              << solver_method_to_string(config.method) << "+"
              << preconditioner_to_string(config.precond) << "\n";
  }

  // With severe ill-conditioning
  features.condition_estimate = 1e7;
  config = route(features);
  std::cout << "\nWith severe ill-conditioning:\n";
  print_solver_config(config);

  // Expected: CG with ILU0 preconditioner
  if (config.method == SolverMethod::CG &&
      config.precond == Preconditioner::ILU0) {
    std::cout << "✅ Test Passed: Correctly chose CG+ILU0 for ill-conditioned "
                 "sparse SPD matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected CG+ILU0, got "
              << solver_method_to_string(config.method) << "+"
              << preconditioner_to_string(config.precond) << "\n";
  }
}

void test_sparse_general() {
  std::cout << "\n=== Test Case 5: Sparse General Matrix ===\n";

  MatrixFeatures features;
  features.n = 10000;
  features.m = 10000;
  features.nnz = 50000;
  features.density = 0.0005;
  features.is_symmetric = false;
  features.is_spd = false;
  features.condition_estimate = 1000.0;
  features.bandwidth = 100;
  features.is_diagonally_dominant = false;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: GMRES with Jacobi preconditioner
  if (config.method == SolverMethod::GMRES &&
      config.precond == Preconditioner::JACOBI) {
    std::cout << "✅ Test Passed: Correctly chose GMRES+Jacobi for sparse "
                 "general matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected GMRES+Jacobi, got "
              << solver_method_to_string(config.method) << "+"
              << preconditioner_to_string(config.precond) << "\n";
  }
}

void test_banded_spd() {
  std::cout << "\n=== Test Case 6: Banded SPD Matrix ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 5000;
  features.density = 0.005;
  features.is_symmetric = true;
  features.is_spd = true;
  features.condition_estimate = 100.0;
  features.bandwidth = 5; // Small bandwidth
  features.is_diagonally_dominant = true;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: BANDED_CHOL
  if (config.method == SolverMethod::BANDED_CHOL) {
    std::cout << "✅ Test Passed: Correctly chose BANDED_CHOL for banded SPD "
                 "matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected BANDED_CHOL, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_tridiagonal() {
  std::cout << "\n=== Test Case 7: Tridiagonal Matrix ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 3000;
  features.density = 0.003;
  features.is_symmetric = true;
  features.is_spd = true;
  features.condition_estimate = 100.0;
  features.bandwidth = 1; // Tridiagonal
  features.is_diagonally_dominant = true;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: TRIDIAG
  if (config.method == SolverMethod::TRIDIAG) {
    std::cout
        << "✅ Test Passed: Correctly chose TRIDIAG for tridiagonal matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected TRIDIAG, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_rank_deficient() {
  std::cout << "\n=== Test Case 8: Rank-Deficient Matrix ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 1000 * 1000;
  features.density = 1.0;
  features.is_symmetric = true;
  features.is_spd = false;
  features.condition_estimate = 1e15;
  features.bandwidth = 999;
  features.is_diagonally_dominant = false;
  features.is_rank_deficient = true;
  features.is_least_squares = false;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: QR
  if (config.method == SolverMethod::QR) {
    std::cout
        << "✅ Test Passed: Correctly chose QR for rank-deficient matrix\n";
  } else {
    std::cout << "❌ Test Failed: Expected QR, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_least_squares() {
  std::cout << "\n=== Test Case 9: Least Squares Problem ===\n";

  MatrixFeatures features;
  features.n = 500;
  features.m = 1000;
  features.nnz = 500 * 1000;
  features.density = 1.0;
  features.is_symmetric = false;
  features.is_spd = false;
  features.condition_estimate = 1000.0;
  features.bandwidth = 499;
  features.is_diagonally_dominant = false;
  features.is_rank_deficient = false;
  features.is_least_squares = true;

  SolverConfig config = route(features);
  print_solver_config(config);

  // Expected: QR
  if (config.method == SolverMethod::QR) {
    std::cout
        << "✅ Test Passed: Correctly chose QR for least squares problem\n";
  } else {
    std::cout << "❌ Test Failed: Expected QR, got "
              << solver_method_to_string(config.method) << "\n";
  }
}

void test_mixed_precision() {
  std::cout << "\n=== Test Case 10: Mixed Precision ===\n";

  MatrixFeatures features;
  features.n = 1000;
  features.m = 1000;
  features.nnz = 1000 * 1000;
  features.density = 1.0;
  features.is_symmetric = true;
  features.is_spd = true;
  features.condition_estimate = 1000.0;
  features.bandwidth = 999;
  features.is_diagonally_dominant = true;
  features.is_rank_deficient = false;
  features.is_least_squares = false;

  // With speed goal
  SolverConfig config = route(features, SolverGoal::SPEED);
  std::cout << "With SPEED goal:\n";
  print_solver_config(config);

  // Expected: MIXED precision
  if (config.precision == Precision::MIXED) {
    std::cout
        << "✅ Test Passed: Correctly chose MIXED precision for speed goal\n";
  } else {
    std::cout << "❌ Test Failed: Expected MIXED precision, got "
              << precision_to_string(config.precision) << "\n";
  }

  // With accuracy goal
  config = route(features, SolverGoal::ACCURACY);
  std::cout << "\nWith ACCURACY goal:\n";
  print_solver_config(config);

  // Expected: FLOAT64 precision
  if (config.precision == Precision::FLOAT64) {
    std::cout << "✅ Test Passed: Correctly chose FLOAT64 precision for "
                 "accuracy goal\n";
  } else {
    std::cout << "❌ Test Failed: Expected FLOAT64 precision, got "
              << precision_to_string(config.precision) << "\n";
  }
}

int main() {
  std::cout << "===== Testing Router Heuristics =====\n";

  test_dense_spd();
  test_dense_symmetric_non_spd();
  test_dense_general();
  test_sparse_spd();
  test_sparse_general();
  test_banded_spd();
  test_tridiagonal();
  test_rank_deficient();
  test_least_squares();
  test_mixed_precision();

  return 0;
}
