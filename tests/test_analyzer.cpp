#include "../include/gms/analyzer.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

/**
 * @brief A helper function to compare two floating-point numbers within a
 * tolerance.
 * @param a The first number.
 * @param b The second number.
 * @param tol The tolerance.
 * @return True if the numbers are nearly equal, false otherwise.
 */
bool nearly_equal(double a, double b, double tol = 1e-9) {
  return std::abs(a - b) < tol;
}

/**
 * @brief Tests for the density() function.
 * Checks zero, identity, dense, and sparse matrices.
 */
void test_density() {
  std::cout << "\n--- Testing density() ---\n";
  const std::size_t n = 4;
  const double tol = 1e-9;

  // Test 1: Zero matrix (all elements are zero)
  std::vector<double> A_zero(n * n, 0.0);
  double d_zero = gms::density(A_zero.data(), n, tol);

  if (nearly_equal(d_zero, 0.0)) {
    std::cout << "✅ Test Case 1 (Zero Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 1 (Zero Matrix) Failed: Expected 0.0, got "
              << d_zero << "\n";
  }

  // Test 2: Identity matrix (n non-zeros)
  std::vector<double> A_identity(n * n, 0.0);
  for (size_t i = 0; i < n; ++i)
    A_identity[i * n + i] = 1.0;

  double d_identity = gms::density(A_identity.data(), n, tol);
  double expected_identity_density = static_cast<double>(n) / (n * n);

  if (nearly_equal(d_identity, expected_identity_density)) {
    std::cout << "✅ Test Case 2 (Identity Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 2 (Identity Matrix) Failed: Expected "
              << expected_identity_density << ", got " << d_identity << "\n";
  }

  // Test 3: Dense matrix (all elements are non-zero)
  std::vector<double> A_dense(n * n, 1.0);
  double d_dense = gms::density(A_dense.data(), n, tol);
  if (nearly_equal(d_dense, 1.0)) {
    std::cout << "✅ Test Case 3 (Dense Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 3 (Dense Matrix) Failed: Expected 1.0, got "
              << d_dense << "\n";
  }

  // Test 4: Sparse matrix with 3 non-zero elements
  std::vector<double> A_sparse(n * n, 0.0);
  A_sparse[1] = 5.0;
  A_sparse[5] = -2.0;
  A_sparse[15] = 10.0;

  double d_sparse = gms::density(A_sparse.data(), n, tol);
  double expected_sparse_density = 3.0 / (n * n);

  if (nearly_equal(d_sparse, expected_sparse_density)) {
    std::cout << "✅ Test Case 4 (Sparse Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 4 (Sparse Matrix) Failed: Expected "
              << expected_sparse_density << ", got " << d_sparse << "\n";
  }
}

/**
 * @brief Tests for the is_symmetric() function.
 * Checks symmetric, non-symmetric, and trivial cases.
 */
void test_is_symmetric() {
  std::cout << "\n--- Testing is_symmetric() ---\n";
  const std::size_t n = 3;

  // Test 1: A genuinely symmetric matrix
  std::vector<double> A_sym = {1, 2, 3, 2, 4, 5, 3, 5, 6};

  if (gms::is_symmetric(A_sym.data(), n)) {
    std::cout << "✅ Test Case 1 (Symmetric) Passed\n";
  } else {
    std::cerr << "❌ Test Case 1 (Symmetric) Failed\n";
  }

  // Test 2: A non-symmetric matrix
  std::vector<double> A_nonsym = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  if (!gms::is_symmetric(A_nonsym.data(), n)) {
    std::cout << "✅ Test Case 2 (Non-Symmetric) Passed\n";
  } else {
    std::cerr << "❌ Test Case 2 (Non-Symmetric) Failed\n";
  }

  // Test 3: A 1x1 matrix (always symmetric)
  std::vector<double> A_1x1 = {100.0};

  if (gms::is_symmetric(A_1x1.data(), 1)) {
    std::cout << "✅ Test Case 3 (1x1 Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 3 (1x1 Matrix) Failed\n";
  }
}

/**
 * @brief Tests for the is_spd() function.
 * Checks a known SPD matrix, a symmetric non-SPD matrix, and a non-symmetric
 * matrix.
 */
void test_is_spd() {
  std::cout << "\n--- Testing is_spd() ---\n";
  const std::size_t n = 3;

  // Test 1: SPD matrix (diagonally dominant with positive diagonal)
  std::vector<double> A_spd = {4, 1, 0, 1, 4, 1, 0, 1, 4};
  if (gms::is_spd(A_spd.data(), n)) {
    std::cout << "✅ Test Case 1 (SPD Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 1 (SPD Matrix) Failed\n";
  }

  // Test 2: Symmetric but not positive definite (eigenvalues are 3 and -1)
  std::vector<double> A_sym_not_pd = {1, 2, 2, 1};
  if (!gms::is_spd(A_sym_not_pd.data(), 2)) {
    std::cout << "✅ Test Case 2 (Symmetric, Not PD) Passed\n";
  } else {
    std::cerr << "❌ Test Case 2 (Symmetric, Not PD) Failed\n";
  }

  // Test 3: Non-symmetric matrix
  std::vector<double> A_nonsym = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  if (!gms::is_spd(A_nonsym.data(), n)) {
    std::cout << "✅ Test Case 3 (Non-Symmetric) Passed\n";
  } else {
    std::cerr << "❌ Test Case 3 (Non-Symmetric) Failed\n";
  }
}

/**
 * @brief Tests for the bandwidth() function.
 * Checks lower, upper, and max bandwidth for a sample matrix.
 */
void test_bandwidth() {
  std::cout << "\n--- Testing bandwidth() ---\n";
  const std::size_t n = 4;
  // Matrix A:
  // [1 2 0 0]
  // [3 4 5 0]
  // [0 6 7 8]
  // [9 0 1 2]
  std::vector<double> A = {1, 2, 0, 0, 3, 4, 5, 0, 0, 6, 7, 8, 9, 0, 1, 2};

  // Expected lower bandwidth = 3 (from A[3,0])
  // Expected upper bandwidth = 1 (tridiagonal-like upper part)
  // Expected max bandwidth = 3
  size_t lower = gms::bandwidth(A.data(), n, gms::BandType::LOWER);
  if (lower == 3) {
    std::cout << "✅ Test Case 1 (Lower Bandwidth) Passed\n";
  } else {
    std::cerr << "❌ Test Case 1 (Lower Bandwidth) Failed: Expected 3, got "
              << lower << "\n";
  }

  size_t upper = gms::bandwidth(A.data(), n, gms::BandType::UPPER);
  if (upper == 1) {
    std::cout << "✅ Test Case 2 (Upper Bandwidth) Passed\n";
  } else {
    std::cerr << "❌ Test Case 2 (Upper Bandwidth) Failed: Expected 1, got "
              << upper << "\n";
  }

  size_t max_bw = gms::bandwidth(A.data(), n, gms::BandType::MAX);
  if (max_bw == 3) {
    std::cout << "✅ Test Case 3 (Max Bandwidth) Passed\n";
  } else {
    std::cerr << "❌ Test Case 3 (Max Bandwidth) Failed: Expected 3, got "
              << max_bw << "\n";
  }

  // Test 4: Diagonal matrix (bandwidth should be 0)
  std::vector<double> A_diag(n * n, 0.0);
  for (size_t i = 0; i < n; ++i)
    A_diag[i * n + i] = 1.0;
  size_t diag_bw = gms::bandwidth(A_diag.data(), n, gms::BandType::MAX);
  if (diag_bw == 0) {
    std::cout << "✅ Test Case 4 (Diagonal Matrix) Passed\n";
  } else {
    std::cerr << "❌ Test Case 4 (Diagonal Matrix) Failed: Expected 0, got "
              << diag_bw << "\n";
  }
}

/**
 * @brief Tests for the is_diagonally_dominant() function.
 * Checks strict, weak, and non-dominant cases.
 */
void test_is_diagonally_dominant() {
  std::cout << "\n--- Testing is_diagonally_dominant() ---\n";
  const std::size_t n = 3;

  // Test 1: Strictly diagonally dominant matrix
  std::vector<double> A_sdd = {4, -1, -1, -1, 4, -1, -1, -1, 4};
  if (gms::is_diagonally_dominant(A_sdd.data(), n, true)) {
    std::cout << "✅ Test Case 1 (Strictly Dominant) Passed\n";
  } else {
    std::cerr << "❌ Test Case 1 (Strictly Dominant) Failed\n";
  }

  // Test 2: Weakly diagonally dominant matrix
  std::vector<double> A_wdd = {2, -1, -1, -1, 2, -1, -1, -1, 2};
  bool weak_pass = gms::is_diagonally_dominant(A_wdd.data(), n, false);
  bool strict_fail = !gms::is_diagonally_dominant(A_wdd.data(), n, true);
  if (weak_pass && strict_fail) {
    std::cout << "✅ Test Case 2 (Weakly Dominant) Passed\n";
  } else {
    std::cerr << "❌ Test Case 2 (Weakly Dominant) Failed\n";
  }

  // Test 3: Not diagonally dominant matrix
  std::vector<double> A_not_dd = {1, 2, 3, 4, 1, 6, 7, 8, 1};
  if (!gms::is_diagonally_dominant(A_not_dd.data(), n, false)) {
    std::cout << "✅ Test Case 3 (Not Dominant) Passed\n";
  } else {
    std::cerr << "❌ Test Case 3 (Not Dominant) Failed\n";
  }
}

/**
 * @brief Tests for the condition_estimate() function.
 * Checks the identity matrix and a well-conditioned matrix.
 */
void test_condition_estimate() {
  std::cout << "\n--- Testing condition_estimate() ---\n";

  // Test 1: Identity matrix. The condition number is exactly 1.
  const std::size_t n_id = 4;
  std::vector<double> A_identity(n_id * n_id, 0.0);
  for (size_t i = 0; i < n_id; ++i)
    A_identity[i * n_id + i] = 1.0;

  double cond_est_id = gms::condition_estimate(A_identity.data(), n_id);
  if (nearly_equal(cond_est_id, 1.0, 1e-5)) {
    std::cout << "✅ Test Case 1 (Identity Matrix) Passed: Estimated "
              << cond_est_id << "\n";
  } else {
    std::cerr << "❌ Test Case 1 (Identity Matrix) Failed: Expected ~1.0, got "
              << cond_est_id << "\n";
  }

  // Test 2: A well-conditioned symmetric matrix
  // A = [4 1 0], [1 4 1], [0 1 4]
  // cond(A) = sigma_max / sigma_min = (4+sqrt(2)) / (4-sqrt(2)) ~= 2.093
  const std::size_t n_well = 3;
  std::vector<double> A_well = {4, 1, 0, 1, 4, 1, 0, 1, 4};
  double cond_est_well = gms::condition_estimate(A_well.data(), n_well, 25);
  double expected_cond = (4.0 + std::sqrt(2.0)) / (4.0 - std::sqrt(2.0));

  if (nearly_equal(cond_est_well, expected_cond, 1e-1)) {
    std::cout << "✅ Test Case 2 (Well-Conditioned) Passed: Expected ~"
              << expected_cond << ", got " << cond_est_well << "\n";
  } else {
    std::cerr << "❌ Test Case 2 (Well-Conditioned) Failed: Expected ~"
              << expected_cond << ", got " << cond_est_well << "\n";
  }
}

/**
 * @brief Main function to run all test suites.
 */
int main() {
  test_density();
  test_is_symmetric();
  test_is_spd();
  test_bandwidth();
  test_is_diagonally_dominant();
  test_condition_estimate();

  return 0;
}
