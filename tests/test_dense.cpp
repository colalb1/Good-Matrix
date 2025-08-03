#include "../include/gms/dense.hpp"
#include <cmath>
#include <iostream>
#include <vector>

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) < tol;
}

void test_cholesky_2x2() {
  std::vector<double> A = {4.0, 12.0, 12.0, 37.0};
  const std::size_t n = 2;
  const std::size_t stride = 2;

  bool ok = gms::cholesky_inplace_lower(A.data(), n, stride);
  if (!ok) {
    std::cerr << "❌ Cholesky decomposition failed\n";
    return;
  }

  // Expect L = [2 0; 6 1]
  bool pass = nearly_equal(A[0], 2.0) && // L(0,0)
              nearly_equal(A[2], 6.0) && // L(1,0)
              nearly_equal(A[3], 1.0);   // L(1,1)

  if (pass)
    std::cout << "✅ Cholesky 2x2 passed\n";
  else
    std::cerr << "❌ Cholesky 2x2 failed\n";
}

void test_solver_3x3(bool use_ldlt) {
  std::vector<double> A = {25, 15, -5, 15, 18, 0, -5, 0, 11};

  std::vector<double> b = {35, 33, 6};
  std::vector<double> x(3);
  std::vector<double> d(3); // Used only for LDL^T

  bool ok = gms::solve_inplace(A.data(), b.data(), x.data(), 3, 3, use_ldlt,
                               d.data());

  if (!ok) {
    std::cerr << (use_ldlt ? "❌ LDLᵀ" : "❌ Cholesky") << " solve failed\n";
    return;
  }

  bool pass = true;
  for (auto xi : x)
    pass = pass && nearly_equal(xi, 1.0);

  if (pass)
    std::cout << (use_ldlt ? "✅ LDLᵀ" : "✅ Cholesky")
              << " solver 3x3 passed\n";
  else
    std::cerr << (use_ldlt ? "❌ LDLᵀ" : "❌ Cholesky")
              << " solver 3x3 failed\n";
}

void test_lu_solve_2x2() {
  std::vector<double> A = {2.0, 3.0, 1.0, 4.0};
  std::vector<double> b = {8.0, 9.0};
  std::vector<double> x(2);
  std::vector<size_t> pivots(2);
  const std::size_t n = 2;
  const std::size_t stride = 2;

  bool ok = gms::lu_solve_inplace(A.data(), b.data(), x.data(), pivots.data(),
                                  n, stride);
  if (!ok) {
    std::cerr << "❌ LU Solver 2x2 failed to solve\n";
    return;
  }

  // Expected solution: x = [1.0, 2.0]
  bool pass = nearly_equal(x[0], 1.0) && nearly_equal(x[1], 2.0);

  if (pass)
    std::cout << "✅ LU Solver 2x2 passed\n";
  else
    std::cerr << "❌ LU Solver 2x2 failed: incorrect solution\n";
}

void test_lu_solve_3x3_pivot() {
  // A[0][0] is 0, forcing a pivot
  std::vector<double> A = {0.0, 1.0, 1.0, 2.0, 1.0, -1.0, 3.0, 2.0, 1.0};
  std::vector<double> b = {2.0, 2.0, 6.0};
  std::vector<double> x(3);
  std::vector<size_t> pivots(3);
  const std::size_t n = 3;
  const std::size_t stride = 3;

  bool ok = gms::lu_solve_inplace(A.data(), b.data(), x.data(), pivots.data(),
                                  n, stride);
  if (!ok) {
    std::cerr << "❌ LU Solver 3x3 (Pivot) failed to solve\n";
    return;
  }

  // Expected solution: x = [1.0, 1.0, 1.0]
  bool pass = nearly_equal(x[0], 1.0) && nearly_equal(x[1], 1.0) &&
              nearly_equal(x[2], 1.0);

  if (pass)
    std::cout << "✅ LU Solver 3x3 (Pivot) passed\n";
  else
    std::cerr << "❌ LU Solver 3x3 (Pivot) failed: incorrect solution\n";
}

void test_lu_solve_singular() {
  // Row 2 is 2 * Row 0, making the matrix singular
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0};
  std::vector<double> b = {1.0, 1.0, 1.0};
  std::vector<double> x(3);
  std::vector<size_t> pivots(3);
  const std::size_t n = 3;
  const std::size_t stride = 3;

  bool ok = gms::lu_solve_inplace(A.data(), b.data(), x.data(), pivots.data(),
                                  n, stride);

  // We expect the function to return false for a singular matrix
  if (!ok)
    std::cout << "✅ LU Solver (Singular) passed: correctly identified "
                 "singular matrix\n";
  else
    std::cerr
        << "❌ LU Solver (Singular) failed: did not identify singular matrix\n";
}

// Test helpers and QR test
inline bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) < tol;
}

inline void test_qr_solve_3x3() {
  std::vector<double> A = {
      2.0,  -1.0, 0.0,  -1.0, 2.0,
      -1.0, 0.0,  -1.0, 2.0}; // Symmetric tridiagonal matrix (SPD)

  std::vector<double> b = {1.0, 0.0, 1.0};
  std::vector<double> x(3);
  std::vector<double> tau(3);
  const std::size_t m = 3, n = 3, stride = 3;

  bool ok = gms::qr_solve_inplace(A.data(), b.data(), x.data(), tau.data(), m,
                                  n, stride);
  if (!ok) {
    std::cerr << "❌ QR Solver 3x3 failed to solve\n";
    return;
  }

  // Expected solution is x = [1.0, 1.0, 1.0]
  bool pass = nearly_equal(x[0], 1.0) && nearly_equal(x[1], 1.0) &&
              nearly_equal(x[2], 1.0);

  if (pass)
    std::cout << "✅ QR Solver 3x3 passed\n";
  else
    std::cerr << "❌ QR Solver 3x3 failed: incorrect solution\n";
}

inline void test_qr_solve_rank_deficient() {
  std::vector<double> A = {
      1.0, 2.0, 3.0, 2.0, 4.0,
      6.0, 1.0, 2.0, 3.0}; // Rank-deficient (row 3 = row 1)

  std::vector<double> b = {6.0, 12.0, 6.0};
  std::vector<double> x(3);
  std::vector<double> tau(3);
  const std::size_t m = 3, n = 3, stride = 3;

  bool ok = gms::qr_solve_inplace(A.data(), b.data(), x.data(), tau.data(), m,
                                  n, stride);

  if (!ok)
    std::cout << "✅ QR Solver (Singular) passed: correctly identified "
                 "rank-deficient matrix\n";
  else
    std::cerr << "❌ QR Solver (Singular) failed: did not identify "
                 "rank-deficient matrix\n";
}

int main() {
  // Test Cholesky
  test_cholesky_2x2();
  test_solver_3x3(false); // Cholesky
  test_solver_3x3(true);  // LDL^T

  std::cout << "\n--- LU Solver Tests ---\n";

  // Test LU Solver
  test_lu_solve_2x2();
  test_lu_solve_3x3_pivot();
  test_lu_solve_singular();

  // Test QR Solver
  test_qr_solve_3x3();
  test_qr_solve_rank_deficient();

  return 0;
}
