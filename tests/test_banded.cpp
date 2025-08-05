#include "../include/gms/banded.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) < tol;
}

void test_banded_cholesky_small() {
  // 3x3 SPD banded matrix with bandwidth m = 1 (tridiagonal)
  // Matrix:
  // [4 1 0]
  // [1 4 1]
  // [0 1 4]
  const std::size_t n = 3, m = 1;
  // Packed band storage: (m + 1) rows by n cols, row-major
  // Rows: diagonal (0), first subdiagonal (1)
  // B layout:
  // B(0, i) = diag A_ii
  // B(1, i) = subdiagonal A_{i + 1, i}
  std::vector<double> B = {
      4.0, 4.0, 4.0, // diagonal
      1.0, 1.0, 0.0  // subdiagonal (last column padded)
  };

  std::vector<double> b = {6.0, 10.0, 6.0};
  std::vector<double> x(n);

  bool ok =
      gms::banded_cholesky_solve_inplace(B.data(), n, m, b.data(), x.data());
  if (!ok) {
    std::cerr << "❌ Banded Cholesky solve failed\n";
    return;
  }

  // Expected solution is x = [1, 2, 1]
  bool pass = nearly_equal(x[0], 1.0) && nearly_equal(x[1], 2.0) &&
              nearly_equal(x[2], 1.0);

  if (pass)
    std::cout << "✅ Banded Cholesky small system passed\n";
  else
    std::cerr << "❌ Banded Cholesky small system failed\n";
}

void test_banded_cholesky_tridiag_larger() {
  // 5x5 tridiagonal SPD matrix with diagonal=4 and off-diagonals=-1
  const std::size_t n = 5, m = 1;
  std::vector<double> B((m + 1) * n, 0.0);

  // Fill diagonal and subdiagonal
  for (std::size_t i = 0; i < n; ++i)
    B[i] = 4.0; // diagonal row is B[0 * n + i]
  for (std::size_t i = 0; i < n - 1; ++i)
    B[n + i] = -1.0; // subdiagonal row is B[1 * n + i]

  std::vector<double> b = {3.0, 2.0, 1.0, 2.0, 3.0};
  std::vector<double> x(n);

  bool ok =
      gms::banded_cholesky_solve_inplace(B.data(), n, m, b.data(), x.data());
  if (!ok) {
    std::cerr << "❌ Banded Cholesky solve failed on larger tridiagonal\n";
    return;
  }

  // Just verify that Ax ~= b by reconstructing Ax with banded multiply
  std::vector<double> Ax(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    Ax[i] += 4.0 * x[i];
    if (i > 0)
      Ax[i] += -1.0 * x[i - 1];
    if (i + 1 < n)
      Ax[i] += -1.0 * x[i + 1];
  }

  bool pass = true;
  for (std::size_t i = 0; i < n; ++i)
    pass &= nearly_equal(Ax[i], b[i]);

  if (pass)
    std::cout << "✅ Banded Cholesky larger tridiagonal system passed\n";
  else
    std::cerr << "❌ Banded Cholesky larger tridiagonal system failed\n";
}

void test_banded_cholesky_singular() {
  // Matrix with zero diagonal to force failure
  const std::size_t n = 2, m = 1;
  std::vector<double> B = {0.0, 1.0, 0.0, 0.0};
  std::vector<double> b = {1.0, 2.0};
  std::vector<double> x(n);

  bool ok =
      gms::banded_cholesky_solve_inplace(B.data(), n, m, b.data(), x.data());
  if (!ok)
    std::cout << "✅ Banded Cholesky correctly failed on singular matrix\n";
  else
    std::cerr << "❌ Banded Cholesky failed to detect singular matrix\n";
}

void test_banded_cholesky_large_random() {
  const std::size_t n = 50;
  const std::size_t m = 3;
  std::vector<double> B((m + 1) * n, 0.0);

  // Generate diagonally dominant banded SPD matrix in packed storage
  for (std::size_t i = 0; i < n; ++i) {
    B[i] = 10.0; // diagonal
    for (std::size_t j = 1; j <= m; ++j) {
      if (i + j < n)
        B[j * n + i] = 1.0 / (j + 1);
    }
  }

  std::vector<double> b(n);
  std::vector<double> x(n);

  std::mt19937 rng(321);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  for (auto &val : b)
    val = dist(rng);

  bool ok =
      gms::banded_cholesky_solve_inplace(B.data(), n, m, b.data(), x.data());
  if (!ok) {
    std::cerr << "❌ Banded Cholesky failed on large random system\n";
    return;
  }
  std::cout << "✅ Banded Cholesky large random system passed\n";
}

int main() {
  std::cout << "\n--- Banded Cholesky Tests ---\n";

  test_banded_cholesky_small();
  test_banded_cholesky_tridiag_larger();
  test_banded_cholesky_singular();
  test_banded_cholesky_large_random();

  return 0;
}
