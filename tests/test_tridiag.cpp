#include "../include/gms/tridiag.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

bool nearly_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) < tol;
}

void test_thomas_3x3_known_solution() {
  // Solve tridiagonal system:
  // [ 2 -1  0 ] [x0]   [1]
  // [-1  2 -1 ] [x1] = [0]
  // [ 0 -1  2 ] [x2]   [1]
  std::vector<double> a = {0.0, -1.0, -1.0}; // sub-diagonal (a[0] unused)
  std::vector<double> b = {2.0, 2.0, 2.0};   // main diagonal
  std::vector<double> c = {-1.0, -1.0, 0.0}; // super-diagonal (c[2] unused)
  std::vector<double> d = {1.0, 0.0, 1.0};   // right-hand side
  std::vector<double> x(3);

  gms::thomas_solve(a.data(), b.data(), c.data(), d.data(), x.data(), 3);

  bool pass = nearly_equal(x[0], 1.0) && nearly_equal(x[1], 1.0) &&
              nearly_equal(x[2], 1.0);

  if (pass)
    std::cout << "✅ Thomas Solver 3x3 passed\n";
  else
    std::cerr << "❌ Thomas Solver 3x3 failed: "
              << "x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
}

void test_thomas_random_100x100() {
  const std::size_t n = 100;
  std::vector<double> a(n, -1.0), b(n, 2.0), c(n, -1.0), d(n), x(n),
      x_expected(n);

  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  for (std::size_t i = 0; i < n; ++i) {
    x_expected[i] = dist(rng); // expected x
  }

  // Build d = A * x_expected where A is tridiagonal with a, b, c
  d[0] = b[0] * x_expected[0] + c[0] * x_expected[1];
  for (std::size_t i = 1; i < n - 1; ++i)
    d[i] = a[i] * x_expected[i - 1] + b[i] * x_expected[i] +
           c[i] * x_expected[i + 1];
  d[n - 1] = a[n - 1] * x_expected[n - 2] + b[n - 1] * x_expected[n - 1];

  gms::thomas_solve(a.data(), b.data(), c.data(), d.data(), x.data(), n);

  bool pass = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (!nearly_equal(x[i], x_expected[i])) {
      pass = false;
      break;
    }
  }

  if (pass)
    std::cout << "✅ Thomas Solver 100x100 passed\n";
  else
    std::cerr << "❌ Thomas Solver 100x100 failed\n";
}

void test_thomas_zero_pivot() {
  // Diagonal element becomes zero due to cancellation
  std::vector<double> a = {0.0, 1.0};
  std::vector<double> b = {1e-12, 1.0};
  std::vector<double> c = {1.0, 0.0};
  std::vector<double> d = {1.0, 2.0};
  std::vector<double> x(2);

  try {
    gms::thomas_solve(a.data(), b.data(), c.data(), d.data(), x.data(), 2);
    std::cerr
        << "❌ Thomas Solver zero pivot test failed: no exception thrown\n";
  } catch (const std::runtime_error &) {
    std::cout << "✅ Thomas Solver zero pivot test passed\n";
  }
}

int main() {
  std::cout << "\n--- Tridiagonal Solver Tests ---\n";
  test_thomas_3x3_known_solution();
  test_thomas_random_100x100();
  test_thomas_zero_pivot();
  return 0;
}
