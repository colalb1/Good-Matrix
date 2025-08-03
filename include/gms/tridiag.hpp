#pragma once
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace gms {

/**
 * @brief Solves Ax = d where A is tridiagonal using the Thomas algorithm.
 * @details A is represented by three arrays: lower (a), diagonal (b), upper
 * (c). The input arrays are not modified. Time complexity: O(n).
 * @param a Sub-diagonal (a[0] unused), size n.
 * @param b Main diagonal, size n.
 * @param c Super-diagonal (c[n-1] unused), size n.
 * @param d Right-hand side vector, size n.
 * @param x Output vector to store solution, size n.
 */
template <class T>
void thomas_solve(const T *a, const T *b, const T *c, const T *d, T *x,
                  std::size_t n) {
  static_assert(std::is_floating_point<T>::value,
                "Thomas algorithm requires floating point type");

  if (!a || !b || !c || !d || !x || n < 2) {
    throw std::invalid_argument("thomas_solve: invalid input");
  }

  std::vector<T> c_prime(n);
  std::vector<T> d_prime(n);

  // Forward sweep
  c_prime[0] = c[0] / b[0];
  d_prime[0] = d[0] / b[0];

  for (std::size_t i = 1; i < n; ++i) {
    const T denom = b[i] - a[i] * c_prime[i - 1];

    if (std::abs(denom) < std::numeric_limits<T>::epsilon() * std::abs(b[i])) {
      throw std::runtime_error("thomas_solve: near-zero pivot encountered");
    }

    c_prime[i] = (i < n - 1) ? c[i] / denom : T{0};
    d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
  }

  // Back substitution
  x[n - 1] = d_prime[n - 1];

  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 2; i >= 0; --i) {
    x[i] = d_prime[i] - c_prime[i] * x[i + 1];
  }
}

} // namespace gms
