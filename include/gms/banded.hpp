#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace gms {

//==============================================================================
// Banded Cholesky (LL^T) Decomposition & Solver
//==============================================================================

/**
 * @brief Performs an in-place Cholesky decomposition of a banded matrix.
 * @details The matrix A is symmetric positive definite and banded. Only the
 * lower band is stored. The Cholesky factor L, which has the same lower
 * bandwidth, overwrites B.
 *
 * Storage format: A symmetric banded matrix A of size n x n with lower
 * semi-bandwidth `m` (i.e., A_ij = 0 for i - j > m) is stored compactly in a
 * (m + 1) x n row-major array B. The element A_ij (with i >= j and i - j <= m)
 * is mapped to B(i - j, j). In the 1D data pointer `B_data`, this corresponds
 * to index `(i - j) * n + j`. The main diagonal A_ii is at B(0, i) ->
 * `B_data[i]`. The first subdiagonal A_{i + 1,i} is at B(1, i) -> `B_data[n +
 * i]`.
 *
 * @param B Pointer to the banded matrix data.
 * @param n Dimension of the matrix.
 * @param m Lower semi-bandwidth (m=0 for diagonal, m=1 for tridiagonal).
 * @return true on success, false if the matrix is not positive definite.
 */
template <class T>
bool banded_cholesky_inplace_lower(T *B, std::size_t n, std::size_t m,
                                   T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point<T>::value,
                "Banded Cholesky requires floating point type");
  if (!B || n == 0) {
    throw std::invalid_argument(
        "banded_cholesky_inplace_lower: invalid B or n");
  }

  const T eps = std::numeric_limits<T>::epsilon();

  // Find the maximum abs val on the diagonal
  T max_diag_abs = *std::max_element(
      B, B + n * ldB, [&](T a, T b) { return std::abs(a) < std::abs(b); });
  const T tol = std::max(eps_rel * max_diag_abs, T(10) * eps * max_diag_abs);

  for (std::size_t k = 0; k < n; ++k) {
    // Update column k using outer products from previous columns j.
    // A_{ik} = A_{ik} - sum_{j} L_{ij} * L_{kj}.
    std::size_t j_start = (k > m) ? k - m : 0;

    for (std::size_t j = j_start; j < k; ++j) {
      // L(k, j)
      T l_kj = B[(k - j) * n + j];
      // Affects rows i >= k.
      // L(i, j) is non-zero for i <= j + m.
      std::size_t i_end = std::min(n, j + m + 1);

      for (std::size_t i = k; i < i_end; ++i) {
        // A(i, k) -= L(i, j) * L(k, j)
        // B(i - k, k) -= B(i - j, j) * B(k - j, j)
        B[(i - k) * n + k] -= B[(i - j) * n + j] * l_kj;
      }
    }

    // Finalize column k: compute L_kk and scale the rest of the column.
    T d = B[k]; // A_kk = L_kk^2
    if (d <= tol) {
      return false; // Matrix not positive definite.
    }

    T l_kk = std::sqrt(d);
    B[k] = l_kk; // Store L_kk

    std::size_t i_end_scale = std::min(n, k + m + 1);
    for (std::size_t i = k + 1; i < i_end_scale; ++i) {
      B[(i - k) * n + k] /= l_kk;
    }
  }
  return true;
}

/**
 * @brief Solves Ly = b for y, where L is a lower-banded matrix.
 */
template <class T>
void banded_forward_subst_lower(const T *B, std::size_t n, std::size_t m,
                                const T *b, T *y) {
  for (std::size_t i = 0; i < n; ++i) {
    T partial_sum = static_cast<T>(0);
    // Sum L_ij * y_j for j from max(0, i - m) to i - 1
    std::size_t j_start = (i > m) ? i - m : 0;

    for (std::size_t j = j_start; j < i; ++j) {
      partial_sum += B[(i - j) * n + j] * y[j];
    }

    y[i] = (b[i] - partial_sum) / B[i];
  }
}

/**
 * @brief Solves L^T x = y for x, where L is a lower-banded matrix.
 */
template <class T>
void banded_backward_subst_upper_from_lower_transpose(const T *B, std::size_t n,
                                                      std::size_t m, const T *y,
                                                      T *x) {
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    T partial_sum = static_cast<T>(0);

    // Sum L_ji * x_j for j from i + 1 to min(n - 1, i + m)
    std::size_t j_end = std::min(n, ui + m + 1);

    for (std::size_t j = ui + 1; j < j_end; ++j) {
      // L_ji at B(j-i, i) -> B[(j-ui)*n + ui]
      partial_sum += B[(j - ui) * n + ui] * x[j];
    }

    // L_ii is at B(0, i) -> B[ui]
    x[ui] = (y[ui] - partial_sum) / B[ui];
  }
}

/**
 * @brief Solves Ax = b for a banded Hermitian matrix A.
 * @param B Pointer to the banded matrix data, overwritten with its Cholesky
 * factor L.
 * @param n Dimension of the matrix.
 * @param m Lower semi-bandwidth.
 * @param b Right-hand side vector.
 * @param x Output solution vector.
 * @return true on success, false if matrix is not positive definite.
 */
template <class T>
bool banded_cholesky_solve_inplace(T *B, std::size_t n, std::size_t m,
                                   const T *b, T *x,
                                   T eps_rel = static_cast<T>(1e-14)) {
  // Factorize A to LL^T
  if (!banded_cholesky_inplace_lower(B, n, m, eps_rel)) {
    return false;
  }

  // Solve Ly = b
  banded_forward_subst_lower(B, n, m, b, x);

  // Solve L^T x = y
  banded_backward_subst_upper_from_lower_transpose(B, n, m, x, x);

  return true;
}

} // namespace gms
