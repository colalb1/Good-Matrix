#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace gms {

//==============================================================================
// Cholesky (LLᵀ) Decomposition & Solver
//==============================================================================

/**
 * @brief Performs an in-place Cholesky (LLᵀ) decomposition.
 * @details The lower triangular part of A is replaced by the Cholesky factor L.
 * @param A Pointer to the matrix data (row-major).
 * @param n Dimension of the matrix.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if the matrix is not positive definite.
 */
template <class T>
bool cholesky_inplace_lower(T *A, std::size_t n, std::size_t row_stride,
                            T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point_v<T>,
                "Cholesky requires floating point type");
  if (!A || row_stride < n) {
    throw std::invalid_argument("cholesky_inplace_lower: invalid A/row_stride");
  }

  const T eps = std::numeric_limits<T>::epsilon();
  T max_diag = static_cast<T>(0);

  for (std::size_t k = 0; k < n; ++k) {
    // Track the largest diagonal value seen so far to define relative tolerance
    max_diag = std::max(max_diag, std::abs(A[k * row_stride + k]));

    T* row_start = A + k * row_stride;

    // Compute $d_k = A_{kk} - \sum_{j=0}^{k-1} L_{kj}^2$
    T residual_diag_correction =
        std::accumulate(row_start, row_start + k, T{0},
                        [](T acc, T val) { return acc + val * val; });
    T d = A[k * row_stride + k] - residual_diag_correction;

    // Check if the matrix is positive definite
    const T tol = std::max(eps_rel * max_diag, T(10) * eps * max_diag);
    if (d <= tol) {
      return false;
    }

    // Compute $L_kk = \sqrt{d}$ and store it in A
    const T L_kk = std::sqrt(d);
    A[k * row_stride + k] = L_kk;

    // Compute $L_ik = \frac{1}{L_kk}(A_ik - \sum_{j=0)^{k - 1} L_ij * L_kj$
    for (std::size_t i = k + 1; i < n; ++i) {
      T inner_product = std::inner_product(
          A + i * row_stride, A + i * row_stride + k, A + k * row_stride, T{0});

      A[i * row_stride + k] = (A[i * row_stride + k] - inner_product) / L_kk;
    }

    // Zero out the upper triangle
    // This is unnecessary, but more of a sanity check
    for (std::size_t j = k + 1; j < n; ++j) {
      A[k * row_stride + j] = 0;
    }
  }

  return true;
}

// Solves Ly = b for y (forward substitution, unit diagonal).
template <class T>
void forward_subst_lower_unitdiag_false(const T *L, std::size_t n,
                                        std::size_t row_stride, const T *b,
                                        T *y) {
  for (std::size_t i = 0; i < n; ++i) {
    T partial_sum =
        std::inner_product(L + i * row_stride, L + i * row_stride + i, y, T{0});

    // Computes $y_i = (b_i - (\sum_{j=0}^{i - 1} L_ij * y_j)) / L_ii$
    y[i] = (b[i] - partial_sum) / L[i * row_stride + i];
  }
}

// Solves Lᵀx = y for x (backward substitution, unit diagonal).
template <class T>
void backward_subst_upper_from_lower_transpose(const T *L, std::size_t n,
                                               std::size_t row_stride,
                                               const T *y, T *x) {
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
    T partial_sum = std::inner_product(
        L + (i + 1) * row_stride + i, L + n * row_stride + i, x + i + 1, T{0},
        std::plus<>{}, [](T l_ji, T x_j) { return l_ji * x_j; });

    // Computes $x_i = (y_i - (\sum_{j=i + 1}^{n - 1} L_ji * x_j)) / L_ii$
    x[static_cast<std::size_t>(i)] =
        (y[static_cast<std::size_t>(i)] - partial_sum) /
        L[static_cast<std::size_t>(i) * row_stride +
          static_cast<std::size_t>(i)];
  }
}

template <class T>
bool cholesky_solve_inplace(T *A, std::size_t n, std::size_t row_stride,
                            const T *b, T *x,
                            T eps_rel = static_cast<T>(1e-14)) {
  // Factorize A to LL^T
  if (!cholesky_inplace_lower(A, n, row_stride, eps_rel))
    return false;

  // Solve Ly = b (result y is stored in x)
  forward_subst_lower_unitdiag_false(A, n, row_stride, b, x);

  // Solve L^Tx = y (result x is stored in x)
  backward_subst_upper_from_lower_transpose(A, n, row_stride, x, x);

  return true;
}

//==============================================================================
// LDLᵀ Decomposition & Solver
//==============================================================================

/**
 * @brief Performs an in-place LDLᵀ decomposition.
 * @details The strictly lower part of A is replaced by the strictly lower
 * part of L. The diagonal of D is stored in the output array `d`.
 * @param A Pointer to the matrix data (row-major). Will be modified.
 * @param d Pointer to an array of size n to store the diagonal of D.
 * @param n Dimension of the matrix.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if the matrix is singular.
 */
template <class T>
bool ldlt_inplace_lower(T *A, T *d, std::size_t n, std::size_t row_stride,
                        T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point_v<T>,
                "LDL^T requires floating point type");
  if (!A || !d || row_stride < n)
    throw std::invalid_argument("ldlt_inplace_lower: invalid A/d/row_stride");

  const T eps = std::numeric_limits<T>::epsilon();
  T max_diag_abs = static_cast<T>(0);

  for (std::size_t j = 0; j < n; ++j) {
    // Numerical tolerance is defined relative to the diagonal element
    max_diag_abs = std::max(max_diag_abs, std::abs(A[j * row_stride + j]));

    // Compute the diagonal element $D_j = A_jj - \sum_{k=0}^{j-1} L_jk ^ 2 *
    // d_k$
    T sum_lower_diag = std::transform_reduce(
        A + j * row_stride, A + j * row_stride + j, // L_jk
        d,                                          // d_k
        T{0}, std::plus<>{},
        [](const T &val, const T &diag) { return val * val * diag; });

    const T D_j = A[j * row_stride + j] - sum_lower_diag;

    const T tol = std::max(eps_rel * max_diag_abs, T(10) * eps * max_diag_abs);

    // If the diagonal element is too small, the matrix is singular
    if (std::abs(D_j) <= tol)
      return false;

    d[j] = D_j;

    // Compute the lower triangular elements $L_ij = (A_ij - \sum_{k=0}^{j - 1}
    // L_ik * L_jk * d_k) / D_j$
    for (std::size_t i = j + 1; i < n; ++i) {
      T sum_l_ld = std::accumulate(size_t{0}, size_t{j}, T{0},
                                   [&](T acc, std::size_t k) {
                                     return acc + row_i[k] * row_j[k] * d[k];
                                   });

      A[i * row_stride + j] = (A[i * row_stride + j] - sum_l_ld) / D_j;
    }
  }
  return true;
}

/**
 * @brief Solves Lz = b for z (forward substitution, unit diagonal).
 */
template <class T>
void forward_subst_lower_unitdiag_true(const T *L, std::size_t n,
                                       std::size_t row_stride, const T *b,
                                       T *z) {
  for (std::size_t i = 0; i < n; ++i) {
    T inner_product =
        std::inner_product(L + i * row_stride, L + i * row_stride + i, z, T{0});

    z[i] = b[i] - inner_product;
  }
}

/**
 * @brief Solves Dy = z for y by scaling.
 */
template <class T> void scale_by_diag(const T *d, std::size_t n, T *y) {
  for (std::size_t i = 0; i < n; ++i) {
    y[i] /= d[i];
  }
}

/**
 * @brief Solves L^Tx = y for x (backward substitution, unit diagonal).
 */
template <class T>
void backward_subst_upper_from_lower_transpose_unitdiag_true(
    const T *L, std::size_t n, std::size_t row_stride, const T *y, T *x) {
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
    const std::size_t ui = static_cast<std::size_t>(i);

    T inner_product = std::inner_product(
        L + (ui + 1) * row_stride + ui, // start of L[j][i] for j = ui + 1
        L + n * row_stride + ui,        // one past last element L[n - 1][i]
        x + ui + 1,                     // x[j] starting from j = ui + 1
        T{0});

    x[ui] = y[ui] - inner_product;
  }
}

/**
 * @brief Solves Ax = b using an in-place LDLᵀ decomposition.
 * @param d Pointer to a temporary array of size n for storing the diagonal of
 * D.
 */
template <class T>
bool ldlt_solve_inplace(T *A, std::size_t n, std::size_t row_stride, const T *b,
                        T *x, T *d, T eps_rel = static_cast<T>(1e-14)) {
  if (!ldlt_inplace_lower(A, d, n, row_stride, eps_rel))
    return false;

  // Solve Lz = b (result z is stored in x)
  forward_subst_lower_unitdiag_true(A, n, row_stride, b, x);

  // Solve Dy = z (result y is stored in x)
  scale_by_diag(d, n, x);

  // Solve L^T x = y (final result x is stored in x)
  backward_subst_upper_from_lower_transpose_unitdiag_true(A, n, row_stride, x,
                                                          x);

  return true;
}

template <class T>
bool solve_inplace(T *A, const T *b, T *x, std::size_t n,
                   std::size_t row_stride, bool use_ldlt = false,
                   T *d = nullptr, T eps_rel = static_cast<T>(1e-14)) {
  if (use_ldlt) {
    if (!d) {
      throw std::invalid_argument(
          "solve_inplace: LDLᵀ requires temporary buffer d.");
    }
    return ldlt_solve_inplace(A, n, row_stride, b, x, d, eps_rel);
  } else {
    return cholesky_solve_inplace(A, n, row_stride, b, x, eps_rel);
  }
}

} // namespace gms
