#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
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
 * @param lda Leading dimension of A.
 * @return true on success, false if the matrix is not positive definite.
 */
template <class T>
bool cholesky_inplace_lower(T *A, std::size_t n, std::size_t lda,
                            T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point_v<T>,
                "Cholesky requires floating point type");
  if (!A || lda < n) {
    throw std::invalid_argument("cholesky_inplace_lower: invalid A/lda");
  }

  const T eps = std::numeric_limits<T>::epsilon();
  T max_diag = static_cast<T>(0);

  for (std::size_t k = 0; k < n; ++k) {
    max_diag = std::max(max_diag, std::abs(A[k * lda + k]));

    T sum = 0;
    for (std::size_t j = 0; j < k; ++j) {
      const T L_kj = A[k * lda + j];
      sum += L_kj * L_kj;
    }
    T d = A[k * lda + k] - sum;

    const T tol = std::max(eps_rel * max_diag, T(10) * eps * max_diag);
    if (d <= tol) {
      return false;
    }

    const T L_kk = std::sqrt(d);
    A[k * lda + k] = L_kk;

    for (std::size_t i = k + 1; i < n; ++i) {
      T s = 0;

      for (std::size_t j = 0; j < k; ++j) {
        s += A[i * lda + j] * A[k * lda + j];
      }

      A[i * lda + k] = (A[i * lda + k] - s) / L_kk;
    }

    for (std::size_t j = k + 1; j < n; ++j) {
      A[k * lda + j] = 0;
    }
  }
  return true;
}

template <class T>
void forward_subst_lower_unitdiag_false(const T *L, std::size_t n,
                                        std::size_t lda, const T *b, T *y) {
  for (std::size_t i = 0; i < n; ++i) {
    T s = 0;

    for (std::size_t j = 0; j < i; ++j) {
      s += L[i * lda + j] * y[j];
    }

    y[i] = (b[i] - s) / L[i * lda + i];
  }
}

template <class T>
void backward_subst_upper_from_lower_transpose(const T *L, std::size_t n,
                                               std::size_t lda, const T *y,
                                               T *x) {
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
    T s = 0;

    for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
      s += L[j * lda + static_cast<std::size_t>(i)] * x[j];
    }
    x[static_cast<std::size_t>(i)] =
        (y[static_cast<std::size_t>(i)] - s) /
        L[static_cast<std::size_t>(i) * lda + static_cast<std::size_t>(i)];
  }
}

template <class T>
bool cholesky_solve_inplace(T *A, std::size_t n, std::size_t lda, const T *b,
                            T *x, T eps_rel = static_cast<T>(1e-14)) {
  if (!cholesky_inplace_lower(A, n, lda, eps_rel))
    return false;
  forward_subst_lower_unitdiag_false(A, n, lda, b, x);
  backward_subst_upper_from_lower_transpose(A, n, lda, x, x);
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
 * @param lda Leading dimension of A.
 * @return true on success, false if the matrix is singular.
 */
template <class T>
bool ldlt_inplace_lower(T *A, T *d, std::size_t n, std::size_t lda,
                        T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point_v<T>,
                "LDL^T requires floating point type");
  if (!A || !d || lda < n)
    throw std::invalid_argument("ldlt_inplace_lower: invalid A/d/lda");

  const T eps = std::numeric_limits<T>::epsilon();
  T max_diag_abs = static_cast<T>(0);

  for (std::size_t j = 0; j < n; ++j) {
    max_diag_abs = std::max(max_diag_abs, std::abs(A[j * lda + j]));

    T sum_ld = 0;

    for (std::size_t k = 0; k < j; ++k) {
      const T L_jk = A[j * lda + k];
      sum_ld += L_jk * L_jk * d[k];
    }

    const T D_j = A[j * lda + j] - sum_ld;

    const T tol = std::max(eps_rel * max_diag_abs, T(10) * eps * max_diag_abs);

    if (std::abs(D_j) <= tol)
      return false;

    d[j] = D_j;

    for (std::size_t i = j + 1; i < n; ++i) {
      T sum_l_ld = 0;

      for (std::size_t k = 0; k < j; ++k) {
        sum_l_ld += A[i * lda + k] * A[j * lda + k] * d[k];
      }

      A[i * lda + j] = (A[i * lda + j] - sum_l_ld) / D_j;
    }
  }
  return true;
}

/**
 * @brief Solves Lz = b for z (forward substitution, unit diagonal).
 */
template <class T>
void forward_subst_lower_unitdiag_true(const T *L, std::size_t n,
                                       std::size_t lda, const T *b, T *z) {
  for (std::size_t i = 0; i < n; ++i) {
    T s = 0;

    for (std::size_t j = 0; j < i; ++j) {
      s += L[i * lda + j] * z[j];
    }

    z[i] = b[i] - s;
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
 * @brief Solves Lᵀx = y for x (backward substitution, unit diagonal).
 */
template <class T>
void backward_subst_upper_from_lower_transpose_unitdiag_true(const T *L,
                                                             std::size_t n,
                                                             std::size_t lda,
                                                             const T *y, T *x) {
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
    T s = 0;
    const auto ui = static_cast<std::size_t>(i);

    for (std::size_t j = ui + 1; j < n; ++j) {
      s += L[j * lda + ui] * x[j];
    }
    
    x[ui] = y[ui] - s;
  }
}

/**
 * @brief Solves Ax = b using an in-place LDLᵀ decomposition.
 * @param d Pointer to a temporary array of size n for storing the diagonal of
 * D.
 */
template <class T>
bool ldlt_solve_inplace(T *A, std::size_t n, std::size_t lda, const T *b, T *x,
                        T *d, T eps_rel = static_cast<T>(1e-14)) {
  if (!ldlt_inplace_lower(A, d, n, lda, eps_rel))
    return false;

  // Solve Lz = b (result z is stored in x)
  forward_subst_lower_unitdiag_true(A, n, lda, b, x);

  // Solve Dy = z (result y is stored in x)
  scale_by_diag(d, n, x);

  // Solve L^T x = y (final result x is stored in x)
  backward_subst_upper_from_lower_transpose_unitdiag_true(A, n, lda, x, x);

  return true;
}

} // namespace gms
