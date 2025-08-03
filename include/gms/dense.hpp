#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace gms {

//==============================================================================
// Cholesky (LL^T) Decomposition & Solver
//==============================================================================

/**
 * @brief Performs an in-place Cholesky (LL^T) decomposition.
 * @details The lower triangular part of A is replaced by the Cholesky factor L.
 * @param A Pointer to the matrix data (row-major).
 * @param n Dimension of the matrix.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if the matrix is not positive definite.
 */
template <class T>
bool cholesky_inplace_lower(T *A, std::size_t n, std::size_t row_stride,
                            T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point<T>::value,
                "Cholesky requires floating point type");
  if (!A || row_stride < n) {
    throw std::invalid_argument("cholesky_inplace_lower: invalid A/row_stride");
  }

  const T eps = std::numeric_limits<T>::epsilon();
  T max_diag = static_cast<T>(0);

  for (std::size_t k = 0; k < n; ++k) {
    // Track the largest diagonal value seen so far to define relative tolerance
    max_diag = std::max(max_diag, std::abs(A[k * row_stride + k]));

    // Compute $d_k = A_{kk} - \sum_{j=0}^{k-1} L_{kj}^2$
    T *row_ptr = A + k * row_stride;
    T residual_diag_correction =
        std::accumulate(row_ptr, row_ptr + k, T{0},
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
    const std::size_t ui = static_cast<std::size_t>(i);
    T partial_sum = T{0};

    // Manually compute the sum of L[j][i] * x[j] for j = i+1 to n-1
    for (std::size_t j = ui + 1; j < n; ++j) {
      partial_sum += L[j * row_stride + ui] * x[j];
    }

    // Computes $x_i = (y_i - (\sum_{j=i + 1}^{n - 1} L_ji * x_j)) / L_ii$
    x[ui] = (y[ui] - partial_sum) / L[ui * row_stride + ui];
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
  static_assert(std::is_floating_point<T>::value,
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
    T sum_lower_diag = T{0};
    for (std::size_t k = 0; k < j; ++k) {
      sum_lower_diag += A[j * row_stride + k] * A[j * row_stride + k] * d[k];
    }

    const T D_j = A[j * row_stride + j] - sum_lower_diag;

    const T tol = std::max(eps_rel * max_diag_abs, T(10) * eps * max_diag_abs);

    // If the diagonal element is too small, the matrix is singular
    if (std::abs(D_j) <= tol)
      return false;

    d[j] = D_j;

    // Compute the lower triangular elements $L_ij = (A_ij - \sum_{k=0}^{j - 1}
    // L_ik * L_jk * d_k) / D_j$
    for (std::size_t i = j + 1; i < n; ++i) {
      T sum_l_ld = 0;

      for (std::size_t k = 0; k < j; ++k) {
        sum_l_ld += A[i * row_stride + k] * A[j * row_stride + k] * d[k];
      }

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
    T inner_product = T{0};

    // Manually compute the sum of L[j][i] * x[j] for j = i+1 to n-1
    for (std::size_t j = ui + 1; j < n; ++j) {
      inner_product += L[j * row_stride + ui] * x[j];
    }

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

/**
 * @brief Performs in-place LU decomposition with partial pivoting.
 * @details A is overwritten with L and U (unit diagonal L). Pivot indices are
 * stored in `pivots`.
 * @param A Pointer to the matrix data (row-major).
 * @param pivots Pointer to array of size n (output).
 * @param n Matrix dimension.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if the matrix is singular.
 */
template <class T>
bool lu_decompose_inplace(T *A, std::size_t *pivots, std::size_t n,
                          std::size_t row_stride,
                          T eps_rel = static_cast<T>(1e-14)) {
  static_assert(std::is_floating_point_v<T>,
                "LU decomposition requires floating point type");

  if (!A || !pivots || row_stride < n)
    throw std::invalid_argument(
        "lu_decompose_inplace: invalid A/pivots/row_stride");

  const T eps = std::numeric_limits<T>::epsilon();
  T max_val = T{0};

  for (std::size_t k = 0; k < n; ++k) {
    // Find pivot row: row with largest absolute value in column k
    std::size_t pivot_row = k;
    T max_in_col = std::abs(A[k * row_stride + k]);

    // Find pivot row: find p such that |A_pk| = max_{i= k,..., n - 1} |A_ik|.
    for (std::size_t i = k + 1; i < n; ++i) {
      T val = std::abs(A[i * row_stride + k]);

      if (val > max_in_col) {
        max_in_col = val;
        pivot_row = i;
      }
    }

    // Check if matrix is singular (pivot too small)
    max_val = std::max(max_val, max_in_col);
    const T tol = std::max(eps_rel * max_val, T(10) * eps * max_val);
    if (max_in_col <= tol)
      return false;

    // Record the pivot row index p for the k-th step
    pivots[k] = pivot_row;

    // If p != k, swap row k and row p
    if (pivot_row != k) {
      for (std::size_t j = 0; j < n; ++j)
        std::swap(A[k * row_stride + j], A[pivot_row * row_stride + j]);
    }

    // For rows i > k, compute the multipliers L_ik and update the submatrix
    for (std::size_t i = k + 1; i < n; ++i) {
      // L_ik = A_ik / A_kk
      A[i * row_stride + k] /= A[k * row_stride + k];

      // For j > k, A_ij := A_ij - L_ik * A_kj
      for (std::size_t j = k + 1; j < n; ++j) {
        A[i * row_stride + j] -= A[i * row_stride + k] * A[k * row_stride + j];
      }
    }
  }

  return true;
}

/**
 * @brief Solves L y = P b using forward substitution.
 * @details L has implicit unit diagonal. Applies permutation to b, then solves.
 * The result is written to y.
 */
template <class T>
void lu_forward_substitute(const T *A, const std::size_t *pivots, std::size_t n,
                           std::size_t row_stride, const T *b, T *y) {
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = b[i];
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (pivots[i] != i) {
      std::swap(y[i], y[pivots[i]]);
    }
  }

  // Now, solve Ly = y (where y was Pb) in-place using the L matrix stored in A
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      y[i] -= A[i * row_stride + j] * y[j];
    }
  }
}

/**
 * @brief Solves U x = y using backward substitution.
 * @details U is the upper triangle of A.
 */
template <class T>
void lu_backward_substitute(const T *A, std::size_t n, std::size_t row_stride,
                            const T *y, T *x) {
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    T inner_product = T{0};

    // inner_product = U_ij * x_j for j = i+1 to n-1
    for (std::size_t j = ui + 1; j < n; ++j) {
      inner_product += A[ui * row_stride + j] * x[j];
    }

    x[ui] = (y[ui] - inner_product) / A[ui * row_stride + ui];
  }
}

/**
 * @brief Solves A x = b using LU decomposition.
 * @details Performs in-place LU decomposition and solves using forward and
 * backward substitution.
 * @param A Pointer to matrix data. Will be overwritten.
 * @param b Right-hand side vector.
 * @param x Output solution vector.
 * @param pivots Temporary buffer of size n.
 * @param n Matrix size.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if the matrix is singular.
 */
template <class T>
bool lu_solve_inplace(T *A, const T *b, T *x, std::size_t *pivots,
                      std::size_t n, std::size_t row_stride,
                      T eps_rel = static_cast<T>(1e-14)) {
  // Step 1: Decompose A into PA = LU.
  if (!lu_decompose_inplace(A, pivots, n, row_stride, eps_rel)) {
    return false; // Matrix is singular, cannot solve.
  }

  // Step 2: Solve Ly = Pb using forward substitution.
  lu_forward_substitute(A, pivots, n, row_stride, b, x);

  // Step 3: Solve Ux = y using backward substitution.
  lu_backward_substitute(A, n, row_stride, x, x);

  return true;
}

/**
 * @brief Performs QR decomposition using Householder reflections.
 * @details The input matrix A (m × n) is overwritten: R in the upper triangle,
 * and the Householder vectors below the diagonal. Tau stores scalar reflection
 * factors.
 * @param A Pointer to the matrix data (row-major), size m × n.
 * @param tau Pointer to output array of size min(m, n) for scalar reflection
 * coefficients.
 * @param m Number of rows.
 * @param n Number of columns.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if numerical instability is detected.
 */
template <class T>
bool qr_decompose_inplace(T *A, T *tau, std::size_t m, std::size_t n,
                          std::size_t row_stride,
                          T eps_rel = static_cast<T>(1e-14)) {
  std::size_t k_max = std::min(m, n);

  for (std::size_t k = 0; k < k_max; ++k) {
    // Compute norm of column k from row k to m - 1
    // $$\|x\|_2 = \sqrt{\sum_{i=k}^{m - 1} A_{ik}^2}$$
    T norm_x = static_cast<T>(0);

    for (std::size_t i = k; i < m; ++i) {
      T val = A[i * row_stride + k];
      norm_x += val * val;
    }
    norm_x = std::sqrt(norm_x);

    if (norm_x <= eps_rel) {
      // If column is negligible, set tau[k] = 0 (don't reflect)
      tau[k] = static_cast<T>(0);
      continue;
    }

    // $$\alpha = A_{kk}
    // $$\beta = -\mathrm{sign}(\alpha)\|x\|_2$$
    T alpha = A[k * row_stride + k];
    T beta = (alpha >= static_cast<T>(0) ? -norm_x : norm_x);

    // $$\tau_k = \frac{\beta - \alpha}{\beta}$$
    T tau_k = (beta - alpha) / beta;
    tau[k] = tau_k;

    // Form Householder vector v in-place: v[0] = 1; A[k,k] = beta;
    // $$v = x - (\beta)e_1$$
    A[k * row_stride + k] = beta;

    for (std::size_t i = k + 1; i < m; ++i) {
      A[i * row_stride + k] /= (alpha - beta);
    }

    // Apply reflector to remaining submatrix A[k:, k+1:]
    for (std::size_t j = k + 1; j < n; ++j) {
      // Compute inner product w = v^T A[:,j]
      // $$w = A_{kj} + \sum_{i=k + 1}^{m - 1} v_i A_{ij}$$
      T w = A[k * row_stride + j];

      for (std::size_t i = k + 1; i < m; ++i) {
        w += A[i * row_stride + k] * A[i * row_stride + j];
      }
      w *= tau_k;

      // Update column j: A[:,j] -= w * v
      // $$A_{ij} \leftarrow A_{ij} - \tau_k * w * v_i$$
      A[k * row_stride + j] -= w;

      for (std::size_t i = k + 1; i < m; ++i) {
        A[i * row_stride + j] -= w * A[i * row_stride + k];
      }
    }
  }
  return true;
}

/**
 * @brief Applies Q^T to vector b using the stored Householder vectors and tau.
 * @details This computes y = Q^T b without explicitly forming Q.
 * @param A The matrix from QR (contains Householder vectors).
 * @param tau Scalar reflection factors from QR.
 * @param m Number of rows of A and b.
 * @param n Number of columns of A (number of Householder reflectors).
 * @param row_stride Leading dimension of A.
 * @param b Input vector (size m).
 * @param y Output vector (size n) containing Q^T b.
 */
template <class T>
void apply_q_transpose_to_vector(const T *A, const T *tau, std::size_t m,
                                 std::size_t n, std::size_t row_stride,
                                 const T *b, T *y) {
  // Copy b into working array y_full
  std::vector<T> y_full(m);
  for (std::size_t i = 0; i < m; ++i) {
    y_full[i] = b[i];
  }

  // For each reflector k=0...n - 1: y_full = (I - tau_k v v^T) y_full
  for (std::size_t k = 0; k < n; ++k) {
    // $$v = [1; A_{k + 1: k + 1...m - 1, k}]$$
    // Compute w = tau_k * (v^T y_full[k:])
    T dot = y_full[k];

    for (std::size_t i = k + 1; i < m; ++i) {
      dot += A[i * row_stride + k] * y_full[i];
    }
    T w = tau[k] * dot;

    // y_full[k: m] -= w * v
    y_full[k] -= w;
    for (std::size_t i = k + 1; i < m; ++i) {
      y_full[i] -= w * A[i * row_stride + k];
    }
  }

  // Output first n entries
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = y_full[i];
  }
}

/**
 * @brief Solves R x = y where R is the upper triangular part of A.
 * @details This is standard backward substitution.
 * @param A The matrix from QR (contains R in upper triangle).
 * @param n Number of columns (size of x and y).
 * @param row_stride Leading dimension of A.
 * @param y The input vector from Q^T b.
 * @param x Output vector containing the solution.
 */
template <class T>
void solve_upper_triangular_from_qr(const T *A, std::size_t n,
                                    std::size_t row_stride, const T *y, T *x) {
  // Backward substitution
  for (std::size_t ii = 0; ii < n; ++ii) {
    std::size_t i = n - 1 - ii;

    // $$x_i = \frac{1}{R_{ii}}\left(y_i - \sum_{j=i+1}^{n-1} R_{ij}
    // x_j\right)$$
    T sum = static_cast<T>(0);

    for (std::size_t j = i + 1; j < n; ++j) {
      sum += A[i * row_stride + j] * x[j];
    }
    x[i] = (y[i] - sum) / A[i * row_stride + i];
  }
}

/**
 * @brief Solves A x = b using QR decomposition (Householder).
 * @details Performs QR factorization, applies Q^T b, then solves R x = y.
 * @param A Matrix A (m × n), overwritten during decomposition.
 * @param b Input vector b (size m).
 * @param x Output vector x (size n).
 * @param tau Temporary array of size n for storing Householder scalar
 * coefficients.
 * @param m Number of rows of A.
 * @param n Number of columns of A.
 * @param row_stride Leading dimension of A.
 * @return true on success, false if matrix is rank-deficient or
 * ill-conditioned.
 */
template <class T>
bool qr_solve_inplace(T *A, const T *b, T *x, T *tau, std::size_t m,
                      std::size_t n, std::size_t row_stride,
                      T eps_rel = static_cast<T>(1e-14)) {
  // $$A \leftarrow Q R$$ via in-place Householder QR
  if (!qr_decompose_inplace(A, tau, m, n, row_stride, eps_rel)) {
    return false;
  }
  // Check for rank-deficiency: zero (or near-zero) diagonal in R
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(A[i * row_stride + i]) <= eps_rel) {
      return false;
    }
  }

  // $$y = Q^T b$$
  std::vector<T> y(n);
  apply_q_transpose_to_vector(A, tau, m, n, row_stride, b, y.data());

  // $$R x = y$$
  solve_upper_triangular_from_qr(A, n, row_stride, y.data(), x);
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
