#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "gms/banded.hpp"

namespace gms {

enum class BandType { MAX, LOWER, UPPER };

/**
 * @brief Computes the density of a matrix A (fraction of non-zero entries).
 */
template <class T>
double density(const T *A, std::size_t n,
               T tol = std::numeric_limits<T>::epsilon()) {
  static_assert(std::is_floating_point_v<T>,
                "density: T must be floating point");
  if (n == 0)
    return 0.0;

  const std::size_t total_elements = n * n;

  std::size_t count =
      std::count_if(std::execution::par, A, A + total_elements,
                    [tol](T val) { return std::abs(val) > tol; });

  return static_cast<double>(count) / static_cast<double>(total_elements);
}

/**
 * @brief Checks if A is symmetric within tolerance tol.
 */
template <class T>
bool is_symmetric(const T *A, std::size_t n,
                  T tol = std::numeric_limits<T>::epsilon()) {
  static_assert(std::is_floating_point_v<T>,
                "is_symmetric: T must be floating point");

  // Trivial matrix is symmetric
  if (n <= 1)
    return true;

  // Create a vector of row indices [0, 1, ..., n - 2]
  std::vector<std::size_t> row_indices(n - 1);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  // Use a parallel `all_of` to check rows
  return std::all_of(std::execution::par, row_indices.cbegin(),
                     row_indices.cend(), [&](std::size_t i) {
                       for (std::size_t j = i + 1; j < n; ++j) {
                         if (std::abs(A[i * n + j] - A[j * n + i]) > tol) {
                           return false;
                         }
                       }
                       return true;
                     });
}

/**
 * @brief Probes SPD-ness by attempting a full-band Cholesky (m = n - 1)
 */
template <class T>
bool is_spd(const T *A, std::size_t n,
            T tol = std::numeric_limits<T>::epsilon()) {
  static_assert(std::is_floating_point_v<T>,
                "is_spd: T must be floating point");
  if (n == 0)
    return true;

  // Check symmetry
  if (!is_symmetric(A, n, tol)) {
    return false;
  }

  std::vector<T> A_copy(A, A + n * n);

  // Check positive definiteness
  return cholesky_inplace_lower(A_copy.data(), n, tol);
}

/**
 * @brief Estimates the bandwidth of a matrix A.
 */
template <class T>
std::size_t bandwidth(const T *A, std::size_t n,
                      BandType type = BandType::MAX) {
  static_assert(std::is_floating_point_v<T>,
                "bandwidth: T must be floating point");
  std::size_t bandwidth = 0;

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (A[i * n + j] == T(0))
        continue;

      std::size_t diff = std::abs(i - j);

      switch (type) {
      case BandType::MAX:
      default:
        bandwidth = std::max(bandwidth, diff);

      case BandType::LOWER:
        if (i >= j)
          bandwidth = std::max(bandwidth, i - j);
        break;

      case BandType::UPPER:
        if (j >= i)
          bandwidth = std::max(bandwidth, j - i);
        break;
      }
    }
  }
  // $$ \text{bandwidth}_{\max} = \max_{a_{ij}\neq 0}|i - j| $$
  return bandwidth;
}

/**
 * @brief Checks diagonal dominance.
 */
template <class T>
bool is_diagonally_dominant(const T *A, std::size_t n, bool strict = false) {
  static_assert(std::is_floating_point_v<T>,
                "is_diagonally_dominant: T must be floating point");
  for (std::size_t i = 0; i < n; ++i) {
    T sum = T(0);
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j)
        continue;
      sum += std::abs(A[i * n + j]);
    }
    // $$ |a_{ii}| %s \sum_{j\neq i}|a_{ij}| $$
    if (strict) {
      if (!(std::abs(A[i * n + i]) > sum))
        return false;
    } else {
      if (!(std::abs(A[i * n + i]) >= sum))
        return false;
    }
  }
  return true;
}

/**
 * @brief Estimates the condition number via power iteration on A^T A.
 */
template <class T>
T condition_estimate(const T *A, std::size_t n, unsigned power_iters = 5) {
  static_assert(std::is_floating_point_v<T>,
                "condition_estimate: T must be floating point");
  // Allocate workspace
  std::vector<T> x(n), y(n), tmp(n);
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_real_distribution<T> dist(T(-1), T(1));

  // init x randomly
  for (auto &xi : x)
    xi = dist(gen);

  // normalize x
  T norm_x = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), T(0)));

  for (auto &xi : x)
    xi /= norm_x;

  // power iteration for max eigenvalue of B = A^T A
  T lambda_max = T(0);
  for (unsigned iter = 0; iter < power_iters; ++iter) {
    // y = A * x
    for (std::size_t i = 0; i < n; ++i) {
      // $$ y_i = \sum_j A_{ij} x_j $$
      y[i] = std::inner_product(A + i * n, A + i * n + n, x.begin(), T(0));
    }
    // tmp = A^T * y
    for (std::size_t j = 0; j < n; ++j) {
      // $$ (A^T y)_j = \sum_i A_{ij} y_i $$
      tmp[j] = T(0);
      for (std::size_t i = 0; i < n; ++i)
        tmp[j] += A[i * n + j] * y[i];
    }
    lambda_max = std::inner_product(x.begin(), x.end(), tmp.begin(), T(0));
    // normalize tmp -> x
    T norm_tmp = std::sqrt(
        std::inner_product(tmp.begin(), tmp.end(), tmp.begin(), T(0)));
    for (std::size_t i = 0; i < n; ++i)
      x[i] = tmp[i] / norm_tmp;
  }

  // Inverse power iteration for min eigenvalue: solve B_data * y = x
  std::size_t m = n > 0 ? n - 1 : 0;
  std::vector<T> B_data((m + 1) * n);
  // pack B_data = A^T A in banded form
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = j; i < n; ++i) {
      // compute (A^T A)_{ij} = row i of A^T dot row j of A^T i.e. col i of A
      // dot col j of A
      T sum = T(0);
      for (std::size_t k = 0; k < n; ++k)
        sum += A[k * n + i] * A[k * n + j];
      B_data[(i - j) * n + j] = sum;
    }
  }
  if (!banded_cholesky_inplace_lower(B_data.data(), n, m)) {
    return std::numeric_limits<T>::infinity();
  }
  auto solve_B = [&](const std::vector<T> &L_data, const std::vector<T> &b,
                     std::vector<T> &out) {
    gms::banded_forward_subst_lower(L_data.data(), n, m, b.data(), out.data());
    gms::banded_backward_subst_upper_from_lower_transpose(
        L_data.data(), n, m, out.data(), out.data());
  };
  // init x randomly
  for (auto &xi : x)
    xi = dist(gen);
  T mu = T(0);
  for (unsigned iter = 0; iter < power_iters; ++iter) {
    solve_B(B_data, x, y);
    mu = std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), T(0)));
    for (std::size_t i = 0; i < n; ++i)
      x[i] = y[i] / mu;
  }
  // \lambda_min ≈ 1 / \mu
  T lambda_min = T(1) / mu;
  // cond(B) = \lambda_max / \lambda_min; cond(A) = sqrt(cond(B))
  return std::sqrt(lambda_max / lambda_min);
}

} // namespace gms
