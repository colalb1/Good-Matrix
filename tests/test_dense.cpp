#include <iostream>
#include <vector>
#include <cmath>
#include "../include/gms/dense.hpp"

bool nearly_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

void test_cholesky_2x2() {
    std::vector<double> A = {
        4.0, 12.0,
        12.0, 37.0
    };
    const std::size_t n = 2;
    const std::size_t stride = 2;

    bool ok = gms::cholesky_inplace_lower(A.data(), n, stride);
    if (!ok) {
        std::cerr << "❌ Cholesky decomposition failed\n";
        return;
    }

    // Expect L = [2 0; 6 1]
    bool pass = nearly_equal(A[0], 2.0) &&  // L(0,0)
                nearly_equal(A[2], 6.0) &&  // L(1,0)
                nearly_equal(A[3], 1.0);    // L(1,1)

    if (pass)
        std::cout << "✅ Cholesky 2x2 passed\n";
    else
        std::cerr << "❌ Cholesky 2x2 failed\n";
}

void test_solver_3x3(bool use_ldlt) {
    std::vector<double> A = {
        25, 15, -5,
        15, 18,  0,
        -5,  0, 11
    };

    std::vector<double> b = {35, 33, 6};
    std::vector<double> x(3);
    std::vector<double> d(3);  // Used only for LDL^T

    bool ok = gms::solve_inplace(
        A.data(), b.data(), x.data(), 3, 3,
        use_ldlt, d.data()
    );

    if (!ok) {
        std::cerr << (use_ldlt ? "❌ LDLᵀ" : "❌ Cholesky")
                  << " solve failed\n";
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

int main() {
    test_cholesky_2x2();
    test_solver_3x3(false);  // Cholesky
    test_solver_3x3(true);   // LDL^T
    return 0;
}
