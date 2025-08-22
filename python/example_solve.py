#!/usr/bin/env python3
"""
Example script demonstrating the use of the unified solve API.
"""

import os
import sys

import numpy as np

# Add the build directory to the Python path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
)

try:
    import gms
except ImportError:
    print(
        "Error: Could not import gms module. Make sure it's built and in the Python path."
    )
    sys.exit(1)


def print_separator():
    print("-" * 80)


def test_dense_spd():
    """Test solving a dense SPD system."""
    print("Testing dense SPD system")
    print_separator()

    # Create a simple SPD matrix
    n = 5
    A = np.zeros((n, n))

    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0

    # Make it symmetric positive definite
    A = A.T @ A + np.eye(n) * 0.1

    # Create a right-hand side
    b = np.ones(n)

    # Solve the system
    x, report = gms.solve(A, b)

    # Print results
    print(f"Matrix A:\n{A}")
    print(f"Right-hand side b: {b}")
    print(f"Solution x: {x}")
    print("\nSolver Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Verify the solution
    residual = np.linalg.norm(A @ x - b)
    print(f"\nResidual ||Ax - b||: {residual}")

    print_separator()


def test_dense_rectangular():
    """Test solving a dense rectangular system (least squares)."""
    print("Testing dense rectangular system (least squares)")
    print_separator()

    # Create a rectangular matrix (more rows than columns)
    m, n = 8, 5
    A = np.random.rand(m, n)

    # Create a right-hand side
    b = np.ones(m)

    # Solve the system
    x, report = gms.solve(A, b)

    # Print results
    print(f"Matrix A shape: {A.shape}")
    print(f"Right-hand side b shape: {b.shape}")
    print(f"Solution x: {x}")
    print("\nSolver Report:")

    for key, value in report.items():
        print(f"  {key}: {value}")

    # Verify the solution (should minimize ||Ax - b||)
    residual = np.linalg.norm(A @ x - b)
    print(f"\nResidual ||Ax - b||: {residual}")

    # Compare with numpy's least squares solution
    x_numpy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual_numpy = np.linalg.norm(A @ x_numpy - b)
    print(f"NumPy least squares residual: {residual_numpy}")
    print(f"Difference between solutions: {np.linalg.norm(x - x_numpy)}")

    print_separator()


def test_strategy_options():
    """Test different strategy options."""
    print("Testing different strategy options")
    print_separator()

    # Create a simple matrix
    n = 5
    A = np.random.rand(n, n)
    A = A.T @ A + np.eye(n)  # Make it SPD
    b = np.ones(n)

    strategies = ["auto", "direct", "iterative", "speed", "accuracy"]

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        x, report = gms.solve(A, b, strategy=strategy)
        print(f"Method: {report['method']}")
        print(f"Preconditioner: {report['preconditioner']}")
        print(f"Precision: {report['precision']}")
        print(f"Rationale: {report['rationale']}")

    print_separator()


def main():
    print("GMS Unified Solver API Example")
    print_separator()

    # Print the GMS module docstring
    print(f"GMS Module: {gms.__doc__}")
    print_separator()

    # Run the tests
    test_dense_spd()
    test_dense_rectangular()
    test_strategy_options()

    print("All tests completed.")


if __name__ == "__main__":
    main()
