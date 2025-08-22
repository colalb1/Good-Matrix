# Good-Matrix
A high-performance C++ matrix meta-solver.

Stub project scaffolding for general matrix solver library in C++23 with Python bindings.

# Good-Matrix Python API

Contains the Python API for the Good-Matrix library, a high-performance system of equations solver that automatically detects matrix properties and routes to the most efficient solver.

Some methods remain unwritten. I may come back to this eventually. Partway through this project, I realized it was a bit foolish since the checks added enough computational overhead to negate the speed improvements that the most efficient solve would yield, and thus offering little practical benefit.

## API Overview

The unified Python API provides a single entry point `solve()` that:
1. Accepts both dense and sparse matrices
2. Analyzes matrix properties automatically
3. Routes to the optimal solver based on these properties
4. Returns both the solution and a detailed solver report

```python
import numpy as np
import gms

# Create a matrix and right-hand side
A = np.array([[4, 1], [1, 3]], dtype=np.float64)
b = np.array([1, 2], dtype=np.float64)

# Solve the system
x, report = gms.solve(A, b)

# Print the solution and report
print(f"Solution: {x}")
print(f"Method used: {report['method']}")
print(f"Rationale: {report['rationale']}")
```

## Function Signature

```python
x, report = gms.solve(A, b, strategy="auto")
```

### Parameters

- `A`: Coefficient matrix. Must be a 2D NumPy array.
- `b`: Right-hand side vector. Must be a 1D NumPy array with length equal to the number of rows in A.
- `strategy`: Solver strategy (optional). Options are:
  - `"auto"`: Automatically select the best solver (default)
  - `"direct"`: Use a direct solver
  - `"iterative"`: Use an iterative solver
  - `"speed"`: Optimize for speed
  - `"accuracy"`: Optimize for accuracy

### Returns

- `x`: Solution vector as a NumPy array
- `report`: Dictionary containing information about the solve process:
  - `method`: Solver method used (e.g., "LU", "QR", "CG", "GMRES")
  - `preconditioner`: Preconditioner used (e.g., "NONE", "JACOBI", "ILU0")
  - `precision`: Precision used for computation (e.g., "FLOAT64", "MIXED")
  - `residual_norm`: Final residual norm $||Ax - b||$
  - `relative_residual`: Relative residual $||Ax - b|| / ||b||$
  - `iterations`: Number of iterations (for iterative methods)
  - `setup_time_ms`: Time spent in setup phase (ms)
  - `solve_time_ms`: Time spent in solve phase (ms)
  - `rationale`: Explanation of solver selection
  - `success`: Whether the solve was successful

## Matrix Properties Detection

The solver automatically analyzes the following matrix properties to determine the most efficient solver:

- **Density**: Ratio of non-zero elements to total elements (threshold: 0.1)
- **Symmetry**: Whether A[i,j] = A[j,i] for all i,j
- **Positive definiteness**: For symmetric matrices, whether $x^T·A·x > 0$ for all non-zero x
- **Bandwidth**: Maximum distance from diagonal with non-zero elements
- **Condition number**: Estimated using power iteration or other methods
- **Diagonal dominance**: Whether $|A[i,i]| > \sum|A[i,j]|$ for all $i\neq j$.
- **Rank deficiency**: Whether the matrix has linearly dependent columns
- **Least squares problem**: Detected for non-square matrices (m > n)

## Routing Mechanism

The solver uses a sophisticated routing algorithm that:

1. **Analyzes matrix structure**:
   - Dense matrices (density > 0.1) → Direct solvers
   - Sparse matrices (density < 0.1) → Iterative solvers
   - Banded matrices with small bandwidth → Specialized banded solvers

2. **Considers matrix properties**:
   - SPD matrices → Cholesky (dense) or CG (sparse)
   - Symmetric non-SPD → LDLT (dense) or GMRES (sparse)
   - Non-symmetric → LU (dense) or GMRES (sparse)
   - Least squares problems → QR
   - Tridiagonal matrices → Thomas algorithm

3. **Adapts to conditioning**:
   - Well-conditioned → Simpler preconditioners or none
   - Ill-conditioned → Stronger preconditioners (ILU0)
   - Severely ill-conditioned → Double precision

4. **Respects strategy parameter**:
   - "speed" → May use mixed precision when safe
   - "accuracy" → Always uses double precision

## Performance Considerations

- **Automatic precision selection**: Uses mixed precision for well-conditioned problems when speed is prioritized
- **Preconditioner selection**: Automatically chooses between None, Jacobi, and ILU(0) based on matrix properties
- **Memory efficiency**: Sparse storage formats for sparse matrices
- **Specialized algorithms**: Uses optimized algorithms for special matrix structures (tridiagonal, banded)

## Examples

See `example_solve.py` for complete examples of using the API with different types of matrices and solver strategies.

## Solver Report Details

The `report` dictionary returned by `solve()` contains:

- **method**: Solver algorithm used (LLT, LDLT, LU, QR, CG, GMRES, etc.)
- **preconditioner**: Preconditioner used for iterative methods (NONE, JACOBI, ILU0)
- **precision**: Numerical precision used (FLOAT64, MIXED)
- **residual_norm**: Final residual norm ||Ax - b||
- **relative_residual**: Relative residual ||Ax - b|| / ||b||
- **iterations**: Number of iterations for iterative methods (0 for direct methods)
- **setup_time_ms**: Time spent in setup phase (factorization, preconditioner setup)
- **solve_time_ms**: Time spent in solve phase (back/forward substitution, iterations)
- **rationale**: Human-readable explanation of solver selection decisions
- **success**: Boolean indicating whether the solve was successful
