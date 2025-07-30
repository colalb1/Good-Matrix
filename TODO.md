Good Matrix Plan

Goal
----
Build a General Matrix Solver (GMS) that accepts any matrix (dense, sparse, banded, tridiagonal, SPD, rank-deficient, least-squares) and automatically routes to the fastest safe backend based on detected properties. Provide Python bindings, structured solver reports, reproducible benchmarks, and a small learned router that beats simple heuristics.

Scope (MVP)
-----------
- Analyzer: infer density, symmetry/SPD, bandwidth, diagonal dominance, condition proxy, rank deficiency, LS detection.
- Backends (dense via Eigen/LAPACK): LLT/LDLT (SPD), LU (GEPP, general), QR with column pivoting (CPQR, LS/rank-deficient), (optional) small SVD.
- Banded/tridiagonal solvers: Thomas algorithm; banded Cholesky.
- Sparse backends: CSR + SpMV, CG, GMRES(m); Preconditioners: Jacobi, ILU(0).
- Mixed-precision iterative refinement (float32 inner solves, float64 correction) when safe.
- Router/meta-solver: heuristic rules + “auto-ml” tiny learned model; fallbacks with rationale trail.
- Python API: accepts NumPy dense arrays and SciPy sparse CSR; SciPy-like ergonomics.
- Benchmarks: dense/sparse/banded/tri-diag/LS/rank-deficient suites; structured results + plots.
- DX: CMake build, unit tests, docs, examples, CI, sdist/wheel packaging.

Architecture
------------
analyze(A)  ─► FeatureReport {n,m,nnz,density,sym,spd?,κ̂,bandwidth,diag_dom,rank_deficient,ls_case,block_hints}
                 │
                 ▼
        route(features, goal, preferences) ─► BackendConfig {method, precond, precision, params}
                 │
                 ▼
           solve(A, b or B) ─► (x, SolverReport{method,precond,precision,iters,rel_res,time_ms,reason,fallbacks})

Repo layout (updated)
---------------------
gms/
  CMakeLists.txt
  /include/gms/
    analyzer.hpp, router.hpp, features.hpp, reports.hpp
    dense.hpp, dense_lapack.hpp
    banded.hpp, tridiag.hpp
    csr.hpp, spmv.hpp, cg.hpp, gmres.hpp, precond.hpp, ilu0.hpp, utils.hpp, mixed_precision.hpp
  /src/
    analyzer.cpp, router.cpp, dense.cpp, banded.cpp, tridiag.cpp
    csr.cpp, spmv.cpp, cg.cpp, gmres.cpp, precond.cpp, ilu0.cpp, mixed_precision.cpp
    pybind_module.cpp
  /python/
    __init__.py
    benchmarks/
      generate_matrices.py
      run_benchmarks.py
      analyze_results.py
  /tests/
    test_analyzer.cpp, test_dense.cpp, test_banded.cpp
    test_csr.cpp, test_krylov.cpp, test_precond.cpp, test_router.cpp
  /examples/
    dense_spd_example.py
    ls_rank_deficient_example.py
    tridiag_pde_example.py
    sparse_poisson_example.py
  /docs/
    README.md, DESIGN.md, BENCHMARKS.md, ROUTING.md, MIXED_PRECISION.md
  .github/workflows/ci.yml

API (Python)
------------
solve(A, b, strategy="auto", tol=1e-8, max_iters=None, max_time_ms=None, prefer="time", budget=None)
analyze(A) -> FeatureReport
auto_solve(A, b, strategy="auto"|"auto-ml") -> (x, SolverReport)
# SolverReport includes explainability string, e.g.:
# "Chose Dense-LLT: SPD=True, κ̂≈2e3, density=1.0, bandwidth=full"

Heuristic rules (v0)
--------------------
- Dense vs Sparse: density = nnz / (n*m). If density > 0.1 → treat as dense; else sparse.
- SPD probe: try LLT with tiny diagonal jitter; if it fails, not SPD.
- Banded: if bandwidth bw << n and density small → banded solver (Thomas/banded Chol).
- Rank-deficient or LS (m≠n or rank<n): prefer CPQR; for tiny n and severe ill-conditioning, allow SVD.
- Sparse SPD: CG + (Jacobi/ILU) depending on κ̂ proxy.
- Sparse nonsymmetric: GMRES(m) with Jacobi; escalate to ILU(0) if slow.
- Mixed precision: enable if κ̂ < κ0 (e.g., 1e4) and target residual moderate; disable for rank-deficient/near-singular.

Acceptance Criteria (MVP)
-------------------------
- solve() correctly handles dense, sparse, banded, tridiagonal, LS, and rank-deficient cases with appropriate backend and explainable routing.
- Auto strategy achieves median regret ≤ 1.2× the best observed per case on the benchmark suite.
- Graceful fallbacks with clear diagnostics; no hard crashes on pathological inputs (status codes + rationale).
- Mixed precision provides equal accuracy with measurable speedups on eligible cases.
- Docs + examples allow a newcomer to solve 3 example problems in < 5 minutes.

30-Day Plan
----------------------

Week 1 — Core, Analyzer, and Router Skeleton
Day 1 — Bootstrap + repo layout
  - Add dense/banded/tridiag/sparse/analyzer/router headers; build stubs; README with general scope.
  - Accept: Build passes; basic structure compiles.

Day 2 — Dense wrappers (Eigen/LAPACK)
  - Implement LLT/LDLT, LU (GEPP), and QR (CPQR) wrappers with unified SolverReport.
  - Accept: Solve small dense systems and cross-check with NumPy/Scipy.

Day 3 — Banded/tridiagonal path
  - Implement Thomas algorithm; banded Cholesky with packed band storage.
  - Accept: Tests pass on synthetic banded SPD and tridiagonal cases.

Day 4 — Analyzer v1 (cheap, robust tests)
  - Compute density, symmetry metric, SPD probe, bandwidth estimate, diagonal dominance, κ̂ via 3–5 power iters on AᵀA.
  - Accept: analyze(A) returns stable features on toy matrices.

Day 5 — Rank deficiency & LS detection
  - CPQR-based rank estimate; detect m≠n least-squares case; flag rank-deficient.
  - Accept: Rank-deficient generators correctly flagged.

Day 6 — Router heuristics v0
  - Initial rules for dense/sparse/banded/tri-diag/SPD/LS; return config + rationale.
  - Accept: route(features) returns sensible choices on canned cases.

Day 7 — Unified Python API
  - solve(A,b,strategy="auto") that accepts dense/sparse; routes and solves; returns x + SolverReport.
  - Accept: Python example handles 4 categories (dense SPD, dense general, sparse SPD, tri-diag).

Week 2 — Solidify Sparse Backends + Mixed Precision & Refinement
Day 8 — CSR + SpMV
  - Implement CSR container and SpMV; unit tests.
  - Accept: Tests pass; numerical parity with dense reference on small cases.

Day 9 — Iterative solvers
  - Implement CG and GMRES(m); tolerances; logs; stopping criteria.
  - Accept: Converge on Poisson-2D and random nonsymmetric matrices.

Day 10 — Preconditioners
  - Jacobi and ILU(0) (factor + triangular solves) and integration with CG/GMRES.
  - Accept: Iteration counts reduced vs no precond.

Day 11 — Mixed precision manager
  - Float32 inner solves with Float64 residual/refinement; API knob precision=("float32","float64").
  - Accept: On eligible cases, matches double accuracy faster.

Day 12 — Analyzer→Precision policy
  - Heuristics to enable/disable mixed precision based on κ̂ and problem type.
  - Accept: Policy toggles as expected across tests.

Day 13 — Fallback ladder
  - Implement safe fallbacks: LLT→LDLT/QR; GMRES stagnation→larger restart or dense LU if density grows.
  - Accept: No hard failures; fallback trail recorded.

Day 14 — Logging & reports
  - Standardize SolverReport{method, precond, precision, iters, rel_res, time_ms, reason, fallbacks}; JSON serialization.
  - Accept: Python captures structured reports.

Week 3 — Benchmarks & Meta-Solver Learning Stub
Day 15 — Benchmark matrix zoo
  - Generators: Dense SPD (Wishart), general Gaussian, Hilbert, Toeplitz, low-rank+diag; Sparse Poisson-2D, AR(1)/Toeplitz SPD, nonsymmetric random, banded.
  - Accept: Saved problems with metadata.

Day 16 — Benchmark harness
  - Sweep sizes/configs; record features + outcomes (time, iters, success); write CSV/JSON.
  - Accept: Results across 10–20 families.

Day 17 — Heuristic audit
  - Compare router choice vs oracle (best observed per case); confusion matrix and regret plot.
  - Accept: Plots saved; issues identified.

Day 18 — Tiny learning model
  - Train logistic regression or small GBDT on features → solver class; export coefficients/rules.
  - Accept: Model lowers median regret vs heuristics on validation.

Day 19 — Embed model
  - Hard-code linear model coefficients or a rule table into C++ router_ml.hpp; add strategy="auto-ml".
  - Accept: auto-ml available; explanation prints feature contributions.

Day 20 — Rank-deficiency & LS polish
  - Ensure CPQR handles over/under-determined systems; return LS solution and residual norm.
  - Accept: Tall/skinny and wide tests pass.

Week 4 — Documentation, Performance, and Release
Day 21 — Performance pass (dense)
  - Ensure BLAS usage (OpenBLAS/MKL if present); avoid copies in bindings; batch small solves.
  - Accept: Dense LLT/LU near NumPy/Eigen performance.

Day 22 — Performance pass (iterative)
  - Optimize SpMV; reduce allocations; precompute ILU workspaces; use restrict/pointer arithmetic.
  - Accept: 10–20% gain vs earlier baseline.

Day 23 — Robust analyzer & tests
  - Stress symmetry/SPD detection under noise; tune tolerances; document false positives/negatives.
  - Accept: Analyzer unit tests with noise pass.

Day 24 — Docs & guides
  - README quickstart; ROUTING.md; MIXED_PRECISION.md; examples updated.
  - Accept: Newcomer solves 3 examples in <5 minutes.

Day 25 — Reproducible benchmark report
  - Publish figures: time vs size; regret vs family; router choice map; include machine specs.
  - Accept: docs/figures/ and BENCHMARKS.md updated.

Day 26 — API ergonomics
  - Add prefer={"time"|"accuracy"}, budget, strategy=("auto","auto-ml","manual"); consistent exceptions.
  - Accept: Clean docstrings and error messages.

Day 27 — CI & packaging
  - scikit-build-core wheels or sdist; GitHub Actions Linux build + Python smoke tests.
  - Accept: Fresh venv pip install . works; examples run.

Day 28 — Final QA & edge cases
  - Very small n, extreme aspect ratios, nearly singular dense, singular sparse; ensure graceful messages and status codes.
  - Accept: No crashes; correct statuses.

Day 29 — v0.1 tag & examples
  - Examples: dense SPD covariance, rank-deficient regression (LS), tri-diag PDE, sparse Poisson.
  - Accept: All examples run and produce SolverReports with rationale.

Day 30 — Release + roadmap
  - Tag v0.1.0; changelog; open v0.2 issues (AMG, GPU, block structures, improved κ̂, randomized SVD).
  - Accept: Public-ready repo.

Stretch (post-v0.1)
-------------------
- Algebraic multigrid (AMG) preconditioner for sparse SPD.
- GPU backends (cuSPARSE for SpMV/ILU; cuSOLVER for dense).
- Block-structured solvers and block-Jacobi/ILU for PDE/finance systems.
- Streaming updates (rank-1 updates) for rolling covariance and graph problems.
