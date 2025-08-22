#include "../include/gms/router.hpp"
#include "../include/gms/solver.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper to convert solver method enum to string
std::string solver_method_to_string(gms::SolverMethod method) {
  switch (method) {
  case gms::SolverMethod::LLT:
    return "LLT";
  case gms::SolverMethod::LDLT:
    return "LDLT";
  case gms::SolverMethod::LU:
    return "LU";
  case gms::SolverMethod::QR:
    return "QR";
  case gms::SolverMethod::BANDED_CHOL:
    return "BANDED_CHOL";
  case gms::SolverMethod::TRIDIAG:
    return "TRIDIAG";
  case gms::SolverMethod::CG:
    return "CG";
  case gms::SolverMethod::GMRES:
    return "GMRES";
  default:
    return "UNKNOWN";
  }
}

// Helper to convert preconditioner enum to string
std::string preconditioner_to_string(gms::Preconditioner precond) {
  switch (precond) {
  case gms::Preconditioner::NONE:
    return "NONE";
  case gms::Preconditioner::JACOBI:
    return "JACOBI";
  case gms::Preconditioner::ILU0:
    return "ILU0";
  default:
    return "UNKNOWN";
  }
}

// Helper to convert precision enum to string
std::string precision_to_string(gms::Precision precision) {
  switch (precision) {
  case gms::Precision::FLOAT32:
    return "FLOAT32";
  case gms::Precision::FLOAT64:
    return "FLOAT64";
  case gms::Precision::MIXED:
    return "MIXED";
  default:
    return "UNKNOWN";
  }
}

// Main solve function
template <typename T>
py::tuple solve_wrapper(py::array_t<T> A, py::array_t<T> b,
                        const std::string &strategy = "auto") {
  py::buffer_info A_info = A.request();
  py::buffer_info b_info = b.request();

  if (A_info.ndim != 2) {
    throw std::runtime_error("A must be a 2-dimensional array");
  }
  if (b_info.ndim != 1) {
    throw std::runtime_error("b must be a 1-dimensional array");
  }

  std::size_t m = A_info.shape[0]; // Num rows
  std::size_t n = A_info.shape[1]; // Num cols

  if (b_info.shape[0] != m) {
    throw std::runtime_error("b must have the same number of rows as A");
  }

  // Determine if A is sparse (this is a placeholder - in a real implementation,
  // we would check the actual sparsity or accept a sparse matrix format)
  bool is_sparse = false;

  // Create output array for x
  py::array_t<T> x(n);
  py::buffer_info x_info = x.request();

  // Get pointers to data
  T *A_ptr = static_cast<T *>(A_info.ptr);
  T *b_ptr = static_cast<T *>(b_info.ptr);
  T *x_ptr = static_cast<T *>(x_info.ptr);

  // Solve the system
  gms::SolverReport report =
      gms::solve_system<T>(A_ptr, b_ptr, x_ptr, n, m, is_sparse, strategy);

  // Create a Python dictionary for the report
  py::dict report_dict;
  report_dict["method"] = solver_method_to_string(report.method);
  report_dict["preconditioner"] =
      preconditioner_to_string(report.preconditioner);
  report_dict["precision"] = precision_to_string(report.precision);
  report_dict["residual_norm"] = report.residual_norm;
  report_dict["relative_residual"] = report.relative_residual;
  report_dict["iterations"] = report.iterations;
  report_dict["setup_time_ms"] = report.setup_time_ms;
  report_dict["solve_time_ms"] = report.solve_time_ms;
  report_dict["rationale"] = report.rationale;
  report_dict["success"] = report.success;

  // Return the solution vector and the report
  return py::make_tuple(x, report_dict);
}

PYBIND11_MODULE(gms, m) {
  m.doc() = "General Matrix Solver - A system of equations solver that "
            "automatically detects matrix properties";

  // Define the solve function
  m.def("solve", &solve_wrapper<double>, py::arg("A"), py::arg("b"),
        py::arg("strategy") = "auto",
        R"(
          Solves a linear system Ax = b using the most efficient method.
          
          Parameters
          ----------
          A : array_like
              Coefficient matrix. Must be a 2D array.
          b : array_like
              Right-hand side vector. Must be a 1D array with length equal to the number of rows in A.
          strategy : str, optional
              Solver strategy. Options are:
              - "auto": Automatically select the best solver (default)
              - "direct": Use a direct solver
              - "iterative": Use an iterative solver
              - "speed": Optimize for speed
              - "accuracy": Optimize for accuracy
          
          Returns
          -------
          x : ndarray
              Solution vector
          report : dict
              Solver report containing information about the solve process:
              - method: Solver method used
              - preconditioner: Preconditioner used (if any)
              - precision: Precision used for computation
              - residual_norm: Final residual norm ||Ax - b||
              - relative_residual: Relative residual ||Ax - b|| / ||b||
              - iterations: Number of iterations (for iterative methods)
              - setup_time_ms: Time spent in setup phase (ms)
              - solve_time_ms: Time spent in solve phase (ms)
              - rationale: Explanation of solver selection
              - success: Whether the solve was successful
          )");

  // Also provide a float version
  m.def("solve", &solve_wrapper<float>, py::arg("A"), py::arg("b"),
        py::arg("strategy") = "auto");
}
