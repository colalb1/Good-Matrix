#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(gms, m) {
    m.doc() = "General Matrix Solver stub module";
    m.def("hello", [](){ return "Hello from gms stub"; });
}
