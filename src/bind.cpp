#include "declare_module.hpp"

PYBIND11_MODULE(_myfm, m) {
  declare_functional<double>(m); 
}

