#include "declare_module.hpp"

PYBIND11_MODULE(_myfm_float, m) {
  declare_functional<float>(m);
}
