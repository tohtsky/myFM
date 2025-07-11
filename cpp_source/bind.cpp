#include "declare_module.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

NB_MODULE(_myfm, m) {
  declare_functional<double>(m);
}
