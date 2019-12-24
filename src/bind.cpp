#include <random>
#include <vector>
#include <tuple>

#include "FM.hpp"
#include "FMLearningConfig.hpp"
#include "FMTrainer.hpp"
#include "definitions.hpp"
#include <functional>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
using Real = double;

namespace py = pybind11;

using FMTrainer = myFM::FMTrainer<Real>;
using FM = myFM::FM<Real>;
using Hyper = myFM::FMHyperParameters<Real>;
using SparseMatrix = typename FM::SparseMatrix;
using FMLearningConfig = typename myFM::FMLearningConfig<Real>;
using Vector = typename FM::Vector;
using ConfigBuilder = FMLearningConfig::Builder;

std::pair<std::vector<FM>, std::vector<Hyper>>
create_train_fm(size_t n_factor, Real init_std, const SparseMatrix &X,
                const Vector &y, int random_seed, FMLearningConfig &config,
                std::function<bool(int, const FM &, const Hyper &)> cb) {
  FMTrainer fm_trainer(X, y, random_seed, config);
  auto fm = fm_trainer.create_FM(n_factor, init_std);
  auto hyper_param = fm_trainer.create_Hyper(fm.n_factors);
  return fm_trainer.learn_with_callback(fm, hyper_param, cb);
}

PYBIND11_MODULE(_myfm, m) {
  m.doc() = "Backend C++ inplementation for myfm.";

  py::enum_<myFM::TASKTYPE>(m, "TaskType", py::arithmetic())
      .value("REGRESSION", myFM::TASKTYPE::REGRESSION)
      .value("CLASSIFICATION", myFM::TASKTYPE::CLASSIFICATION);

  py::class_<FMLearningConfig>(m, "FMLearningConfig");

  py::class_<ConfigBuilder>(m, "ConfigBuilder")
      .def(py::init<>())
      .def("set_alpha_0", &ConfigBuilder::set_alpha_0)
      .def("set_beta_0", &ConfigBuilder::set_beta_0)
      .def("set_gamma_0", &ConfigBuilder::set_gamma_0)
      .def("set_mu_0", &ConfigBuilder::set_mu_0)
      .def("set_reg_0", &ConfigBuilder::set_reg_0)
      .def("set_n_iter", &ConfigBuilder::set_n_iter)
      .def("set_n_kept_samples", &ConfigBuilder::set_n_kept_samples)
      .def("set_task_type", &ConfigBuilder::set_task_type)
      .def("set_group_index", &ConfigBuilder::set_group_index)
      .def("set_indentical_groups", &ConfigBuilder::set_indentical_groups)
      .def("build", &ConfigBuilder::build);

  py::class_<FM>(m, "FM")
      .def_readwrite("V", &FM::V)
      .def_readwrite("w", &FM::w)
      .def_readwrite("w0", &FM::w0);

  py::class_<Hyper>(m, "FMHyperParameters")
    .def_readonly("alpha", &Hyper::alpha)
    .def_readonly("mu_w", &Hyper::mu_w)
    .def_readonly("lambda_w", &Hyper::lambda_w)
    .def_readonly("mu_V", &Hyper::mu_V)
    .def_readonly("lambda_V", &Hyper::lambda_V);

  py::class_<FMTrainer>(m, "FMTrainer")
      .def(py::init<const SparseMatrix &, const Vector &, int,
                    FMLearningConfig>())
      .def("create_FM", &FMTrainer::create_FM)
      .def("create_Hyper", &FMTrainer::create_Hyper)
      .def("learn", &FMTrainer::learn);

  m.def("create_train_fm", &create_train_fm, "create and train fm.");
}