#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "myfm/FM.hpp"
#include "myfm/FMLearningConfig.hpp"
#include "myfm/FMTrainer.hpp"
#include "myfm/LearningHistory.hpp"
#include "myfm/OProbitSampler.hpp"
#include "myfm/definitions.hpp"
#include "myfm/util.hpp"
#include "myfm/variational.hpp"

using namespace std;

namespace py = pybind11;

template <typename Real> using FMTrainer = myFM::GibbsFMTrainer<Real>;

template <typename Real>
std::pair<myFM::Predictor<Real>, myFM::GibbsLearningHistory<Real>>
create_train_fm(
    size_t n_factor, Real init_std,
    const typename myFM::FM<Real>::SparseMatrix &X,
    const vector<myFM::relational::RelationBlock<Real>> &relations,
    const typename myFM::FM<Real>::Vector &y, int random_seed,
    myFM::FMLearningConfig<Real> &config,
    std::function<bool(int, myFM::FM<Real> *, myFM::FMHyperParameters<Real> *,
                       myFM::GibbsLearningHistory<Real> *)>
        cb) {
  FMTrainer<Real> fm_trainer(X, relations, y, random_seed, config);
  auto fm = fm_trainer.create_FM(n_factor, init_std);
  auto hyper_param = fm_trainer.create_Hyper(fm.n_factors);
  return fm_trainer.learn_with_callback(fm, hyper_param, cb);
}

template <typename Real>
std::pair<myFM::variational::VariationalPredictor<Real>,
          myFM::variational::VariationalLearningHistory<Real>>
create_train_vfm(
    size_t n_factor, Real init_std,
    const typename myFM::FM<Real>::SparseMatrix &X,
    const vector<myFM::relational::RelationBlock<Real>> &relations,
    const typename myFM::FM<Real>::Vector &y, int random_seed,
    myFM::FMLearningConfig<Real> &config,
    std::function<bool(int, myFM::variational::VariationalFM<Real> *,
                       myFM::variational::VariationalFMHyperParameters<Real> *,
                       myFM::variational::VariationalLearningHistory<Real> *)>
        cb) {
  myFM::variational::VariationalFMTrainer<Real> fm_trainer(X, relations, y,
                                                           random_seed, config);
  auto fm = fm_trainer.create_FM(n_factor, init_std);
  auto hyper_param = fm_trainer.create_Hyper(fm.n_factors);
  return fm_trainer.learn_with_callback(fm, hyper_param, cb);
}

template <typename Real> void declare_functional(py::module &m) {
  using FMTrainer = FMTrainer<Real>;
  using VFMTrainer = myFM::variational::VariationalFMTrainer<Real>;
  using FM = myFM::FM<Real>;
  using VFM = myFM::variational::VariationalFM<Real>;
  using Hyper = myFM::FMHyperParameters<Real>;
  using VHyper = myFM::variational::VariationalFMHyperParameters<Real>;
  using History = myFM::GibbsLearningHistory<Real>;
  using VHistory = myFM::variational::VariationalLearningHistory<Real>;
  using SparseMatrix = typename FM::SparseMatrix;
  using FMLearningConfig = typename myFM::FMLearningConfig<Real>;
  using Vector = typename FM::Vector;
  using DenseMatrix = typename FM::DenseMatrix;
  using ConfigBuilder = typename FMLearningConfig::Builder;
  using RelationBlock = typename myFM::relational::RelationBlock<Real>;
  using Predictor = typename myFM::Predictor<Real>;
  using VPredictor = typename myFM::variational::VariationalPredictor<Real>;
  using TASKTYPE = typename myFM::FMLearningConfig<Real>::TASKTYPE;

  m.doc() = "Backend C++ implementation for myfm.";

  py::enum_<TASKTYPE>(m, "TaskType", py::arithmetic())
      .value("REGRESSION", TASKTYPE::REGRESSION)
      .value("CLASSIFICATION", TASKTYPE::CLASSIFICATION)
      .value("ORDERED", TASKTYPE::ORDERED);

  py::class_<FMLearningConfig>(m, "FMLearningConfig");

  py::class_<RelationBlock>(m, "RelationBlock",
                            R"delim(The RelationBlock Class.)delim")
      .def(py::init<vector<size_t>, const SparseMatrix &>(), R"delim(
    Initializes relation block.

    Parameters
    ----------

    original_to_block: List[int]
        describes which entry points to to which row of the data (second argument).
    data: scipy.sparse.csr_matrix[float64]
        describes repeated pattern. 
      
    Note
    -----
    The entries of `original_to_block` must be in the [0, data.shape[0]-1].)delim",
           py::arg("original_to_block"), py::arg("data"))
      .def_readonly("original_to_block", &RelationBlock::original_to_block)
      .def_readonly("data", &RelationBlock::X)
      .def_readonly("mapper_size", &RelationBlock::mapper_size)
      .def_readonly("block_size", &RelationBlock::block_size)
      .def_readonly("feature_size", &RelationBlock::feature_size)
      .def("__repr__",
           [](const RelationBlock &block) {
             return (myFM::StringBuilder{})(
                        "<RelationBlock with mapper size = ")(
                        block.mapper_size)(", block data size = ")(
                        block.block_size)(", feature size = ")(
                        block.feature_size)(">")
                 .build();
           })
      .def(py::pickle(
          [](const RelationBlock &block) {
            return py::make_tuple(block.original_to_block, block.X);
          },
          [](py::tuple t) {
            if (t.size() != 2) {
              throw std::runtime_error("invalid state for Relationblock.");
            }
            return new RelationBlock(
                t[0].cast<vector<size_t>>(),
                t[1].cast<typename RelationBlock::SparseMatrix>());
          }));

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
      .def("set_nu_oprobit", &ConfigBuilder::set_nu_oprobit)
      .def("set_fit_w0", &ConfigBuilder::set_fit_w0)
      .def("set_fit_linear", &ConfigBuilder::set_fit_linear)
      .def("set_group_index", &ConfigBuilder::set_group_index)
      .def("set_identical_groups", &ConfigBuilder::set_identical_groups)
      .def("set_cutpoint_scale", &ConfigBuilder::set_cutpoint_scale)
      .def("set_cutpoint_groups", &ConfigBuilder::set_cutpoint_groups)
      .def("build", &ConfigBuilder::build);

  py::class_<FM>(m, "FM")
      .def_readwrite("w0", &FM::w0)
      .def_readwrite("w", &FM::w)
      .def_readwrite("V", &FM::V)
      .def_readwrite("cutpoints", &FM::cutpoints)
      .def("predict_score", &FM::predict_score)

      .def("__repr__",
           [](const FM &fm) {
             return (myFM::StringBuilder{})(
                        "<Factorization Machine sample with feature size = ")(
                        fm.w.rows())(", rank = ")(fm.V.cols())(">")
                 .build();
           })
      .def(py::pickle(
          [](const FM &fm) {
            Real w0 = fm.w0;
            Vector w(fm.w);
            DenseMatrix V(fm.V);
            vector<Vector> cutpoints(fm.cutpoints);
            return py::make_tuple(w0, w, V, cutpoints);
          },
          [](py::tuple t) {
            if (t.size() == 3) {
              /* For the compatibility with earlier versions */
              return new FM(t[0].cast<Real>(), t[1].cast<Vector>(),
                            t[2].cast<DenseMatrix>());
            } else if (t.size() == 4) {
              return new FM(t[0].cast<Real>(), t[1].cast<Vector>(),
                            t[2].cast<DenseMatrix>(),
                            t[3].cast<vector<Vector>>());
            } else {
              throw std::runtime_error("invalid state for FM.");
            }
          }));

  py::class_<VFM>(m, "VariationalFM")
      .def_readwrite("w0", &VFM::w0)
      .def_readwrite("w0_var", &VFM::w0_var)
      .def_readwrite("w", &VFM::w)
      .def_readwrite("w_var", &VFM::w_var)
      .def_readwrite("V", &VFM::V)
      .def_readwrite("V_var", &VFM::V_var)
      .def_readwrite("cutpoints", &VFM::cutpoints)
      .def("predict_score", &VFM::predict_score)
      .def("__repr__",
           [](const VFM &fm) {
             return (myFM::StringBuilder{})(
                        "<Factorization Machine sample with feature size = ")(
                        fm.w.rows())(", rank = ")(fm.V.cols())(">")
                 .build();
           })
      .def(py::pickle(
          [](const VFM &fm) {
            Real w0 = fm.w0;
            Real w0_var = fm.w0_var;
            Vector w(fm.w);
            Vector w_var(fm.w_var);
            DenseMatrix V(fm.V);
            DenseMatrix V_var(fm.V_var);
            vector<Vector> cutpoints(fm.cutpoints);
            return py::make_tuple(w0, w0_var, w, w_var, V, V_var, cutpoints);
          },
          [](py::tuple t) {
            if (t.size() == 6) {
              /* For the compatibility with earlier versions */
              return new VFM(t[0].cast<Real>(), t[1].cast<Real>(),
                             t[2].cast<Vector>(), t[3].cast<Vector>(),
                             t[4].cast<DenseMatrix>(),
                             t[5].cast<DenseMatrix>());
            } else if (t.size() == 7) {
              return new VFM(t[0].cast<Real>(), t[1].cast<Real>(),
                             t[2].cast<Vector>(), t[3].cast<Vector>(),
                             t[4].cast<DenseMatrix>(), t[5].cast<DenseMatrix>(),
                             t[6].cast<vector<Vector>>());
            } else {
              throw std::runtime_error("invalid state for FM.");
            }
          }));

  py::class_<Hyper>(m, "FMHyperParameters")
      .def_readonly("alpha", &Hyper::alpha)
      .def_readonly("mu_w", &Hyper::mu_w)
      .def_readonly("lambda_w", &Hyper::lambda_w)
      .def_readonly("mu_V", &Hyper::mu_V)
      .def_readonly("lambda_V", &Hyper::lambda_V)
      .def(py::pickle(
          [](const Hyper &hyper) {
            Real alpha = hyper.alpha;
            Vector mu_w(hyper.mu_w);
            Vector lambda_w(hyper.lambda_w);
            DenseMatrix mu_V(hyper.mu_V);
            DenseMatrix lambda_V(hyper.lambda_V);
            return py::make_tuple(alpha, mu_w, lambda_w, mu_V, lambda_V);
          },
          [](py::tuple t) {
            if (t.size() != 5) {
              throw std::runtime_error("invalid state for FMHyperParameters.");
            }
            // placement new
            return new Hyper(t[0].cast<Real>(), t[1].cast<Vector>(),
                             t[2].cast<Vector>(), t[3].cast<DenseMatrix>(),
                             t[4].cast<DenseMatrix>());
          }));

  py::class_<VHyper>(m, "VariationalFMHyperParameters")
      .def_readonly("alpha", &VHyper::alpha)
      .def_readonly("alpha_rate", &VHyper::alpha_rate)
      .def_readonly("mu_w", &VHyper::mu_w)
      .def_readonly("mu_w_var", &VHyper::mu_w_var)
      .def_readonly("lambda_w", &VHyper::lambda_w)
      .def_readonly("lambda_w_rate", &VHyper::lambda_w_rate)
      .def_readonly("mu_V", &VHyper::mu_V)
      .def_readonly("mu_V_var", &VHyper::mu_V_var)
      .def_readonly("lambda_V", &VHyper::lambda_V)
      .def_readonly("lambda_V_rate", &VHyper::lambda_V_rate)
      .def(py::pickle(
          [](const VHyper &hyper) {
            Real alpha = hyper.alpha;
            Real alpha_rate = hyper.alpha_rate;
            Vector mu_w(hyper.mu_w);
            Vector mu_w_var(hyper.mu_w_var);
            Vector lambda_w(hyper.lambda_w);
            Vector lambda_w_rate(hyper.lambda_w_rate);
            DenseMatrix mu_V(hyper.mu_V);
            DenseMatrix mu_V_var(hyper.mu_V_var);
            DenseMatrix lambda_V(hyper.lambda_V);
            DenseMatrix lambda_V_rate(hyper.lambda_V_rate);

            return py::make_tuple(alpha, alpha_rate, mu_w, mu_w_var, lambda_w,
                                  lambda_w_rate, mu_V, mu_V_var, lambda_V,
                                  lambda_V_rate);
          },
          [](py::tuple t) {
            if (t.size() != 10) {
              throw std::runtime_error("invalid state for FMHyperParameters.");
            }
            // placement new
            return new VHyper(
                t[0].cast<Real>(), t[1].cast<Real>(), t[2].cast<Vector>(),
                t[3].cast<Vector>(), t[4].cast<Vector>(), t[5].cast<Vector>(),
                t[6].cast<DenseMatrix>(), t[7].cast<DenseMatrix>(),
                t[8].cast<DenseMatrix>(), t[9].cast<DenseMatrix>());
          }));

  py::class_<Predictor>(m, "Predictor")
      .def_readonly("samples", &Predictor::samples)
      .def("predict", &Predictor::predict)
      .def("predict_parallel", &Predictor::predict_parallel)
      .def(py::pickle(
          [](const Predictor &predictor) {
            return py::make_tuple(predictor.rank, predictor.feature_size,
                                  static_cast<int>(predictor.type),
                                  predictor.samples);
          },
          [](py::tuple t) {
            if (t.size() != 4) {
              throw std::runtime_error("invalid state for FMHyperParameters.");
            }
            Predictor *p =
                new Predictor(t[0].cast<size_t>(), t[1].cast<size_t>(),
                              static_cast<TASKTYPE>(t[2].cast<int>()));
            p->set_samples(std::move(t[3].cast<vector<FM>>()));
            return p;
          }));

  py::class_<VPredictor>(m, "VariationalPredictor")
      .def("predict", &VPredictor::predict)
      .def(py::pickle(
          [](const VPredictor &predictor) {
            return py::make_tuple(predictor.rank, predictor.feature_size,
                                  static_cast<int>(predictor.type),
                                  predictor.samples);
          },
          [](py::tuple t) {
            if (t.size() != 4) {
              throw std::runtime_error("invalid state for FMHyperParameters.");
            }
            VPredictor *p =
                new VPredictor(t[0].cast<size_t>(), t[1].cast<size_t>(),
                               static_cast<TASKTYPE>(t[2].cast<int>()));
            p->set_samples(std::move(t[3].cast<vector<VFM>>()));
            return p;
          }))
      .def("weights", [](VPredictor &predictor) {
        VFM returned = predictor.samples.at(0);
        return returned;
      });

  py::class_<FMTrainer>(m, "FMTrainer")
      .def(py::init<const SparseMatrix &, const vector<RelationBlock> &,
                    const Vector &, int, FMLearningConfig>())
      .def("create_FM", &FMTrainer::create_FM)
      .def("create_Hyper", &FMTrainer::create_Hyper);

  py::class_<VFMTrainer>(m, "VariationalFMTrainer")
      .def(py::init<const SparseMatrix &, const vector<RelationBlock> &,
                    const Vector &, int, FMLearningConfig>())
      .def("create_FM", &VFMTrainer::create_FM)
      .def("create_Hyper", &VFMTrainer::create_Hyper);

  py::class_<History>(m, "LearningHistory")
      .def_readonly("hypers", &History::hypers)
      .def_readonly("train_log_losses", &History::train_log_losses)
      .def_readonly("n_mh_accept", &History::n_mh_accept)
      .def(py::pickle(
          [](const History &h) {
            return py::make_tuple(h.hypers, h.train_log_losses, h.n_mh_accept);
          },
          [](py::tuple t) {
            if (t.size() != 3) {
              throw std::runtime_error("invalid state for LearningHistory.");
            }
            History *result = new History();
            result->hypers = t[0].cast<vector<Hyper>>();
            result->train_log_losses = t[1].cast<vector<Real>>();
            result->n_mh_accept = t[2].cast<vector<size_t>>();
            return result;
          }));

  py::class_<VHistory>(m, "VariationalLearningHistory")
      .def_readonly("hypers", &VHistory::hyper)
      .def_readonly("elbos", &VHistory::elbos)
      .def(py::pickle(
          [](const VHistory &h) { return py::make_tuple(h.hyper, h.elbos); },
          [](py::tuple t) {
            if (t.size() != 2) {
              throw std::runtime_error(
                  "invalid state for VariationalLearningHistory.");
            }
            VHistory *result =
                new VHistory(t[0].cast<Hyper>(), t[1].cast<vector<Real>>());
            return result;
          }));
  m.def("create_train_fm", &create_train_fm<Real>, "create and train fm.",
        py::return_value_policy::move);
  m.def("create_train_vfm", &create_train_vfm<Real>, "create and train fm.",
        py::return_value_policy::move, py::arg("rank"), py::arg("init_std"),
        py::arg("X"), py::arg("relations"), py::arg("y"),
        py::arg("random_seed"), py::arg("learning_config"),
        py::arg("callback"));
  m.def("mean_var_truncated_normal_left",
        &myFM::mean_var_truncated_normal_left<Real>);
  m.def("mean_var_truncated_normal_right",
        &myFM::mean_var_truncated_normal_right<Real>);
}