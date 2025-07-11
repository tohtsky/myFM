#pragma once

#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

#include "myfm/FM.hpp"
#include "myfm/FMLearningConfig.hpp"
#include "myfm/FMTrainer.hpp"
#include "myfm/LearningHistory.hpp"
#include "myfm/definitions.hpp"
#include "myfm/util.hpp"
#include "myfm/variational.hpp"

using namespace std;

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

template <typename Real> void declare_functional(nanobind::module_ &m) {
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

  nanobind::enum_<TASKTYPE>(m, "TaskType")
      .value("REGRESSION", TASKTYPE::REGRESSION)
      .value("CLASSIFICATION", TASKTYPE::CLASSIFICATION)
      .value("ORDERED", TASKTYPE::ORDERED);

  nanobind::class_<FMLearningConfig>(m, "FMLearningConfig");

  nanobind::class_<RelationBlock>(m, "RelationBlock",
                                  R"delim(The RelationBlock Class.)delim")
      .def(nanobind::init<vector<size_t>, const SparseMatrix &>(), R"delim(
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
           nanobind::arg("original_to_block"), nanobind::arg("data"))
      .def_ro("original_to_block", &RelationBlock::original_to_block)
      .def_ro("data", &RelationBlock::X)
      .def_ro("mapper_size", &RelationBlock::mapper_size)
      .def_ro("block_size", &RelationBlock::block_size)
      .def_ro("feature_size", &RelationBlock::feature_size)
      .def("__repr__",
           [](const RelationBlock &block) {
             return (myFM::StringBuilder{})(
                        "<RelationBlock with mapper size = ")(
                        block.mapper_size)(", block data size = ")(
                        block.block_size)(", feature size = ")(
                        block.feature_size)(">")
                 .build();
           })
      .def("__getstate__",
           [](const RelationBlock &block) {
             return std::make_tuple(block.original_to_block, block.X);
           })
      .def("__setstate__",
           [](RelationBlock &block,
              const std::tuple<vector<size_t>,
                               typename RelationBlock::SparseMatrix> &state) {
             new (&block) RelationBlock(std::get<0>(state), std::get<1>(state));
           });

  nanobind::class_<ConfigBuilder>(m, "ConfigBuilder")
      .def(nanobind::init<>())
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

  nanobind::class_<FM>(m, "FM")
      .def_rw("w0", &FM::w0)
      .def_rw("w", &FM::w)
      .def_rw("V", &FM::V)
      .def_rw("cutpoints", &FM::cutpoints)
      .def("predict_score", &FM::predict_score)
      .def("oprobit_predict_proba", &FM::oprobit_predict_proba)
      .def("__repr__",
           [](const FM &fm) {
             return (myFM::StringBuilder{})(
                        "<Factorization Machine sample with feature size = ")(
                        fm.w.rows())(", rank = ")(fm.V.cols())(">")
                 .build();
           })
      .def("__getstate__",
           [](const FM &fm) {
             Real w0 = fm.w0;
             Vector w(fm.w);
             DenseMatrix V(fm.V);
             vector<Vector> cutpoints(fm.cutpoints);
             return std::make_tuple(w0, w, V, cutpoints);
           })
      .def("__setstate__",
           [](FM &fm, const std::tuple<Real, Vector, DenseMatrix,
                                       vector<Vector>> &state) {
             new (&fm) FM(std::get<0>(state), std::get<1>(state),
                          std::get<2>(state), std::get<3>(state));
           });

  nanobind::class_<VFM>(m, "VariationalFM")
      .def_rw("w0", &VFM::w0)
      .def_rw("w0_var", &VFM::w0_var)
      .def_rw("w", &VFM::w)
      .def_rw("w_var", &VFM::w_var)
      .def_rw("V", &VFM::V)
      .def_rw("V_var", &VFM::V_var)
      .def_rw("cutpoints", &VFM::cutpoints)
      .def("predict_score", &VFM::predict_score)
      .def("__repr__",
           [](const VFM &fm) {
             return (myFM::StringBuilder{})(
                        "<Factorization Machine sample with feature size = ")(
                        fm.w.rows())(", rank = ")(fm.V.cols())(">")
                 .build();
           })
      .def("__getstate__",
           [](const VFM &fm) {
             Real w0 = fm.w0;
             Real w0_var = fm.w0_var;
             Vector w(fm.w);
             Vector w_var(fm.w_var);
             DenseMatrix V(fm.V);
             DenseMatrix V_var(fm.V_var);
             vector<Vector> cutpoints(fm.cutpoints);
             return std::make_tuple(w0, w0_var, w, w_var, V, V_var, cutpoints);
           })
      .def(
          "__setstate__",
          [](VFM &vfm, const std::tuple<Real, Real, Vector, Vector, DenseMatrix,
                                        DenseMatrix, vector<Vector>> &state) {
            new (&vfm)
                VFM(std::get<0>(state), std::get<1>(state), std::get<2>(state),
                    std::get<3>(state), std::get<4>(state), std::get<5>(state),
                    std::get<6>(state)

                );
          });

  nanobind::class_<Hyper>(m, "FMHyperParameters")
      .def_ro("alpha", &Hyper::alpha)
      .def_ro("mu_w", &Hyper::mu_w)
      .def_ro("lambda_w", &Hyper::lambda_w)
      .def_ro("mu_V", &Hyper::mu_V)
      .def_ro("lambda_V", &Hyper::lambda_V)
      .def("__getstate__",
           [](const Hyper &hyper) {
             Real alpha = hyper.alpha;
             Vector mu_w(hyper.mu_w);
             Vector lambda_w(hyper.lambda_w);
             DenseMatrix mu_V(hyper.mu_V);
             DenseMatrix lambda_V(hyper.lambda_V);
             return std::make_tuple(alpha, mu_w, lambda_w, mu_V, lambda_V);
           })
      .def("__setstate__",
           [](Hyper &hyper, const std::tuple<Real, Vector, Vector, DenseMatrix,
                                             DenseMatrix> &state) {
             new (&hyper) Hyper(std::get<0>(state), std::get<1>(state),
                                std::get<2>(state), std::get<3>(state),
                                std::get<4>(state));
           });

  nanobind::class_<VHyper>(m, "VariationalFMHyperParameters")
      .def_ro("alpha", &VHyper::alpha)
      .def_ro("alpha_rate", &VHyper::alpha_rate)
      .def_ro("mu_w", &VHyper::mu_w)
      .def_ro("mu_w_var", &VHyper::mu_w_var)
      .def_ro("lambda_w", &VHyper::lambda_w)
      .def_ro("lambda_w_rate", &VHyper::lambda_w_rate)
      .def_ro("mu_V", &VHyper::mu_V)
      .def_ro("mu_V_var", &VHyper::mu_V_var)
      .def_ro("lambda_V", &VHyper::lambda_V)
      .def_ro("lambda_V_rate", &VHyper::lambda_V_rate)
      .def("__getstate__",
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

             return nanobind::make_tuple(alpha, alpha_rate, mu_w, mu_w_var,
                                         lambda_w, lambda_w_rate, mu_V,
                                         mu_V_var, lambda_V, lambda_V_rate);
           })
      .def(
          "__setstate__",
          [](VHyper &hyper, const std::tuple<Real, Real, Vector, Vector, Vector,
                                             Vector, DenseMatrix, DenseMatrix,
                                             DenseMatrix, DenseMatrix> &state) {
            new (&hyper) VHyper(std::get<0>(state), std::get<1>(state),
                                std::get<2>(state), std::get<3>(state),
                                std::get<4>(state), std::get<5>(state),
                                std::get<6>(state), std::get<7>(state),
                                std::get<8>(state), std::get<9>(state));
          });

  nanobind::class_<Predictor>(m, "Predictor")
      .def_ro("samples", &Predictor::samples)
      .def("predict", &Predictor::predict)
      .def("predict_parallel", &Predictor::predict_parallel)
      .def("predict_parallel_oprobit", &Predictor::predict_parallel_oprobit)
      .def("__getstate__",
           [](const Predictor &predictor) {
             return std::make_tuple(predictor.rank, predictor.feature_size,
                                    static_cast<int>(predictor.type),
                                    predictor.samples);
           })
      .def("__setstate__",
           [](Predictor &predictor,
              const std::tuple<size_t, size_t, TASKTYPE, vector<FM>> &state) {
             new (&predictor) Predictor(std::get<0>(state), std::get<1>(state),
                                        std::get<2>(state), std::get<3>(state));
           });
  nanobind::class_<VPredictor>(m, "VariationalPredictor")
      .def("predict", &VPredictor::predict)
      .def("__getstate__",
           [](const VPredictor &predictor) {
             return std::make_tuple(predictor.rank, predictor.feature_size,
                                    static_cast<int>(predictor.type),
                                    predictor.samples);
           })
      .def("__setstate__",
           [](VPredictor &predictor,
              const std::tuple<size_t, size_t, TASKTYPE, vector<VFM>> &state) {
             new (&predictor) VPredictor(std::get<0>(state), std::get<1>(state),
                                         std::get<2>(state), std::get<3>(state));
           });
  nanobind::class_<FMTrainer>(m, "FMTrainer")
      .def(nanobind::init<const SparseMatrix &, const vector<RelationBlock> &,
                          const Vector &, int, FMLearningConfig>())
      .def("create_FM", &FMTrainer::create_FM)
      .def("create_Hyper", &FMTrainer::create_Hyper);

  nanobind::class_<VFMTrainer>(m, "VariationalFMTrainer")
      .def(nanobind::init<const SparseMatrix &, const vector<RelationBlock> &,
                          const Vector &, int, FMLearningConfig>())
      .def("create_FM", &VFMTrainer::create_FM)
      .def("create_Hyper", &VFMTrainer::create_Hyper);

  nanobind::class_<History>(m, "LearningHistory")
      .def_ro("hypers", &History::hypers)
      .def_ro("train_log_losses", &History::train_log_losses)
      .def_ro("n_mh_accept", &History::n_mh_accept)
      .def("__getstate__",
           [](const History &h) {
             return std::make_tuple(h.hypers, h.train_log_losses,
                                    h.n_mh_accept);
           })
      .def("__setstate__",
           [](History &h, const std::tuple<vector<Hyper>, vector<Real>,
                                           vector<size_t>> &state) {
             History *result = new History();
             new (&h) History();
             h.hypers = std::get<0>(state);
             h.train_log_losses = std::get<1>(state);
             h.n_mh_accept = std::get<2>(state);
           });

  nanobind::class_<VHistory>(m, "VariationalLearningHistory")
      .def_ro("hypers", &VHistory::hyper)
      .def_ro("elbos", &VHistory::elbos)
      .def("__getstate__",
           [](const VHistory &h) { return std::make_tuple(h.hyper, h.elbos); })
      .def("__setstate__",

           [](VHistory &result, const std::tuple<Hyper, vector<Real>> &state) {
             new (&result) VHistory(std::get<0>(state), std::get<1>(state)

             );
           });
  m.def("create_train_fm", &create_train_fm<Real>, "create and train fm.",
        nanobind::rv_policy::move);
  m.def("create_train_vfm", &create_train_vfm<Real>, "create and train fm.",
        nanobind::rv_policy::move, nanobind::arg("rank"),
        nanobind::arg("init_std"), nanobind::arg("X"),
        nanobind::arg("relations"), nanobind::arg("y"),
        nanobind::arg("random_seed"), nanobind::arg("learning_config"),
        nanobind::arg("callback"));
  m.def("mean_var_truncated_normal_left",
        &myFM::mean_var_truncated_normal_left<Real>);
  m.def("mean_var_truncated_normal_right",
        &myFM::mean_var_truncated_normal_right<Real>);
}
