#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "FMLearningConfig.hpp"
#include "HyperParams.hpp"
#include "OProbitSampler.hpp"
#include "definitions.hpp"
#include "predictor.hpp"
#include "util.hpp"

namespace myFM {
template <typename Real, class Derived, class FMType, class HyperType,
          class RelationWiseCache, class HistoryType>
struct BaseFMTrainer {
  // typedef typename Derived::FMType FMType;
  // typedef typename Derived::HyperType HyperType;

  typedef typename FMType::Vector Vector;
  typedef typename FMType::DenseMatrix DenseMatrix;
  typedef typename FMType::SparseMatrix SparseMatrix;

  typedef relational::RelationBlock<Real> RelationBlock;
  // typedef relational::RelationWiseCache<Real> RelationWiseCache;

  typedef FMLearningConfig<Real> Config;
  typedef typename Config::TASKTYPE TASKTYPE;

  typedef pair<Predictor<Real>, HistoryType> learn_result_type;

  typedef OprobitSampler<Real> OprobitSamplerType;

  SparseMatrix X;
  vector<RelationBlock> relations;
  SparseMatrix X_t; // transposed

  const size_t dim_all;
  const Vector y;

  const int n_train;
  int n_class = 0; // Used by ordered probit

  Vector e_train;
  Vector q_train;
  vector<RelationWiseCache> relation_caches;

  const Config learning_config;

  size_t n_nan_occurred = 0;

  inline BaseFMTrainer(const SparseMatrix &X,
                       const vector<RelationBlock> &relations, int random_seed,
                       Config learning_config) {}

  inline BaseFMTrainer(const SparseMatrix &X,
                       const vector<RelationBlock> &relations, const Vector &y,
                       int random_seed, Config learning_config)
      : X(X), relations(relations), X_t(X.transpose()),
        dim_all(check_row_consistency_return_column(X, relations)), y(y),
        n_train(X.rows()), e_train(X.rows()), q_train(X.rows()),
        relation_caches(), learning_config(learning_config),
        random_seed(random_seed), gen_(random_seed) {
    for (auto it = relations.begin(); it != relations.end(); it++) {
      relation_caches.emplace_back(*it);
    }
    if (X.rows() != y.rows()) {
      throw std::runtime_error(StringBuilder{}
                                   .add("Shape mismatch: X has size")
                                   .space_and_add(X.rows())
                                   .space_and_add("and y has size")
                                   .space_and_add(y.rows())
                                   .build());
    }
    this->X.makeCompressed();
    this->X_t.makeCompressed();
    if (learning_config.task_type == Config::TASKTYPE::ORDERED) {

      const size_t rows = this->X.rows();
      std::vector<bool> existence(rows, false);
      for (auto &group_config : learning_config.cutpoint_groups()) {
        for (size_t k : group_config.second) {
          if (k >= rows) {
            throw std::invalid_argument(
                "out of range for cutpoint group config.");
          }
          if (existence[k]) {
            std::stringstream ss;
            ss << "index " << k << " overlapping in cutpoint config.";
            throw std::invalid_argument(ss.str());
          }
          existence[k] = true;
        }
      }
      for (size_t i_ = 0; i_ < rows; i_++) {
        if (!existence[i_]) {
          std::stringstream ss;
          ss << "cutpoint group not specified for " << i_ << ".";
          throw std::invalid_argument(ss.str());
        }
      }
    }
  }

  inline FMType create_FM(int rank, Real init_std) {
    FMType fm(rank);
    fm.initialize_weight(dim_all, init_std, gen_);
    return fm;
  }

  inline HyperType create_Hyper(size_t rank) {
    return HyperType{rank, learning_config.get_n_groups()};
  }


  inline learn_result_type
  learn_with_callback(FMType &fm, HyperType &hyper,
                      std::function<bool(int, FMType *, HyperType *, Predictor<Real> *, HistoryType *)> cb);

  inline void initialize_hyper(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).initialize_alpha();
    static_cast<Derived &>(*this).initialize_mu_w();
    static_cast<Derived &>(*this).initialize_lambda_w();

    static_cast<Derived &>(*this).initialize_mu_V();
    static_cast<Derived &>(*this).initialize_lambda_V();
  }

  inline void initialize_e(FMType &fm, const HyperType &hyper) {
    static_cast<Derived &>(*this).initialize_e(fm, hyper);
  }

  inline void update_all(FMType &fm, HyperType &hyper) {
    update_alpha_(fm, hyper);

    update_w0_(fm, hyper);

    update_lambda_w_(fm, hyper);

    update_mu_w_(fm, hyper);

    update_w_(fm, hyper);

    update_lambda_V_(fm, hyper);
    update_mu_V_(fm, hyper);

    update_V_(fm, hyper);

    update_e_(fm, hyper);
  }

  inline void update_alpha_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_alpha(fm, hyper);
  }

  inline void update_w0_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_w0(fm, hyper);
  }

  inline void update_lambda_w_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_lambda_w(fm, hyper);
  }

  inline void update_mu_w_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_mu_w(fm, hyper);
  }

  inline void update_lambda_V_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_lambda_V(fm, hyper);
  }

  inline void update_mu_V_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_mu_V(fm, hyper);
  }

  inline void update_w_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_w(fm, hyper);
  }

  inline void update_e_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_e(fm, hyper);
  }

  inline void update_V_(FMType &fm, HyperType &hyper) {
    static_cast<Derived &>(*this).update_V(fm, hyper);
  }

  const int random_seed;

protected:
  mt19937 gen_;
  // std::vector<OprobitSamplerType> cutpoint_sampler;

}; // BaseFMTrainer
} // namespace myFM
