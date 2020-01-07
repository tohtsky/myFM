#pragma once

#include <cmath>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "FM.hpp"
#include "FMLearningConfig.hpp"
#include "HyperParams.hpp"
#include "definitions.hpp"
#include "predictor.hpp"
#include "util.hpp"

constexpr size_t checkIndex = 79999;
namespace myFM {

template <typename Real> struct FMTrainer {

  typedef FM<Real> FMType;
  typedef FMHyperParameters<Real> HyperType;
  typedef typename FMType::Vector Vector;
  typedef typename FMType::DenseMatrix DenseMatrix;
  typedef typename FMType::SparseMatrix SparseMatrix;

  typedef relational::RelationBlock<Real> RelationBlock;
  typedef relational::RelationWiseCache<Real> RelationWiseCache;

  typedef FMLearningConfig<Real> Config;
  typedef typename Config::TASKTYPE TASKTYPE;

  SparseMatrix X;
  vector<RelationBlock> relations;
  SparseMatrix X_t; // transposed

  const size_t dim_all;

  const Vector y;

  const int n_train;

  Vector e_train;
  Vector q_train;

  vector<RelationWiseCache> relation_caches;

  const Config learning_config;

  size_t n_nan_occurred = 0;

  inline FMTrainer(const SparseMatrix &X,
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
  }

  inline FMType create_FM(int rank, Real init_std) {
    FMType fm(rank);
    fm.initialize_weight(dim_all, init_std, gen_);
    return fm;
  }

  inline FMHyperParameters<Real> create_Hyper(size_t rank) {
    return FMHyperParameters<Real>{rank, learning_config.get_n_groups()};
  }

  inline pair<Predictor<Real>, vector<HyperType>> learn(FMType &fm,
                                                        HyperType &hyper) {
    return learn_with_callback(
        fm, hyper, [](int i, FMType *fm, HyperType *hyper) { return false; });
  }

  /**
   *  Main routine for Gibbs sampling.
   */
  inline pair<Predictor<Real>, vector<HyperType>>
  learn_with_callback(FMType &fm, HyperType &hyper,
                      std::function<bool(int, FMType *, HyperType *)> cb) {
    pair<Predictor<Real>, vector<HyperType>> result{
        {static_cast<size_t>(fm.n_factors), dim_all, learning_config.task_type},
        {}};
    initialize_hyper(fm, hyper);
    initialize_e(fm, hyper);
    result.first.samples.reserve(learning_config.n_kept_samples);
    for (int mcmc_iteration = 0; mcmc_iteration < learning_config.n_iter;
         mcmc_iteration++) {
      mcmc_step(fm, hyper);
      if (learning_config.n_iter <=
          (mcmc_iteration + learning_config.n_kept_samples)) {
        result.first.samples.emplace_back(fm);
      }
      // for tracing
      result.second.emplace_back(hyper);

      bool should_stop = cb(mcmc_iteration, &fm, &hyper);
      if (should_stop) {
        break;
      }
    }
    return result;
  }

  inline void initialize_hyper(FMType &fm, HyperType &hyper) {
    hyper.alpha = static_cast<Real>(1);

    hyper.mu_w.array() = static_cast<Real>(0);
    hyper.lambda_w.array() = static_cast<Real>(1e-5);

    hyper.mu_V.array() = static_cast<Real>(0);
    hyper.lambda_V.array() = static_cast<Real>(1e-5);
  }

  inline void initialize_e(const FMType &fm, const HyperType &h) {
    fm.predict_score_write_target(e_train, X, relations);
    e_train -= y;
  }

  inline void mcmc_step(FMType &fm, HyperType &hyper) {
    sample_alpha(fm, hyper);

    sample_w0(fm, hyper);

    sample_lambda_w(fm, hyper);

    sample_mu_w(fm, hyper);

    sample_w(fm, hyper);

    sample_lambda_V(fm, hyper);
    sample_mu_V(fm, hyper);

    sample_V(fm, hyper);

    sample_e(fm, hyper);
  }

private:
  // sample from quad x ^2 - 2 * first x + ... = quad (x - first / quad) ^2
  inline Real sample_normal(const Real &quad, const Real &first) {
    return (first / quad) +
           normal_distribution<Real>(0, 1)(gen_) / std::sqrt(quad);
  }

  inline void sample_alpha(FMType &fm, HyperType &hyper) {
    // If the task is classification, take alpha = 1.
    if (learning_config.task_type == TASKTYPE::CLASSIFICATION) {
      hyper.alpha = static_cast<Real>(1);
      return;
    }
    Real e_all = e_train.array().square().sum();
    Real exponent = (learning_config.alpha_0 + X.rows()) / 2;
    Real variance = (learning_config.beta_0 + e_all) / 2;
    Real new_alpha = gamma_distribution<Real>(exponent, 1 / variance)(gen_);
    hyper.alpha = new_alpha;
  }

  /*
   The sampling method for both $\lambda _g ^{(w)}$ and $\lambda _{g,r} ^{(v)}$.
   */
  inline void sample_lambda_generic(const Vector &mu, Eigen::Ref<Vector> lambda,
                                    const Vector &weight) {
    const vector<vector<size_t>> &group_vs_feature_index =
        learning_config.group_vs_feature_index();
    size_t group_index = 0;
    for (const auto &group_feature_indices : group_vs_feature_index) {
      Real mean = mu(group_index);
      Real alpha = learning_config.alpha_0 + group_feature_indices.size();
      Real beta = learning_config.beta_0;

      for (auto feature_index : group_feature_indices) {
        auto dev = weight(feature_index) - mean;
        beta += dev * dev;
      }
      Real new_lambda = gamma_distribution<Real>(alpha / 2, 2 / beta)(gen_);
      lambda(group_index) = new_lambda;
      group_index++;
    }
  }

  /*
   The sampling method for both $\mu _g ^{(w)}$ and $\mu _{g,r} ^{(v)}$.
   */
  inline void sample_mu_generic(Eigen::Ref<Vector> mu, const Vector &lambda,
                                const Vector &weight) {
    const vector<vector<size_t>> &group_vs_feature_index =
        learning_config.group_vs_feature_index();
    size_t group_index = 0;
    for (const auto &group_feature_indices : group_vs_feature_index) {
      size_t n_feature_in_groups = group_feature_indices.size();
      Real square =
          lambda(group_index) * (learning_config.gamma_0 + n_feature_in_groups);
      Real linear = learning_config.gamma_0 * learning_config.mu_0;
      for (auto &f : group_feature_indices) {
        linear += weight(f);
      }
      linear *= lambda(group_index);
      Real new_mu = sample_normal(square, linear);
      mu(group_index) = new_mu;
      group_index++;
    }
  }

  inline void sample_lambda_w(FMType &fm, HyperType &hyper) {
    sample_lambda_generic(hyper.mu_w, hyper.lambda_w, fm.w);
  }

  inline void sample_mu_w(FMType &fm, HyperType &hyper) {
    sample_mu_generic(hyper.mu_w, hyper.lambda_w, fm.w);
  }

  inline void sample_lambda_V(FMType &fm, HyperType &hyper) {
    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      sample_lambda_generic(hyper.mu_V.col(factor_index),
                            hyper.lambda_V.col(factor_index),
                            fm.V.col(factor_index));
    }
  }

  inline void sample_mu_V(FMType &fm, HyperType &hyper) {
    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      sample_mu_generic(hyper.mu_V.col(factor_index),
                        hyper.lambda_V.col(factor_index),
                        fm.V.col(factor_index));
    }
  }

  inline void sample_w0(FMType &fm, HyperType &hyper) {
    Real w0_lin_term = hyper.alpha * (fm.w0 - e_train.array()).sum();
    Real w0_quad_term = hyper.alpha * n_train + learning_config.reg_0;
    Real w0_new = sample_normal(w0_quad_term, w0_lin_term);
    e_train.array() += (w0_new - fm.w0);
    fm.w0 = w0_new;
  }

  inline void sample_w(FMType &fm, HyperType &hyper) {
    // main table
    for (int feature_index = 0; feature_index < X.cols(); feature_index++) {
      int group = learning_config.group_index(feature_index);

      const Real w_old = fm.w(feature_index);
      e_train.array() -= X_t.row(feature_index) * w_old;
      Real lambda = hyper.lambda_w(group);
      Real mu = hyper.mu_w(group);
      Real square_term =
          lambda + hyper.alpha * X_t.row(feature_index).cwiseAbs2().sum();
      Real linear_term =
          -hyper.alpha * X_t.row(feature_index) * e_train + lambda * mu;

      Real w_new = sample_normal(square_term, linear_term);
      e_train.array() += X_t.row(feature_index) * w_new;
      fm.w(feature_index) = w_new;
    }

    // relational blocks
    size_t offset = X.cols();
    for (size_t relation_index = 0; relation_index < relations.size();
         relation_index++) {
      RelationBlock &relation_data = relations[relation_index];
      RelationWiseCache &relation_cache = relation_caches[relation_index];
      relation_cache.e.array() = 0;
      relation_cache.q.array() = 0;

      relation_cache.q =
          relation_data.X * fm.w.segment(offset, relation_data.feature_size);

      {
        size_t train_data_index = 0;
        for (auto i : relation_data.original_to_block) {
          relation_cache.e(i) += e_train(train_data_index);
          e_train(train_data_index++) -= relation_cache.q(i); // un-synchronize
        }
      }
      for (size_t inner_feature_index = 0;
           inner_feature_index < relation_data.feature_size;
           inner_feature_index++) {
        int group = learning_config.group_index(offset + inner_feature_index);
        const Real w_old = fm.w(offset + inner_feature_index);
        Real lambda = hyper.lambda_w(group);
        Real mu = hyper.mu_w(group);

        Real square_term =
            relation_cache.X_t.row(inner_feature_index).cwiseAbs2() *
            relation_cache.cardinality;
        Real linear_term =
            -relation_cache.X_t.row(inner_feature_index) * relation_cache.e;

        linear_term += square_term * w_old;

        square_term = lambda + hyper.alpha * square_term;
        linear_term = hyper.alpha * linear_term + lambda * mu;

        Real w_new = sample_normal(square_term, linear_term);
        fm.w(offset + inner_feature_index) = w_new;
        relation_cache.e += relation_cache.X_t.row(inner_feature_index)
                                .transpose()
                                .cwiseProduct(relation_cache.cardinality) *
                            (w_new - w_old);
      }

      relation_cache.q =
          relation_data.X * fm.w.segment(offset, relation_data.feature_size);
      {
        size_t train_data_index = 0;
        for (auto i : relation_data.original_to_block) {
          e_train(train_data_index++) += relation_cache.q(i); // un-sync
        }
      }
      offset += relation_data.feature_size;
    }
  }

  inline void sample_V(FMType &fm, HyperType &hyper) {
    using itertype = typename SparseMatrix::InnerIterator;

    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      q_train = X * fm.V.col(factor_index).head(X.cols());

      // compute contribution of blocks
      {
        // initialize block q caches
        size_t offset = X.cols();
        for (size_t relation_index = 0; relation_index < relations.size();
             relation_index++) {
          const RelationBlock &relation_data = relations[relation_index];
          RelationWiseCache &relation_cache = relation_caches[relation_index];
          relation_cache.q = relation_data.X *
                             (fm.V.col(factor_index)
                                  .segment(offset, relation_data.feature_size));
          size_t train_data_index = 0;
          for (auto i : relation_data.original_to_block) {
            q_train(train_data_index++) += relation_cache.q(i);
          }
          offset += relation_data.feature_size;
        }
      }

      // main table
      for (int feature_index = 0; feature_index < X_t.rows(); feature_index++) {
        auto g = learning_config.group_index(feature_index);
        Real v_old = fm.V(feature_index, factor_index);

        Real square_coeff = 0;
        Real linear_coeff = 0;

        for (itertype it(X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto h =
              it.value() * (q_train(train_data_index) - it.value() * v_old);
          square_coeff += h * h;
          linear_coeff += (-e_train(train_data_index)) * h;
        }
        linear_coeff += square_coeff * v_old;

        square_coeff *= hyper.alpha;
        linear_coeff *= hyper.alpha;

        square_coeff += hyper.lambda_V(g, factor_index);
        linear_coeff +=
            hyper.lambda_V(g, factor_index) * hyper.mu_V(g, factor_index);

        Real v_new = sample_normal(square_coeff, linear_coeff);
        fm.V(feature_index, factor_index) = v_new;
        for (itertype it(X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto h =
              it.value() * (q_train(train_data_index) - it.value() * v_old);
          q_train(train_data_index) += it.value() * (v_new - v_old);
          e_train(train_data_index) += h * (v_new - v_old);
        }
      }

      // draw v for relations
      size_t offset = X.cols();
      // initialize caches
      for (size_t relation_index = 0; relation_index < relations.size();
           relation_index++) {
        const RelationBlock &relation_data = relations[relation_index];
        RelationWiseCache &relation_cache = relation_caches[relation_index];

        // initialize block caches.
        relation_cache.q_S = relation_data.X.cwiseAbs2() *
                             (fm.V.col(factor_index)
                                  .segment(offset, relation_data.feature_size)
                                  .array()
                                  .square()
                                  .matrix());
        size_t train_data_index = 0;

        relation_cache.c.array() = 0;
        relation_cache.c_S.array() = 0;
        relation_cache.e.array() = 0;
        relation_cache.e_q.array() = 0;

        for (auto i : relation_data.original_to_block) {
          Real temp = (q_train(train_data_index) - relation_cache.q(i));
          relation_cache.c(i) += temp;
          relation_cache.c_S(i) += temp * temp;
          relation_cache.e(i) += e_train(train_data_index);
          relation_cache.e_q(i) += e_train(train_data_index) * temp;
          // un-synchronization of q and e
          q_train(train_data_index) -= relation_cache.q(i);
          // q_B
          // 1/ 2 ( (q_B + q_other) **2 - (q_B_S + other) )
          // q_B * q_other + 0.5 q_B **2 - 0.5 * q_B_S
          e_train(train_data_index) -=
              (q_train(train_data_index) * relation_cache.q(i) +
               0.5 * relation_cache.q(i) * relation_cache.q(i) -
               0.5 * relation_cache.q_S(i));
          train_data_index++;
        }
        // Initialized block-wise caches.
        for (size_t inner_feature_index = 0;
             inner_feature_index < relation_data.feature_size;
             inner_feature_index++) {
          auto g = learning_config.group_index(offset + inner_feature_index);
          Real v_old = fm.V(offset + inner_feature_index, factor_index);
          Real square_coeff = 0;
          Real linear_coeff = 0;

          Real x_il;
          for (itertype it(relation_cache.X_t, inner_feature_index); it; ++it) {
            auto block_data_index = it.col();
            x_il = it.value();
            auto h_B = (relation_cache.q(block_data_index) - x_il * v_old);
            auto h_squared =
                h_B * h_B * relation_cache.cardinality(block_data_index) +
                2 * relation_cache.c(block_data_index) * h_B +
                relation_cache.c_S(block_data_index);
            h_squared = x_il * x_il * h_squared;
            square_coeff += h_squared;
            linear_coeff += (-relation_cache.e(block_data_index) * h_B -
                             relation_cache.e_q(block_data_index)) *
                            x_il;
          }
          linear_coeff += square_coeff * v_old;
          square_coeff *= hyper.alpha;
          linear_coeff *= hyper.alpha;
          square_coeff += hyper.lambda_V(g, factor_index);
          linear_coeff +=
              hyper.lambda_V(g, factor_index) * hyper.mu_V(g, factor_index);

          Real v_new = sample_normal(square_coeff, linear_coeff);
          Real delta = v_new - v_old;
          fm.V(offset + inner_feature_index, factor_index) = v_new;
          for (itertype it(relation_cache.X_t, inner_feature_index); it; ++it) {
            auto block_data_index = it.col();
            const Real x_il = it.value();
            auto h_B = relation_cache.q(block_data_index) - x_il * v_old;
            relation_cache.q(block_data_index) += delta * x_il;
            relation_cache.q_S(block_data_index) +=
                delta * (v_new + v_old) * x_il * x_il;

            relation_cache.e(block_data_index) +=
                x_il * delta *
                (h_B * relation_cache.cardinality(block_data_index) +
                 relation_cache.c(block_data_index));
            relation_cache.e_q(block_data_index) +=
                x_il * delta *
                (h_B * relation_cache.c(block_data_index) +
                 relation_cache.c_S(block_data_index));
          }
        }
        // resync
        train_data_index = 0;
        for (auto i : relation_data.original_to_block) {
          e_train(train_data_index) +=
              (q_train(train_data_index) * relation_cache.q(i) +
               0.5 * relation_cache.q(i) * relation_cache.q(i) -
               0.5 * relation_cache.q_S(i));
          q_train(train_data_index) += relation_cache.q(i);
          train_data_index++;
        }
        offset += relation_data.feature_size;
      }
    }

    // relations
  }

  inline void sample_e(FMType &fm, HyperType &hyper) {
    fm.predict_score_write_target(e_train, X, relations);

    if (learning_config.task_type == TASKTYPE::REGRESSION) {
      e_train -= y;
    } else if (learning_config.task_type == TASKTYPE::CLASSIFICATION) {
      Real zero = static_cast<Real>(0);
      Real std = static_cast<Real>(1); // 1/ sqrt(hyper.alpha);
      for (int train_data_index = 0; train_data_index < X.rows();
           train_data_index++) {
        Real gt = y(train_data_index);
        Real pred = e_train(train_data_index);
        Real n;
        if (gt > 0) {
          n = sample_truncated_normal_left(gen_, pred, std, zero);
        } else {
          n = sample_truncated_normal_right(gen_, pred, std, zero);
        }
        e_train(train_data_index) -= n;
      }
    }
  }

  const int random_seed;
  mt19937 gen_;
};

} // namespace myFM