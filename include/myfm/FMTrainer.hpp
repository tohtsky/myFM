#pragma once
#include <sstream>
#include <string>
#include <tuple>

#include "FM.hpp"
#include "FMLearningConfig.hpp"
#include "HyperParams.hpp"
#include "LearningHistory.hpp"
#include "OProbitSampler.hpp"
#include "definitions.hpp"
#include "predictor.hpp"
#include "util.hpp"

#include "BaseFMTrainer.hpp"

namespace myFM {

template <typename Real>
using GibbsRelationWiseCache = relational::RelationWiseCache<Real>;

template <typename RealType>
struct GibbsFMTrainer
    : public BaseFMTrainer<RealType, class GibbsFMTrainer<RealType>,
                           FM<RealType>, FMHyperParameters<RealType>,
                           GibbsRelationWiseCache<RealType>,
                           GibbsLearningHistory<RealType>> {

  typedef RealType Real;

  typedef FM<Real> FMType;
  typedef FMHyperParameters<Real> HyperType;
  typedef GibbsRelationWiseCache<Real> RelationWiseCache;
  typedef GibbsLearningHistory<Real> LearningHistory;

  typedef BaseFMTrainer<RealType, GibbsFMTrainer<RealType>, FMType, HyperType,
                        RelationWiseCache, LearningHistory>
      BaseType;

  typedef typename BaseType::RelationBlock RelationBlock;
  typedef typename FMType::Vector Vector;
  typedef typename FMType::DenseMatrix DenseMatrix;
  typedef typename FMType::SparseMatrix SparseMatrix;

  using Config = FMLearningConfig<Real>;
  using TASKTYPE = typename Config::TASKTYPE;

  typedef OprobitSampler<Real> OprobitSamplerType;

public:
  using BaseType::BaseType;

  /**
   *  Main routine for Gibbs sampling.
   */
  inline pair<Predictor<Real>, LearningHistory> learn_with_callback(
      FMType &fm, HyperType &hyper,
      std::function<bool(int, FMType *, HyperType *, LearningHistory *)> cb) {
    std::pair<Predictor<Real>, LearningHistory> result{
        {static_cast<size_t>(fm.n_factors), this->dim_all,
         this->learning_config.task_type},
        {},
    };
    initialize_hyper(fm, hyper);
    initialize_e(fm, hyper);

    result.first.samples.reserve(this->learning_config.n_kept_samples);
    for (int mcmc_iteration = 0; mcmc_iteration < this->learning_config.n_iter;
         mcmc_iteration++) {
      this->update_all(fm, hyper);
      if (this->learning_config.n_iter <=
          (mcmc_iteration + this->learning_config.n_kept_samples)) {
        result.first.samples.emplace_back(fm);
      }
      // for tracing
      result.second.hypers.emplace_back(hyper);

      bool should_stop = cb(mcmc_iteration, &fm, &hyper, &(result.second));
      if (should_stop) {
        break;
      }
    }
    for (OprobitSamplerType &cs : cutpoint_sampler) {
      result.second.n_mh_accept.emplace_back(cs.accept_count);
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

  inline void initialize_e(FMType &fm, const HyperType &hyper) {
    fm.predict_score_write_target(this->e_train, this->X, this->relations);
    if (this->learning_config.task_type == TASKTYPE::ORDERED) {
      int i = 0;
      for (auto &config : this->learning_config.cutpoint_groups()) {
        fm.cutpoints.emplace_back(config.first - 1);
        cutpoint_sampler.emplace_back(
            this->e_train, this->y, config.first, config.second, this->gen_,
            this->learning_config.reg_0, this->learning_config.nu_oprobit);
        cutpoint_sampler[i].start_sample();
        cutpoint_sampler[i].alpha_to_gamma(fm.cutpoints[i],
                                           cutpoint_sampler[i].alpha_now);

        cutpoint_sampler[i].sample_z_given_cutpoint();
        i++;
      }

      return;
    }
    this->e_train -= this->y;
  }

  // sample from quad x ^2 - 2 * first x + ... = quad (x - first / quad) ^2
  inline Real sample_normal(const Real &quad, const Real &first) {
    return (first / quad) +
           normal_distribution<Real>(0, 1)(this->gen_) / std::sqrt(quad);
  }

  inline void update_alpha(FMType &fm, HyperType &hyper) {
    // If the task is classification, take alpha = 1.
    if ((this->learning_config.task_type == TASKTYPE::CLASSIFICATION)) {
      hyper.alpha = static_cast<Real>(1);
      return;
    }
    if ((this->learning_config.task_type == TASKTYPE::ORDERED)) {
      hyper.alpha = static_cast<Real>(1);
      return;
    }

    Real e_all = this->e_train.array().square().sum();

    Real exponent = (this->learning_config.alpha_0 + this->X.rows()) / 2;
    Real variance = (this->learning_config.beta_0 + e_all) / 2;
    Real new_alpha =
        gamma_distribution<Real>(exponent, 1 / variance)(this->gen_);
    hyper.alpha = new_alpha;
  }

  /*
 The sampling method for both $\lambda _g ^{(w)}$ and $\lambda _{g,r} ^{(v)}$.
 */
  inline void update_lambda_generic(const Vector &mu, Eigen::Ref<Vector> lambda,
                                    const Vector &weight) {
    const vector<vector<size_t>> &group_vs_feature_index =
        this->learning_config.group_vs_feature_index();
    size_t group_index = 0;
    for (const auto &group_feature_indices : group_vs_feature_index) {
      Real mean = mu(group_index);
      Real alpha = this->learning_config.alpha_0 + group_feature_indices.size();
      Real beta = this->learning_config.beta_0;

      for (auto feature_index : group_feature_indices) {
        auto dev = weight(feature_index) - mean;
        beta += dev * dev;
      }
      Real new_lambda =
          gamma_distribution<Real>(alpha / 2, 2 / beta)(this->gen_);
      lambda(group_index) = new_lambda;
      group_index++;
    }
  }

  /*
 The sampling method for both $\mu _g ^{(w)}$ and $\mu _{g,r} ^{(v)}$.
 */
  inline void update_mu_generic(Eigen::Ref<Vector> mu, const Vector &lambda,
                                const Vector &weight) {
    const vector<vector<size_t>> &group_vs_feature_index =
        this->learning_config.group_vs_feature_index();
    size_t group_index = 0;
    for (const auto &group_feature_indices : group_vs_feature_index) {
      size_t n_feature_in_groups = group_feature_indices.size();
      Real square = lambda(group_index) *
                    (this->learning_config.gamma_0 + n_feature_in_groups);
      Real linear = this->learning_config.gamma_0 * this->learning_config.mu_0;
      for (auto &f : group_feature_indices) {
        linear += weight(f);
      }
      linear *= lambda(group_index);
      Real new_mu = sample_normal(square, linear);
      mu(group_index) = new_mu;
      group_index++;
    }
  }

  inline void update_lambda_w(FMType &fm, HyperType &hyper) {
    update_lambda_generic(hyper.mu_w, hyper.lambda_w, fm.w);
  }

  inline void update_mu_w(FMType &fm, HyperType &hyper) {
    update_mu_generic(hyper.mu_w, hyper.lambda_w, fm.w);
  }

  inline void update_lambda_V(FMType &fm, HyperType &hyper) {
    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      update_lambda_generic(hyper.mu_V.col(factor_index),
                            hyper.lambda_V.col(factor_index),
                            fm.V.col(factor_index));
    }
  }

  inline void update_mu_V(FMType &fm, HyperType &hyper) {
    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      update_mu_generic(hyper.mu_V.col(factor_index),
                        hyper.lambda_V.col(factor_index),
                        fm.V.col(factor_index));
    }
  }

  inline void update_w0(FMType &fm, HyperType &hyper) {
    if (!this->learning_config.fit_w0) {
      fm.w0 = 0;
      return;
    }
    Real w0_lin_term = hyper.alpha * (fm.w0 - this->e_train.array()).sum();
    Real w0_quad_term =
        hyper.alpha * this->n_train + this->learning_config.reg_0;
    Real w0_new = sample_normal(w0_quad_term, w0_lin_term);
    this->e_train.array() += (w0_new - fm.w0);
    fm.w0 = w0_new;
  }

  inline void update_w(FMType &fm, HyperType &hyper) {
    if (!this->learning_config.fit_linear) {
      fm.w.array() = 0;
      return;
    }
    // main table
    for (int feature_index = 0; feature_index < this->X.cols();
         feature_index++) {
      int group = this->learning_config.group_index(feature_index);

      const Real w_old = fm.w(feature_index);
      this->e_train.array() -= this->X_t.row(feature_index) * w_old;
      Real lambda = hyper.lambda_w(group);
      Real mu = hyper.mu_w(group);
      Real square_term =
          lambda + hyper.alpha * this->X_t.row(feature_index).cwiseAbs2().sum();
      Real linear_term =
          -hyper.alpha * this->X_t.row(feature_index) * this->e_train +
          lambda * mu;

      Real w_new = sample_normal(square_term, linear_term);
      this->e_train.array() += this->X_t.row(feature_index) * w_new;
      fm.w(feature_index) = w_new;
    }

    // relational blocks
    size_t offset = this->X.cols();
    for (size_t relation_index = 0; relation_index < this->relations.size();
         relation_index++) {
      RelationBlock &relation_data = this->relations[relation_index];
      RelationWiseCache &relation_cache = this->relation_caches[relation_index];
      relation_cache.e.array() = 0;
      relation_cache.q.array() = 0;

      relation_cache.q =
          relation_data.X * fm.w.segment(offset, relation_data.feature_size);

      {
        size_t train_data_index = 0;
        for (auto i : relation_data.original_to_block) {
          relation_cache.e(i) += this->e_train(train_data_index);
          this->e_train(train_data_index++) -=
              relation_cache.q(i); // un-synchronize
        }
      }
      for (size_t inner_feature_index = 0;
           inner_feature_index < relation_data.feature_size;
           inner_feature_index++) {
        int group =
            this->learning_config.group_index(offset + inner_feature_index);
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
          this->e_train(train_data_index++) += relation_cache.q(i); // un-sync
        }
      }
      offset += relation_data.feature_size;
    }
  }

  inline void update_V(FMType &fm, HyperType &hyper) {
    using itertype = typename SparseMatrix::InnerIterator;

    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      this->q_train = this->X * fm.V.col(factor_index).head(this->X.cols());

      // compute contribution of blocks
      {
        // initialize block q caches
        size_t offset = this->X.cols();
        for (size_t relation_index = 0; relation_index < this->relations.size();
             relation_index++) {
          const RelationBlock &relation_data = this->relations[relation_index];
          RelationWiseCache &relation_cache =
              this->relation_caches[relation_index];
          relation_cache.q = relation_data.X *
                             (fm.V.col(factor_index)
                                  .segment(offset, relation_data.feature_size));
          size_t train_data_index = 0;
          for (auto i : relation_data.original_to_block) {
            this->q_train(train_data_index++) += relation_cache.q(i);
          }
          offset += relation_data.feature_size;
        }
      }

      // main table
      for (int feature_index = 0; feature_index < this->X_t.rows();
           feature_index++) {
        auto g = this->learning_config.group_index(feature_index);
        Real v_old = fm.V(feature_index, factor_index);

        Real square_coeff = 0;
        Real linear_coeff = 0;

        for (itertype it(this->X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto h = it.value() *
                   (this->q_train(train_data_index) - it.value() * v_old);
          square_coeff += h * h;
          linear_coeff += (-this->e_train(train_data_index)) * h;
        }
        linear_coeff += square_coeff * v_old;

        square_coeff *= hyper.alpha;
        linear_coeff *= hyper.alpha;

        square_coeff += hyper.lambda_V(g, factor_index);
        linear_coeff +=
            hyper.lambda_V(g, factor_index) * hyper.mu_V(g, factor_index);

        Real v_new = sample_normal(square_coeff, linear_coeff);
        fm.V(feature_index, factor_index) = v_new;
        for (itertype it(this->X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto h = it.value() *
                   (this->q_train(train_data_index) - it.value() * v_old);
          this->q_train(train_data_index) += it.value() * (v_new - v_old);
          this->e_train(train_data_index) += h * (v_new - v_old);
        }
      }

      // draw v for relations
      size_t offset = this->X.cols();
      // initialize caches
      for (size_t relation_index = 0; relation_index < this->relations.size();
           relation_index++) {
        const RelationBlock &relation_data = this->relations[relation_index];
        RelationWiseCache &relation_cache =
            this->relation_caches[relation_index];

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
          Real temp = (this->q_train(train_data_index) - relation_cache.q(i));
          relation_cache.c(i) += temp;
          relation_cache.c_S(i) += temp * temp;
          relation_cache.e(i) += this->e_train(train_data_index);
          relation_cache.e_q(i) += this->e_train(train_data_index) * temp;
          // un-synchronization of q and e
          this->q_train(train_data_index) -= relation_cache.q(i);
          // q_B
          // 1/ 2 ( (q_B + q_other) **2 - (q_B_S + other) )
          // q_B * q_other + 0.5 q_B **2 - 0.5 * q_B_S
          this->e_train(train_data_index) -=
              (this->q_train(train_data_index) * relation_cache.q(i) +
               0.5 * relation_cache.q(i) * relation_cache.q(i) -
               0.5 * relation_cache.q_S(i));
          train_data_index++;
        }
        // Initialized block-wise caches.
        for (size_t inner_feature_index = 0;
             inner_feature_index < relation_data.feature_size;
             inner_feature_index++) {
          auto g =
              this->learning_config.group_index(offset + inner_feature_index);
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
          this->e_train(train_data_index) +=
              (this->q_train(train_data_index) * relation_cache.q(i) +
               0.5 * relation_cache.q(i) * relation_cache.q(i) -
               0.5 * relation_cache.q_S(i));
          this->q_train(train_data_index) += relation_cache.q(i);
          train_data_index++;
        }
        offset += relation_data.feature_size;
      }
    }

    // relations
  }

  inline void sample_cutpoint_z_marginalized(FMType &fm) {
    cutpoint_sampler->step();
    cutpoint_sampler->alpha_to_gamma(fm.cutpoint, cutpoint_sampler->alpha_now);
  }

  inline void update_e(FMType &fm, HyperType &hyper) {
    fm.predict_score_write_target(this->e_train, this->X, this->relations);

    if (this->learning_config.task_type == TASKTYPE::REGRESSION) {
      this->e_train -= this->y;
    } else if (this->learning_config.task_type == TASKTYPE::CLASSIFICATION) {
      Real zero = static_cast<Real>(0);
      Real std = static_cast<Real>(1); // 1/ sqrt(hyper.alpha);
      for (int train_data_index = 0; train_data_index < this->X.rows();
           train_data_index++) {
        Real gt = this->y(train_data_index);
        Real pred = this->e_train(train_data_index);
        Real n;
        if (gt > 0) {
          n = sample_truncated_normal_left(this->gen_, pred, std, zero);
        } else {
          n = sample_truncated_normal_right(this->gen_, pred, std, zero);
        }
        this->e_train(train_data_index) -= n;
      }
    } else if (this->learning_config.task_type == TASKTYPE::ORDERED) {
      int i = 0;
      for (auto &sampler_ : cutpoint_sampler) {
        sampler_.step();
        sampler_.alpha_to_gamma(fm.cutpoints[i], sampler_.alpha_now);
        sampler_.sample_z_given_cutpoint();
        i++;
      }
    }
  }
  std::vector<OprobitSamplerType> cutpoint_sampler;
};

} // namespace myFM