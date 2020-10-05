#pragma once

#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "FM.hpp"
#include "FMLearningConfig.hpp"
#include "HyperParams.hpp"
#include "OProbitSampler.hpp"
#include "definitions.hpp"
#include "predictor.hpp"
#include "util.hpp"

#include "BaseFMTrainer.hpp"

namespace myFM {

namespace variational {

template <typename Real>
struct VariationalFMHyperParameters : public FMHyperParameters<Real> {
public:
  using BaseType = FMHyperParameters<Real>;
  using Vector = typename BaseType::Vector;
  using DenseMatrix = typename BaseType::DenseMatrix;
  Real alpha_rate;
  Vector mu_w_var;
  Vector lambda_w_rate;

  DenseMatrix mu_V_var;
  DenseMatrix lambda_V_rate;
  inline VariationalFMHyperParameters(size_t n_factors, size_t n_groups)
      : BaseType(n_factors, n_groups), mu_w_var(n_groups),
        lambda_w_rate(n_groups), mu_V_var(n_groups, n_factors),
        lambda_V_rate(n_groups, n_factors) {}

  inline VariationalFMHyperParameters(size_t n_factors)
      : VariationalFMHyperParameters(n_factors, 1) {}

  inline VariationalFMHyperParameters(
      Real alpha, Real alpha_rate, const Vector &mu_w, const Vector &mu_w_var,
      const Vector &lambda_w, const Vector &lambda_w_rate,
      const DenseMatrix &mu_V, const DenseMatrix &mu_V_var,
      const DenseMatrix &lambda_V, const DenseMatrix &lambda_V_rate)
      : BaseType(alpha, mu_w, lambda_w, mu_V, lambda_V), alpha_rate(alpha_rate),
        mu_w_var(mu_w_var), lambda_w_rate(lambda_w_rate), mu_V_var(mu_V_var),
        lambda_V_rate(lambda_V_rate) {}

  inline VariationalFMHyperParameters(const VariationalFMHyperParameters &other)
      : BaseType(other.alpha, other.mu_w, other.lambda_w, other.mu_V,
                 other.lambda_V),
        alpha_rate(other.alpha_rate), mu_w_var(other.mu_w_var),
        lambda_w_rate(other.lambda_w_rate), mu_V_var(other.mu_V_var),
        lambda_V_rate(other.lambda_V_rate) {}
};

template <typename Real> struct VariationalFM : public FM<Real> {
  using BaseType = FM<Real>;
  using typename BaseType::DenseMatrix;
  using typename BaseType::SparseMatrix;
  using typename BaseType::Vector;

  using BaseType::BaseType;
  inline void initialize_weight(int n_features, Real init_std, mt19937 &gen) {
    this->initialized = false;
    normal_distribution<Real> nd;

    auto get_rand = [&gen, &nd, init_std](Real dummy) {
      return nd(gen) * init_std;
    };
    this->V = DenseMatrix{n_features, this->n_factors}.unaryExpr(get_rand);
    this->w = Vector{n_features}.unaryExpr(get_rand);
    this->w0 = get_rand(1);
    this->V_var = DenseMatrix{n_features, this->n_factors};
    this->V_var.array() = init_std * init_std;
    this->w_var = Vector{n_features};
    this->w_var.array() = init_std * init_std;
    this->w0 = 1;

    this->initialized = true;
  }

  inline VariationalFM(const VariationalFM &other)
      : BaseType(other.w0, other.w, other.V), w0_var(other.w0_var),
        w_var(other.w_var), V_var(other.V_var) {}

  inline VariationalFM(Real w0, Real w0_var, const Vector &w,
                       const Vector &w_var, const DenseMatrix &V,
                       const DenseMatrix &V_var)
      : BaseType(w0, w, V), w0_var(w0_var), w_var(w_var), V_var(V_var) {}

  inline VariationalFM(Real w0, Real w0_var, const Vector &w,
                       const Vector &w_var, const DenseMatrix &V,
                       const DenseMatrix &V_var,
                       const vector<Vector> &cutpoints)
      : BaseType(w0, w, V, cutpoints), w0_var(w0_var), w_var(w_var),
        V_var(V_var) {}

  Real w0_var;
  Vector w_var;
  DenseMatrix V_var;
};

template <typename Real>
using VariationalPredictor = Predictor<Real, VariationalFM<Real>>;

template <typename Real>
struct VariationalRelationWiseCache
    : public relational::RelationWiseCache<Real> {
  using BaseType = relational::RelationWiseCache<Real>;
  using Vector = typename BaseType::Vector;
  using RelationBlock = relational::RelationBlock<Real>;
  inline VariationalRelationWiseCache(const RelationBlock &source)
      : BaseType(source), x2s(source.X.rows()), x3sv(source.X.rows()),
        x4s2(source.X.rows()), x4sv2(source.X.rows()) {}
  Vector x2s;
  Vector x3sv;
  Vector x4s2;
  Vector x4sv2;
};

template <typename RealType>
struct VariationalFMTrainer
    : public BaseFMTrainer<RealType, class VariationalFMTrainer<RealType>,
                           VariationalFM<RealType>,
                           VariationalFMHyperParameters<RealType>,
                           VariationalRelationWiseCache<RealType>> {

  typedef RealType Real;

  typedef VariationalFM<Real> FMType;
  typedef VariationalFMHyperParameters<Real> HyperType;
  typedef VariationalRelationWiseCache<Real> RelationWiseCache;

  typedef BaseFMTrainer<RealType, VariationalFMTrainer<RealType>, FMType,
                        HyperType, RelationWiseCache>
      BaseType;

  typedef typename BaseType::RelationBlock RelationBlock;
  typedef typename FMType::Vector Vector;
  typedef typename FMType::DenseMatrix DenseMatrix;
  typedef typename FMType::SparseMatrix SparseMatrix;
  typedef OprobitSampler<Real> OprobitSamplerType;
  typedef FMLearningConfig<Real> Config;

  using TASKTYPE = typename BaseType::TASKTYPE;

public:
  Vector x2s;
  Vector x3sv;

  inline VariationalFMTrainer(const SparseMatrix &X,
                              const vector<RelationBlock> &relations,
                              const Vector &y, int random_seed,
                              Config learning_config)
      : BaseType(X, relations, y, random_seed, learning_config), x2s(X.rows()),
        x3sv(X.rows()) {}

  inline pair<VariationalPredictor<Real>, HyperType> learn(FMType &fm,
                                                           HyperType &hyper) {
    return learn_with_callback(
        fm, hyper, [](int i, FMType *fm, HyperType *hyper) { return false; });
  }

  /**
   *  Main routine for Variational update.
   */
  inline pair<VariationalPredictor<Real>, HyperType>
  learn_with_callback(FMType &fm, HyperType &hyper,
                      std::function<bool(int, FMType *, HyperType *)> cb) {
    initialize_hyper(fm, hyper);
    initialize_e(fm, hyper);

    for (int iteration = 0; iteration < this->learning_config.n_iter;
         iteration++) {
      this->update_all(fm, hyper);

      bool should_stop = cb(iteration, &fm, &hyper);
      if (should_stop) {
        break;
      }
    }
    pair<VariationalPredictor<Real>, HyperType> result{
        {static_cast<size_t>(fm.n_factors), this->dim_all,
         this->learning_config.task_type},
        hyper};
    result.first.samples.emplace_back(fm);
    return result;
  }

  inline void initialize_hyper(FMType &fm, HyperType &hyper) {
    hyper.alpha = static_cast<Real>(1);
    hyper.alpha_rate = this->n_train * .5;

    hyper.mu_w.array() = static_cast<Real>(0);
    hyper.mu_w_var.array() = 1;
    hyper.lambda_w.array() = static_cast<Real>(1e-5);
    hyper.lambda_w_rate.array() = 1;

    hyper.mu_V.array() = static_cast<Real>(0);
    hyper.mu_V_var.array() = 1;
    hyper.lambda_V.array() = static_cast<Real>(1e-5);
    hyper.lambda_V_rate.array() = 1;
  }

  inline void initialize_e(FMType &fm, const HyperType &hyper) {
    fm.predict_score_write_target(this->e_train, this->X, this->relations);
    this->e_train -= this->y;
  }

  // sample from quad x ^2 - 2 * first x + ... = quad (x - first / quad) ^2
  inline Real sample_normal(const Real &quad, const Real &first) {
    return (first / quad);
  }

  inline void update_alpha(FMType &fm, HyperType &hyper) {
    // If the task is classification, take alpha = 1.
    using itertype = typename SparseMatrix::InnerIterator;

    if ((this->learning_config.task_type == TASKTYPE::CLASSIFICATION)) {
      hyper.alpha = static_cast<Real>(1);
      return;
    }

    Real e_all = fm.w0_var * fm.w0_var * this->n_train +
                 this->e_train.array().square().sum();
    e_all += (this->X.cwiseAbs2() * fm.w_var).sum();

    for (int r = 0; r < fm.n_factors; r++) {
      const Vector &V_ref = fm.V.col(r);
      const Vector &V_var_ref = fm.V_var.col(r);
      for (int train_index = 0; train_index < this->n_train; train_index++) {
        Real x2s = 0;
        Real x3sv = 0;
        Real x4s2 = 0;
        Real x4sv2 = 0;
        Real q = 0;
        for (itertype it(this->X, train_index); it; ++it) {
          Real x = it.value();
          Real x2 = x * x;
          Real x4 = x2 * x2;
          int col = it.col();
          q += x * V_ref(col);
          x2s += x * V_var_ref(col);
          x3sv += x2 * x * V_var_ref(col) * V_ref(col);
          x4s2 += x4 * V_var_ref(col) * V_var_ref(col);
          x4sv2 += x4 * V_var_ref(col) * V_ref(col) * V_ref(col);
        }
        size_t offset = this->X.cols();
        for (size_t relation_index = 0; relation_index < this->relations.size();
             relation_index++) {
          throw std::runtime_error("not implemented");
        }
        e_all +=
            (q * q * x2s + 0.5 * x2s * x2s - 2 * x3sv * q - 0.5 * x4s2 + x4sv2);
      }
    }

    Real exponent = (this->learning_config.alpha_0 + this->X.rows()) / 2;
    Real rate = (this->learning_config.beta_0 + e_all) / 2;
    Real new_alpha = gamma_distribution<Real>(exponent, 1 / rate)(this->gen_);
    hyper.alpha = new_alpha;
    hyper.alpha_rate = rate;
  }

  /*
 The sampling method for both $\lambda _g ^{(w)}$ and $\lambda _{g,r} ^{(v)}$.
 */
  inline void update_lambda_generic(const Vector &mu, const Vector &mu_var,
                                    Eigen::Ref<Vector> lambda,
                                    Eigen::Ref<Vector> lambda_rate,
                                    const Vector &weight,
                                    const Vector &weight_var) {
    const vector<vector<size_t>> &group_vs_feature_index =
        this->learning_config.group_vs_feature_index();
    size_t group_index = 0;
    for (const auto &group_feature_indices : group_vs_feature_index) {
      Real mean = mu(group_index);
      Real alpha = this->learning_config.alpha_0 + group_feature_indices.size();
      Real beta = this->learning_config.beta_0;

      for (auto feature_index : group_feature_indices) {
        auto dev = weight(feature_index) - mean;
        beta += dev * dev + mu_var(group_index) + weight_var(feature_index);
      }
      Real new_lambda = alpha / beta;
      Real new_rate = beta / 2;
      lambda(group_index) = new_lambda;
      lambda_rate(group_index) = new_rate;
      group_index++;
    }
  }

private:
  /*
 The sampling method for both $\mu _g ^{(w)}$ and $\mu _{g,r} ^{(v)}$.
 */
  inline void update_mu_generic(Eigen::Ref<Vector> mu,
                                Eigen::Ref<Vector> mu_var, const Vector &lambda,
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
      mu_var(group_index) = 1 / square;
      group_index++;
    }
  }

public:
  inline void update_lambda_w(FMType &fm, HyperType &hyper) {
    this->update_lambda_generic(hyper.mu_w, hyper.mu_w_var, hyper.lambda_w,
                                hyper.lambda_w_rate, fm.w, fm.w_var);
  }

  inline void update_mu_w(FMType &fm, HyperType &hyper) {
    this->update_mu_generic(hyper.mu_w, hyper.mu_w_var, hyper.lambda_w, fm.w);
  }

  inline void update_lambda_V(FMType &fm, HyperType &hyper) {
    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      this->update_lambda_generic(
          hyper.mu_V.col(factor_index), hyper.mu_V_var.col(factor_index),
          hyper.lambda_V.col(factor_index),
          hyper.lambda_V_rate.col(factor_index), fm.V.col(factor_index),
          fm.V_var.col(factor_index));
    }
  }

  inline void update_mu_V(FMType &fm, HyperType &hyper) {
    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      this->update_mu_generic(
          hyper.mu_V.col(factor_index), hyper.mu_V_var.col(factor_index),
          hyper.lambda_V.col(factor_index), fm.V.col(factor_index));
    }
  }

  inline void update_w0(FMType &fm, HyperType &hyper) {
    /*
    if ( (learning_config.task_type == TASKTYPE::ORDERED) &&
    (learning_config.cutpoint_sample_method ==
    CutpointSampleMethod::AlbertChib01)) { fm.w0 = 0; return;
    }
    */
    Real w0_lin_term = hyper.alpha * (fm.w0 - this->e_train.array()).sum();
    Real w0_quad_term =
        hyper.alpha * this->n_train + this->learning_config.reg_0;
    Real w0_new = w0_lin_term / w0_quad_term;
    this->e_train.array() += (w0_new - fm.w0);
    fm.w0 = w0_new;
    fm.w0_var = 1 / w0_quad_term;
  }

  inline void update_w(FMType &fm, HyperType &hyper) {
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
      fm.w_var(feature_index) = 1 / square_term;
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
        fm.w_var(offset + inner_feature_index) = 1 / square_term;

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
      this->q_train.array() = 0;
      this->x2s.array() = 0;
      this->x3sv.array() = 0;
      const auto &V_ref = fm.V.col(factor_index);
      const auto &V_var_ref = fm.V_var.col(factor_index);
      for (int train_index = 0; train_index < this->n_train; train_index++) {
        for (itertype it(this->X, train_index); it; ++it) {
          Real x = it.value();
          int col = it.col();
          this->q_train(train_index) += x * V_ref(col);
          this->x2s(train_index) += x * x * V_var_ref(col);
          this->x3sv(train_index) += x * x * x * V_var_ref(col) * V_ref(col);
        }
      }
      // compute contribution of blocks
      {
        // initialize block q caches
        size_t offset = this->X.cols();
        for (size_t relation_index = 0; relation_index < this->relations.size();
             relation_index++) {

          throw std::runtime_error("not implemented");
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
        Real v_var_old = fm.V_var(feature_index, factor_index);

        Real square_coeff = 0;
        Real linear_coeff = 0;
        Real square_coeff_var = 0;
        Real linear_coeff_var = 0;

        for (itertype it(this->X_t, feature_index); it; ++it) {
          auto x = it.value();
          auto train_data_index = it.col();
          auto h = x * (this->q_train(train_data_index) - x * v_old);
          Real x2s = this->x2s(train_data_index);
          Real x3sv = this->x3sv(train_data_index);
          x2s -= x * x * v_var_old;
          x3sv -= x * x * x * v_var_old * v_old;
          square_coeff += h * h;
          linear_coeff += (-this->e_train(train_data_index)) * h;
          square_coeff_var += x2s * x * x;
          linear_coeff_var += h * x2s - x * x3sv;
        }
        linear_coeff += square_coeff * v_old;
        linear_coeff -= linear_coeff_var;
        square_coeff += square_coeff_var;

        square_coeff *= hyper.alpha;
        linear_coeff *= hyper.alpha;

        square_coeff += hyper.lambda_V(g, factor_index);
        linear_coeff +=
            hyper.lambda_V(g, factor_index) * hyper.mu_V(g, factor_index);

        Real v_new = sample_normal(square_coeff, linear_coeff);
        Real v_var_new = 1 / square_coeff;
        fm.V(feature_index, factor_index) = v_new;
        fm.V_var(feature_index, factor_index) = v_var_new;
        for (itertype it(this->X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto x = it.value();
          auto h = x * (this->q_train(train_data_index) - x * v_old);
          this->q_train(train_data_index) += x * (v_new - v_old);
          this->e_train(train_data_index) += h * (v_new - v_old);

          this->x2s(train_data_index) += x * x * (v_var_new - v_var_old);
          this->x3sv(train_data_index) +=
              x * x * x * (v_var_new * v_new - v_var_old * v_old);
        }
      }

      // draw v for relations
      size_t offset = this->X.cols();
      // initialize caches
      for (size_t relation_index = 0; relation_index < this->relations.size();
           relation_index++) {
        throw std::runtime_error("not implemented");

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

  inline void update_e(FMType &fm, HyperType &hyper) {
    fm.predict_score_write_target(this->e_train, this->X, this->relations);

    if (this->learning_config.task_type == TASKTYPE::REGRESSION) {
      this->e_train -= this->y;
    } else if (this->learning_config.task_type == TASKTYPE::CLASSIFICATION) {
      throw std::runtime_error("not implemented");

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
      throw std::runtime_error("not implemented");
    }
  }
};

} // namespace variational

} // namespace myFM