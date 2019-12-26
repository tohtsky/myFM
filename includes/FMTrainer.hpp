#ifndef MYFM_FM_TRAIN_HPP
#define MYFM_FM_TRAIN_HPP

#include <cmath>
#include <memory>
#include <set>
#include <tuple>
#include <string>

#include "FM.hpp"
#include "FMLearningConfig.hpp"
#include "HyperParams.hpp"
#include "util.hpp"
namespace myFM {

template <typename Real> struct FMTrainer {

  typedef FM<Real> FMType;
  typedef FMHyperParameters<Real> HyperType;
  typedef typename FMType::Vector Vector;
  typedef typename FMType::DenseMatrix DenseMatrix;
  typedef typename FMType::SparseMatrix SparseMatrix;

  typedef FMLearningConfig<Real> Config;
  typedef typename Config::TASKTYPE TASKTYPE;

  SparseMatrix X;
  SparseMatrix X_t; // transposed

  const Vector y;

  const int n_train;

  Vector e_train;
  Vector q_train;

  const Config learning_config;

  size_t n_nan_occured = 0;

  inline FMTrainer(const SparseMatrix &X, const Vector &y, int random_seed)
      : FMTrainer(X, y, random_seed,
                  Config::Builder::get_default_config(X.cols())) {}

  inline FMTrainer(const SparseMatrix &X, const Vector &y, int random_seed,
                   Config learning_config)
      : X(X), X_t(X.transpose()), y(y), n_train(X.rows()), e_train(X.rows()),
        q_train(X.rows()), learning_config(learning_config),
        random_seed(random_seed), gen_(random_seed) {
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
    fm.initialize_weight(X.cols(), init_std, gen_);
    return fm;
  }

  inline FMHyperParameters<Real> create_Hyper(size_t rank) {
    return FMHyperParameters<Real>{rank, learning_config.get_n_groups()};
  }

  inline pair<vector<FMType>, vector<HyperType>> learn(FMType &fm, HyperType &hyper) {
    return learn_with_callback(fm, hyper, [](int i, const FMType & fm, const HyperType & hyper){
      cout << "iteration = " << i << endl;
      return false; 
    });
  }

  /**
   *  Main routine for Gibbs sampling.
   */
  inline pair<vector<FMType>, vector<HyperType>>
  learn_with_callback(FMType &fm, HyperType &hyper, std::function<bool(int, const FMType&, const HyperType&)> cb) {
    initialize_hyper(fm, hyper);
    initialize_e(fm, hyper);
    vector<FMType> result_fm;
    vector<HyperType> result_hyper;
    for (int mcmc_iteration = 0; mcmc_iteration < learning_config.n_iter;
         mcmc_iteration++) {
      mcmc_step(fm, hyper);
      if (learning_config.n_iter <=
          (mcmc_iteration + learning_config.n_kept_samples)) {
        result_fm.emplace_back(fm);
      }
      // for tracing
      result_hyper.emplace_back(hyper);

      bool should_stop = cb(mcmc_iteration, fm, hyper);
      if (should_stop){
        break;
      }
    }
    return {result_fm, result_hyper};
  }

  inline void initialize_hyper(FMType &fm, HyperType &hyper) {
    hyper.alpha = static_cast<Real>(1);

    hyper.mu_w.array() = static_cast<Real>(0);
    hyper.lambda_w.array() = static_cast<Real>(1e-5);

    hyper.mu_V.array() = static_cast<Real>(0);
    hyper.lambda_V.array() = static_cast<Real>(1e-5);
  }

  inline void initialize_e(const FMType &fm, const HyperType &h) {
    e_train = fm.predict_score(X);
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
      Real new_lambda =
          gamma_distribution<Real>(alpha / 2, 2 / beta)(gen_);
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
                      hyper.lambda_V.col(factor_index), fm.V.col(factor_index));
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
    for (int feature_index = 0; feature_index < fm.w.rows(); feature_index++) {
      int group = learning_config.group_index()[feature_index];
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
  }

  inline void sample_V(FMType &fm, HyperType &hyper) {
    Vector cache(X.rows());
    Vector coeffCache(X.rows());
    using itertype = typename SparseMatrix::InnerIterator;

    for (int factor_index = 0; factor_index < fm.n_factors; factor_index++) {
      cache = X * fm.V.col(factor_index);

      for (int feature_index = 0; feature_index < X_t.rows(); feature_index++) {
        auto g = learning_config.group_index()[feature_index];
        Real v_old = fm.V(feature_index, factor_index);

        Real square_coeff = 0;
        Real linear_coeff = 0;

        for (itertype it(X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto h = it.value() * (cache(train_data_index) - it.value() * v_old);
          square_coeff += h * h;
          linear_coeff += (-e_train(train_data_index) + v_old * h) * h;
        }

        square_coeff *= hyper.alpha;
        linear_coeff *= hyper.alpha;

        square_coeff += hyper.lambda_V(g, factor_index);
        linear_coeff +=
            hyper.lambda_V(g, factor_index) * hyper.mu_V(g, factor_index);

        Real v_new = sample_normal(square_coeff, linear_coeff);
        fm.V(feature_index, factor_index) = v_new;
        for (itertype it(X_t, feature_index); it; ++it) {
          auto train_data_index = it.col();
          auto h = it.value() * (cache(train_data_index) - v_old);
          cache(train_data_index) += it.value() * (v_new - v_old);
          e_train(train_data_index) += h * (v_new - v_old);
        }
      }
    }
  }

  inline void sample_e(FMType &fm, HyperType &hyper) {
    e_train = fm.predict_score(X);

    if (learning_config.task_type == TASKTYPE::REGRESSION) {
      e_train -= y;
    } else if (learning_config.task_type == TASKTYPE::CLASSIFICATION) {

      for (int train_case_index = 0; train_case_index < X.rows();
           train_case_index++) {
        Real gt = y(train_case_index);
        Real pred = e_train(train_case_index);
        Real n;
        Real std = 1; // 1/ sqrt(hyper.alpha);
        Real zero = static_cast<Real>(0);
        if (gt > 0) {
          n = sample_truncated_normal_left(gen_, pred, std, zero);
        } else {
          n = sample_truncated_normal_right(gen_, pred, std, zero);
        }
        e_train(train_case_index) -= n;
      }
    }
  }

  const int random_seed;
  mt19937 gen_;
};

} // end namespace myFM
#endif
