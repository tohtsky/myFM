#pragma once

#include "Faddeeva/Faddeeva.hh"
#include "util.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <random>

#include <fstream>
#include <sstream>

namespace myFM {
template <typename Real> struct OprobitSampler {

  using DenseVector = types::Vector<Real>;
  using DenseMatrix = types::DenseMatrix<Real>;
  using IntVector = Eigen::Matrix<int, Eigen::Dynamic, 1>;
  static constexpr Real SQRT2 = 1.4142135623730951;
  static constexpr Real SQRTPI = 1.7724538509055159;
  static constexpr Real SQRT2PI = SQRT2 * SQRTPI;
  static constexpr Real PI = 3.141592653589793;

  OprobitSampler(DenseVector &x, const DenseVector &y, int K,
                 const std::vector<size_t> &indices, std::mt19937 &rng,
                 Real reg, Real nu)
      : x_(x), y_(y), K(K), indices_(indices), reg(reg), nu(nu), rng(rng),
        zmins(K), zmaxs(K), histogram(K), accept_count(0) {
    this->alpha_now = DenseVector::Zero(K - 1);
    this->gamma_now = DenseVector::Zero(K - 1);
    this->alpha_to_gamma(gamma_now, alpha_now);
    this->H = DenseMatrix::Zero(K - 1, K - 1);
    for (auto i : indices_) {
      int y_label = static_cast<int>(y_(i));
      if (std::abs(y_label - y(i)) > 1e-3) {
        throw std::invalid_argument("y has a floating-point element.");
      }
      if (y_label < 0) {
        throw std::invalid_argument("y has a negative element.");
      }
      if (y_label >= K) {
        std::stringstream ss;
        ss << "y[ " << i << "] is greater than " << (K - 1) << ".";
        throw std::invalid_argument(ss.str());
      }
      histogram[y_label]++;
    }
  }

  inline Real log_p_mvt(const DenseMatrix &SigmaInverse, const DenseVector mu,
                        Real nu, const DenseVector &x) {
    Real log_p = (x - mu).transpose() * SigmaInverse * (x - mu);
    return std::log(1 + log_p / nu) * (-nu - SigmaInverse.rows()) / 2;
  }

  inline DenseVector sample_mvt(const DenseMatrix &SigmaInverse, Real nu) {
    /*Sample From multivariate t-distribution*/
    DenseVector result(SigmaInverse.rows());
    std::normal_distribution<Real> base_dist(0, 1);
    std::gamma_distribution<Real> chi_gen(nu / 2);
    for (int i = 0; i < result.rows(); i++) {
      result(i) = base_dist(rng);
    }
    Eigen::LLT<DenseMatrix, Eigen::Upper> L(SigmaInverse);
    result = L.matrixU().solve(result);
    result /= std::sqrt(chi_gen(rng) * 2 / nu);
    if (fix_gamma0) {
      result(0) = 0;
    }
    return result;
  }

  static inline void jacobian_dgamma_dalpha(DenseMatrix &J,
                                            const DenseVector &alpha) {
    /*
    J_{ij} with i=> alpha, j=>gamma
    */
    J.array() = 0;
    J(0, 0) = 1;
    if (!fix_gamma0) {
      for (int j = 1; j < alpha.rows(); j++) {
        J(0, j) = 1;
      }
    }
    for (int i = 1; i < alpha.rows(); i++) {
      Real ed = std::exp(alpha(i));
      for (int j = i; j < alpha.rows(); j++) {
        J(i, j) = ed;
      }
    }
    // d f / d alpha_0 = (df / d gamma_i) (d gamma_i / d alpha_0 )
  }

  static inline void alpha_to_gamma(DenseVector &target,
                                    const DenseVector &alpha) {
    target(0) = alpha(0);
    for (int i = 1; i < alpha.rows(); i++) {
      target(i) = target(i - 1) + std::exp(alpha(i));
    }
  }

  static inline void gamma_to_alpha(DenseVector &target,
                                    const DenseVector &gamma) {
    target(0) = gamma(0);
    for (int i = 1; i < gamma.rows(); i++) {
      target(i) = std::log(gamma(i) - gamma(i - 1));
    }
  }

  static inline void safe_ldiff(Real x, Real y, Real &loss, Real &dx, Real &dy,
                                DenseMatrix *HessianTarget = nullptr,
                                int label = 0) {
    // assert(x >= y);
    Real denominator;
    Real exp_factor;
    if (y > 0) {
      // both positive
      // erfcy = erfc * exp( y**2 / 2)
      exp_factor = std::exp((y * y - x * x) / 2);
      denominator =
          Faddeeva::erfcx(y / SQRT2) - exp_factor * Faddeeva::erfcx(x / SQRT2);

      loss -= y * y / 2;
      loss += std::log(denominator / 2);
      dx += (2 / SQRT2PI) * exp_factor / denominator;
      dy -= (2 / SQRT2PI) / denominator;
      if (HessianTarget != nullptr) {
        (*HessianTarget)(label, label) +=
            -(SQRT2PI * x * denominator * std::exp((y * y - x * x) / 2) +
              2 * std::exp(y * y - x * x)) /
            denominator / denominator / PI;
        (*HessianTarget)(label - 1, label - 1) +=
            (SQRT2PI * y * denominator - 2) / denominator / denominator / PI;
        Real off_diag =
            2 * std::exp((y * y - x * x) / 2) / PI / denominator / denominator;
        (*HessianTarget)(label, label - 1) += off_diag;
        (*HessianTarget)(label - 1, label) += off_diag;
      }
    } else if (x < 0) {
      // both negative
      loss -= x * x / 2;

      exp_factor = std::exp((x * x - y * y) / 2);
      denominator = Faddeeva::erfcx(-x / SQRT2) -
                    exp_factor * Faddeeva::erfcx(-y / SQRT2);
      loss += std::log(denominator / 2);
      dx += (2 / SQRT2PI) / denominator;
      dy -= (2 / SQRT2PI) * exp_factor / denominator;
      if (HessianTarget != nullptr) {
        (*HessianTarget)(label, label) +=
            -(SQRT2PI * x * denominator + 2) / PI / denominator / denominator;
        (*HessianTarget)(label - 1, label - 1) +=
            (SQRT2PI * y * exp_factor * denominator -
             2 * (exp_factor * exp_factor)) /
            PI / denominator / denominator;
        Real off_diag = 2 * exp_factor / PI / denominator / denominator;
        (*HessianTarget)(label, label - 1) += off_diag;
        (*HessianTarget)(label - 1, label) += off_diag;
      }
    } else {
      // x positive, y negative. safe to use erf
      denominator = Faddeeva::erf(x / SQRT2) - Faddeeva::erf(y / SQRT2);
      Real expxx = std::exp(-x * x / 2);
      Real expyy = std::exp(-y * y / 2);
      dx += 2 * expxx / denominator / SQRT2PI;
      dy -= 2 * expyy / denominator / SQRT2PI;
      loss += std::log(denominator / 2);
      if (HessianTarget != nullptr) {
        (*HessianTarget)(label, label) +=
            -(SQRT2PI * x * denominator * expxx + 2 * expxx * expxx) / PI /
            denominator / denominator;
        (*HessianTarget)(label - 1, label - 1) +=
            -(-SQRT2PI * y * denominator * expyy + 2 * expyy * expyy) / PI /
            denominator / denominator;
        Real off_diag = 2 * expxx * expyy / PI / denominator / denominator;
        (*HessianTarget)(label, label - 1) += off_diag;
        (*HessianTarget)(label - 1, label) += off_diag;
      }
    }
  }

  static inline void safe_lcdf(Real x, Real &loss, Real &dx,
                               DenseMatrix *HessianTarget = nullptr,
                               int label = 0) {
    Real denominator;
    Real exp_factor;
    if (x > 1) {
      exp_factor = std::exp(-x * x / 2);
      denominator = 1 + Faddeeva::erf(x / SQRT2);
      dx += (2 / SQRT2PI) * exp_factor / denominator;
      loss += std::log(denominator / 2);
      if (HessianTarget != nullptr) {
        (*HessianTarget)(label, label) +=
            -(SQRT2PI * x * denominator * exp_factor +
              2 * exp_factor * exp_factor) /
            PI / denominator / denominator;
      }
    } else {
      denominator = Faddeeva::erfcx(-x / SQRT2);
      dx += (2 / SQRT2PI) / denominator;
      loss -= x * x / 2;
      loss += std::log(denominator / 2);
      if (HessianTarget != nullptr) {
        (*HessianTarget)(label, label) +=
            -(SQRT2PI * x * denominator + 2) / PI / denominator / denominator;
      }
    }
  }

  inline void safe_lccdf(Real x, Real &loss, Real &dx,
                         DenseMatrix *HessianTarget, int label = 0) {
    Real denominator;
    if (x > -1) {
      denominator = Faddeeva::erfcx(x / SQRT2);
      dx -= (2 / SQRT2PI) / denominator;
      loss += std::log(denominator / 2);
      loss -= x * x / 2;
      if (HessianTarget != nullptr) {
        (*HessianTarget)(label - 1, label - 1) +=
            (SQRT2PI * x * denominator - 2) / denominator / denominator / PI;
      }
    } else {
      // safe to use erf
      denominator = 1 - Faddeeva::erf(x / SQRT2);
      dx -= (2 / SQRT2PI) * std::exp(-x * x / 2) / denominator;
      loss += std::log(denominator / 2);
      if (HessianTarget != nullptr) {
        Real exp_factor = std::exp(-(x * x) / 2);
        (*HessianTarget)(label - 1, label - 1) +=
            -(-SQRT2PI * x * denominator * exp_factor +
              2 * exp_factor * exp_factor) /
            PI / denominator / denominator;
      }
    }
  }

  inline void sample_z_given_cutpoint() {
    zmins.array() = std::numeric_limits<Real>::max();
    zmaxs.array() = std::numeric_limits<Real>::lowest();
    Real deviation = 1;

    for (int train_data_index : indices_) {
      int class_index = static_cast<int>(y_(train_data_index));
      Real pred_score = x_(train_data_index);
      Real z_new;

      if (class_index == 0) {
        z_new = deviation * sample_truncated_normal_right(
                                rng, (gamma_now(class_index) - pred_score) /
                                         deviation) +
                pred_score;
        zmaxs(0) = std::max(zmaxs(0), z_new);
      } else if (class_index == (K - 1)) {
        z_new =
            deviation * sample_truncated_normal_left(
                            rng, (gamma_now(K - 2) - pred_score) / deviation) +
            pred_score;
        zmins(K - 1) = std::min(zmins(K - 1), z_new);
      } else {
        z_new =
            deviation *
                sample_truncated_normal_twoside(
                    rng, (gamma_now(class_index - 1) - pred_score) / deviation,
                    (gamma_now(class_index) - pred_score) / deviation) +
            pred_score;
        zmins(class_index) = std::min(zmins(class_index), z_new);
        zmaxs(class_index) = std::max(zmaxs(class_index), z_new);
      }
      x_(train_data_index) -= z_new;
    }
  }

  inline void start_sample() {
    DenseVector alpha_hat = DenseVector::Zero(K - 1);
    find_minimum(alpha_hat);
    alpha_now = alpha_hat;
    alpha_to_gamma(gamma_now, alpha_now);
  }

  inline void sample_cutpoint_given_z() {
    for (int i = 1; i <= (K - 3); i++) {
      Real lower = zmaxs(i);
      Real upper = zmins(i + 1);
      gamma_now(i) = std::uniform_real_distribution<Real>(lower, upper)(rng);
    }
  }

  inline void find_minimum(DenseVector &alpha_hat, bool verbose = false) {
    int max_iter = 10000;
    Real epsilon = 1e-5;
    Real epsilon_rel = 1e-5;
    Real delta = 1e-5;
    int past = 3;
    DenseVector history(past);
    DenseVector alpha_new(alpha_hat);
    DenseVector dalpha(alpha_hat);
    DenseVector direction(alpha_hat);
    Real ll_current;
    bool first = true;
    int i = 0;
    while (true) {
      if (first) {
        ll_current = (*this)(alpha_hat, dalpha, &H);
        if (verbose) {
          print_to_stream(std::cout, "ll_current = ", ll_current,
                          "\ndalpha = ", dalpha);
          std::cout << std::endl;
        }
      }
      {

        Real alpha2 = alpha_hat.norm();
        Real dalpha2 = dalpha.norm();
        if (verbose) {
          print_to_stream(std::cout, "ll = ", ll_current,
                          "\nalpha_hat =", alpha_hat);
          std::cout << std::endl;

          print_to_stream(std::cout, "dalpha2 = ", dalpha2);
          std::cout << std::endl;
        }

        if (dalpha2 < epsilon || dalpha2 < epsilon_rel * alpha2) {
          break;
        }
      }

      direction = -H.llt().solve(dalpha);
      if (verbose) {
        print_to_stream(std::cout, "H = ", H);
        std::cout << std::endl;

        print_to_stream(std::cout, "direction = ", direction);
        std::cout << std::endl;
      }

      Real step_size = 1;
      int lsc = 0;
      while (true) {
        alpha_new = alpha_hat + step_size * direction;
        Real ll_new;
        try {
          ll_new = (*this)(alpha_new, dalpha, &H);
        } catch (std::runtime_error) {
          step_size /= 2;
          continue;
        }

        if (ll_new >= (ll_current * (1 + delta))) {
          step_size /= 2;
        } else {
          alpha_hat = alpha_new;
          ll_current = ll_new;
          break;
        }
        if (++lsc > 1000)
          break;
      }
      first = false;
      if (i >= past) {
        Real past_loss = history(i % past);
        if (std::abs(past_loss - ll_current) <=
            delta *
                std::max(std::max(abs(ll_current), abs(past_loss)), Real(1))) {
          break;
        }
      }
      history(i % past) = ll_current;
      i++;
      if (i >= max_iter)
        break;
    }
    if (i == max_iter) {
      throw std::runtime_error("Failed to converge. See fail-log.txt");
    }
  }

  inline bool step(bool verbose = false) {
    DenseVector alpha_hat = alpha_now;
    DenseVector gamma(alpha_hat);
    find_minimum(alpha_hat, verbose);
    DenseVector alpha_candidate = sample_mvt(H, nu) + alpha_hat;

    Real ll_candidate, ll_old;
    try {
      ll_candidate = -(*this)(alpha_candidate, gamma);
      ll_old = -(*this)(alpha_now, gamma);
    } catch (std::runtime_error e) {
      // should be NaN encounter
      return false;
    }
    Real log_p_transition_candidate =
        log_p_mvt(H, alpha_hat, nu, alpha_candidate);
    Real log_p_transition_old = log_p_mvt(H, alpha_hat, nu, alpha_now);
    Real test_ratio = std::exp(ll_candidate - log_p_transition_candidate -
                               ll_old + log_p_transition_old);
    Real u = std::uniform_real_distribution<Real>{0, 1}(rng);
    if (u < test_ratio) {
      alpha_now = alpha_candidate;
      alpha_to_gamma(gamma_now, alpha_now);
      accept_count++;
      return true;
    } else {
      return false;
    }
  }

  inline Real operator()(const DenseVector &alpha, DenseVector &dalpha,
                         DenseMatrix *HessianTarget = nullptr) {
    DenseVector gamma = DenseVector::Zero(alpha.rows());
    dalpha.array() = 0;
    alpha_to_gamma(gamma, alpha);

    DenseMatrix dGammadAlpha = DenseMatrix(alpha.rows(), alpha.rows());
    jacobian_dgamma_dalpha(dGammadAlpha, alpha);
    Real ll = 0;
    if (HessianTarget != nullptr) {

      (*HessianTarget).array() = 0;
    }
    for (auto i : indices_) {
      int label = y_(i);
      if (label == 0) {
        safe_lcdf(gamma(0) - x_(i), ll, dalpha(0), HessianTarget, label);
      } else if (label == (K - 1)) {
        safe_lccdf(gamma(K - 2) - x_(i), ll, dalpha(K - 2), HessianTarget,
                   label);
      } else {
        safe_ldiff(gamma(label) - x_(i), gamma(label - 1) - x_(i), ll,
                   dalpha(label), dalpha(label - 1), HessianTarget, label);
      }
    }

    if (HessianTarget != nullptr) {
      DenseMatrix &H = (*HessianTarget);
      DenseVector expAlpha(alpha.array().exp().matrix());
      H = dGammadAlpha * H * dGammadAlpha.transpose();
      {
        // m = 0
        // gamma 0 = alpha_0 does not contribute
        // since \partial^2 gamma_0 / \partial alpha_i \partial alpha_j = 0 for
        // all gamma_m = alpha_0 + \sum _{s=1}^{m}(exp\alpha_s)
        for (int m = 1; m < (K - 1); m++) {
          { // i =0, j > 0
            for (int j = 1; j <= m; j++) {
              H(j, j) += dalpha(m) * expAlpha(j);
            }
          }
        }
      }
      H(0, 0) -= reg;
      for (int m = 1; m < (K - 1); m++) {
        H(m, m) -= reg;
      }
      H.array() *= -1;
      if (H.hasNaN()) {
        fail_dump();
        throw std::runtime_error(print_to_string(
            __FILE__, ":", __LINE__, " H has NaN, alpha = ", alpha));
      }
    }
    if (fix_gamma0) {
      dalpha(0) = 0;
      if (HessianTarget != nullptr) {
        (*HessianTarget).row(0).array() = 0;
        (*HessianTarget).col(0).array() = 0;
        (*HessianTarget)(0, 0) = 1;
      }
    }
    dalpha = -dGammadAlpha * dalpha;
    if (dalpha.hasNaN()) {
      fail_dump();
      throw std::runtime_error(print_to_string(
          __FILE__, ":", __LINE__, " dalpha has NaN, alpha = ", alpha));
    }

    dalpha(0) += reg * alpha(0);
    ll -= 0.5 * reg * alpha(0) * alpha(0);
    for (int m = 1; m < (K - 1); m++) {
      dalpha(m) += reg * alpha(m);
      ll -= 0.5 * reg * alpha(m) * alpha(m);
    }
    return -ll;
  }

  template <class ostype> inline void show_info(ostype &os) {
    os << "{\"xs\": [";
    bool first = true;
    for (auto i : indices_) {
      if (!first)
        os << ", ";
      os << x_[i];
      first = false;
    }
    os << "], \"ys\":[";
    first = true;
    for (auto i : indices_) {
      if (!first)
        os << ", ";
      os << y_[i];
      first = false;
    }
    os << "]}";
  }

  inline void fail_dump() {
    std::ofstream fail_log("fail-log.json");
    show_info(fail_log);
  }

  DenseVector &x_;
  const DenseVector &y_;

  int K;
  const std::vector<size_t> indices_;
  Real tune = 1;
  Real reg;
  Real nu;
  std::mt19937 &rng;
  DenseVector alpha_now;
  DenseVector gamma_now;
  DenseMatrix H;
  static constexpr bool fix_gamma0 = false;
  DenseVector zmins, zmaxs;
  std::vector<size_t> histogram;
  size_t accept_count;
};

} // namespace myFM