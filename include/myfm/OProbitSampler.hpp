#pragma once 

#include <iostream>
#include <random>
#include "Faddeeva/Faddeeva.hh"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <memory>

namespace myFM {
template <typename Real> struct AC01Sampler {
  using DenseVector = Eigen::VectorXd;
  using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
  using IntVector = Eigen::Matrix<int, Eigen::Dynamic, 1>;
  static constexpr Real SQRT2 = 1.4142135623730951;
  static constexpr Real SQRTPI = 1.7724538509055159;
  static constexpr Real SQRT2PI = SQRT2 * SQRTPI;
  static constexpr Real PI = 3.141592653589793;

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
    return result;
  }

  inline void jacobian_dgamma_dalpha(DenseMatrix &J, const DenseVector &alpha) {
    /*
    J_{ij} with i=> alpha, j=>gamma
    */
    J.array() = 0;
    for (int j = 0; j < alpha.rows(); j++) {
      J(0, j) = 1;
    }
    for (int i = 1; i < alpha.rows(); i++) {
      Real ed = std::exp(alpha(i));
      for (int j = i; j < alpha.rows(); j++) {
        J(i, j) = ed;
      }
    }
    // d f / d alpha_0 = (df / d gamma_i) (d gamma_i / d alpha_0 )
  }

  inline void alpha_to_gamma(DenseVector &target, const DenseVector &alpha) {
    target(0) = alpha(0);
    for (int i = 1; i < alpha.rows(); i++) {
      target(i) = target(i - 1) + std::exp(alpha(i));
    }
  }

  inline void gamma_to_alpha(DenseVector &target, const DenseVector &gamma) {
    target(0) = gamma(0);
    for (int i = 1; i < gamma.rows(); i++) {
      target(i) = std::log(gamma(i) - gamma(i - 1));
    }
  }

  inline void safe_ldiff(Real x, Real y, Real &loss, Real &dx, Real &dy,
                         DenseMatrix *HessianTarget = nullptr, int label = 0) {
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
      Real expyy = std::exp(-x * x / 2);
      dx += 2 * expxx / denominator / SQRT2PI;
      dy -= 2 * std::exp(-y * y / 2) / denominator / SQRT2PI;
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

  inline void safe_lcdf(Real x, Real &loss, Real &dx,
                        DenseMatrix *HessianTarget = nullptr, int label = 0) {
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

  AC01Sampler(DenseVector & x, const DenseVector& y, int K, std::mt19937& rng)
      : x_(x), y_(y), K(K), rng(rng) {
  }

  void start_sample() {
    // this->alpha_now = find_minimum();
    this->alpha_now = DenseVector::Zero(K - 1);
    this->H = DenseMatrix::Zero(K - 1, K - 1);
  }

  bool step() {
    DenseVector alpha_hat = alpha_now;
    DenseVector gamma(alpha_hat);
    Real ll;

    {
      int max_iter = 10000;
      Real epsilon = 1e-5;
      Real epsilon_rel = 1e-5;
      Real delta = 1e-10;
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
        }
        {

          Real alpha2 = alpha_hat.norm();
          Real dalpha2 = dalpha.norm();
          if (dalpha2 < epsilon || dalpha2 < epsilon_rel * alpha2) {
            break;
          }
        }
        direction = -H.llt().solve(dalpha);
        Real step_size = 1;
        int lsc = 0;
        while (true) {
          alpha_new = alpha_hat + step_size * direction;
          Real ll_new = (*this)(alpha_new, dalpha, &H);
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
              delta * std::max(std::max(abs(ll_current), abs(past_loss)),
                               Real(1))) {
            break;
          }
        }
        history(i % past) = ll_current;
        i++;
        if (i >= max_iter)
          break;
      }
      if (i == max_iter)
        throw std::runtime_error("Failed to converge");
    }

    DenseVector alpha_candidate = sample_mvt(H, nu) + alpha_hat;

    Real ll_candidate = -(*this)(alpha_candidate, gamma);
    Real ll_old = -(*this)(alpha_now, gamma);
    Real log_p_transition_candidate =
        log_p_mvt(H, alpha_hat, nu, alpha_candidate);
    Real log_p_transition_old = log_p_mvt(H, alpha_hat, nu, alpha_now);
    Real test_ratio = std::exp(ll_candidate - log_p_transition_candidate -
                               ll_old + log_p_transition_old);
    Real u = std::uniform_real_distribution<Real>{0, 1}(rng);
    if (u < test_ratio) {
      alpha_now = alpha_candidate;
      return true;
    } else {
      return false;
    }
  }

  /*
    inline DenseVector find_minimum(const DenseVector &initial) {
      DenseVector alpha_hat(initial);
      Real ll = 0;
      int niter = solver_->minimize(*this, alpha_hat, ll);
      return alpha_hat;
    }
    inline DenseVector find_minimum() {
      DenseVector initial = DenseVector::Zero(K - 1);
      return find_minimum(initial);
    }
    */

  inline DenseMatrix hessian(const DenseVector &alpha) {
    DenseMatrix H = DenseMatrix::Zero(alpha.rows(), alpha.rows());
    DenseVector gamma = DenseVector::Zero(alpha.rows());
    DenseMatrix dGammadAlpha = DenseMatrix(alpha.rows(), alpha.rows());
    alpha_to_gamma(gamma, alpha);
    jacobian_dgamma_dalpha(dGammadAlpha, alpha);
    for (int i = 0; i < x_.rows(); i++) {
      int label = y_(i);
      Real denominator;
      if (label == 0) {
        Real x = gamma(0) - x_(i);
        if (x > 0) {
          denominator = Faddeeva::erf(x / SQRT2) + 1;
          H(label, label) +=
              -(SQRT2PI * x * denominator * std::exp(-(x * x) / 2) +
                2 * std::exp(-x * x)) /
              PI / denominator / denominator;
        } else {
          denominator = Faddeeva::erfcx(-x / SQRT2);
          H(label, label) +=
              -(SQRT2PI * x * denominator + 2) / PI / denominator / denominator;
        }
      } else if (label == (K - 1)) {
        Real y = gamma(K - 2) - x_(i);
        if (y > 0) {
          denominator = Faddeeva::erfcx(y / SQRT2);
          H(label - 1, label - 1) +=
              (SQRT2PI * y * denominator - 2) / denominator / denominator / PI;
        } else {
          denominator = 1 - Faddeeva::erf(y / SQRT2);
          H(label - 1, label - 1) +=
              -(-SQRT2PI * y * denominator * std::exp(-(y * y) / 2) +
                2 * std::exp(-y * y)) /
              PI / denominator / denominator;
        }
      } else {
        Real x = gamma(label) - x_(i);
        Real y = gamma(label - 1) - x_(i);
        if (y > 0) {
          denominator =
              Faddeeva::erfcx(y / SQRT2) -
              std::exp((y * y - x * x) / 2) * Faddeeva::erfcx(x / SQRT2);
          H(label, label) +=
              -(SQRT2PI * x * denominator * std::exp((y * y - x * x) / 2) +
                2 * std::exp(y * y - x * x)) /
              denominator / denominator / PI;
          H(label - 1, label - 1) +=
              (SQRT2PI * y * denominator - 2) / denominator / denominator / PI;
          Real off_diag = 2 * std::exp((y * y - x * x) / 2) / PI / denominator /
                          denominator;
          H(label, label - 1) += off_diag;
          H(label - 1, label) += off_diag;
        } else if (x < 0) {
          denominator =
              Faddeeva::erfcx(-x / SQRT2) -
              std::exp((x * x - y * y) / 2) * Faddeeva::erfcx(-y / SQRT2);
          H(label, label) +=
              -(SQRT2PI * x * denominator + 2) / PI / denominator / denominator;
          H(label - 1, label - 1) +=
              (SQRT2PI * y * std::exp((x * x - y * y) / 2) * denominator -
               2 * std::exp(x * x - y * y)) /
              PI / denominator / denominator;
          Real off_diag = 2 * std::exp((x * x - y * y) / 2) / PI / denominator /
                          denominator;
          H(label, label - 1) += off_diag;
          H(label - 1, label) += off_diag;
        } else {
          denominator = Faddeeva::erf(x / SQRT2) - Faddeeva::erf(y / SQRT2);
          H(label, label) +=
              -(SQRT2PI * x * denominator * std::exp(-(x * x) / 2) +
                2 * std::exp(-x * x)) /
              PI / denominator / denominator;
          H(label - 1, label - 1) +=
              -(-SQRT2PI * y * denominator * std::exp(-(y * y) / 2) +
                2 * std::exp(-y * y)) /
              PI / denominator / denominator;
          Real off_diag = 2 * std::exp((-x * x - y * y) / 2) / PI /
                          denominator / denominator;
          H(label, label - 1) += off_diag;
          H(label - 1, label) += off_diag;
        }
      }
    }
    H = dGammadAlpha * H * dGammadAlpha.transpose();
    return H;
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
    for (int i = 0; i < x_.rows(); i++) {
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
      H.array() *= -1;
    }
    dalpha = -dGammadAlpha * dalpha;
    return -ll;
  }
  DenseVector& x_;
  const DenseVector & y_;
  int K;
  Real tune = 1;
  Real nu = 5;
  std::mt19937& rng;
  DenseVector alpha_now;
  DenseMatrix H;
};

} // namespace myfm