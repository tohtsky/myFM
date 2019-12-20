#ifndef MYFM_HYPER_PARAMS_HPP
#define MYFM_HYPER_PARAMS_HPP

#include "FM.hpp"
#include "definitions.hpp"

namespace myFM {

template <typename Real> struct FMHyperParameters {
  Real alpha;
  using FMType = FM<Real>;
  using Vector = typename FMType::Vector;
  using DenseMatrix = typename FMType::DenseMatrix;

  Vector mu_w;      // mean for w. will be (n_group) - vector
  DenseMatrix mu_V; // mean for V. will be (n_group x n_factor) matrix

  Vector lambda_w;      // variances for w. will be (n_group) - vector
  DenseMatrix lambda_V; // variances for V (n_group x n_factor) - matrix

  inline FMHyperParameters(size_t n_factors, size_t n_groups)
      : mu_w(n_groups), mu_V(n_groups, n_factors), lambda_w(n_groups),
        lambda_V(n_groups, n_factors) {}

  inline FMHyperParameters(size_t n_factors)
      : FMHyperParameters(n_factors, 1) {}

  inline FMHyperParameters(const FMHyperParameters &other)
      : alpha(other.alpha), mu_w(other.mu_w), mu_V(other.mu_w),
        lambda_w(other.lambda_w), lambda_V(other.lambda_V) {}
};

} // namespace myFM

#endif
