#pragma once

#include "FM.hpp"
#include "definitions.hpp"

namespace myFM {

template <typename Real> struct FMHyperParameters {
  using FMType = FM<Real>;
  using Vector = typename FMType::Vector;
  using DenseMatrix = typename FMType::DenseMatrix;

  Real alpha;

  Vector mu_w;     // mean for w. will be (n_group) - vector
  Vector lambda_w; // variances for w. will be (n_group) - vector

  DenseMatrix mu_V;     // mean for V. will be (n_group x n_factor) matrix
  DenseMatrix lambda_V; // variances for V (n_group x n_factor) - matrix

  inline FMHyperParameters(size_t n_factors, size_t n_groups)
      : mu_w(n_groups), lambda_w(n_groups), mu_V(n_groups, n_factors),
        lambda_V(n_groups, n_factors) {}

  inline FMHyperParameters(size_t n_factors)
      : FMHyperParameters(n_factors, 1) {}

  inline FMHyperParameters(Real alpha, const Vector &mu_w,
                           const Vector &lambda_w, const DenseMatrix &mu_V,
                           const DenseMatrix &lambda_V)
      : alpha(alpha), mu_w(mu_w), lambda_w(lambda_w), mu_V(mu_V),
        lambda_V(lambda_V) {}

  inline FMHyperParameters(const FMHyperParameters &other)
      : alpha(other.alpha), mu_w(other.mu_w), lambda_w(other.lambda_w),
        mu_V(other.mu_V), lambda_V(other.lambda_V) {}
};

} // namespace myFM