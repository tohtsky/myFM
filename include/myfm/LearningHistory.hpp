#pragma once

#include "HyperParams.hpp"

namespace myFM {
template <typename Real> struct GibbsLearningHistory {
  std::vector<FMHyperParameters<Real>> hypers;
  std::vector<size_t>
      n_mh_accept; // will be used for M-H step in ordered probit regression;
  std::vector<Real> train_log_losses;
};
} // namespace myFM
