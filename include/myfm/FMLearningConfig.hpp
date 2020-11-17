#pragma once

#include "OProbitSampler.hpp"
#include "definitions.hpp"
#include "util.hpp"
#include <cstddef>
#include <set>
#include <tuple>
#include <vector>

namespace myFM {
template <typename Real> struct FMLearningConfig {
public:
  enum class TASKTYPE { REGRESSION, CLASSIFICATION, ORDERED };
  using CutpointGroupType = vector<pair<size_t, vector<size_t>>>;

  inline FMLearningConfig(Real alpha_0, Real beta_0, Real gamma_0, Real mu_0,
                          Real reg_0, TASKTYPE task_type, Real nu_oprobit,
                          bool fit_w0, bool fit_linear,
                          const vector<size_t> &group_index, int n_iter,
                          int n_kept_samples, Real cutpoint_scale,
                          const CutpointGroupType &cutpoint_groups)
      : alpha_0(alpha_0), beta_0(beta_0), gamma_0(gamma_0), mu_0(mu_0),
        reg_0(reg_0), task_type(task_type), nu_oprobit(nu_oprobit),
        fit_w0(fit_w0), fit_linear(fit_linear), n_iter(n_iter),
        n_kept_samples(n_kept_samples), cutpoint_scale(cutpoint_scale),
        group_index_(group_index), cutpoint_groups_(cutpoint_groups) {

    /* check group_index consistency */
    set<size_t> all_index(group_index.begin(), group_index.end());
    n_groups_ = all_index.size();
    /* verify that groups from 0 - (n_groups - 1)  are contained.*/
    for (size_t i = 0; i < n_groups_; i++) {
      if (all_index.find(i) == all_index.cend()) {
        throw invalid_argument(
            (StringBuilder{})("No matching index for group index ")(i)(
                " found.")
                .build());
      }
    }
    group_vs_feature_index_ = vector<vector<size_t>>{n_groups_};

    size_t feature_index = 0;
    for (auto iter = group_index.cbegin(); iter != group_index.cend(); iter++) {
      group_vs_feature_index_[*iter].push_back(feature_index++);
    }

    if (n_kept_samples < 0) {
      throw invalid_argument("n_kept_samples must be non-negative,");
    }
    if (n_iter <= 0) {
      throw invalid_argument("n_iter must be positive.");
    }
    if (n_iter < n_kept_samples) {
      throw invalid_argument("n_kept_samples must not exceed n_iter.");
    }
  }

  FMLearningConfig(const FMLearningConfig &other) = default;

  const Real alpha_0, beta_0, gamma_0;
  const Real mu_0;
  const Real reg_0;

  const TASKTYPE task_type;
  const Real nu_oprobit;
  bool fit_w0, fit_linear;

  const int n_iter, n_kept_samples;

  const Real cutpoint_scale;

private:
  const vector<size_t> group_index_;
  size_t n_groups_;
  vector<vector<size_t>> group_vs_feature_index_;

  const CutpointGroupType cutpoint_groups_;

public:
  inline size_t get_n_groups() const { return n_groups_; }

  inline size_t group_index(int at) const { return group_index_.at(at); }
  const CutpointGroupType &cutpoint_groups() const {
    return this->cutpoint_groups_;
  }

  const vector<vector<size_t>> &group_vs_feature_index() const {
    return group_vs_feature_index_;
  }

  struct Builder {
    Real alpha_0 = 1;
    Real beta_0 = 1;
    Real gamma_0 = 1;
    Real mu_0 = 1;
    Real reg_0 = 1;
    int n_iter = 100;
    int n_kept_samples = 10;
    TASKTYPE task_type = TASKTYPE::REGRESSION;
    Real nu_oprobit = 5;
    bool fit_w0 = true;
    bool fit_linear = true;
    vector<size_t> group_index;
    Real cutpoint_scale = 10;
    CutpointGroupType cutpoint_groups;

    Builder() {}

    inline Builder &set_alpha_0(Real arg) {
      this->alpha_0 = arg;
      return *this;
    }

    inline Builder &set_beta_0(Real arg) {
      this->beta_0 = arg;
      return *this;
    }

    inline Builder &set_gamma_0(Real arg) {
      this->gamma_0 = arg;
      return *this;
    }

    inline Builder &set_mu_0(Real arg) {
      this->mu_0 = arg;
      return *this;
    }
    inline Builder &set_reg_0(Real arg) {
      this->reg_0 = arg;
      return *this;
    }

    inline Builder &set_n_iter(int arg) {
      this->n_iter = arg;
      return *this;
    }

    inline Builder &set_n_kept_samples(int arg) {
      this->n_kept_samples = arg;
      return *this;
    }

    inline Builder &set_task_type(TASKTYPE arg) {
      this->task_type = arg;
      return *this;
    }

    inline Builder &set_group_index(const vector<size_t> arg) {
      this->group_index = arg;
      return *this;
    }

    inline Builder &set_identical_groups(size_t n_features) {
      vector<size_t> default_group_index(n_features);
      for (auto c = default_group_index.begin(); c != default_group_index.end();
           c++) {
        *c = 0;
      }
      return set_group_index(default_group_index);
    }

    inline Builder &set_nu_oprobit(size_t nu_oprobit) {
      this->nu_oprobit = nu_oprobit;
      return *this;
    }

    inline Builder &set_fit_w0(bool fit_w0) {
      this->fit_w0 = fit_w0;
      return *this;
    }

    inline Builder &set_fit_linear(bool fit_linear) {
      this->fit_linear = fit_linear;
      return *this;
    }

    inline Builder &set_cutpoint_scale(Real cutpoint_scale) {
      this->cutpoint_scale = cutpoint_scale;
      return *this;
    }

    inline Builder &
    set_cutpoint_groups(const CutpointGroupType &cutpoint_groups) {
      this->cutpoint_groups = cutpoint_groups;
      return *this;
    }

    FMLearningConfig build() {
      return FMLearningConfig(alpha_0, beta_0, gamma_0, mu_0, reg_0, task_type,
                              nu_oprobit, fit_w0, fit_linear, group_index,
                              n_iter, n_kept_samples, cutpoint_scale,
                              this->cutpoint_groups);
    }

    static FMLearningConfig get_default_config(size_t n_features) {
      Builder builder;
      return builder.set_identical_groups(n_features).build();
    }

  }; // end Builder
};

} // namespace myFM