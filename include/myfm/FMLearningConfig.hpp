#pragma once

#include <vector>
#include <set> 
#include "util.hpp"
#include "definitions.hpp"

namespace myFM{
template <typename Real> struct FMLearningConfig {

  enum class TASKTYPE { REGRESSION, CLASSIFICATION };
  

  inline FMLearningConfig(Real alpha_0, Real beta_0, Real gamma_0, Real mu_0, Real reg_0, TASKTYPE task_type,
                          const vector<size_t> &group_index, int n_iter, int n_kept_samples)
      : alpha_0(alpha_0), beta_0(beta_0), gamma_0(gamma_0), mu_0(mu_0), reg_0(reg_0), task_type(task_type),
        n_iter(n_iter), n_kept_samples(n_kept_samples), group_index_(group_index){
    /* check group_index consistency */
    set<size_t> all_index(group_index.begin(), group_index.end());
    n_groups_ = all_index.size();
    /* verify that groups from 0 - (n_groups - 1)  are contained.*/
    for (size_t i = 0; i < n_groups_; i++) {
      if (all_index.find(i) == all_index.cend()) {
        throw invalid_argument((StringBuilder{})
                                   ("No matching index for group index ") (i) (" found.")
                                   .build());
      }
    }
    group_vs_feature_index_ = vector<vector<size_t>>{n_groups_};

    size_t feature_index = 0;
    for (auto iter = group_index.cbegin(); iter != group_index.cend(); iter++) {
      group_vs_feature_index_[*iter].push_back(feature_index++);
    }

    if (n_kept_samples < 0){
      throw invalid_argument("n_kept_samples must be non-negative,");
    }
    if (n_iter <=0){
      throw invalid_argument("n_iter must be positive.");
    }
    if (n_iter < n_kept_samples) {
      throw invalid_argument("n_kept_samples must not exceed n_iter.");
    }
  }

  FMLearningConfig(const FMLearningConfig & other) = default;

  const Real alpha_0, beta_0, gamma_0;
  const Real mu_0;
  const Real reg_0;

  const TASKTYPE task_type;

  const int n_iter, n_kept_samples;

private:
  const vector<size_t> group_index_;
  size_t n_groups_;
  vector<vector<size_t>> group_vs_feature_index_;

public:
  inline size_t get_n_groups() const { return n_groups_; }

  inline size_t group_index(int at) const { return group_index_.at(at); }

  const vector<vector<size_t>> & group_vs_feature_index() const {
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
    vector<size_t> group_index;

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
    inline Builder & set_reg_0(Real arg) {
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

    inline Builder &set_indentical_groups(size_t n_features) {
      vector<size_t> default_group_index(n_features);
      for (auto c = default_group_index.begin(); c != default_group_index.end();
           c++) {
        *c = 0;
      }
      return set_group_index(default_group_index);
    }

    FMLearningConfig build() {
      return FMLearningConfig(alpha_0, beta_0, gamma_0, mu_0, reg_0, task_type, group_index,
                              n_iter, n_kept_samples);
    }

    static FMLearningConfig get_default_config(size_t n_features) {
      Builder builder;
      return builder.set_indentical_groups(n_features).build();
    }

  }; // end Builder
};

}