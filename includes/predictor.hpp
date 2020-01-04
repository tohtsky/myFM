#pragma once

#include <mutex>
#include <thread>
#include <atomic>

#include "definitions.hpp"
#include "FM.hpp"
#include "FMLearningConfig.hpp"

namespace myFM {

template <typename Real> struct Predictor {
  typedef typename FMLearningConfig<Real>::TASKTYPE TASKTYPE;
  typedef typename FM<Real>::SparseMatrix SparseMatrix;
  typedef typename FM<Real>::Vector Vector;
  typedef typename FM<Real>::RelationBlock RelationBlock;
  inline Predictor(TASKTYPE type) : samples(), type(type) {}

  inline Vector predict_parallel(const SparseMatrix &X,
                                 const vector<RelationBlock> &relations,
                                 size_t n_workers) const {
    if (samples.empty()) {
      throw std::runtime_error("Empty samples!");
    }
    Vector result = Vector::Zero(X.rows());
    const size_t n_samples = this->samples.size();

    std::mutex mtx;
    std::atomic<size_t> currently_done(0);
    std::vector<std::thread> workers;

    for (size_t i = 0; i < n_workers; i++) {
      workers.emplace_back([this, i, n_samples, &result, &X, &relations, &currently_done, &mtx] {
        Vector cache(X.rows());
        while (true) {
          size_t cd = currently_done++;
          if (cd >= n_samples) break;
          this->samples[cd].predict_score_write_target(cache, X, relations);
          if (this->type == TASKTYPE::CLASSIFICATION) {
            cache.array() =
                ((cache.array() * static_cast<Real>(std::sqrt(0.5))).erf() + static_cast<Real>(1)) /
                static_cast<Real>(2);
          }
          {
            std::lock_guard<std::mutex> lock{mtx};
            result += cache;
          }
        }
      });
    }
    for (auto & worker:workers) {
      worker.join();
    }
    result.array() /= static_cast<Real>(n_samples);
    return result;
  }

  inline Vector predict(const SparseMatrix & X, const vector<RelationBlock> & relations) const{
      if (samples.empty()) {
          throw std::runtime_error("Empty samples!");
      }
      Vector result = Vector::Zero(X.rows());
      Vector cache = Vector(X.rows());
      for (auto iter = samples.cbegin(); iter != samples.cend(); iter++) {
        iter->predict_score_write_target(cache, X, relations);
        if (type == TASKTYPE::REGRESSION) {
          result += cache;
        } else if (type == TASKTYPE::CLASSIFICATION) {
          result.array() += (
              ( cache.array() * static_cast<Real>(std::sqrt(0.5)) ).erf() + static_cast<Real>(1)
          ) / static_cast<Real>(2);
        }
      }
      result.array() /= static_cast<Real>(samples.size());
      return result;
  }

  inline void set_samples(vector<FM<Real>>&& samples_from) {
      samples = std::forward<vector<FM<Real>>>(samples_from);
  }

  vector<FM<Real>> samples;
  const TASKTYPE type;
};

} // namespace myfm