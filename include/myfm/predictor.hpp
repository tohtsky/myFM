#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include "FM.hpp"
#include "FMLearningConfig.hpp"
#include "definitions.hpp"
#include "util.hpp"

namespace myFM {

template <typename Real, class FMType = FM<Real>> struct Predictor {
  typedef typename FMLearningConfig<Real>::TASKTYPE TASKTYPE;
  typedef typename FMType::SparseMatrix SparseMatrix;
  typedef typename FMType::Vector Vector;
  typedef typename FMType::DenseMatrix DenseMatrix;
  typedef typename FMType::RelationBlock RelationBlock;

  inline Predictor(size_t rank, size_t feature_size, TASKTYPE type)
      : rank(rank), feature_size(feature_size), type(type), samples() {}

  inline void check_input(const SparseMatrix &X,
                          const vector<RelationBlock> &relations) const {
    auto given_feature_size = check_row_consistency_return_column(X, relations);
    if (feature_size != given_feature_size) {
      throw std::invalid_argument(
          StringBuilder{}("Told to predict for ")(
              given_feature_size)(" but this->feature_size is ")(feature_size)
              .build());
    }
  }

  inline Vector predict_parallel(const SparseMatrix &X,
                                 const vector<RelationBlock> &relations,
                                 size_t n_workers) const {
    check_input(X, relations);
    if (samples.empty()) {
      throw std::runtime_error("Told to predict but no sample available.");
    }
    Vector result = Vector::Zero(X.rows());
    const size_t n_samples = this->samples.size();

    std::mutex mtx;
    std::atomic<size_t> currently_done(0);
    std::vector<std::thread> workers;

    for (size_t i = 0; i < n_workers; i++) {
      workers.emplace_back(
          [this, n_samples, &result, &X, &relations, &currently_done, &mtx] {
            Vector cache(X.rows());
            while (true) {
              size_t cd = currently_done++;
              if (cd >= n_samples)
                break;
              this->samples[cd].predict_score_write_target(cache, X, relations);
              if (this->type == TASKTYPE::CLASSIFICATION) {
                cache.array() =
                    ((cache.array() * static_cast<Real>(std::sqrt(0.5))).erf() +
                     static_cast<Real>(1)) /
                    static_cast<Real>(2);
              }
              {
                std::lock_guard<std::mutex> lock{mtx};
                result += cache;
              }
            }
          });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    result.array() /= static_cast<Real>(n_samples);
    return result;
  }

  inline DenseMatrix
  predict_parallel_oprobit(const SparseMatrix &X,
                           const vector<RelationBlock> &relations,
                           size_t n_workers, size_t cutpoint_index) const {
    check_input(X, relations);
    if (samples.empty()) {
      throw std::runtime_error("Told to predict but no sample available.");
    }
    if (this->type != TASKTYPE::ORDERED) {
      throw std::runtime_error(
          "predict_parallel_oprobit must be called for oprobit model.");
    }
    int n_cpt = (this->samples.at(0)).cutpoints.at(cutpoint_index).size();
    DenseMatrix result = DenseMatrix::Zero(X.rows(), n_cpt + 1);
    const size_t n_samples = this->samples.size();

    std::mutex mtx;
    std::atomic<size_t> currently_done(0);
    std::vector<std::thread> workers;

    for (size_t i = 0; i < n_workers; i++) {
      workers.emplace_back([this, n_samples, &result, &X, &relations,
                            &currently_done, &mtx, cutpoint_index, n_cpt] {
        Vector score(X.rows());

        while (true) {
          size_t cd = currently_done.fetch_add(1);
          if (cd >= n_samples)
            break;

          DenseMatrix sample_result =
              this->samples.at(cd).oprobit_predict_proba(X, relations,
                                                         cutpoint_index);

          {
            std::lock_guard<std::mutex> lock{mtx};
            result += sample_result;
          }
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    result.array() /= static_cast<Real>(n_samples);
    return result;
  }

  inline Vector predict(const SparseMatrix &X,
                        const vector<RelationBlock> &relations) const {
    check_input(X, relations);
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
        result.array() +=
            ((cache.array() * static_cast<Real>(std::sqrt(0.5))).erf() +
             static_cast<Real>(1)) /
            static_cast<Real>(2);
      }
    }
    result.array() /= static_cast<Real>(samples.size());
    return result;
  }

  inline void set_samples(vector<FMType> &&samples_from) {
    samples = std::forward<vector<FMType>>(samples_from);
  }

  inline void add_sample(const FMType &fm) {
    if (fm.w0.rows() != feature_size) {
      throw std::invalid_argument("feature size mismatch!");
    }
    if (fm.V.cols() != rank) {
      throw std::invalid_argument("rank mismatch!");
    }
    samples.emplace_back(fm);
  }

  const size_t rank;
  const size_t feature_size;
  const TASKTYPE type;
  vector<FMType> samples;
};

} // namespace myFM
