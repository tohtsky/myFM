#ifndef MYFM_PREDICTOR_HPP
#define MYFM_PREDICTOR_HPP
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

  inline Vector predict(const SparseMatrix & X, const vector<RelationBlock> & relations) const{
      if (samples.empty()) {
          throw std::runtime_error("Empty samples!");
      }
      Vector result = Vector::Zero(X.rows());
      for (auto iter = samples.cbegin(); iter != samples.cend(); iter++) {
        if (type == TASKTYPE::REGRESSION) {
          result.array() += iter->predict_score(X, relations).array();
        } else if (type == TASKTYPE::CLASSIFICATION) {

          result.array() += (
              ( iter->predict_score(X, relations).array() * static_cast<Real>(std::sqrt(0.5)) ).erf() + static_cast<Real>(1)
          ) / static_cast<Real>(2);
        }
      }
      result.array() /= samples.size();
      return result;
  }

  inline void set_samples(vector<FM<Real>>&& samples_from) {
      samples = std::forward<vector<FM<Real>>>(samples_from);
  }

  vector<FM<Real>> samples;
  const TASKTYPE type;

};
};

#endif