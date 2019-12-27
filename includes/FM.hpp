#ifndef MYFM_FM_HPP
#define MYFM_FM_HPP
#include "definitions.hpp"
#include <cmath>

namespace myFM {

using namespace std;

template <typename Real> struct FM {

  using DenseMatrix = Eigen::Matrix<Real, -1, -1, Eigen::ColMajor>;

  using Vector = Eigen::Matrix<Real, -1, 1>;

  using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

  using SparseVector = Eigen::SparseVector<Real>;

  typedef relational::RelationBlock<Real> RelationBlock;

  inline FM(int n_factors, size_t n_groups)
      : n_factors(n_factors), initialized(false) {}
  inline FM(int n_factors) : FM(n_factors, 1) {}

  inline FM(const FM &other)
      : n_factors(other.n_factors), w0(other.w0), w(other.w), V(other.V),
        initialized(other.initialized) {}

  inline FM(Real w0, const Vector &w, const DenseMatrix &V)
      : n_factors(V.cols()), w0(w0), w(w), V(V), initialized(true) {}

  inline void initialize_weight(int n_features, Real init_std, mt19937 &gen) {
    initialized = false;
    normal_distribution<Real> nd;

    auto get_rand = [&gen, &nd, init_std, this](Real dummy) {
      return nd(gen) * init_std;
    };
    V = DenseMatrix{n_features, n_factors}.unaryExpr(get_rand);
    w = Vector{n_features}.unaryExpr(get_rand);
    w0 = get_rand(1);
    initialized = true;
  }

  inline Vector add_q(const SparseMatrix &X, Eigen::Ref<Vector> q) {
    q += X * V;
  }

  inline Vector predict_score(const SparseMatrix &X) const {
    if (!initialized) {
      throw std::runtime_error("get_score called before initialization");
    }

    // Vector result = Vector::Constant(X.rows(), w0_);
    Vector result = w0 + (X * w).array();
    result.array() += (X * V).array().square().rowwise().sum() * 0.5;
    result -=
        (X.cwiseAbs2()) * ((0.5 * V.array().square().rowwise().sum()).matrix());

    return result;
  }

  inline Vector predict_score(
    const SparseMatrix &X, const vector<RelationBlock> & relations
  ) const {

    if (!initialized) {
      throw std::runtime_error("get_score called before initialization");
    }
    Vector result = w0 + (X * w.head(X.cols())).array();
    size_t offset = X.cols();
    for (auto iter=relations.begin(); iter != relations.end(); iter++) { 
      Vector w0_cache = (iter->X) * w.segment(offset, iter->feature_size);
      size_t j = 0;
      for (auto i : (iter->original_to_block)) {
        result(j++) += w0_cache(i);
      }
      offset += iter->feature_size;
    }

    Vector q_cache(result.rows());
    vector<Vector> block_q_caches;
    for(auto & relation: relations) { 
      block_q_caches.emplace_back(relation.feature_size);
    }

    for (int factor_index=0; factor_index < this->n_factors; factor_index++){
      q_cache = X * V.col(factor_index).head(X.cols());
      size_t offset = X.cols();
      size_t relation_index = 0;
      for (auto iter = relations.begin(); iter != relations.end(); iter++, relation_index++) {
        Eigen::Ref<Vector> block_cache = block_q_caches[relation_index];
        block_cache = iter->X * V.col(factor_index).segment(offset, iter->feature_size);
        offset += iter->feature_size;
        size_t train_case_index = 0;
        for (auto i: iter->original_to_block){ 
          q_cache(train_case_index++) += block_cache(i);
        }
      }
      result.array() += q_cache.array().square() * static_cast<Real>(0.5);

      offset = X.cols();
      relation_index = 0;
      q_cache = X.cwiseAbs2() * ( V.col(factor_index).head(X.cols()).array().square().matrix());
      for (auto iter = relations.begin(); iter != relations.end(); iter++, relation_index++) {
        Eigen::Ref<Vector> block_cache = block_q_caches[relation_index];
        block_cache = ( iter->X.cwiseAbs2() ) * (
          V.col(factor_index).segment(offset, iter->feature_size).array().square().matrix()
        );
        offset += iter->feature_size;
        size_t train_case_index = 0;
        for (auto i: iter->original_to_block){
          q_cache(train_case_index++) += block_cache(i);
        }
      }
      result -= q_cache * static_cast<Real>(0.5);
    }
    return result;
  }


  const int n_factors;
  Real w0;
  Vector w;
  DenseMatrix V; // (n_feature, n_factor) - matrix

private:
  bool initialized;
};

} // namespace myFM
#endif
