#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <unsupported/Eigen/SpecialFunctions>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace myFM {

using namespace std;
namespace types {
template <typename Real>
using DenseMatrix = Eigen::Matrix<Real, -1, -1, Eigen::ColMajor>;

template <typename Real> using Vector = Eigen::Matrix<Real, -1, 1>;

template <typename Real>
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

template <typename Real> using SparseVector = Eigen::SparseVector<Real>;

} // namespace types

namespace relational {

template <typename Real> struct RelationBlock {
  typedef Eigen::SparseMatrix<Real, Eigen::RowMajor> SparseMatrix;
  typedef Eigen::Matrix<Real, -1, 1> Vector;

  inline RelationBlock(vector<size_t> original_to_block, const SparseMatrix &X)
      : original_to_block(original_to_block),
        mapper_size(original_to_block.size()), X(X), block_size(X.rows()),
        feature_size(X.cols()) {
    for (auto c : original_to_block) {
      if (c >= block_size)
        throw runtime_error("index mapping points to non-existing row.");
    }
  }

  inline RelationBlock(const RelationBlock &other)
      : RelationBlock(other.original_to_block, other.X) {}

  const vector<size_t> original_to_block;
  const size_t mapper_size;
  const SparseMatrix X;
  const size_t block_size;
  const size_t feature_size;
};

template <typename Real> struct RelationWiseCache {
  typedef typename RelationBlock<Real>::Vector Vector;
  typedef typename RelationBlock<Real>::SparseMatrix SparseMatrix;

  inline RelationWiseCache(const RelationBlock<Real> &source)
      : target(source), X_t(source.X.transpose()), cardinality(source.X.rows()),
        y(source.X.rows()), q(source.X.rows()), q_S(source.X.rows()),
        c(source.X.rows()), c_S(source.X.rows()), e(source.X.rows()),
        e_q(source.X.rows()) {
    X_t.makeCompressed();
    cardinality.array() = static_cast<Real>(0);
    for (auto v : source.original_to_block) {
      cardinality(v)++;
    }
  }

  const RelationBlock<Real> &target;
  SparseMatrix X_t;
  Vector cardinality; // for each

  Vector y;

  Vector q;
  Vector q_S;

  Vector c;
  Vector c_S;

  Vector e;
  Vector e_q;
};
} // namespace relational

} // namespace myFM