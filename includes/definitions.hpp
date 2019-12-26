#ifndef MYFM_DEFINITIONS_HPP
#define MYFM_DEFINITIONS_HPP

#include <random>
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace myFM {

using namespace std;

namespace relational {

template<typename Real>


  struct RelationBlock {
    typedef SparseMatrix<Real, Eigen::RowMajor> SparseMatrix;

    inline RelationBlock(vector<size_t> original_to_block, const SparseMatrix & X)
      : original_to_block(original_to_block), X(X)
    {
      for (auto c : original_to_block) {
        if ( c >= X.rows() )
          throw runtime_error("index mapping points to non-existing row.");
      }
    }

    inline vector<size_t> cardinarity() const {
      vector<size_t> result(X.rows(), static_cast<Real>(0));
      for (auto v : original_to_block ) {
        result[v]++;
      } 
      result;
    }


    const vector<size_t> original_to_block;
    const SparseMatrix X;

    const vector<size_t> indptr;
    const vector<size_t> indices;
  };


template<typename Real>
struct RelationWiseCache {
  const RelationBlock<Real> & target;
  vector<size_t> cardinarity; // for each
  vector<size_t> indptr;
  vector<size_t> indices;
  Vector<Real> q_b_i_f;
  Vector<Real> q_s_b_i_f;
};

}


} // namespace myFM

#endif
