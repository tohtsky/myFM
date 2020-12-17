#pragma once
#include "Faddeeva/Faddeeva.hh"
#include "definitions.hpp"
#include <random>
#include <sstream>

namespace myFM {
using namespace std;

/*
Sample from truncated normal distribution.
https://arxiv.org/pdf/0907.4010.pdf
Proposition 2.3.
*/
template <typename Real>
inline Real sample_truncated_normal_left(mt19937 &gen, Real mu_minus) {
  if (mu_minus < 0) {
    normal_distribution<Real> dist(0, 1);
    while (true) {
      Real z = dist(gen);
      if (z > mu_minus) {
        return z;
      }
    }
  } else {
    Real alpha_star = (mu_minus + std::sqrt(mu_minus * mu_minus + 4)) / 2;
    uniform_real_distribution<Real> dist(0, 1);
    while (true) {
      Real z = -std::log(dist(gen)) / alpha_star + mu_minus;
      Real rho = std::exp(-(z - alpha_star) * (z - alpha_star) / 2);
      Real u = dist(gen);
      if (u < rho) {
        return z;
      }
    }
  }
}

template <typename Real>
inline Real sample_truncated_normal_twoside(mt19937 &gen, Real mu_minus,
                                            Real mu_plus) {
  uniform_real_distribution<Real> proposal(mu_minus, mu_plus);
  uniform_real_distribution<Real> acceptance(0, 1);
  Real rho;
  while (true) {
    Real z = proposal(gen);
    if ((mu_minus <= static_cast<Real>(0)) &&
        (mu_plus >= static_cast<Real>(0))) {
      rho = std::exp(-z * z / 2);
    } else if (mu_plus < static_cast<Real>(0)) {
      rho = std::exp((mu_plus * mu_plus - z * z) / 2);
    } else {
      rho = std::exp((mu_minus * mu_minus - z * z) / 2);
    }
    Real u = acceptance(gen);
    if (u < rho) {
      return z;
    }
  }
}
template <typename Real>
inline Real sample_truncated_normal_left(mt19937 &gen, Real mean, Real std,
                                         Real mu_minus) {
  return mean +
         std * sample_truncated_normal_left(gen, (mu_minus - mean) / std);
}

template <typename Real>
inline Real sample_truncated_normal_right(mt19937 &gen, Real mu_plus) {
  return -sample_truncated_normal_left(gen, -mu_plus);
}

template <typename Real>
inline Real sample_truncated_normal_right(mt19937 &gen, Real mean, Real std,
                                          Real mu_plus) {
  return mean +
         std * sample_truncated_normal_right(gen, (mu_plus - mean) / std);
}

template <typename Real>
inline std::tuple<Real, Real, Real> mean_var_truncated_normal_left(Real mu) {
  static constexpr Real SQRT2 = 1.4142135623730951;
  static constexpr Real SQRTPI = 1.7724538509055159;
  static constexpr Real SQRT2PI = SQRT2 * SQRTPI;

  // mean, variance, log(Z)

  /*
  q(z)  = 1{z > 0} exp( - frac{1}{2}(z-mu)^2) / Z
  Z = 1 - \Phi(-mu)
  E_q[z] = \mu + 1/\sqrt{2\pi} exp(-\mu^2/2) / (1 - \Phi(-mu))
  */
  Real phi_Z;
  Real lnZ;
  Real mu_square = mu * mu / 2;
  if (mu > 0) {
    Real Z = (1 - Faddeeva::erf(-mu / SQRT2));
    phi_Z = 2 * std::exp(-mu_square) / SQRT2PI / Z;
    lnZ = std::log(Z);
  } else {
    Real Z = (Faddeeva::erfcx(-mu / SQRT2));
    phi_Z = 2 / Z / SQRT2PI;
    lnZ = std::log(Z) - mu_square;
  }
  std::tuple<Real, Real, Real> result(mu + phi_Z,
                                      1 - mu * phi_Z - phi_Z * phi_Z, lnZ);
  return result;
}

template <typename Real>
inline std::tuple<Real, Real, Real> mean_var_truncated_normal_right(Real mu) {
  auto result = mean_var_truncated_normal_left(-mu);
  std::get<0>(result) *= -1;
  return result;
}

struct StringBuilder {
  inline StringBuilder() : oss_() {}

  template <typename T> inline StringBuilder &add(const T &arg) {
    oss_ << arg;
    return *this;
  }

  template <typename T> inline StringBuilder &operator()(const T &arg) {
    oss_ << arg;
    return *this;
  }

  template <typename T> inline StringBuilder &space_and_add(const T &arg) {
    oss_ << " " << arg;
    return *this;
  }

  template <typename T, typename F>
  inline StringBuilder &add(const T &arg, const T &fmt) {
    oss_ << fmt << arg;
    return *this;
  }

  inline string build() { return oss_.str(); }

private:
  ostringstream oss_;
};

template <typename Real>
inline size_t check_row_consistency_return_column(
    const types::SparseMatrix<Real> &X,
    const vector<relational::RelationBlock<Real>> &relations) {
  size_t row = X.rows();
  size_t col = X.cols();
  int i = 0;
  for (const auto &rel : relations) {
    if (row != rel.original_to_block.size()) {
      throw std::runtime_error(
          (StringBuilder{})("main table has size ")(row)(" but the relation[")(
              i)("] has size ")(rel.original_to_block.size())
              .build());
    }
    col += rel.feature_size;
    i++;
  }
  return col;
}

template <typename... Cs> void print_to_stream(std::ostream &ss, Cs &&... args);

template <typename C, typename... Cs>
inline void print_to_stream(std::ostream &ss, C &&c0, Cs &&... args) {
  ss << c0;
  print_to_stream(ss, std::forward<Cs>(args)...);
}

template <> inline void print_to_stream(std::ostream &ss) {}

template <typename... Cs> std::string print_to_string(Cs &&... args) {
  std::stringstream ss;
  print_to_stream(ss, std::forward<Cs>(args)...);
  return ss.str();
}

} // namespace myFM
