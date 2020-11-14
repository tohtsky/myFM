#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file
#include "catch.hpp"
#include "myfm/OProbitSampler.hpp"

using namespace myFM;
using OpS = OprobitSampler<double>;

template <typename Real>
void ldiff_straightforward(Real x, Real y, Real &loss, Real &dx, Real &dy,
                           OpS::DenseMatrix *HessianTarget = nullptr,
                           int label = 0) {
  // x positive, y negative. safe to use erf
  Real denominator =
      Faddeeva::erf(x / OpS::SQRT2) - Faddeeva::erf(y / OpS::SQRT2);
  Real expxx = std::exp(-x * x / 2);
  Real expyy = std::exp(-y * y / 2);
  dx += 2 * expxx / denominator / OpS::SQRT2PI;
  dy -= 2 * expyy / denominator / OpS::SQRT2PI;
  loss += std::log(denominator / 2);
  if (HessianTarget != nullptr) {
    (*HessianTarget)(label, label) +=
        -(OpS::SQRT2PI * x * denominator * expxx + 2 * expxx * expxx) /
        OpS::PI / denominator / denominator;
    (*HessianTarget)(label - 1, label - 1) +=
        -(-OpS::SQRT2PI * y * denominator * expyy + 2 * expyy * expyy) /
        OpS::PI /

        denominator / denominator;
    Real off_diag = 2 * expxx * expyy / OpS::PI / denominator / denominator;
    (*HessianTarget)(label, label - 1) += off_diag;
    (*HessianTarget)(label - 1, label) += off_diag;
  }
}

TEST_CASE("Factorials are computed", "[factorial]") {
  OpS::DenseMatrix H(3, 3);
  OpS::DenseMatrix H_gt(3, 3);
  double dx, dy, dx_gt, dy_gt, loss, loss_gt;

  /* region 1 */

  loss = 0;
  loss_gt = 0;
  H.array() = 0;
  H_gt.array() = 0;

  ldiff_straightforward(0.5, 0.1, loss_gt, dx_gt, dy_gt, &H_gt, 1);
  OpS::safe_ldiff(0.5, 0.1, loss, dx, dy, &H, 1);
  REQUIRE(dx == Approx(dx_gt));
  REQUIRE(dy == Approx(dy_gt));
  REQUIRE(loss == Approx(loss_gt));
  REQUIRE((H - H_gt).array().abs().sum() < 1e-10);

  /* region 2 */
  loss = 0;
  loss_gt = 0;
  H.array() = 0;
  H_gt.array() = 0;

  ldiff_straightforward(-0.1, -0.5, loss_gt, dx_gt, dy_gt, &H_gt, 1);
  OpS::safe_ldiff(-0.1, -0.5, loss, dx, dy, &H, 1);
  REQUIRE(dx == Approx(dx_gt));
  REQUIRE(dy == Approx(dy_gt));
  REQUIRE(loss == Approx(loss_gt));
  REQUIRE((H - H_gt).array().abs().sum() < 1e-10);

  /* region 3 */
  loss = 0;
  loss_gt = 0;
  H.array() = 0;
  H_gt.array() = 0;

  ldiff_straightforward(0.1, -0.2, loss_gt, dx_gt, dy_gt, &H_gt, 1);
  OpS::safe_ldiff(0.1, -0.2, loss, dx, dy, &H, 1);
  REQUIRE(dx == Approx(dx_gt));
  REQUIRE(dy == Approx(dy_gt));
  REQUIRE(loss == Approx(loss_gt));
  REQUIRE((H - H_gt).array().abs().sum() < 1e-10);
}

TEST_CASE("minimization test.", "[probit-minimization]") {
  OpS::DenseVector x(6);
  OpS::DenseVector y(6);
  x << -5, -4, -1, 1, 4, 5; // symmetric
  x.array() *= 10;
  y << 0, 1, 0, 2, 1, 2;
  std::mt19937 rng(32);
  OpS sampler(x, y, 3, {0, 1, 2, 3, 4, 5}, rng, 0, 5);
  sampler.start_sample();
  REQUIRE(sampler.gamma_now(0) == Approx(-sampler.gamma_now(1)));
}