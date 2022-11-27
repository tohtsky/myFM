#pragma once

namespace myFM {
template <typename Real> struct OprobitMinimizationConfig {
  OprobitMinimizationConfig(int max_iter, Real epsilon, Real epsilon_rel,
                            Real delta, int history_window)
      : max_iter(max_iter), epsilon(epsilon), epsilon_rel(epsilon_rel),
        delta(delta), history_window(history_window) {}
  int max_iter;
  Real epsilon;
  Real epsilon_rel;
  Real delta;
  int history_window;
};

} // namespace myFM
