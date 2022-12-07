import numpy as np
import pytest
from scipy.special import erf

from myfm._myfm import OprobitMinimizationConfig, OprobitSampler


def Phi(x: float) -> float:
    result: float = (1 + erf(x / np.sqrt(2))) / 2
    return result


def ll(X: np.ndarray, Y: np.ndarray, K: int, alpha: np.ndarray, reg: float) -> float:
    gammas = [alpha[0]]
    current_alpha = alpha[0]
    for alpha_i in alpha[1:]:
        current_alpha += np.exp(alpha_i)
        gammas.append(current_alpha)
    loss: float = (alpha**2).sum() / 2 * reg
    for x, y_ in zip(X, Y):
        y = int(y_)
        gamma_plus: float
        gamma_minus: float
        if y == 0:
            gamma_plus = gammas[0]
            gamma_minus = -np.inf
        elif y == (K - 1):
            gamma_plus = np.inf
            gamma_minus = gammas[K - 2]
        else:
            gamma_plus = gammas[y]
            gamma_minus = gammas[y - 1]
        p = Phi(gamma_plus - x) - Phi(gamma_minus - x)
        loss += -np.log(p)
    return loss


@pytest.mark.parametrize(["reg"], [(0.001,), (1.0,)])
def test_minimization(reg: float) -> None:
    config = OprobitMinimizationConfig(10000, 1e-10, 1e-10, 1e-10)
    X = np.asfarray([1, 1, 2, 2, 4, 5, 5])
    y = np.asfarray([0, 0, 0, 1, 1, 2, 0])
    sampler = OprobitSampler(X, y, 3, np.arange(X.shape[0]), 0, reg, 5, config)

    sampler.start_sample(X, y)

    alpha_hat = sampler.find_minimum(X, y, np.asfarray([0, 2]))

    minvalue_candidate = ll(X, y, 3, alpha_hat, reg)
    for _ in range(1000):
        cand_value = ll(X, y, 3, alpha_hat + np.random.randn(2) * 0.1, reg)
        assert cand_value > minvalue_candidate


def test_minimization_fails() -> None:
    config = OprobitMinimizationConfig(1, 1e-10, 1e-10, 1e-10)
    X = np.asfarray([1, 1, 2, 2, 4, 5, 5])
    y = np.asfarray([0, 0, 0, 1, 1, 2, 0])
    sampler = OprobitSampler(X, y, 3, np.arange(X.shape[0]), 0, 0.1, 5, config)
    with pytest.raises(RuntimeError):
        sampler.start_sample(X, y)
        sampler.find_minimum(X, y, np.asfarray([0, 2]))
