from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from abc import ABC, abstractmethod
from scipy import special, sparse as sps
import pandas as pd
import numpy as np

from ..._myfm import RelationBlock, FM, FMHyperParameters, LearningHistory
from ...base import ArrayLike, check_data_consistency, REAL


def std_cdf(x: np.ndarray) -> np.ndarray:
    return (1 + special.erf(x * np.sqrt(0.5))) / 2


class LibFMLikeCallbackBase(ABC):
    def __init__(
        self,
        n_iter: int,
        X_test: Optional[ArrayLike],
        X_rel_test: List[RelationBlock],
        y_test: np.ndarray,
        trace_path: Optional[str] = None,
    ):
        """Provides a LibFM-like callback after each iteration.
        This will be helpful when we cannot afford enough memory to store
        all posterior samples."""
        self.n_test_data = check_data_consistency(X_test, X_rel_test)

        self.n_iter = n_iter
        if X_test is not None:
            self.X_test: ArrayLike = X_test
        else:
            self.X_test = sps.csr_matrix((self.n_test_data, 0), dtype=REAL)
        self.X_rel_test = X_rel_test
        self.y_test: np.ndarray = y_test
        self.result_trace: List[Dict[str, float]] = []
        self.trace_path = trace_path
        self.n_samples = 0

    @abstractmethod
    def _measure_score(
        self, i: int, fm: FM, hyper: FMHyperParameters
    ) -> Tuple[str, Dict[str, float]]:
        raise NotImplementedError("must be implemented")

    def __call__(
        self, i: int, fm: FM, hyper: FMHyperParameters, history: LearningHistory
    ) -> Tuple[bool, Optional[str]]:
        description, trace_result = self._measure_score(i, fm, hyper)
        self.result_trace.append(trace_result)

        if self.trace_path is not None:
            df = pd.DataFrame(self.result_trace)
            df.to_csv(self.trace_path, index=False)

        return False, description


class RegressionCallback(LibFMLikeCallbackBase):
    def __init__(
        self,
        n_iter: int,
        X_test: Optional[ArrayLike],
        y_test: np.ndarray,
        X_rel_test: List[RelationBlock] = [],
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        trace_path: Optional[str] = None,
    ):
        super(RegressionCallback, self).__init__(
            n_iter, X_test, X_rel_test, y_test, trace_path=trace_path
        )
        self.predictions = np.zeros((self.n_test_data,), dtype=np.float64)
        self.prediction_all_but_5 = np.zeros((self.n_test_data,), dtype=np.float64)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def clip_value(self, arr: np.ndarray) -> None:
        if self.clip_min is not None:
            arr[arr <= self.clip_min] = self.clip_min
        if self.clip_max is not None:
            arr[arr >= self.clip_max] = self.clip_max

    def _measure_score(
        self, i: int, fm: FM, hyper: FMHyperParameters
    ) -> Tuple[str, Dict[str, float]]:
        score = fm.predict_score(self.X_test, self.X_rel_test)
        self.predictions += score
        self.n_samples += 1
        prediction_mean = self.predictions / self.n_samples
        self.clip_value(prediction_mean)
        if i >= 5:
            self.prediction_all_but_5 += score
            prediction_mean_all_but_5 = self.prediction_all_but_5 / (i + 1 - 5)
            self.clip_value(prediction_mean_all_but_5)
            rmse_all_but_5 = float(
                ((self.y_test - prediction_mean_all_but_5) ** 2).mean() ** 0.5
            )
        else:
            rmse_all_but_5 = float("nan")

        rmse = float(((self.y_test - prediction_mean) ** 2).mean() ** 0.5)
        rmse_this = float(((self.y_test - score) ** 2).mean() ** 0.5)
        description = "alpha={0:.4f}, rmse_mean={1:.4f}, rmse_this={2:.4f}, rmse_all_but_5={3:.4f}".format(
            hyper.alpha, rmse, rmse_this, rmse_all_but_5
        )
        result = OrderedDict(
            [
                ("alpha", hyper.alpha),
                ("rmse", rmse),
                ("rmse_this", rmse_this),
                ("rmse_all_but_5", rmse_all_but_5),
            ]
        )
        return description, result


class ClassificationCallback(LibFMLikeCallbackBase):
    def __init__(
        self,
        n_iter: int,
        X_test: Optional[ArrayLike],
        y_test: np.ndarray,
        X_rel_test: List[RelationBlock] = [],
        eps: Optional[float] = 1e-15,
        trace_path: Optional[str] = None,
    ):
        super(ClassificationCallback, self).__init__(
            n_iter, X_test, X_rel_test, y_test, trace_path=trace_path
        )
        self.predictions = np.zeros((self.n_test_data,), dtype=np.float64)
        self.prediction_all_but_5 = np.zeros((self.n_test_data,), dtype=np.float64)
        self.eps = eps

    def clip_value(self, arr: np.ndarray) -> None:
        if self.eps is not None:
            arr[arr <= self.eps] = self.eps
            arr[arr >= (1 - self.eps)] = 1 - self.eps

    def __log_loss(self, arr: np.ndarray) -> float:
        result = 0
        result += np.log(arr[self.y_test == 1]).sum()
        result += np.log(1 - arr[self.y_test == 0]).sum()
        return -result

    def __accuracy(self, arr: np.ndarray) -> float:
        return float((self.y_test == (arr >= 0.5)).mean())

    def _measure_score(
        self, i: int, fm: FM, hyper: FMHyperParameters
    ) -> Tuple[str, Dict[str, float]]:
        prob_this = fm.predict_score(self.X_test, self.X_rel_test)
        self.predictions += prob_this
        self.n_samples += 1
        prediction_mean = self.predictions / self.n_samples
        self.clip_value(prediction_mean)
        if i >= 5:
            self.prediction_all_but_5 += prob_this
            prediction_mean_all_but_5 = self.prediction_all_but_5 / (i + 1 - 5)
            self.clip_value(prediction_mean_all_but_5)
            ll_all_but_5 = self.__log_loss(prediction_mean_all_but_5)
            accuracy_all_but_5 = self.__accuracy(prediction_mean_all_but_5)
        else:
            ll_all_but_5 = float("nan")
            accuracy_all_but_5 = float("nan")

        ll = self.__log_loss(prediction_mean)
        accuracy = self.__accuracy(prediction_mean)
        ll_this = self.__log_loss(prob_this)
        accuracy_this = self.__accuracy(prob_this)
        description = "ll_mean={0:.4f}, ll_this={1:.4f}, ll_all_but_5={2:.4f}".format(
            ll, ll_this, ll_all_but_5
        )
        result = OrderedDict(
            [
                ("log_loss", ll),
                ("log_loss_this", ll_this),
                ("log_loss_all_but_5", ll_all_but_5),
                ("accuracy", accuracy),
                ("accuracy_this", accuracy_this),
                ("accuracy_all_but_5", accuracy_all_but_5),
            ]
        )
        return description, result


class OrderedProbitCallback(LibFMLikeCallbackBase):
    def __init__(
        self,
        n_iter: int,
        X_test: Optional[ArrayLike],
        y_test: np.ndarray,
        n_class: int,
        X_rel_test: List[RelationBlock] = [],
        eps: Optional[float] = 1e-15,
        trace_path: Optional[str] = None,
    ):
        super(OrderedProbitCallback, self).__init__(
            n_iter, X_test, X_rel_test, y_test, trace_path=trace_path
        )
        self.predictions = np.zeros((self.n_test_data, n_class), dtype=np.float64)
        self.prediction_all_but_5 = np.zeros(
            (self.n_test_data, n_class), dtype=np.float64
        )
        self.n_class = n_class
        self.eps = eps
        self.y_test = self.y_test.astype(np.int32)
        assert (self.y_test.min() >= 0) and (self.y_test.max() <= (self.n_class - 1))

    def __log_loss(self, arr: np.ndarray) -> float:
        ps = arr[np.arange(self.y_test.shape[0]), self.y_test].copy()
        ps[ps <= self.eps] = self.eps
        return -float(np.log(ps).sum())

    def __accuracy(self, arr: np.ndarray) -> float:
        return float((self.y_test == (arr.argmax(axis=1))).mean())

    def __rmse(self, arr: np.ndarray) -> float:
        return (
            float(((self.y_test - arr.dot(np.arange(self.n_class))) ** 2).mean()) ** 0.5
        )

    def _measure_score(
        self, i: int, fm: FM, hyper: FMHyperParameters
    ) -> Tuple[str, Dict[str, float]]:
        score = fm.predict_score(self.X_test, self.X_rel_test)
        score = std_cdf(fm.cutpoints[0][np.newaxis, :] - score[:, np.newaxis])
        score = np.hstack(
            [
                np.zeros((score.shape[0], 1), dtype=score.dtype),
                score,
                np.ones((score.shape[0], 1), dtype=score.dtype),
            ]
        )
        prob_this = score[:, 1:] - score[:, :-1]
        self.predictions += prob_this
        self.n_samples += 1
        prediction_mean = self.predictions / self.n_samples
        if i >= 5:
            self.prediction_all_but_5 += prob_this
            prediction_mean_all_but_5 = self.prediction_all_but_5 / (i + 1 - 5)
            ll_all_but_5 = self.__log_loss(prediction_mean_all_but_5)
            accuracy_all_but_5 = self.__accuracy(prediction_mean_all_but_5)
            rmse_all_but_5 = self.__rmse(prediction_mean_all_but_5)
        else:
            ll_all_but_5 = float("nan")
            accuracy_all_but_5 = float("nan")
            rmse_all_but_5 = float("nan")

        ll = self.__log_loss(prediction_mean)
        accuracy = self.__accuracy(prediction_mean)
        rmse = self.__rmse(prediction_mean)
        ll_this = self.__log_loss(prob_this)
        accuracy_this = self.__accuracy(prob_this)
        rmse_this = self.__rmse(prob_this)
        description = "ll_mean={0:.4f}, ll_this={1:.4f}, ll_all_but_5={2:.4f}".format(
            ll, ll_this, ll_all_but_5
        )
        result = OrderedDict(
            [
                ("log_loss", ll),
                ("log_loss_this", ll_this),
                ("log_loss_all_but_5", ll_all_but_5),
                ("accuracy", accuracy),
                ("accuracy_this", accuracy_this),
                ("accuracy_all_but_5", accuracy_all_but_5),
                ("rmse", rmse),
                ("rmse_this", rmse_this),
                ("rmse_all_but_5", rmse_all_but_5),
            ]
        )
        return description, result
