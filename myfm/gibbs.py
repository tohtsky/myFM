from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
from scipy import sparse as sps

from ._myfm import (
    FM,
    FMLearningConfig,
    ConfigBuilder,
    FMHyperParameters,
    LearningHistory,
    Predictor,
    RelationBlock,
    TaskType,
    create_train_fm,
)
from .base import (
    REAL,
    ArrayLike,
    MyFMBase,
    RegressorMixin,
    ClassifierMixin,
    check_data_consistency,
    std_cdf,
    DenseArray,
    BinaryClassificationTarget,
)


try:
    import pandas as pd
except:
    pd = None


class MyFMGibbsBase(
    MyFMBase[
        FM,
        FMHyperParameters,
        Predictor,
        LearningHistory,
    ]
):
    def _predict_core(
        self,
        X: Optional[ArrayLike],
        X_rel: List[RelationBlock] = [],
        n_workers: Optional[int] = None,
    ) -> DenseArray:

        predictor = self._fetch_predictor()
        shape = check_data_consistency(X, X_rel)
        if X is None:
            X = sps.csr_matrix((shape, 0), dtype=REAL)
        else:
            X = sps.csr_matrix(X)
        if n_workers is None:
            return predictor.predict(X, X_rel)
        else:
            return predictor.predict_parallel(X, X_rel, n_workers)

    @classmethod
    def _train_core(
        cls,
        rank: int,
        init_stdev: float,
        X: sps.csr_matrix,
        X_rel: List[RelationBlock],
        y: np.ndarray,
        random_seed: int,
        config: FMLearningConfig,
        callback: Callable[[int, FM, FMHyperParameters, LearningHistory], bool],
    ) -> Tuple[Predictor, LearningHistory]:
        return create_train_fm(
            rank, init_stdev, X, X_rel, y, random_seed, config, callback
        )

    def get_hyper_trace(self) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("Require pandas for get_hyper_trace.")

        if (self.n_groups_ is None) or (self.history_ is None):
            raise RuntimeError("Sampler not run yet.")

        columns = (
            ["alpha"]
            + ["mu_w[{}]".format(g) for g in range(self.n_groups_)]
            + ["lambda_w[{}]".format(g) for g in range(self.n_groups_)]
            + [
                "mu_V[{},{}]".format(g, r)
                for g in range(self.n_groups_)
                for r in range(self.rank)
            ]
            + [
                "lambda_V[{},{}]".format(g, r)
                for g in range(self.n_groups_)
                for r in range(self.rank)
            ]
        )

        res: List[DenseArray] = []
        for hyper in self.history_.hypers:
            res.append(
                np.concatenate(
                    [
                        np.asfarray([hyper.alpha]),
                        hyper.mu_w,
                        hyper.lambda_w,
                        hyper.mu_V.ravel(),
                        hyper.lambda_V.ravel(),
                    ]
                )
            )
        res_as_array = np.vstack(res)

        df = pd.DataFrame(res_as_array)
        df.columns = columns
        return df


class MyFMGibbsRegressor(RegressorMixin[FM, FMHyperParameters], MyFMGibbsBase):
    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_rel: List[RelationBlock] = [],
        X_test: Optional[ArrayLike] = None,
        y_test: Optional[np.ndarray] = None,
        X_rel_test: List[RelationBlock] = [],
        n_iter: int = 100,
        n_kept_samples: Optional[int] = None,
        grouping: Optional[List[int]] = None,
        group_shapes: Optional[List[int]] = None,
        callback: Optional[
            Callable[
                [int, FM, FMHyperParameters, LearningHistory],
                Tuple[bool, Optional[str]],
            ]
        ] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ) -> "MyFMGibbsRegressor":
        r"""Performs Gibbs sampling to fit the data.

        Parameters
        ----------
        X : 2D array-like.
            Input variable.

        y : 1D array-like.
            Target variable.

        X_rel: list of RelationBlock, optional (default=[])
               Relation blocks which supplements X.

        n_iter : int, optional (default = 100)
            Iterations to perform.

        n_kept_samples: int, optional (default = None)
            The number of samples to store.
            If `None`, the value is set to `n_iter` - 5.

        grouping: Integer List, optional (default = None)
            If not `None`, this specifies which column of X belongs to which group.
            That is, if grouping[i] is g, then, :math:`w_i` and :math:`V_{i, r}`
            will be distributed according to
            :math:`\mathcal{N}(\mu_w[g], \lambda_w[g])` and :math:`\mathcal{N}(\mu_V[g, r], \lambda_V[g,r])`,
            respectively.
            If `None`, all the columns of X are assumed to belong to a single group, 0.

        group_shapes: Integer array, optional (default = None)
            If not `None`, this specifies each variable group's size.
            Ignored if grouping is not None.
            For example, if ``group_shapes = [n_1, n_2]``,
            this is equivalent to ``grouping = [0] * n_1 + [1] * n_2``

        callback: function(int, fm, hyper, history) -> (bool, str), optional(default = None)
            Called at the every end of each Gibbs iteration.
        """
        self._fit(
            X,
            y,
            X_rel=X_rel,
            X_test=X_test,
            X_rel_test=X_rel_test,
            y_test=y_test,
            n_iter=n_iter,
            n_kept_samples=n_kept_samples,
            grouping=grouping,
            callback=callback,
            group_shapes=group_shapes,
            config_builder=config_builder,
        )
        return self

    def predict(
        self,
        X: Optional[ArrayLike],
        X_rel: List[RelationBlock] = [],
        n_workers: Optional[int] = None,
    ) -> DenseArray:
        """Make a prediction by compute the posterior predictive mean.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Main table. When None, treated as a matrix with no column.
        X_rel : List[RelationBlock]
            Relations.
        n_workers : Optional[int], optional
            The number of threads to compute the posterior predictive mean, by default None

        Returns
        -------
            One-dimensional array of predictions.
        """
        return self._predict(X, X_rel, n_workers=n_workers)


class MyFMGibbsClassifier(ClassifierMixin[FM, FMHyperParameters], MyFMGibbsBase):
    r"""Bayesian Factorization Machines for binary classification tasks."""

    def fit(
        self,
        X: ArrayLike,
        y: BinaryClassificationTarget,
        X_rel: List[RelationBlock] = [],
        X_test: Optional[ArrayLike] = None,
        y_test: Optional[np.ndarray] = None,
        X_rel_test: List[RelationBlock] = [],
        n_iter: int = 100,
        n_kept_samples: Optional[int] = None,
        grouping: Optional[List[int]] = None,
        group_shapes: Optional[List[int]] = None,
        callback: Optional[
            Callable[
                [int, FM, FMHyperParameters, LearningHistory],
                Tuple[bool, Optional[str]],
            ]
        ] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ) -> "MyFMGibbsClassifier":
        r"""Performs Gibbs sampling to fit the data.

        Parameters
        ----------
        X : 2D array-like.
            Input variable.

        y : 1D array-like.
            Target variable.

        X_rel: list of RelationBlock, optional (default=[])
            Relation blocks which supplements X.

        n_iter : int, optional (default = 100)
            Iterations to perform.

        n_kept_samples: int, optional (default = None)
            The number of samples to store.
            If `None`, the value is set to `n_iter` - 5.

        grouping: Integer List, optional (default = None)
            If not `None`, this specifies which column of X belongs to which group.
            That is, if grouping[i] is g, then, :math:`w_i` and :math:`V_{i, r}`
            will be distributed according to
            :math:`\mathcal{N}(\mu_w[g], \lambda_w[g])` and :math:`\mathcal{N}(\mu_V[g, r], \lambda_V[g,r])`,
            respectively.
            If `None`, all the columns of X are assumed to belong to a single group, 0.

        group_shapes: Integer array, optional (default = None)
            If not `None`, this specifies each variable group's size.
            Ignored if grouping is not None.
            For example, if ``group_shapes = [n_1, n_2]``,
            this is equivalent to ``grouping = [0] * n_1 + [1] * n_2``

        callback: function(int, fm, hyper, history) -> (bool, str), optional(default = None)
            Called at the every end of each Gibbs iteration.
        """
        self._fit(
            X,
            y,
            X_rel=X_rel,
            X_test=X_test,
            X_rel_test=X_rel_test,
            y_test=y_test,
            n_iter=n_iter,
            n_kept_samples=n_kept_samples,
            grouping=grouping,
            callback=callback,
            group_shapes=group_shapes,
            config_builder=config_builder,
        )
        return self

    def predict(
        self,
        X: Optional[ArrayLike],
        X_rel: List[RelationBlock] = [],
        n_workers: Optional[int] = None,
    ) -> BinaryClassificationTarget:
        """Based on the class probability, return binary classified outcome based on threshold = 0.5.
        If you want class probability instead, use `predict_proba` method.

        Parameters
        ----------
        Parameters
        ----------
        X : Optional[ArrayLike]
            When None, treated as a matrix with no column.
        X_rel : List[RelationBlock]
            Relations.
        n_workers : Optional[int], optional
            The number of threads to compute the posterior predictive mean, by default None

        Returns
        -------
        np.ndarray
            One-dimensional array of predicted outcomes.
        """
        return self._predict(X, X_rel, n_workers=n_workers)

    def predict_proba(
        self,
        X: Optional[ArrayLike],
        X_rel: List[RelationBlock] = [],
        n_workers: Optional[int] = None,
    ) -> DenseArray:
        """Compute the probability that the outcome will be 1 based on posterior predictive mean.

        Parameters
        ----------
        Parameters
        ----------
        X : Optional[ArrayLike]
            When None, treated as a matrix with no column.
        X_rel : List[RelationBlock]
            Relations.
        n_workers : Optional[int], optional
            The number of threads to compute the posterior predictive mean, by default None

        Returns
        -------
        np.ndarray
            One-dimensional array of probabilities.

        """
        return self._predict_proba(X, X_rel, n_workers=n_workers)


class MyFMOrderedProbit(MyFMGibbsBase):
    """Bayesian Factorization Machines for Ordinal Regression Tasks."""

    @property
    def _task_type(self) -> TaskType:
        return TaskType.ORDERED

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_rel: List[RelationBlock] = [],
        X_test: Optional[ArrayLike] = None,
        y_test: Optional[np.ndarray] = None,
        X_rel_test: List[RelationBlock] = [],
        n_iter: int = 100,
        n_kept_samples: Optional[int] = None,
        grouping: Optional[List[int]] = None,
        group_shapes: Optional[List[int]] = None,
        callback: Optional[
            Callable[
                [int, FM, FMHyperParameters, LearningHistory],
                Tuple[bool, Optional[str]],
            ]
        ] = None,
        callback_default_freq: int = 5,
    ) -> "MyFMOrderedProbit":
        config_builder = ConfigBuilder()
        y = np.asarray(y)
        n_class = y.max() + 1
        cutpoint_group_configs = [(int(n_class), list(range(y.shape[0])))]
        self.n_cutpoint_groups = len(cutpoint_group_configs)
        config_builder.set_cutpoint_groups(cutpoint_group_configs)
        super()._fit(
            X,
            y,
            X_rel=X_rel,
            X_test=X_test,
            y_test=y_test,
            X_rel_test=X_rel_test,
            n_iter=n_iter,
            n_kept_samples=n_kept_samples,
            grouping=grouping,
            callback=callback,
            group_shapes=group_shapes,
            config_builder=config_builder,
            callback_default_freq=callback_default_freq,
        )
        return self

    @classmethod
    def _score_to_class_prob(
        cls, score: np.ndarray, cutpoints: np.ndarray
    ) -> DenseArray:
        score = std_cdf((cutpoints[np.newaxis, :] - score[:, np.newaxis]))
        score_enlarged = np.hstack(
            [
                np.zeros((score.shape[0], 1), dtype=score.dtype),
                score,
                np.ones((score.shape[0], 1), dtype=score.dtype),
            ]
        )
        result: DenseArray = score_enlarged[:, 1:] - score_enlarged[:, :-1]
        return result

    def _prepare_prediction_for_test(
        self, fm: FM, X: ArrayLike, X_rel: List[RelationBlock]
    ) -> np.ndarray:
        score = fm.predict_score(X, X_rel)
        return self._score_to_class_prob(score, fm.cutpoints[0])

    def _process_y(self, y: np.ndarray) -> np.ndarray:
        y_as_float = y.astype(np.float64)
        assert y.min() >= 0
        return y_as_float

    def _measure_score(cls, prediction: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        result: Dict[str, float] = OrderedDict()
        accuracy = (np.argmax(prediction, axis=1) == y).mean()
        result["accuracy"] = accuracy
        log_loss = -np.log(
            prediction[np.arange(prediction.shape[0]), y.astype(np.int64)] + 1e-15
        ).mean()
        result["log_loss"] = log_loss
        return result

    def _status_report(cls, fm: FM, hyper: FMHyperParameters) -> str:
        log_str = "w0 = {:.2f}, ".format(fm.w0)
        if len(fm.cutpoints) == 1:
            log_str += "cutpoint = {} ".format(
                ["{:.3f}".format(c) for c in list(fm.cutpoints[0])]
            )
        return log_str

    def predict_proba(
        self,
        X: ArrayLike,
        X_rel: List[RelationBlock] = [],
    ) -> np.ndarray:
        """Compute the ordinal class probability.

        Parameters
        ----------
        X : array_like
            The input data.
        X_rel : List[RelationBlock], optional
            Relational Block part of the data., by default []

        Returns
        -------
        np.float
            The class probability
        """

        predictor = self._fetch_predictor()

        shape = check_data_consistency(X, X_rel)
        if X is None:
            X = sps.csr_matrix((shape, 0), dtype=REAL)
        else:
            X = sps.csr_matrix(X)

        if X.dtype != REAL:
            X.data = X.data.astype(REAL)

        p = 0.0
        for sample in predictor.samples:
            score = sample.predict_score(X, X_rel)
            p += self._score_to_class_prob(score, sample.cutpoints[0])
        return p / len(predictor.samples)

    def predict(
        self,
        X: ArrayLike,
        X_rel: List[RelationBlock] = [],
    ) -> DenseArray:
        r"""Predict the class outcome according to the class probability.

        Parameters
        ----------
        X : array_like
            The input data.
        X_rel : List[RelationBlock], optional
            Relational Block part of the data., by default []

        Returns
        -------
        np.int64
            The class prediction
        """

        return self.predict_proba(X, X_rel=X_rel).argmax(axis=1)
