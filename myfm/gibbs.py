from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import sparse as sps
from scipy import special

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
    std_cdf,
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
        callback: Callable[[int, FM, FMHyperParameters], bool],
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

        res = []
        for hyper in self.history_.hypers:
            res.append(
                np.concatenate(
                    [
                        [hyper.alpha],
                        hyper.mu_w,
                        hyper.lambda_w,
                        hyper.mu_V.ravel(),
                        hyper.lambda_V.ravel(),
                    ]
                )
            )
        res = np.vstack(res)

        df = pd.DataFrame(res)
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
        callback: Optional[Callable[[int, FM, FMHyperParameters], bool]] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ):
        """Performs Gibbs sampling to fit the data.

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

        callback: function(int, fm, hyper) -> bool, optional(default = None)
            Called at the every end of each Gibbs iteration.
        """
        return self._fit(
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


class MyFMGibbsClassifier(
    ClassifierMixin[FM, FMHyperParameters], MyFMGibbsBase
):
    r"""Bayesian Factorization Machines for binary classification tasks."""

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
        callback: Optional[Callable[[int, FM, FMHyperParameters], bool]] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ):
        """Performs Gibbs sampling to fit the data.

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

        callback: function(int, fm, hyper) -> bool, optional(default = None)
            Called at the every end of each Gibbs iteration.
        """
        return self._fit(
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


class MyFMOrderedProbit(MyFMGibbsBase):
    """Bayesian Factorization Machines for Ordinal Regression Tasks."""

    @property
    def _task_type(self):
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
        callback: Optional[Callable[[int, FM, FMHyperParameters], bool]] = None,
        cutpoint_group_configs: Optional[List[Tuple[int, np.ndarray]]] = None,
        callback_default_freq: int = 5,
    ):
        config_builder = ConfigBuilder()
        y = np.asarray(y)
        if cutpoint_group_configs is None:
            n_class = y.max() + 1
            cutpoint_group_configs = [
                (int(n_class), np.arange(y.shape[0], dtype=np.int64))
            ]
        self.n_cutpoint_groups = len(cutpoint_group_configs)
        config_builder.set_cutpoint_groups(cutpoint_group_configs)
        return super()._fit(
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

    def _process_score(self, score):
        return (1 + special.erf(score * np.sqrt(0.5))) / 2

    def _process_y(self, y):
        y_as_float = y.astype(np.float64)
        assert y.min() >= 0
        return y_as_float

    def _measure_score(self, prediction, y):
        raise NotImplementedError("not implemented")

    def _status_report(cls, fm: FM, hyper: FMHyperParameters):
        log_str = "w0= {:2f}".format(fm.w0)
        if len(fm.cutpoints) == 1:
            log_str += ", cutpoint = {} ".format(
                ["{:.3f}".format(c) for c in list(fm.cutpoints[0])]
            )
        return log_str

    def predict_proba(
        self,
        X: ArrayLike,
        X_rel: List[RelationBlock] = [],
        cutpoint_index: Optional[int] = None,
    ):
        """Compute the ordinal class probability.

        Parameters
        ----------
        X : array_like
            The input data.
        X_rel : List[RelationBlock], optional
            Relational Block part of the data., by default []
        cutpoint_index : int, optional
            if not ``None`` and multiple cutpoints are enabled
            when ``fit``, compute the class probability
            based on the ``cutpoint_index``-th cutpoint, by default None.
            Must not be ``None`` when there are multiple cutpoints.

        Returns
        -------
        np.float
            The class probability
        """

        if self.predictor_ is None:
            raise RuntimeError("Not fit yet.")

        if cutpoint_index is None:
            if self.n_cutpoint_groups == 1:
                cutpoint_index = 0
            else:
                raise ValueError("specify the cutpoint index")

        X = sps.csr_matrix(X)
        if X.dtype != np.float64:
            X.data = X.data.astype(np.float64)
        p = 0

        for sample in self.predictor_.samples:
            score = sample.predict_score(X, X_rel)
            score = std_cdf(
                (
                    sample.cutpoints[cutpoint_index][np.newaxis, :]
                    - score[:, np.newaxis]
                )
            )
            score = np.hstack(
                [
                    np.zeros((score.shape[0], 1), dtype=score.dtype),
                    score,
                    np.ones((score.shape[0], 1), dtype=score.dtype),
                ]
            )
            p += score[:, 1:] - score[:, :-1]
        return p / len(self.predictor_.samples)

    def predict(
        self,
        X: ArrayLike,
        X_rel: List[RelationBlock] = [],
        cutpoint_index: Optional[int] = None,
    ):
        """Predict the class outcome according to the class probability.

        Parameters
        ----------
        X : array_like
            The input data.
        X_rel : List[RelationBlock], optional
            Relational Block part of the data., by default []
        cutpoint_index : int, optional
            if not ``None`` and multiple cutpoints are enabled
            when ``fit``, compute the class probability
            based on the ``cutpoint_index``-th cutpoint, by default None
            Must not be ``None`` when there are multiple cutpoints.

        Returns
        -------
        np.int64
            The class prediction
        """

        return self.predict_proba(
            X, X_rel=X_rel, cutpoint_index=cutpoint_index
        ).argmax(axis=1)
