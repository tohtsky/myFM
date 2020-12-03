from typing import Tuple, List, Callable, Optional
import warnings
import numpy as np
from .base import (
    MyFMBase,
    RegressorMixin,
    ClassifierMixin,
    ArrayLike,
    check_data_consistency,
    REAL,
)
import scipy.sparse as sps
from ._myfm import (
    ConfigBuilder,
    FMLearningConfig,
    RelationBlock,
    VariationalFM,
    VariationalFMHyperParameters,
    VariationalPredictor,
    VariationalLearningHistory,
    create_train_vfm,
)


class MyFMVariationalBase(
    MyFMBase[
        VariationalFM,
        VariationalFMHyperParameters,
        VariationalPredictor,
        VariationalLearningHistory,
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
        callback: Callable[
            [
                int,
                VariationalFM,
                VariationalFMHyperParameters,
                VariationalLearningHistory,
            ],
            bool,
        ],
    ) -> Tuple[VariationalPredictor, VariationalLearningHistory]:
        return create_train_vfm(
            rank, init_stdev, X, X_rel, y, random_seed, config, callback
        )

    def _predict_core(
        self,
        X: Optional[ArrayLike],
        X_rel: List[RelationBlock] = [],
        n_workers: Optional[int] = None,
    ) -> np.ndarray:
        if n_workers is not None:
            warnings.warn(
                "Currently variational api does not support multi-thread prediction."
            )
        predictor = self._fetch_predictor()
        shape = check_data_consistency(X, X_rel)
        if X is None:
            X = sps.csr_matrix((shape, 0), dtype=REAL)
        else:
            X = sps.csr_matrix(X)
        return predictor.predict(X, X_rel)


class VariationalFMRegressor(
    RegressorMixin[VariationalFM, VariationalFMHyperParameters],
    MyFMVariationalBase,
):
    """Variational Inference for Regression Task."""

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
                [
                    int,
                    VariationalFM,
                    VariationalFMHyperParameters,
                    VariationalLearningHistory,
                ],
                Tuple[bool, Optional[str]],
            ]
        ] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ) -> "VariationalFMRegressor":
        """Performs batch variational inference fit the data.

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

        callback: function(int, fm, hyper, history) -> bool, optional(default = None)
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
        self, X: Optional[ArrayLike], X_rel: List[RelationBlock] = []
    ) -> np.ndarray:
        """Make a prediction based on variational mean.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Main Table. When None, treated as a matrix without columns.
        X_rel : List[RelationBlock], optional
            Relations, by default []

        Returns
        -------
        np.ndarray
            [description]
        """
        return self._predict(X, X_rel)


class VariationalFMClassifier(
    ClassifierMixin[VariationalFM, VariationalFMHyperParameters],
    MyFMVariationalBase,
):
    """Variational Inference for Classification Task."""

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
                [
                    int,
                    VariationalFM,
                    VariationalFMHyperParameters,
                    VariationalLearningHistory,
                ],
                Tuple[bool, Optional[str]],
            ]
        ] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ) -> "VariationalFMClassifier":
        """Performs batch variational inference fit the data.

        Parameters
        ----------
        X : Optional[ArrayLike].
            Main table. When None, treated as a matrix without columns.

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
        self, X: Optional[ArrayLike], X_rel: List[RelationBlock] = []
    ) -> np.ndarray:
        """Based on the class probability, return binary classified outcome based on threshold = 0.5.
        If you want class probability instead, use `predict_proba` method.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Main Table. When None, treated as a matrix without columns.
        X_rel : List[RelationBlock], optional
            Relations, by default []

        Returns
        -------
        np.ndarray
            0/1 predictions based on the probability.
        """
        return self._predict(X, X_rel)

    def predict_proba(
        self, X: Optional[ArrayLike], X_rel: List[RelationBlock] = []
    ) -> np.ndarray:
        """Compute the probability that the outcome will be 1 based on variational mean.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Main Table. When None, treated as a matrix without columns.
        X_rel : List[RelationBlock], optional
            Relations, by default []

        Returns
        -------
        np.ndarray
            the probability.
        """
        return self._predict_proba(X, X_rel)
