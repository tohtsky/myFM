from typing import Tuple, List, Callable, Optional
import numpy as np
from .base import MyFMBase, RegressorMixin, ClassifierMixin, ArrayLike
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
            [int, VariationalFM, VariationalFMHyperParameters], bool
        ],
    ) -> Tuple[VariationalPredictor, VariationalLearningHistory]:
        return create_train_vfm(
            rank, init_stdev, X, X_rel, y, random_seed, config, callback
        )


class VariationalFMRegressor(
    RegressorMixin[VariationalFM, VariationalFMHyperParameters],
    MyFMVariationalBase,
):
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
            Callable[[int, VariationalFM, VariationalFMHyperParameters], bool]
        ] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ):
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


class VariationalFMClassifier(
    ClassifierMixin[VariationalFM, VariationalFMHyperParameters],
    MyFMVariationalBase,
):
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
            Callable[[int, VariationalFM, VariationalFMHyperParameters], bool]
        ] = None,
        config_builder: Optional[ConfigBuilder] = None,
    ):
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
