from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from tqdm import tqdm
from scipy import sparse as sps

from . import _myfm
from ._myfm import ConfigBuilder, FMLearningConfig, RelationBlock, TaskType

REAL = np.float64

ArrayLike = Union[np.ndarray, sps.csr_matrix]


def _infinite_int_generator():
    i: int = 0
    while True:
        yield i
        i += 1


def check_data_consistency(
    X: Optional[ArrayLike], X_rel: List[RelationBlock]
) -> int:
    shape: Optional[int] = None
    if X_rel:
        shape_rel_all = {rel.mapper_size for rel in X_rel}
        if len(shape_rel_all) > 1:
            raise ValueError("Inconsistent case size for X_rel.")
        shape = list(shape_rel_all)[0]
        if X is not None:
            if X.shape[0] != shape:
                raise ValueError("X and X_rel have different shape.")
        return shape
    else:
        if X is None:
            raise ValueError("At lease X or X_rel must be provided.")
        shape = int(X.shape[0])
        return shape


FM = TypeVar("FM", _myfm.FM, _myfm.VariationalFM)
Predictor = TypeVar("Predictor", _myfm.Predictor, _myfm.VariationalPredictor)
History = TypeVar(
    "History", _myfm.LearningHistory, _myfm.VariationalLearningHistory
)
Hyper = TypeVar(
    "Hyper", _myfm.FMHyperParameters, _myfm.VariationalFMHyperParameters
)

CallBackType = Callable[[int, FM, Hyper], bool]


class MyFMBase(Generic[FM, Hyper, Predictor, History], ABC):
    r"""Bayesian Factorization Machines for regression tasks."""
    _create_function: Callable[
        [
            int,  # rank,
            float,  # init_stdev
            sps.csr_matrix,  # X
            List[RelationBlock],  # X_rel
            np.ndarray,  # y
            int,  # random_seed
            FMLearningConfig,  # config
            Callable[[int, FM, Hyper], bool],  # callback
        ],
        Tuple[Any, Any],
    ]

    @abstractproperty
    def _task_type(self) -> TaskType:
        raise NotImplementedError("must be specified in child")

    def __init__(
        self,
        rank: int,
        init_stdev: float = 0.1,
        random_seed: int = 42,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma_0: float = 1.0,
        mu_0: float = 0.0,
        reg_0: float = 1.0,
    ):
        """Setup the configuration.

        Parameters
        ----------
        rank : int
            The number of factors.

        init_stdev : float, optional (defalult = 0.1)
            The standard deviation for initialization.
            The factorization machine weights are randomely sampled from
            `Normal(0, init_stdev ** 2)`.

        random_seed : integer, optional (default = 0.1)
            The random seed used inside the whole learning process.

        alpha_0 : float, optional (default = 1.0)
            The half of alpha parameter for the gamma-distribution
            prior for alpha, lambda_w and lambda_V.
            Together with beta_0, the priors for these parameters are
            alpha, lambda_w, lambda_v ~ Gamma(alpha_0 / 2, beta_0 / 2)

        beta_0 : float, optioal (default = 1.0)
            See the explanation for alpha_0 .

        gamma_0: float optional (default = 1.0)
            Inverse variance of the prior for mu_w, mu_v.
            Together with mu_0, the priors for these parameters are
            mu_w, mu_v ~ Normal(mu_0, 1 / gamma_0)

        mu_0:
            See the explanation for gamma_0.

        reg_0:
            Inverse variance of tthe prior for w0.
            w0 ~ Normal(0, 1 / reg_0)
        """
        self.rank = rank

        self.init_stdev = init_stdev
        self.random_seed = random_seed

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma_0 = gamma_0
        self.mu_0 = mu_0

        self.reg_0 = reg_0

        self.predictor_: Optional[Predictor] = None
        self.history_: Optional[History] = None

        self.n_groups_: Optional[int] = None

    def __str__(self):
        return "{class_name}(init_stdev={init_stdev}, alpha_0={alpha_0}, beta_0={beta_0}, gamma_0={gamma_0}, mu_0={mu_0}, reg_0={reg_0})".format(
            class_name=self.__class__.__name__,
            init_stdev=self.init_stdev,
            alpha_0=self.alpha_0,
            beta_0=self.beta_0,
            gamma_0=self.gamma_0,
            mu_0=self.mu_0,
            reg_0=self.reg_0,
        )

    def _create_default_callback(
        self,
        pbar: tqdm,
        callback_default_freq: int,
        do_test: float,
        X_test: Optional[sps.csr_matrix] = None,
        X_rel_test: List[RelationBlock] = [],
        y_test: Optional[np.ndarray] = None,
    ) -> Callable[[int, FM, Hyper], bool]:
        def callback(i: int, fm: FM, hyper: Hyper) -> bool:
            pbar.update(1)

            if i % callback_default_freq:
                return False

            log_str = self._status_report(fm, hyper)

            if do_test:
                pred_this = self._process_score(
                    fm.predict_score(X_test, X_rel_test or [])
                )
                val_results = self._measure_score(pred_this, y_test)
                for key, metric in val_results.items():
                    log_str += " {}_this: {:.2f}".format(key, metric)

            pbar.set_description(log_str)
            return False

        return callback

    def _fit(
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
        callback: Optional[Callable[[int, FM, Hyper], bool]] = None,
        config_builder: Optional[ConfigBuilder] = None,
        callback_default_freq: int = 10,
    ) -> "MyFMBase":
        """Performs Gibbs sampling to fit the data.

        Parameters
        ----------
        X : 2D array-like.
            Input variable.

        y : 1D array-like.
            Target variable.

        X_rel: list of RelationBlock, optional (defalult=[])
               Relation blocks which supplements X.

        n_iter : int, optional (defalult = 100)
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

        if config_builder is None:
            config_builder = ConfigBuilder()
        train_size = check_data_consistency(X, X_rel)
        if X is None:
            X = sps.csr_matrix((train_size, 0), dtype=REAL)

        assert X.shape[0] == y.shape[0]
        dim_all = X.shape[1] + sum([rel.feature_size for rel in X_rel])

        if n_kept_samples is None:
            n_kept_samples = n_iter - 10
        else:
            assert n_iter >= n_kept_samples

        for key in ["alpha_0", "beta_0", "gamma_0", "mu_0", "reg_0"]:
            value = getattr(self, key)
            getattr(config_builder, "set_{}".format(key))(value)

        if group_shapes is not None and grouping is None:
            grouping = [
                i for i, gsize in enumerate(group_shapes) for _ in range(gsize)
            ]

        if grouping is None:
            self.n_groups_ = 1
            config_builder.set_identical_groups(dim_all)
        else:
            assert dim_all == len(grouping)
            self.n_groups_ = np.unique(grouping).shape[0]
            config_builder.set_group_index(grouping)

        if X_test is not None or X_rel_test:
            if y_test is None:
                raise RuntimeError(
                    "Must specify both (X_test or X_rel_test) and y_test."
                )
            test_size = check_data_consistency(X_test, X_rel_test)
            assert test_size == y_test.shape[0]
            if X_test is None:
                X_test = sps.csr_matrix((test_size, 0), dtype=np.float64)
            else:
                X_test = sps.csr_matrix(X_test)
            do_test = True
        elif y_test is not None:
            raise RuntimeError(
                "Must specify both (X_test or X_rel_test) and y_test."
            )
        else:
            do_test = False

        config_builder.set_n_iter(n_iter).set_n_kept_samples(n_kept_samples)

        X = sps.csr_matrix(X)
        if X.dtype != np.float64:
            X.data = X.data.astype(np.float64)
        y = self._process_y(y)
        self._set_tasktype(config_builder)

        config = config_builder.build()

        pbar = None
        if callback is None:
            pbar = tqdm(total=n_iter)
            callback = self._create_default_callback(
                pbar,
                callback_default_freq=callback_default_freq,
                do_test=do_test,
                X_test=X_test,
                X_rel_test=X_rel_test,
                y_test=y_test,
            )

        self.predictor_, self.history_ = self.__class__._create_function(
            self.rank,
            self.init_stdev,
            X,
            X_rel,
            y,
            self.random_seed,
            config,
            callback,
        )
        return self

    def _set_tasktype(self, config_builder: ConfigBuilder) -> None:
        config_builder.set_task_type(self._task_type)

    def predict(
        self,
        X: ArrayLike,
        X_rel: List[RelationBlock] = [],
        n_workers: Optional[int] = None,
    ) -> np.ndarray:
        """Predict the outcome by posterior mean.

        Parameters
        ----------
        X : array-like
            input matrix.
        X_rel : list, optional
            Relation blocks that supplements X, by default []
        n_workers : [int], optional
            if not None, compute the prediction of each Gibbs sample on
            different threads, by default None

        Returns
        -------
        np.float64
            The prediction value.

        """
        shape = check_data_consistency(X, X_rel)
        if self.predictor_ is None:
            raise RuntimeError("MyFM instance not fit yet.")
        if X is None:
            X = sps.csr_matrix((shape, 0), dtype=np.float64)
        if n_workers is not None:
            if isinstance(self.predictor_, _myfm.Predictor):
                return self.predictor_.predict_parallel(X, X_rel, n_workers)
            else:
                return self.predictor_.predict(
                    X,
                    X_rel,
                )
        else:
            return self.predictor_.predict(X, X_rel)

    @abstractmethod
    def _status_report(cls, fm: FM, hyper: Hyper):
        raise NotImplementedError("must implement status report")

    def _process_score(cls, y):
        return y

    def _process_y(cls, y):
        return y.astype(np.float64)

    @abstractmethod
    def _measure_score(cls, prediction: np.ndarray, y: np.ndarray):
        raise NotImplementedError("")
