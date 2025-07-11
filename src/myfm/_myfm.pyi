import enum
from collections.abc import Callable, Sequence
from typing import Annotated

import scipy.sparse
from numpy.typing import ArrayLike

class TaskType(enum.Enum):
    REGRESSION = 0

    CLASSIFICATION = 1

    ORDERED = 2

class FMLearningConfig:
    pass

class RelationBlock:
    """The RelationBlock Class."""

    def __init__(
        self, original_to_block: Sequence[int], data: scipy.sparse.csr_matrix[float]
    ) -> None:
        """
        Initializes relation block.

        Parameters
        ----------

        original_to_block: List[int]
            describes which entry points to to which row of the data (second argument).
        data: scipy.sparse.csr_matrix[float64]
            describes repeated pattern.

        Note
        -----
        The entries of `original_to_block` must be in the [0, data.shape[0]-1].
        """

    @property
    def original_to_block(self) -> list[int]: ...
    @property
    def data(self) -> scipy.sparse.csr_matrix[float]: ...
    @property
    def mapper_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def feature_size(self) -> int: ...
    def __repr__(self) -> str: ...
    def __getstate__(self) -> tuple[list[int], scipy.sparse.csr_matrix[float]]: ...
    def __setstate__(
        self, arg: tuple[Sequence[int], scipy.sparse.csr_matrix[float]], /
    ) -> None: ...

class ConfigBuilder:
    def __init__(self) -> None: ...
    def set_alpha_0(self, arg: float, /) -> None: ...
    def set_beta_0(self, arg: float, /) -> None: ...
    def set_gamma_0(self, arg: float, /) -> None: ...
    def set_mu_0(self, arg: float, /) -> None: ...
    def set_reg_0(self, arg: float, /) -> None: ...
    def set_n_iter(self, arg: int, /) -> None: ...
    def set_n_kept_samples(self, arg: int, /) -> None: ...
    def set_task_type(self, arg: TaskType, /) -> None: ...
    def set_nu_oprobit(self, arg: int, /) -> None: ...
    def set_fit_w0(self, arg: bool, /) -> None: ...
    def set_fit_linear(self, arg: bool, /) -> None: ...
    def set_group_index(self, arg: Sequence[int], /) -> None: ...
    def set_identical_groups(self, arg: int, /) -> None: ...
    def set_cutpoint_scale(self, arg: float, /) -> None: ...
    def set_cutpoint_groups(
        self, arg: Sequence[tuple[int, Sequence[int]]], /
    ) -> None: ...
    def build(self) -> FMLearningConfig: ...

class FM:
    @property
    def w0(self) -> float: ...
    @w0.setter
    def w0(self, arg: float, /) -> None: ...
    @property
    def w(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @w.setter
    def w(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        /,
    ) -> None: ...
    @property
    def V(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @V.setter
    def V(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        /,
    ) -> None: ...
    @property
    def cutpoints(
        self,
    ) -> list[Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]]: ...
    @cutpoints.setter
    def cutpoints(
        self,
        arg: Sequence[
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]
        ],
        /,
    ) -> None: ...
    def predict_score(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: Sequence[RelationBlock], /
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    def oprobit_predict_proba(
        self,
        arg0: scipy.sparse.csr_matrix[float],
        arg1: Sequence[RelationBlock],
        arg2: int,
        /,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    def __repr__(self) -> str: ...
    def __getstate__(
        self,
    ) -> tuple[
        float,
        Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        list[Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]],
    ]: ...
    def __setstate__(
        self,
        arg: tuple[
            float,
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Sequence[
                Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]
            ],
        ],
        /,
    ) -> None: ...

class VariationalFM:
    @property
    def w0(self) -> float: ...
    @w0.setter
    def w0(self, arg: float, /) -> None: ...
    @property
    def w0_var(self) -> float: ...
    @w0_var.setter
    def w0_var(self, arg: float, /) -> None: ...
    @property
    def w(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @w.setter
    def w(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        /,
    ) -> None: ...
    @property
    def w_var(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @w_var.setter
    def w_var(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        /,
    ) -> None: ...
    @property
    def V(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @V.setter
    def V(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        /,
    ) -> None: ...
    @property
    def V_var(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @V_var.setter
    def V_var(
        self,
        arg: Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        /,
    ) -> None: ...
    @property
    def cutpoints(
        self,
    ) -> list[Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]]: ...
    @cutpoints.setter
    def cutpoints(
        self,
        arg: Sequence[
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]
        ],
        /,
    ) -> None: ...
    def predict_score(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: Sequence[RelationBlock], /
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    def __repr__(self) -> str: ...
    def __getstate__(
        self,
    ) -> tuple[
        float,
        float,
        Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        list[Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]],
    ]: ...
    def __setstate__(
        self,
        arg: tuple[
            float,
            float,
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Sequence[
                Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]
            ],
        ],
        /,
    ) -> None: ...

class FMHyperParameters:
    @property
    def alpha(self) -> float: ...
    @property
    def mu_w(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @property
    def lambda_w(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @property
    def mu_V(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @property
    def lambda_V(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    def __getstate__(
        self,
    ) -> tuple[
        float,
        Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
    ]: ...
    def __setstate__(
        self,
        arg: tuple[
            float,
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        ],
        /,
    ) -> None: ...

class VariationalFMHyperParameters:
    @property
    def alpha(self) -> float: ...
    @property
    def alpha_rate(self) -> float: ...
    @property
    def mu_w(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @property
    def mu_w_var(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @property
    def lambda_w(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @property
    def lambda_w_rate(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    @property
    def mu_V(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @property
    def mu_V_var(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @property
    def lambda_V(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    @property
    def lambda_V_rate(
        self,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(
        self,
        arg: tuple[
            float,
            float,
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
            Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")],
        ],
        /,
    ) -> None: ...

class Predictor:
    @property
    def samples(self) -> list[FM]: ...
    def predict(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: Sequence[RelationBlock], /
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    def predict_parallel(
        self,
        arg0: scipy.sparse.csr_matrix[float],
        arg1: Sequence[RelationBlock],
        arg2: int,
        /,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    def predict_parallel_oprobit(
        self,
        arg0: scipy.sparse.csr_matrix[float],
        arg1: Sequence[RelationBlock],
        arg2: int,
        arg3: int,
        /,
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None, None), order="F")]: ...
    def __getstate__(self) -> tuple[int, int, int, list[FM]]: ...
    def __setstate__(self, arg: tuple[int, int, TaskType, Sequence[FM]], /) -> None: ...

class VariationalPredictor:
    def predict(
        self, arg0: scipy.sparse.csr_matrix[float], arg1: Sequence[RelationBlock], /
    ) -> Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")]: ...
    def weights(self) -> VariationalFM: ...
    def __getstate__(self) -> tuple[int, int, int, list[VariationalFM]]: ...
    def __setstate__(
        self, arg: tuple[int, int, TaskType, Sequence[VariationalFM]], /
    ) -> None: ...

class FMTrainer:
    def __init__(
        self,
        arg0: scipy.sparse.csr_matrix[float],
        arg1: Sequence[RelationBlock],
        arg2: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        arg3: int,
        arg4: FMLearningConfig,
        /,
    ) -> None: ...
    def create_FM(self, arg0: int, arg1: float, /) -> FM: ...
    def create_Hyper(self, arg: int, /) -> FMHyperParameters: ...

class VariationalFMTrainer:
    def __init__(
        self,
        arg0: scipy.sparse.csr_matrix[float],
        arg1: Sequence[RelationBlock],
        arg2: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
        arg3: int,
        arg4: FMLearningConfig,
        /,
    ) -> None: ...
    def create_FM(self, arg0: int, arg1: float, /) -> VariationalFM: ...
    def create_Hyper(self, arg: int, /) -> VariationalFMHyperParameters: ...

class LearningHistory:
    @property
    def hypers(self) -> list[FMHyperParameters]: ...
    @property
    def train_log_losses(self) -> list[float]: ...
    @property
    def n_mh_accept(self) -> list[int]: ...
    def __getstate__(
        self,
    ) -> tuple[list[FMHyperParameters], list[float], list[int]]: ...
    def __setstate__(
        self, arg: tuple[Sequence[FMHyperParameters], Sequence[float], Sequence[int]], /
    ) -> None: ...

class VariationalLearningHistory:
    @property
    def hypers(self) -> FMHyperParameters: ...
    @property
    def elbos(self) -> list[float]: ...
    def __getstate__(self) -> tuple[FMHyperParameters, list[float]]: ...
    def __setstate__(
        self, arg: tuple[FMHyperParameters, Sequence[float]], /
    ) -> None: ...

def create_train_fm(
    arg0: int,
    arg1: float,
    arg2: scipy.sparse.csr_matrix[float],
    arg3: Sequence[RelationBlock],
    arg4: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
    arg5: int,
    arg6: FMLearningConfig,
    arg7: Callable[[int, FM, FMHyperParameters, LearningHistory], bool],
    /,
) -> tuple[Predictor, LearningHistory]:
    """create and train fm."""

def create_train_vfm(
    rank: int,
    init_std: float,
    X: scipy.sparse.csr_matrix[float],
    relations: Sequence[RelationBlock],
    y: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")],
    random_seed: int,
    learning_config: FMLearningConfig,
    callback: Callable[
        [int, VariationalFM, VariationalFMHyperParameters, VariationalLearningHistory],
        bool,
    ],
) -> tuple[VariationalPredictor, VariationalLearningHistory]:
    """create and train fm."""

def mean_var_truncated_normal_left(arg: float, /) -> tuple[float, float, float]: ...
def mean_var_truncated_normal_right(arg: float, /) -> tuple[float, float, float]: ...
