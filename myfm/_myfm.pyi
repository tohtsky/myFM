"""Backend C++ implementation for myfm."""
import myfm._myfm
from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64

_Shape = Tuple[int, ...]
import numpy
import scipy.sparse

__all__ = [
    "ConfigBuilder",
    "FM",
    "FMHyperParameters",
    "FMLearningConfig",
    "FMTrainer",
    "LearningHistory",
    "Predictor",
    "RelationBlock",
    "TaskType",
    "VariationalFM",
    "VariationalFMHyperParameters",
    "VariationalFMTrainer",
    "VariationalLearningHistory",
    "VariationalPredictor",
    "create_train_fm",
    "create_train_vfm",
    "mean_var_truncated_normal_left",
    "mean_var_truncated_normal_right",
]

m: int
n: int


class ConfigBuilder:
    def __init__(self) -> None:
        ...

    def build(self) -> FMLearningConfig:
        ...

    def set_alpha_0(self, arg0: float) -> ConfigBuilder:
        ...

    def set_beta_0(self, arg0: float) -> ConfigBuilder:
        ...

    def set_cutpoint_groups(
        self, arg0: List[Tuple[int, List[int]]]
    ) -> ConfigBuilder:
        ...

    def set_cutpoint_scale(self, arg0: float) -> ConfigBuilder:
        ...

    def set_gamma_0(self, arg0: float) -> ConfigBuilder:
        ...

    def set_group_index(self, arg0: List[int]) -> ConfigBuilder:
        ...

    def set_identical_groups(self, arg0: int) -> ConfigBuilder:
        ...

    def set_mu_0(self, arg0: float) -> ConfigBuilder:
        ...

    def set_n_iter(self, arg0: int) -> ConfigBuilder:
        ...

    def set_n_kept_samples(self, arg0: int) -> ConfigBuilder:
        ...

    def set_nu_oprobit(self, arg0: int) -> ConfigBuilder:
        ...

    def set_reg_0(self, arg0: float) -> ConfigBuilder:
        ...

    def set_task_type(self, arg0: TaskType) -> ConfigBuilder:
        ...

    pass


class FM:
    def __getstate__(self) -> tuple:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    def predict_score(
        self, arg0: scipy.sparse.csr_matrix[float64], arg1: List[RelationBlock]
    ) -> numpy.ndarray[float64, _Shape[m, 1]]:
        ...

    @property
    def V(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @V.setter
    def V(self, arg0: numpy.ndarray[float64, _Shape[m, n]]) -> None:
        pass

    @property
    def cutpoints(self) -> List[numpy.ndarray[float64, _Shape[m, 1]]]:
        """
        :type: List[numpy.ndarray[float64, _Shape[m, 1]]]
        """

    @cutpoints.setter
    def cutpoints(
        self, arg0: List[numpy.ndarray[float64, _Shape[m, 1]]]
    ) -> None:
        pass

    @property
    def w(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @w.setter
    def w(self, arg0: numpy.ndarray[float64, _Shape[m, 1]]) -> None:
        pass

    @property
    def w0(self) -> float:
        """
        :type: float
        """

    @w0.setter
    def w0(self, arg0: float) -> None:
        pass

    pass


class FMHyperParameters:
    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def alpha(self) -> float:
        """
        :type: float
        """

    @property
    def lambda_V(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @property
    def lambda_w(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @property
    def mu_V(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @property
    def mu_w(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    pass


class FMLearningConfig:
    pass


class FMTrainer:
    def __init__(
        self,
        arg0: scipy.sparse.csr_matrix[float64],
        arg1: List[RelationBlock],
        arg2: numpy.ndarray[float64, _Shape[m, 1]],
        arg3: int,
        arg4: FMLearningConfig,
    ) -> None:
        ...

    def create_FM(self, arg0: int, arg1: float) -> FM:
        ...

    def create_Hyper(self, arg0: int) -> FMHyperParameters:
        ...

    pass


class LearningHistory:
    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def hypers(self) -> List[FMHyperParameters]:
        """
        :type: List[FMHyperParameters]
        """

    @property
    def n_mh_accept(self) -> List[int]:
        """
        :type: List[int]
        """

    @property
    def train_log_losses(self) -> List[float]:
        """
        :type: List[float]
        """

    pass


class Predictor:
    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    def predict(
        self, arg0: scipy.sparse.csr_matrix[float64], arg1: List[RelationBlock]
    ) -> numpy.ndarray[float64, _Shape[m, 1]]:
        ...

    def predict_parallel(
        self,
        arg0: scipy.sparse.csr_matrix[float64],
        arg1: List[RelationBlock],
        arg2: int,
    ) -> numpy.ndarray[float64, _Shape[m, 1]]:
        ...

    @property
    def samples(self) -> List[FM]:
        """
        :type: List[FM]
        """

    pass


class RelationBlock:
    """
    The RelationBlock Class.
    """

    def __getstate__(self) -> tuple:
        ...

    def __init__(
        self,
        original_to_block: List[int],
        data: scipy.sparse.csr_matrix[float64],
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

    def __repr__(self) -> str:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def block_size(self) -> int:
        """
        :type: int
        """

    @property
    def data(self) -> scipy.sparse.csr_matrix[float64]:
        """
        :type: scipy.sparse.csr_matrix[float64]
        """

    @property
    def feature_size(self) -> int:
        """
        :type: int
        """

    @property
    def mapper_size(self) -> int:
        """
        :type: int
        """

    @property
    def original_to_block(self) -> List[int]:
        """
        :type: List[int]
        """

    pass


class TaskType:
    """
    Members:

      REGRESSION

      CLASSIFICATION

      ORDERED
    """

    def __init__(self, arg0: int) -> None:
        ...

    def __int__(self) -> int:
        ...

    @property
    def name(self) -> str:
        """
        (self: handle) -> str

        :type: str
        """

    CLASSIFICATION: myfm._myfm.TaskType  # value = TaskType.CLASSIFICATION
    ORDERED: myfm._myfm.TaskType  # value = TaskType.ORDERED
    REGRESSION: myfm._myfm.TaskType  # value = TaskType.REGRESSION
    __entries: dict  # value = {'REGRESSION': (TaskType.REGRESSION, None), 'CLASSIFICATION': (TaskType.CLASSIFICATION, None), 'ORDERED': (TaskType.ORDERED, None)}
    __members__: dict  # value = {'REGRESSION': TaskType.REGRESSION, 'CLASSIFICATION': TaskType.CLASSIFICATION, 'ORDERED': TaskType.ORDERED}
    pass


class VariationalFM:
    def __getstate__(self) -> tuple:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    def predict_score(
        self, arg0: scipy.sparse.csr_matrix[float64], arg1: List[RelationBlock]
    ) -> numpy.ndarray[float64, _Shape[m, 1]]:
        ...

    @property
    def V(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @V.setter
    def V(self, arg0: numpy.ndarray[float64, _Shape[m, n]]) -> None:
        pass

    @property
    def V_var(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @V_var.setter
    def V_var(self, arg0: numpy.ndarray[float64, _Shape[m, n]]) -> None:
        pass

    @property
    def cutpoints(self) -> List[numpy.ndarray[float64, _Shape[m, 1]]]:
        """
        :type: List[numpy.ndarray[float64, _Shape[m, 1]]]
        """

    @cutpoints.setter
    def cutpoints(
        self, arg0: List[numpy.ndarray[float64, _Shape[m, 1]]]
    ) -> None:
        pass

    @property
    def w(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @w.setter
    def w(self, arg0: numpy.ndarray[float64, _Shape[m, 1]]) -> None:
        pass

    @property
    def w0(self) -> float:
        """
        :type: float
        """

    @w0.setter
    def w0(self, arg0: float) -> None:
        pass

    @property
    def w0_var(self) -> float:
        """
        :type: float
        """

    @w0_var.setter
    def w0_var(self, arg0: float) -> None:
        pass

    @property
    def w_var(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @w_var.setter
    def w_var(self, arg0: numpy.ndarray[float64, _Shape[m, 1]]) -> None:
        pass

    pass


class VariationalFMHyperParameters:
    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def alpha(self) -> float:
        """
        :type: float
        """

    @property
    def alpha_rate(self) -> float:
        """
        :type: float
        """

    @property
    def lambda_V(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @property
    def lambda_V_rate(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @property
    def lambda_w(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @property
    def lambda_w_rate(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @property
    def mu_V(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @property
    def mu_V_var(self) -> numpy.ndarray[float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, n]]
        """

    @property
    def mu_w(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    @property
    def mu_w_var(self) -> numpy.ndarray[float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[float64, _Shape[m, 1]]
        """

    pass


class VariationalFMTrainer:
    def __init__(
        self,
        arg0: scipy.sparse.csr_matrix[float64],
        arg1: List[RelationBlock],
        arg2: numpy.ndarray[float64, _Shape[m, 1]],
        arg3: int,
        arg4: FMLearningConfig,
    ) -> None:
        ...

    def create_FM(self, arg0: int, arg1: float) -> VariationalFM:
        ...

    def create_Hyper(self, arg0: int) -> VariationalFMHyperParameters:
        ...

    pass


class VariationalLearningHistory:
    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    @property
    def elbos(self) -> List[float]:
        """
        :type: List[float]
        """

    @property
    def hypers(self) -> FMHyperParameters:
        """
        :type: FMHyperParameters
        """

    pass


class VariationalPredictor:
    def __getstate__(self) -> tuple:
        ...

    def __setstate__(self, arg0: tuple) -> None:
        ...

    def predict(
        self, arg0: scipy.sparse.csr_matrix[float64], arg1: List[RelationBlock]
    ) -> numpy.ndarray[float64, _Shape[m, 1]]:
        ...

    def weights(self) -> VariationalFM:
        ...

    pass


def create_train_fm(
    arg0: int,
    arg1: float,
    arg2: scipy.sparse.csr_matrix[float64],
    arg3: List[RelationBlock],
    arg4: numpy.ndarray[float64, _Shape[m, 1]],
    arg5: int,
    arg6: FMLearningConfig,
    arg7: Callable[[int, FM, FMHyperParameters, LearningHistory], bool],
) -> Tuple[Predictor, LearningHistory]:
    """
    create and train fm.
    """


def create_train_vfm(
    rank: int,
    init_std: float,
    X: scipy.sparse.csr_matrix[float64],
    relations: List[RelationBlock],
    y: numpy.ndarray[float64, _Shape[m, 1]],
    random_seed: int,
    learning_config: FMLearningConfig,
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
    """
    create and train fm.
    """


def mean_var_truncated_normal_left(arg0: float) -> Tuple[float, float, float]:
    pass


def mean_var_truncated_normal_right(arg0: float) -> Tuple[float, float, float]:
    pass
