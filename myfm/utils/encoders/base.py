from abc import abstractmethod, ABC
from typing import List, Any
import scipy.sparse as sps


class SparseEncoderBase(ABC):
    @abstractmethod
    def to_sparse(self, x: List[Any]) -> sps.csr_matrix:
        raise NotImplementedError("must be implemented")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("must be implemented")
