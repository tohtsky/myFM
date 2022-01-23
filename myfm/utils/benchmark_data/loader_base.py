import urllib.request
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Optional, Tuple, Union
from zipfile import ZipFile

import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import KFold

RandomStateType = Union[int, RandomState]


def train_test_split_with_kfold(
    df: pd.DataFrame,
    K: int,
    fold: int,
    random_state: Optional[RandomStateType] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not ((0 <= fold) and (fold < K)):
        raise ValueError("0 <= fold < K")
    kf = KFold(K, shuffle=True, random_state=random_state)
    for i, (tr, te) in enumerate(kf.split(df)):
        if i == fold:
            return df.iloc[tr], df.iloc[te]
    raise RuntimeError("should not reach here.")


class DataLoaderBase(ABC):
    zf: ZipFile

    @abstractproperty
    def DOWNLOAD_URL(self) -> str:
        raise NotImplementedError("must be implemented")  # pragma: no cover

    @abstractproperty
    def DEFAULT_PATH(self) -> Path:
        raise NotImplementedError("must be implemented")  # pragma: no cover

    def __init__(self, zippath: Optional[Path] = None):
        zippath = Path(zippath or self.DEFAULT_PATH)
        if not zippath.exists():
            permission = input(
                "Could not find {}.\nCan I download and save it there?[y/N]".format(
                    zippath
                )
            ).lower()
            download = permission == "y"
            if download:
                print("start download...")
                urllib.request.urlretrieve(self.DOWNLOAD_URL, zippath)
                print("complete")
            else:
                raise RuntimeError("abort.")
        self.zf = ZipFile(zippath)


class MovieLensBase(DataLoaderBase, ABC):
    @abstractmethod
    def load_rating_all(self) -> pd.DataFrame:
        raise NotImplementedError("must be implemented")

    def load_rating_kfold_split(
        self, K: int, fold: int, random_state: Optional[RandomStateType] = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the entire dataset and split it into train/test set.
        K-fold

        Parameters
        ----------
        K : int
            K in the K-fold splitting scheme.
        fold : int
            fold index.
        random_state : Union[np.RandomState, int, None], optional
            Controlls random state of the split.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train and test dataframes.

        Raises
        ------
        ValueError
            When 0 <= fold < K  is not met.
        """
        if not ((0 <= fold) and (fold < K)):
            raise ValueError("0 <= fold < K")
        df_all = self.load_rating_all()
        return train_test_split_with_kfold(df_all, K, fold, random_state)
