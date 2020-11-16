from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Union, Tuple
import pandas as pd
import os
import numpy as np
from zipfile import ZipFile

from sklearn.model_selection import KFold
from numpy.random import RandomState
import urllib.request

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
        raise NotImplementedError("must be implemented")

    @abstractproperty
    def DEFAULT_PATH(self) -> str:
        raise NotImplementedError("must be implemented")

    def __init__(self, zippath: Optional[str] = None):
        if zippath is None:
            zippath = self.DEFAULT_PATH
            if not os.path.exists(zippath):
                download = input(
                    "Could not find {}.\nCan I download and save it there?[y/N]".format(
                        zippath
                    )
                )
                if download.lower() == "y":
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
