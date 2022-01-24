from io import BytesIO
from pathlib import Path

import pandas as pd

from .loader_base import MovieLensBase
from .movielens1M_data import read_ml1m10m_df


class MovieLens10MDataManager(MovieLensBase):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DEFAULT_PATH = Path("~/.ml-10m.zip").expanduser()

    def load_rating_all(self) -> pd.DataFrame:
        with BytesIO(self.zf.read("ml-10M100K/ratings.dat")) as ifs:
            return read_ml1m10m_df(ifs)
