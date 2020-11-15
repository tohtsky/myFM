import os
from io import BytesIO
from .movielens100k_data import MovieLensBase
import pandas as pd


class MovieLens1MDataManager(MovieLensBase):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-1m.zip")

    def load_rating_all(self) -> pd.DataFrame:
        """Read all (1M) interactions.

        Returns
        -------
        pd.DataFrame
            [description]
        """
        with BytesIO(self.zf.read("ml-1m/ratings.dat")) as ifs:
            import pandas as pd

            df = pd.read_csv(
                ifs,
                sep="\:\:",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                engine="python",
            )
            df["timestamp"] = pd.to_datetime(df.timestamp, unit="s")
            return df
