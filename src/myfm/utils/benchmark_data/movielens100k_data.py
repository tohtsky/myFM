from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .loader_base import MovieLensBase


class MovieLens100kDataManager(MovieLensBase):
    """The Data manager for MovieLens 100k dataset."""

    @property
    def DOWNLOAD_URL(self) -> str:
        return "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

    @property
    def DEFAULT_PATH(self) -> Path:
        return Path("~/.ml-100k.zip").expanduser()

    def _read_interaction(self, byte_stream: bytes) -> pd.DataFrame:
        with BytesIO(byte_stream) as ifs:
            data = pd.read_csv(
                ifs,
                sep="\t",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
            )
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
            return data

    def load_rating_all(self) -> pd.DataFrame:
        """Load the entire rating dataset.

        Returns
        -------
        pd.DataFrame
            all the available ratings.
        """
        return self._read_interaction(self.zf.read("ml-100k/u.data"))

    def load_rating_predefined_split(
        self,
        fold: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read the pre-defined train/test split.
        Fold index ranges from 1 to 5.

        Parameters
        ----------
        fold : int
            specifies the fold index.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train and test dataframes.

        """
        assert fold >= 1 and fold <= 5
        train_path = "ml-100k/u{}.base".format(fold)
        test_path = "ml-100k/u{}.test".format(fold)
        df_train = self._read_interaction(self.zf.read(train_path))
        df_test = self._read_interaction(self.zf.read(test_path))

        return df_train, df_test

    def load_user_info(self) -> pd.DataFrame:
        """load user meta information.

        Returns
        -------
        pd.DataFrame
            user infomation
        """
        user_info_bytes = self.zf.read("ml-100k/u.user")
        with BytesIO(user_info_bytes) as ifs:
            return pd.read_csv(
                ifs,
                sep="|",
                header=None,
                names=["user_id", "age", "gender", "occupation", "zipcode"],
            )

    def genres(self) -> List[str]:
        with BytesIO(self.zf.read("ml-100k/u.genre")) as ifs:
            genres: List[str] = list(pd.read_csv(ifs, sep="|", header=None)[0])
        return genres

    def load_movie_info(self) -> pd.DataFrame:
        r"""load movie meta information.

        Returns
        -------
        pd.DataFrame
            A dataframe containing meta-information (id, title, release_date, url, genres) about the movies.
            Multiple genres per movie will be concatenated by "|".
        """
        MOVIE_COLUMNS = ["movie_id", "title", "release_date", "unk", "url"]
        genres = self.genres()

        with BytesIO(self.zf.read("ml-100k/u.item")) as ifs:
            df_mov = pd.read_csv(
                ifs,
                sep="|",
                encoding="latin-1",
                header=None,
            )
            df_mov.columns = MOVIE_COLUMNS + genres
        df_mov["release_date"] = pd.to_datetime(df_mov.release_date)
        movie_index, genre_index = df_mov[genres].values.nonzero()
        genre_df = (
            (
                pd.DataFrame(
                    dict(
                        movie_id=df_mov.movie_id.values[movie_index],
                        genre=[genres[i] for i in genre_index],
                    )
                )
                .groupby("movie_id")
                .genre.agg(lambda x: "|".join(x))
            )
            .reindex(df_mov.movie_id)
            .fillna("")
        )
        df_mov["genres"] = genre_df.values
        return df_mov
