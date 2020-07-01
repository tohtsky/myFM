import os
from io import BytesIO
from .movielens100k_data import MovieLens100kDataManager
import pandas as pd
from sklearn.model_selection import train_test_split

class MovieLens10MDataManager(MovieLens100kDataManager):
    DOWNLOAD_URL = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
    DEFAULT_PATH = os.path.expanduser('~/.ml-10m.zip')
    
    def load_rating(self):
        with BytesIO(self.zf.read('ml-10M100K/ratings.dat')) as ifs:
            import pandas as pd
            df = pd.read_csv(ifs, sep='\:\:', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
            df['timestamp'] = pd.to_datetime(df.timestamp, unit='s')
            return df