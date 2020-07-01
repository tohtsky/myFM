import os
from io import BytesIO
from .movielens100k_data import MovieLens100kDataManager
import pandas as pd
from sklearn.model_selection import train_test_split

class MovieLens1MDataManager(MovieLens100kDataManager):
    DOWNLOAD_URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    DEFAULT_PATH = os.path.expanduser('~/.ml-1m.zip')
    
    def load_rating(self, random_state=114514, test_size=0.1):
        with BytesIO(self.zf.read('ml-1m/ratings.dat')) as ifs:
            import pandas as pd
            df = pd.read_csv(ifs, sep='\:\:', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
            df['timestamp'] = pd.to_datetime(df.timestamp, unit='s')
            return train_test_split(df, random_state=random_state, test_size=test_size)