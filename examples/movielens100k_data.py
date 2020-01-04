from zipfile import ZipFile
from io import BytesIO
import os
import urllib.request

import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split


class MovieLens100kDataManager:
    DOWNLOAD_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    DEFAULT_PATH = os.path.expanduser('~/.ml-100k.zip')
    
    @classmethod
    def _read_interaction(cls, byte_stream):
        with BytesIO(byte_stream) as ifs:
            data = pd.read_csv(ifs, sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            return data

    def __init__(self, zippath=None):
        if zippath is None:
            zippath = self.DEFAULT_PATH
            if not os.path.exists(zippath):
                download = input('Could not find {}.\nCan I download and save it there?[y/N]'.format(zippath))
                if download.lower() == 'y':
                    print('start download...')
                    urllib.request.urlretrieve(self.DOWNLOAD_URL, zippath)
                    print('complete')
                else:
                    zippath = None
                    
        if zippath is not None:
            self.zf = ZipFile(zippath)
        else:
            self.zf = None
            
    def load_rating(self, random_state=114514, fold=None):
        if fold is None:
            df_all = self._read_interaction(self.zf.read('ml-100k/u.data'))
            df_train, df_test = train_test_split(df_all, random_state=random_state)
        else:
            assert fold >=1 and fold <=5
            train_path = 'ml-100k/u{}.base'.format(fold)
            test_path = 'ml-100k/u{}.test'.format(fold)
            df_train = self._read_interaction(self.zf.read(train_path))
            df_test = self._read_interaction(self.zf.read(test_path))
        return df_train, df_test
    
    def load_userinfo(self):
        user_info_bytes = self.zf.read('ml-100k/u.user')
        with BytesIO(user_info_bytes) as ifs:
            return pd.read_csv(ifs, sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zipcode'])
    
    def load_movieinfo(self):
        MOVIE_COLUMNS = ['movie_id', 'title', 'release_date', 'unk', 'url']
        with BytesIO(self.zf.read('ml-100k/u.genre')) as ifs:
            genres = pd.read_csv(ifs, sep='|', header=None)[0]
        with BytesIO(self.zf.read('ml-100k/u.item')) as ifs:
            df_mov = pd.read_csv(
                ifs, sep='|', encoding='latin-1',
                header=None,
            )
            df_mov.columns = (
                MOVIE_COLUMNS + list(genres) 
            )
        df_mov['release_date'] = pd.to_datetime(df_mov.release_date)
        return df_mov, list(genres)