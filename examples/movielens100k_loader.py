from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import urllib.request

MOVIELENS100K_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

def _read_interaction(byte_stream):
    with BytesIO(byte_stream) as ifs:
        return pd.read_csv(ifs, sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

def load_dataset(zippath=None, id_only=True, random_state=114514, fold=None):
    if zippath is None:
        zippath = os.path.expanduser('~/.ml-100k.zip')
        if not os.path.exists(zippath):
            download = input('Could not find {}.\nCan I download and save it there?[y/N]'.format(zippath))
            if download.lower() == 'y':
                print('start download...')
                urllib.request.urlretrieve(MOVIELENS100K_URL, zippath)
                print('complete')
            else:
                return None
    zf = ZipFile(zippath) 
    if fold is None:
        df_all = _read_interaction(zf.read('ml-100k/u.data'))
        df_train, df_test = train_test_split(df_all, random_state=random_state)
    else:
        assert fold >=1 and fold <=5
        train_path = 'ml-100k/u{}.base'.format(fold)
        test_path = 'ml-100k/u{}.test'.format(fold)
        df_train = _read_interaction(zf.read(train_path))
        df_test = _read_interaction(zf.read(test_path))
    if id_only:
        return df_train, df_test