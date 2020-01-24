import time 

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

from fastFM.mcmc import FMClassification, FMRegression
import myfm
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import dump_svmlight_file


# このデータ読み出しクラスは `examples/` 内で定義されている
from movielens100k_data import MovieLens100kDataManager

def test_libFM(X_train, y_train, X_test, rank=8, n_iter=100, classification=False):
    pass


def test_fastFM(X_train, y_train, X_test, rank=8, n_iter=100, classification=False):
    if classification:
        fm = FMClassification(rank=rank, n_iter=n_iter
    else:
        fm = FMRegression(rank=rank, n_iter=n_iter)

    return fm.fit_predict(X_train, y_train, X_test)



def test_myfm(X_train, y_train, X_test, rank=8, n_iter=100, classification=False):
    if classification:
        fm = myfm.MyFMClassifier(rank=rank)
    else:
        fm = myfm.MyFMRegressor(rank=rank) 

    fm.fit(X_train, y_train, n_iter=n_iter, n_kept_samples=n_iter)

    if classification:
        return fm.predict_proba(X_test)
    else:
        return fm.predict(X_test)

def test_usual():
    explanation_columns = ['user_id', 'movie_id']
    data_manager = MovieLens100kDataManager()
    df_train, df_test = data_manager.load_rating(fold=3) # Note the dependence on the fold 
    ohe = OneHotEncoder(handle_unknown='ignore')

    X_train = ohe.fit_transform(df_train[explanation_columns])
    X_test = ohe.transform(df_test[explanation_columns])
    y_train = df_train.rating.values
    y_test = df_test.rating.values

    for test_method in [ test_fastFM, test_myfm]:
        start = time.time()
        p = test_method(X_train, y_train, X_test, rank=8);
        spent  = time.time() - start
        rmse = ( (y_test - p) ** 2 ).mean() ** .5
        print(f'rmse = {rmse}, spent={spent}')

if __name__ == "__main__":
    test_usual()
