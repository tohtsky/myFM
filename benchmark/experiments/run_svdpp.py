import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import myfm
import pandas as pd
from scipy import sparse as sps
from mapper import DefaultMapper
# read movielens 100k data.
from movielens100k_loader import load_dataset

df_train, df_test = load_dataset(
    id_only=True, fold=2
) # Note the dependence on the fold

from myfm._myfm import RelationBlock

user_to_internal = DefaultMapper(df_train.user_id.values)
movie_to_internal = DefaultMapper(df_train.movie_id)

user_vs_watched = dict()
movie_vs_watched = dict()
for row in df_train.itertuples():
    user_id = row.user_id
    movie_id = row.movie_id
    user_vs_watched.setdefault(user_id, list()).append(movie_id)
    movie_vs_watched.setdefault(movie_id, list()).append(user_id)

X_user = sps.lil_matrix((len(user_to_internal), len(user_to_internal) + len(movie_to_internal)))
for i in range(len(user_to_internal)):
    X_user[i, i] = 1

for uid, watched in user_vs_watched.items():
    if not watched:
        continue
    u_iid = user_to_internal[uid]
    m_iids = [len(user_to_internal) + movie_to_internal[mid] for mid in watched]
    X_user[u_iid, m_iids] = 1 / max(1, len(watched)) ** 0.5

X_user = X_user.tocsr()

X_movie = sps.lil_matrix(
    (len(movie_to_internal), len(movie_to_internal) + len(user_to_internal))
)

for i in range(len(movie_to_internal)):
    X_movie[i, i] = 1

for mid, watched in movie_vs_watched.items():
    if not watched:
        continue
    m_iid = movie_to_internal[mid]
    u_iids = [len(movie_to_internal) + user_to_internal[uid] for uid in watched]
    X_movie[m_iid, u_iids] = 1 / max(1, len(watched)) ** 0.5

X_movie = X_movie.tocsr()

rb_1 = RelationBlock(
    df_train.user_id.map(user_to_internal).values,
    X_user
)
rb_2 = RelationBlock(
    df_train.movie_id.map(movie_to_internal).values,
    X_movie
)

fmr = myfm.MyFMRegressor(rank=12)

fmr.fit(None, df_train.rating.values, X_rel=[rb_1, rb_2], n_iter=200, n_kept_samples=100)
print(fmr.fms_)
main = sps.csr_matrix((80000, 0), dtype=np.float64)
print( 
    (
        (fmr.predict(main, [rb_1, rb_2]) - df_train.rating.values) ** 2
    ).mean() ** 0.5
)
with open('test.pkl', 'wb') as ofs:
    import pickle
    pickle.dump(fmr, ofs)
