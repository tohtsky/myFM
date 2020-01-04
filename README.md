# myFM
An implementation of Bayesian [Factorization Machine](https://ieeexplore.ieee.org/abstract/document/5694074/) based on Gibbs sampling, which I believe is a wheel worth reinventing.

The goal of this project is to

1. Implement Gibbs sampler easy to use from Python.
2. Use modern technology like [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and [pybind11](https://github.com/pybind/pybind11) for simpler and faster implementation.

Currently this supports most options for libFM MCMC engine, such as
 - Grouping of input variables (`-meta` option of [libFM](https://github.com/srendle/libfm))
 - Relation Data format (See the paper ["Scaling Factorization Machine to relational data"](https://dl.acm.org/citation.cfm?id=2488340))


# Requirements
Recent version of gcc/clang with C++ 11 support.

# Installation

For Linux / Mac OSX, type
```
pip install git+https://github.com/tohtsky/myFM
```
In addition to installing python dependencies (`numpy`, `scipy`, `pybind11`, ...),  the above command will automatically download eigen (ver 3.3.7) to its build directory and use it for the build.

If you want to use another version of eigen, you can also do
```
EIGEN3_INCLUDE_DIR=/path/to/eigen pip install git+https://github.com/tohtsky/myFM
```

# Examples

## A Toy example
This example is taken from [pyfm](https://github.com/coreylynch/pyFM) with some modification.
```Python
import myfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print(X.toarray()) 
# print
# [[ 19.   0.   0.   0.   1.   1.   0.   0.   0.]
#  [ 33.   0.   0.   1.   0.   0.   1.   0.   0.]
#  [ 55.   0.   1.   0.   0.   0.   0.   1.   0.]
#  [ 20.   1.   0.   0.   0.   0.   0.   0.   1.]]
y = np.asarray([0, 1, 1, 0])
fm = myfm.MyFMClassifier(rank=4)
fm.fit(X,y)
fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
```

## A Movielens-100k Example
This example will require `pandas` and `scikit-learn`. `movielens100k_loader` is present in `examples/movielens100k_loader.py`.

You will be able to obtain a result comparable to SOTA algorithms like GC-MC. See `examples/ml-100k.ipynb` for the detailed version.
```Python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

import myfm

from movielens100k_data import MovieLens100kDataManager
data_manager = MovieLens100kDataManager()
df_train, df_test = data_manager.load_rating(fold=3) # Note the dependence on the fold

def test_myfm(df_train, df_test, rank=8, grouping=None, n_iter=100, samples=95):
    explanation_columns = ['user_id', 'movie_id']
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train = ohe.fit_transform(df_train[explanation_columns])
    X_test = ohe.transform(df_test[explanation_columns])
    y_train = df_train.rating.values
    y_test = df_test.rating.values
    fm = myfm.MyFMRegressor(rank=rank, random_seed=114514)
    
    if grouping:
        # assign group index for each column of X_train.
        grouping = [ i for i, category in enumerate(ohe.categories_) for _ in category]
        assert len(grouping) == X_train.shape[1]

    fm.fit(X_train, y_train, grouping=grouping, n_iter=n_iter, n_kept_samples=samples)
    prediction = fm.predict(X_test)
    rmse = ((y_test - prediction) ** 2).mean() ** .5
    mae = np.abs(y_test - prediction).mean()
    print('rmse={rmse}, mae={mae}'.format(rmse=rmse, mae=mae))
    return fm

# basic regression
test_myfm(df_train, df_test, rank=8);
# rmse=0.90321, mae=0.71164

# with grouping
fm = test_myfm(df_train, df_test, rank=8, grouping=True)
# rmse=0.89594, mae=0.70481
```

## Examples for Relational Data format 
Below is a toy movielens-like example which utilizes relational data format proposed in [Rendle,].  

This example, however, is too simplistic to exhibit the computational advantage of this data format. For an example with drastically reduced computational complexity, see `examples/ml-100k-extended.ipynb`; 

```Python
import pandas as pd
import numpy as np
from myfm import MyFMRegressor, RelationBlock
from sklearn.preprocessing import OneHotEncoder

users = pd.DataFrame([
    {'user_id': 1, 'age': '20s', 'married': False},
    {'user_id': 2, 'age': '30s', 'married': False},
    {'user_id': 3, 'age': '40s', 'married': True}
]).set_index('user_id')

movies = pd.DataFrame([
    {'movie_id': 1, 'comedy': True, 'action': False },
    {'movie_id': 2, 'comedy': False, 'action': True },
    {'movie_id': 3, 'comedy': True, 'action': True}
]).set_index('movie_id')

ratings = pd.DataFrame([
    {'user_id': 1, 'movie_id': 1, 'rating': 2},
    {'user_id': 1, 'movie_id': 2, 'rating': 5},
    {'user_id': 2, 'movie_id': 2, 'rating': 4},
    {'user_id': 2, 'movie_id': 3, 'rating': 3},
    {'user_id': 3, 'movie_id': 3, 'rating': 3},
])

user_ids, user_indices = np.unique(ratings.user_id, return_inverse=True)
movie_ids, movie_indices = np.unique(ratings.movie_id, return_inverse=True)

user_ohe = OneHotEncoder(handle_unknown='ignore').fit(users.reset_index()) # include user id as feature
movie_ohe = OneHotEncoder(handle_unknown='ignore').fit(movies.reset_index())

X_user = user_ohe.transform(
    users.reindex(user_ids).reset_index()
)
X_movie = movie_ohe.transform(
    movies.reindex(movie_ids).reset_index()
)

block_user = RelationBlock(user_indices, X_user)
block_movie = RelationBlock(movie_indices, X_movie)

fm = MyFMRegressor(rank=2).fit(None, ratings.rating, X_rel=[block_user, block_movie])

prediction_df = pd.DataFrame([
    dict(user_id=user_id,movie_id=movie_id,
         user_index=user_index, movie_index=movie_index)
    for user_index, user_id in enumerate(user_ids)
    for movie_index, movie_id in enumerate(movie_ids)
])
predicted_rating = fm.predict(None, [
    RelationBlock(prediction_df.user_index, X_user),
    RelationBlock(prediction_df.movie_index, X_movie)
])

prediction_df['prediction']  = predicted_rating

print(
    prediction_df.merge(ratings.rename(columns={'rating':'ground_truth'}), how='left')
)
```


# References
1. Rendle, Steffen. "Factorization machines." 2010 IEEE International Conference on Data Mining. IEEE, 2010.
1. Rendle, Steffen. "Factorization machines with libfm." ACM Transactions on Intelligent Systems and Technology (TIST) 3.3 (2012): 57.
1. Rendle, Steffen. "Scaling factorization machines to relational data." Proceedings of the VLDB Endowment. Vol. 6. No. 5. VLDB Endowment, 2013.
1. Bayer, Immanuel. "fastfm: A library for factorization machines." arXiv preprint arXiv:1505.00641 (2015).
