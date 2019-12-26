# myFM
An implementation of Bayesian Factorization Machine based on Gibbs sampling.  
The goal of this project is to

1. Implement Gibbs sampler easy to use from Python.
2. Use modern technology like [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and [pybind11](https://github.com/pybind/pybind11) for simpler and faster implementation.

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

# Comparison with other libraries (which implement MCMC functionality)
## with [libFM](https://github.com/srendle/libfm) : 
- pros:
  - faster (at least in my environment)
  - availability from Python
- cons:
  - lacks some functionality, especially relational data (TODO)
  - lacks other algorithms like SGD

## with [fastFM](https://github.com/ibayer/fastFM) : 
- pros:
  - faster (I'm not sure if I have configured the build for fastFM in an optimal way, though)
  - support grouping (i.e., assume different std or mean for subsets of features)
- cons:
  - lacks other algorithms like SGD

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
This example will require `pandas` and `scikit-learn`.

`movielens100k_loader` is defined in `examples/movielens100k_loader.py`.

See `examples/ml-100k.ipynb` for detailed version.
```Python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

import myfm

# read movielens 100k data into pandas DataFrame.
from movielens100k_loader import load_dataset

df_train, df_test = load_dataset(
    zippath='/home/tomoki/ml-100k.zip',
    id_only=True, fold=3
) # folding dependence

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
# rmse=0.9032126256432311, mae=0.7116432524241615

# with grouping
fm = test_myfm(df_train, df_test, rank=8, grouping=True)
# rmse=0.8959382764109612, mae=0.7048050699780434
```