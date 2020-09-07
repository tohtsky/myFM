.. _MovielensIndex:

=========================================
A Tutorial with Movielens
=========================================

FMs are believed to perform remarkably well on the datasets with
huge and sparse feature matrices,
and most common examples are (explicit) collaborative filtering tasks.

Here, let us see the power of Bayesian Factorization Machine
using the well-known Movielens 100k and go through the ``myFM``'s API.

-------------------------
Pure Matrix Factorization
-------------------------

First let us consider the pure Matrix Factorization.
That is, we model the user :math:`u`'s rating response to movie :math:`i`,
which we write :math:`r_{ui}`, as

.. math::
    r_{ui} \sim w_0 + b_u + d_i + \vec{u}_u \cdot \vec{v}_j

This formulation is equivalent to Factorization Machine with

1. User IDs treated as a categorical feature with one-hot encoding
2. Movie IDs treated as a categorical feature with one-hot encoding

So you can efficiently use encoder like sklearn's `OneHotEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_
to prepare the input matrix.

::

    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn import metrics

    import myfm
    from myfm.utils.benchmark_data import MovieLens100kDataManager

    FM_RANK = 10

    data_manager = MovieLens100kDataManager()
    df_train, df_test = data_manager.load_rating(fold=3)

    FEATURE_COLUMNS = ['user_id', 'movie_id']
    ohe = OneHotEncoder(handle_unknown='ignore')

    X_train = ohe.fit_transform(df_train[FEATURE_COLUMNS])
    X_test = ohe.transform(df_test[FEATURE_COLUMNS])
    y_train = df_train.rating.values
    y_test = df_test.rating.values

    fm = myfm.MyFMRegressor(rank=FM_RANK, random_seed=42)
    fm.fit(X_train, y_train, n_iter=200, n_kept_samples=200)

    prediction = fm.predict(X_test)
    rmse = ((y_test - prediction) ** 2).mean() ** .5
    mae = np.abs(y_test - prediction).mean()
    print(f'rmse={rmse}, mae={mae}')

The above script should give you rmse=0.8944, mae=0.7031 which is already
impressive compared with other recent methods.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Assuming Separate Variance for movie & user
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Probabilistic Matrix Factorization, we usually assume
user vectors and item vectors are drawn from separate normal priors:

.. math::
    u_i & \sim \mathcal{N}(\mu_U, \Sigma_U) \\
    v_i & \sim \mathcal{N}(\mu_I, \Sigma_I)

However, we haven't provided any information about which columns are users' and items'.

In fact you can tell these information ``MyFMRegressor`` by ``group_shapes`` option: ::

    fm_grouped = myfm.MyFMRegressor(
        rank=FM_RANK, random_seed=42,
        group_shapes=[len(group) for group in ohe.categories_]
    )
    fm_grouped.fit(X_train, y_train, n_iter=200, n_kept_samples=200)

    prediction_grouped = fm_grouped.predict(X_test)
    rmse = ((y_test - prediction_grouped) ** 2).mean() ** .5
    mae = np.abs(y_test - prediction_grouped).mean()
    print(f'rmse={rmse}, mae={mae}')