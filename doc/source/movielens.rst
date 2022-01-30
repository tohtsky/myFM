.. _MovielensIndex:

=========================================
A Basic Tutorial with Movielens 100K
=========================================

FMs perform remarkably well on datasets with huge and sparse feature matrices,
and the most common examples are (explicit) collaborative filtering tasks.

Let us examine the power of the Bayesian Factorization Machines
by testing a series of APIs in myFM using the well-known Movielens 100k dataset.


-------------------------
Pure Matrix Factorization
-------------------------

First let us consider the probabilistic Matrix Factorization.
That is, we model the user :math:`u`'s rating response to movie :math:`i`,
which we write :math:`r_{ui}`, as

.. math::
    r_{ui} \sim w_0 + b_u + d_i + \vec{u}_u \cdot \vec{v}_j

This formulation is equivalent to Factorization Machines with

1. User IDs treated as a categorical feature with one-hot encoding
2. Movie IDs treated as a categorical feature with one-hot encoding

So you can efficiently use encoder like sklearn's `OneHotEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_
to prepare the input matrix.

.. testcode ::

    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
    from sklearn import metrics

    import myfm
    from myfm.utils.benchmark_data import MovieLens100kDataManager

    FM_RANK = 10

    data_manager = MovieLens100kDataManager()
    df_train, df_test = data_manager.load_rating_predefined_split(fold=3)

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

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    rmse=..., mae=...

The above script should give you RMSE=0.8944, MAE=0.7031 which is already
impressive compared with other recent methods.

.. _grouping:

-------------------------------------------
Assuming Separate Variance for movie & user
-------------------------------------------

In Probabilistic Matrix Factorization, we usually assume
user vectors and item vectors are drawn from separate normal priors:

.. math::
    u_i & \sim \mathcal{N}(\mu_U, \Sigma_U) \\
    v_i & \sim \mathcal{N}(\mu_I, \Sigma_I)

However, we haven't provided any information about which columns are users' and items'.

You can tell  :py:class:`myfm.MyFMRegressor` these information (i.e., which parameters share a common mean and variance) by ``group_shapes`` option:

.. testcode ::

    fm_grouped = myfm.MyFMRegressor(
        rank=FM_RANK, random_seed=42,
    )
    fm_grouped.fit(
        X_train, y_train, n_iter=200, n_kept_samples=200,
        group_shapes=[len(group) for group in ohe.categories_]
    )

    prediction_grouped = fm_grouped.predict(X_test)
    rmse = ((y_test - prediction_grouped) ** 2).mean() ** .5
    mae = np.abs(y_test - prediction_grouped).mean()
    print(f'rmse={rmse}, mae={mae}')

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    rmse=..., mae=...


This will slightly improve the performance to RMSE=0.8925, MAE=0.7001.


-------------------------------------------
Adding Side information
-------------------------------------------

It is straightforward to include user/item side information.

First we retrieve the side information from ``Movielens100kDataManager``:

.. testcode ::

    user_info = data_manager.load_user_info().set_index('user_id')
    user_info["age"] = user_info.age // 5 * 5
    user_info["zipcode"] = user_info.zipcode.str[0]
    user_info_ohe = OneHotEncoder(handle_unknown='ignore').fit(user_info)

    movie_info = data_manager.load_movie_info().set_index('movie_id')
    movie_info['release_year'] = [
        str(x) for x in movie_info['release_date'].dt.year.fillna('NaN')
    ]
    movie_info = movie_info[['release_year', 'genres']]
    movie_info_ohe = OneHotEncoder(handle_unknown='ignore').fit(movie_info[['release_year']])
    movie_genre_mle = MultiLabelBinarizer(sparse_output=True).fit(
        movie_info.genres.apply(lambda x: x.split('|'))
    )



Note that the way movie genre information is represented in ``movie_info`` DataFrame is a bit tricky (it is already binary encoded).

We can then augment ``X_train`` / ``X_test`` with auxiliary information. The `hstack <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html>`_ function of ``scipy.sparse`` is very convenient for this purpose:

.. testcode ::

    import scipy.sparse as sps
    X_train_extended = sps.hstack([
        X_train,
        user_info_ohe.transform(
            user_info.reindex(df_train.user_id)
        ),
        movie_info_ohe.transform(
            movie_info.reindex(df_train.movie_id).drop(columns=['genres'])
        ),
        movie_genre_mle.transform(
            movie_info.genres.reindex(df_train.movie_id).apply(lambda x: x.split('|'))
        )
    ])

    X_test_extended = sps.hstack([
        X_test,
        user_info_ohe.transform(
            user_info.reindex(df_test.user_id)
        ),
        movie_info_ohe.transform(
            movie_info.reindex(df_test.movie_id).drop(columns=['genres'])
        ),
        movie_genre_mle.transform(
            movie_info.genres.reindex(df_test.movie_id).apply(lambda x: x.split('|'))
        )
    ])

Then we can regress ``X_train_extended`` against ``y_train``

.. testcode ::

    group_shapes_extended = (
        [len(group) for group in ohe.categories_] +
        [len(group) for group in user_info_ohe.categories_] +
        [len(group) for group in movie_info_ohe.categories_] +
        [ len(movie_genre_mle.classes_)]
    )

    fm_side_info = myfm.MyFMRegressor(
        rank=FM_RANK, random_seed=42,
    )
    fm_side_info.fit(
        X_train_extended, y_train, n_iter=200, n_kept_samples=200,
        group_shapes=group_shapes_extended
    )

    prediction_side_info = fm_side_info.predict(X_test_extended)
    rmse = ((y_test - prediction_side_info) ** 2).mean() ** .5
    mae = np.abs(y_test - prediction_side_info).mean()
    print(f'rmse={rmse}, mae={mae}')

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    rmse=..., mae=...

The result should improve further with RMSE = 0.8855, MAE = 0.6944.

Unfortunately, the running time is somewhat (~ 4 times) slower compared to
the pure matrix-factorization described above. This is as it should be:
the complexity of Bayesian FMs is proportional to :math:`O(\mathrm{NNZ})`
(i.e., non-zero elements of input sparse matrix),
and we have incorporated various non-zero elements (user/item features) for each row.

Surprisingly, we can still train the equivalent model
in a running time close to pure MF if represent the data in Relational Data Format.
See :ref:`next section <RelationBlockTutorial>` for how Relational Data Format works.
