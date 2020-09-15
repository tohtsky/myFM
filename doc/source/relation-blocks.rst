.. _RelationBlockTutorial :

--------------------------------------
TimeSVD++ Flipped with Relation Blocks
--------------------------------------

As mentioned in the :ref:`Movielens example <MovielensIndex>`,
the complexity of Bayesian FMs is proportional to :math:`O(\mathrm{NNZ})`.
This is especially troublesome when we include SVD++-like features in the feature matrix.
In such a case, for each user, we include all of the item IDs that the user had interacted with,
and the complexity grows further by a factor of :math:`O(\mathrm{NNZ} / N_U)`.

However, we can get away with this catastrophic complexity if we notice the repeated pattern in the input matrix.
Interested readers can refer to `[Rendle, '13] <https://dl.acm.org/doi/abs/10.14778/2535573.2488340>`_ 
and `libFM's Manual <http://www.libfm.org/libfm-1.40.manual.pdf>`_ for details.

Below let us see how we can incorporate SVD++-like features efficiently
using the relational data again using Movielens 100K dataset.

^^^^^^^^^^^^^^^^^^^^^^^^
Building SVD++ Features
^^^^^^^^^^^^^^^^^^^^^^^^

In `[Rendle, et al., '19] <https://arxiv.org/abs/1905.01395>`_,
in addition to the user/movie id, they have made use of the following features to improve the accuracy considerably:

1. User Implicit Features: All the movies the user had watched
2. Movie Implicit Features: All the users who have watched the movie
3. Time Variable: The day of watch event (regarded as a categorical variable)

Let us construct these features. ::

    from collections import defaultdict
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn import metrics
    import myfm
    from myfm import RelationBlock
    from scipy import sparse as sps

    from myfm.utils.benchmark_data import MovieLens100kDataManager

    data_manager = MovieLens100kDataManager()
    df_train, df_test = data_manager.load_rating(fold=1) # fold 1 is the toughest one

    date_ohe = OneHotEncoder(handle_unknown='ignore').fit(
        df_train.timestamp.dt.date.values.reshape(-1, 1)
    )
    def categorize_date(df):
        return date_ohe.transform(df.timestamp.dt.date.values[:, np.newaxis])

    # index "0" is reserved for unknown ids.
    user_to_index = defaultdict(lambda : 0, { uid: i+1 for i,uid in enumerate(np.unique(df_train.user_id)) })
    movie_to_index = defaultdict(lambda: 0, { mid: i+1 for i,mid in enumerate(np.unique(df_train.movie_id))})
    USER_ID_SIZE = len(user_to_index) + 1
    MOVIE_ID_SIZE = len(movie_to_index) + 1

Above we constructed dictionaries which map user/movie id to the corresponding indices.
We have preserved the index ''0'' for ''Unknown'' user/movies, respectively.

To do the feature-engineering stated above, we have to memoize which users/movies had interactions with which movies/users. ::

    # The flags to control the included features.
    use_date = True # use date info or not
    use_iu = True # use implicit user feature
    use_ii = True # use implicit item feature

    movie_vs_watched = dict()
    user_vs_watched = dict()
    for row in df_train.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id
        movie_vs_watched.setdefault(movie_id, list()).append(user_id)
        user_vs_watched.setdefault(user_id, list()).append(movie_id)

    if use_date:
        X_date_train = categorize_date(df_train)
        X_date_test  = categorize_date(df_test)
    else:
        X_date_train, X_date_test = (None, None)


We can then define functions which maps a list of user/movie ids to the features represented in sparse matrix format ::

    # given user/movie ids, add additional infos and return it as sparse
    def augment_user_id(user_ids):
        Xs = []
        X_uid = sps.lil_matrix((len(user_ids), USER_ID_SIZE))
        for index, user_id in enumerate(user_ids):
            X_uid[index, user_to_index[user_id]] = 1
        Xs.append(X_uid)
        if use_iu:
            X_iu = sps.lil_matrix((len(user_ids), MOVIE_ID_SIZE))
            for index, user_id in enumerate(user_ids):
                watched_movies = user_vs_watched.get(user_id, [])
                normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                for uid in watched_movies:
                    X_iu[index, movie_to_index[uid]] = normalizer
            Xs.append(X_iu)
        return sps.hstack(Xs, format='csr')

    def augment_movie_id(movie_ids):
        Xs = []
        X_movie = sps.lil_matrix((len(movie_ids), MOVIE_ID_SIZE))
        for index, movie_id in enumerate(movie_ids):
            X_movie[index, movie_to_index[movie_id]] = 1
        Xs.append(X_movie)

        if use_ii:
            X_ii = sps.lil_matrix((len(movie_ids), USER_ID_SIZE))
            for index, movie_id in enumerate(movie_ids):
                watched_users = movie_vs_watched.get(movie_id, [])
                normalizer = 1 / max(len(watched_users), 1) ** 0.5
                for uid in watched_users:
                    X_ii[index, user_to_index[uid]] = normalizer
            Xs.append(X_ii)    


        return sps.hstack(Xs, format='csr')

^^^^^^^^^^^^
A naive way
^^^^^^^^^^^^

We now setup the problem in a non-relational way: ::

    train_uid_unique, train_uid_index = np.unique(df_train.user_id, return_inverse=True)
    train_mid_unique, train_mid_index = np.unique(df_train.movie_id, return_inverse=True)
    user_data_train = augment_user_id(train_uid_unique)
    movie_data_train = augment_movie_id(train_mid_unique)

    test_uid_unique, test_uid_index = np.unique(df_test.user_id, return_inverse=True)
    test_mid_unique, test_mid_index = np.unique(df_test.movie_id, return_inverse=True)
    user_data_test = augment_user_id(test_uid_unique)
    movie_data_test = augment_movie_id(test_mid_unique)

    X_train_naive = sps.hstack([
        X_date_train,
        user_data_train[train_uid_index],
        movie_data_train[train_mid_index]
    ])

    X_test_naive = sps.hstack([
        X_date_test,
        user_data_test[test_uid_index],
        movie_data_test[test_mid_index]
    ])

    fm_naive = myfm.MyFMRegressor(rank=10).fit(X_train_naive, df_train.rating, n_iter=5, n_kept_samples=5)

In my environment, it takes ~ 2s per iteration, which is much slower than pure MF example.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The problem formulation with RelationBlock.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above code, we have already seen a hint to optimize the performance.
The line ::

        user_data_train[train_uid_index],

says that the sparse matrix  ``user_data_train`` is repeated many times,
and we will compute the same combination of factors repeatedly.

The role of :py:class:`myfm.RelationBlock` is to tell such a repeated pattern explicitly
so that we can drastically reduce the complexity ::

    block_user_train = RelationBlock(train_uid_index, user_data_train)
    block_movie_train = RelationBlock(train_mid_index, movie_data_train)
    block_user_test = RelationBlock(test_uid_index, user_data_test)
    block_movie_test = RelationBlock(test_mid_index, movie_data_test)

We can now feed these blocks into :py:meth:`myfm.MyFMRegressor.fit` by ::

    fm_rb = myfm.MyFMRegressor(rank=10).fit(
        X_date_train, df_train.rating,
        X_rel=[block_user_train, block_movie_train],
        n_iter=300, n_kept_samples=300
    )

Note that we cannot express ``X_date_train`` as a relation block, and we have
supplied such a non-repeated data for the first argument.
This time, the speed is 20iters / s, almost 40x speed up compared to the naive version.

What the relation format does is to reorganize the computation, but it does not 
change the result up to a floating point artifact: ::

    for i in range(5):
        sample_naive = fm_naive.predictor_.samples[i].w
        sample_rb = fm_rb.predictor_.samples[i].w
        print(np.max(np.abs(sample_naive - sample_rb)))
        # should print tiny numbers

The resulting performance measures are RMSE=0.889, MAE=0.7000 : ::

    rmse = ((df_test.rating.values - test_prediction) ** 2).mean() ** 0.5
    mae = np.abs(df_test.rating.values - test_prediction).mean()
    print(f'rmse={rmse}, mae={mae}')

Note that we still haven't exploited all the available ingredients such as
user/item side-information and :ref:`grouping of the input variables <grouping>`.
See also `examples notebooks & scripts <https://github.com/tohtsky/myFM/blob/master/examples/>`_
for further improved results.