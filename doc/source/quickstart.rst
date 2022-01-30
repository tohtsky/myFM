===========
Quick Start
===========


------------
Installation
------------

On MacOS/Linux First try::

    pip install myfm

If it works, you can now try the examples.

If there is something nasty, then read the :ref:`detailed installation guide <DetailedInstallationGuide>`
and figure out what went wrong.
Of course, feel free to create an issue on `GitHub <https://github.com/tohtsky/myFM>`_!


-------------
A toy example
-------------

Let us first look at how :py:class:`myfm.MyFMClassifier` works for `a toy example provided in pyFM <https://github.com/coreylynch/pyFM>`_.

.. doctest ::

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

    # Note that X is a sparse matrix
    print(X.toarray())

    # The target variable to be classified.
    y = np.asarray([0, 1, 1, 0])
    fm = myfm.MyFMClassifier(rank=4)
    fm.fit(X,y)

    # It also supports prediction for new unseen items.
    fm.predict_proba(v.transform([{"user": "1", "item": "10", "age": 24}]))

.. testoutput ::
    :hide:
    :options: +ELLIPSIS

     [[ 19.   0.   0.   0.   1.   1.   0.   0.   0.]
      [ 33.   0.   0.   1.   0.   0.   1.   0.   0.]
      [ 55.   0.   1.   0.   0.   0.   0.   1.   0.]
      [ 20.   1.   0.   0.   0.   0.   0.   0.   1.]]


As the example suggests, :py:class:`myfm.MyFMClassifier` takes
sparse matrices of `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ as its input.
In the above example, `sklearn's DictVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`_
transforms the categorical variables (user id and movie id) into a one-hot encoded vectors.

As you can see, :py:class:MyFMClassifier: can make predictions against
new (unseen) items despite the fact that it is an MCMC solver.
This is possible because it simply retains all the intermediate (noisy) samples.

For more practical example with larger data, move on to :ref:`Movielens examples <MovielensIndex>` .
