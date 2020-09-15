----------------------------
Ordinal Regression Tutorial
----------------------------

^^^^^^^^^^^^^^^^^^^^^^
UCLA Dataset
^^^^^^^^^^^^^^^^^^^^^^

Let us first explain the API of :py:class:`myfm.MyFMOrderedProbit`
using `UCLA dataset <https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/>`_.

It says

    This hypothetical data set has a three level variable called apply, with levels “unlikely”, “somewhat likely”, and “very likely”, coded 1, 2, and 3, respectively, that we will use as our outcome variable. We also have three variables that we will use as predictors: pared, which is a 0/1 variable indicating whether at least one parent has a graduate degree; public, which is a 0/1 variable where 1 indicates that the undergraduate institution is public and 0 private, and gpa, which is the student’s grade point average. 

We can read the data (which is in Stata format) using pandas: ::

    import pandas as pd
    df = pd.read_stata("https://stats.idre.ucla.edu/stat/data/ologit.dta")
    df.head()

this should print

.. csv-table::
    :header-rows: 1

    ,apply,pared,public,gpa
    0,very likely,0,0,3.26
    1,somewhat likely,1,0,3.21
    2,unlikely,1,1,3.94
    3,somewhat likely,0,0,2.81
    4,somewhat likely,0,0,2.53

We regard the target label ``apply`` as a label with ordering, 

.. math::
    (\text{unlikely} = 0) < (\text{somewhat likely} = 1) < (\text{very likely} = 2)

so we map ``apply`` as ::

    y = df['apply'].map({'unlikely': 0, 'somewhat likely': 1, 'very likely': 2}).values

Prepare other features as usual. ::

    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    X = df[['pared', 'public', 'gpa']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Now we can feed the data into :py:class:`myfm.MyFMOrderedProbit`. ::

    from myfm import MyFMOrderedProbit
    clf = MyFMOrderedProbit(rank=0).fit(X_train, y_train, n_iter=200)

    p = clf.predict_proba(X_test)

    print(f'rmse={metrics.log_loss(y_test, p)}')
    # 0.8403, slightly better than constant model baseline.
    print(f'p={p}')
    # a 2D array

Note that unlike binary probit regression, :py:meth:`myfm.MyFMOrderedProbit.predict_proba` 
returns 2D (N_item x N_class) array of class probability.