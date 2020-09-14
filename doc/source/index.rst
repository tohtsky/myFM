.. myfm documentation master file, created by
   sphinx-quickstart on Wed Aug 19 13:39:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to myFM's documentation!
================================

myFM is an unofficial implementation of Bayesian Factorization Machines.

The goals of myFM are to

* implement a `libFM <http://libfm.org/>`_ - like functionality that is easy to use from Python
* provide a simpler and faster implementation with `Pybind11 <https://github.com/pybind/pybind11>`_ and `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_

If you have a standard Python environment on MacOS/Linux, you can install the library from PyPI: ::

   pip install myFM

It has an interface similar to sklearn, and you can use them for wide variety of prediction tasks.
For example, ::

   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn import metrics

   from myfm import MyFMClassifier

   dataset = load_breast_cancer()
   X = StandardScaler().fit_transform(dataset['data'])
   y = dataset['target']

   X_train, X_test, y_train, y_test = train_test_split(
      X, y, random_state=42
   )
   fm = MyFMClassifier(rank=2).fit(X_train, y_train)

   metrics.roc_auc_score(y_test, fm.predict_proba(X_test))
   # 0.9954

Currently, in addition to regression and binary classification, it also supports
ordinal regression with probit link function.

Try out the following :ref:`examples <MovielensIndex>` to see how Bayesian approaches to explicit collaborative filtering
are still very competitive (almost unbeaten)!


Contents:

.. toctree::
   :caption: Basic Usage
   :maxdepth: 1

   quickstart

.. toctree::
   :caption: Details
   :maxdepth: 1

   dependencies
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Hoge


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
