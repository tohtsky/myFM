.. myfm documentation master file, created by
   sphinx-quickstart on Wed Aug 19 13:39:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


myFM - Bayesian Factorization Machines in Python/C++
====================================================

**myFM** is an unofficial implementation of Bayesian Factorization Machines in Python/C++.
Notable features include:

* Implementation most functionalities of `libFM <http://libfm.org/>`_ MCMC engine (including grouping & relation block)
* A simpler and faster implementation with `Pybind11 <https://github.com/pybind/pybind11>`_ and `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_
* Gibbs sampling for **ordinal regression** with probit link function. See :ref:`the tutorial <OrdinalRegression>` for its usage.
* Variational inference which converges faster and requires lower memory (but usually less accurate than the Gibbs sampling).


In most cases, you can install the library from PyPI: ::

   pip install myfm

It has an interface similar to sklearn, and you can use them for wide variety of prediction tasks.
For example,

.. testcode::

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

   print(metrics.roc_auc_score(y_test, fm.predict_proba(X_test)))
   # 0.9954

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   0.99...


Try out the following :ref:`examples <MovielensIndex>` to see how Bayesian approaches to explicit collaborative filtering
are still very competitive (almost unbeaten)!

.. toctree::
   :caption: Basic Usage
   :maxdepth: 1

   quickstart
   movielens
   relation-blocks
   ordinal-regression

.. toctree::
   :caption: Details
   :maxdepth: 1

   api_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
