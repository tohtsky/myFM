.. myfm documentation master file, created by
   sphinx-quickstart on Wed Aug 19 13:39:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to myFM's documentation!
================================

myFM is an unofficial implementation of Bayesian Factorization Machines.

The goals of ``myFM`` are to

* implement a `libFM <http://libfm.org/>`_ - like functionality that is easy to use from Python
* provide a simpler and faster implementation with `Pybind11 <https://github.com/pybind/pybind11>`_ and `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_

If you have a standard Python environment on MacOS/Linux, you can install the library from PyPI: ::

   pip install myFM

It has interfaces similar to sklearn, and you can use them for wide variety of prediction tasks, such as 

* regression (``MyFMRegressor``)
* binary classification (``MyFMClassifier``)
* ordered probit regression (``MyFMOrderedProbit``)

Checkout the :ref:`examples <MovielensIndex>` below to see how Bayesian approaches to explicit collaborative filtering
are still very competitive (almost unbeaten)!

Note that it can also be used to high-performance Bayesian linear regression.

Contents:

.. toctree::
   :caption: Basic Usage
   :maxdepth: 1

   quickstart
   movielens
   relation-blocks

.. toctree::
   :caption: Details
   :maxdepth: 1

   dependencies


.. automodule::  myfm

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
