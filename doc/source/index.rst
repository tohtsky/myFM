.. myfm documentation master file, created by
   sphinx-quickstart on Wed Aug 19 13:39:04 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to myFM's documentation!
================================

myFM provides an unofficial implementation of Bayesian Factorization Machines.

The goal of this project is to

* implement a `libFM <http://libfm.org/>`_ - like functionality that is easy to use from Python
* provide a simpler and faster implementation with `Pybind11 <https://github.com/pybind/pybind11>`_ and `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_
* implement Ordred Probit Regression models and Factorization Machine generalization thereof.

Contents:

.. toctree::
   :caption: Basic Usage
   :maxdepth: 1

   quickstart
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
