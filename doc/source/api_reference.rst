.. _APIReference:

==============
API References
==============

.. currentmodule:: myfm

Training API
------------
.. autosummary::
    :toctree: api_reference 

    RelationBlock
    MyFMRegressor
    MyFMClassifier
    MyFMGibbsRegressor
    MyFMGibbsClassifier
    MyFMOrderedProbit
    VariationalFMRegressor
    VariationalFMClassifier

.. currentmodule:: myfm

Benchmark Dataset
-----------------
.. autosummary::
    :toctree: api_reference 

    utils.benchmark_data.MovieLens100kDataManager
    utils.benchmark_data.MovieLens1MDataManager
    utils.benchmark_data.MovieLens10MDataManager


Utilities for Sparse Matrix Construction
----------------------------------------

.. autosummary::
    :toctree: api_reference

    utils.encoders.DataFrameEncoder
    utils.encoders.CategoryValueToSparseEncoder
    utils.encoders.MultipleValuesToSparseEncoder
    utils.encoders.BinningEncoder
