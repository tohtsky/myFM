from . import _myfm as core
from ._myfm import RelationBlock

# from .wrapper import MyFMRegressor, MyFMClassifier, MyFMOrderedProbit
from .gibbs import MyFMGibbsClassifier, MyFMGibbsRegressor, MyFMOrderedProbit

MyFMRegressor = MyFMGibbsRegressor
MyFMClassifier = MyFMGibbsClassifier

__all__ = [
    "core",
    "RelationBlock",
    "MyFMOrderedProbit",
    "MyFMRegressor",
    "MyFMClassifier",
    "MyFMGibbsRegressor",
    "MyFMGibbsClassifier",
]
