from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

try:
    __version__ = get_distribution("myfm").version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover

from ._myfm import RelationBlock
from .gibbs import MyFMGibbsClassifier, MyFMGibbsRegressor, MyFMOrderedProbit
from .variational import VariationalFMClassifier, VariationalFMRegressor

MyFMRegressor = MyFMGibbsRegressor
MyFMClassifier = MyFMGibbsClassifier

__all__ = [
    "RelationBlock",
    "MyFMOrderedProbit",
    "MyFMRegressor",
    "MyFMClassifier",
    "MyFMGibbsRegressor",
    "MyFMGibbsClassifier",
    "VariationalFMRegressor",
    "VariationalFMClassifier",
]
