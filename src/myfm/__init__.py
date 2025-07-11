from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

try:
    from ._version import __version__
except:  # pragma: no cover
    __version__ = "0.0.0"

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
