import numpy as np
from scipy import (special, sparse as sps)

from ._myfm import (
    TaskType, FMLearningConfig, ConfigBuilder,
    FMTrainer, FM, create_train_fm
)

class MyFMRegressor(object):
    def __init__(self, rank, init_stdev=0.1, random_seed=42, **learning_configs):
        self.rank = rank
        self.init_stdev = init_stdev
        self.random_seed = random_seed 
        config_builder = ConfigBuilder()

        self.group_index = None
        for key, value in learning_configs.items():
            if key in [ 'alpha_0', 'beta_0', 'gamma_0', 'mu_0', 'reg_0']:
                value = learning_configs[key]
                getattr(config_builder, "set_{}".format(key))(value)
            elif key == "group_index":
                self.group_index = learning_configs[key]
            else:
                raise RuntimeError("Got unknown keyword argument {}.".format(key))

        self.config_builder = config_builder
        self.set_tasktype()
        self.fms_ = []

    def set_tasktype(self):
        self.config_builder.set_task_type(TaskType.REGRESSION)

    def predict(self, X):
        if not self.fms_:
            raise RuntimeError("No available sample.")
        X = sps.csr_matrix(X)
        predictions = 0
        for sample in self.fms_:
            sqt = (sample.V **2).sum(axis=1)
            pred = ((X.dot(sample.V) ** 2 ).sum(axis=1) - X.dot(sqt)) / 2
            pred += X.dot(sample.w)
            pred += sample.w0
            predictions += self.process_score(pred)
        return predictions / len(self.fms_)

    @classmethod
    def process_score(cls, y):
        return y

    @classmethod
    def process_y(cls, y):
        return y

    def fit(self, X, y, n_iter=100, n_kept_samples=10, group_index=None):
        if self.group_index is not None:
            assert X.shape[1] == len(self.group_index)
            self.config_builder.set_group_index(self.group_index)
        else:
            self.config_builder.set_indentical_groups(X.shape[1])

        self.config_builder.set_n_iter(n_iter).set_n_kept_samples(n_kept_samples)

        X = sps.csr_matrix(X) 
        y = self.process_y(y)
        config = self.config_builder.build()
        self.fms_ = create_train_fm(self.rank, self.init_stdev, X, y, self.random_seed, config)
        return self


class MyFMClassifier(MyFMRegressor):
    def set_tasktype(self):
        self.config_builder.set_task_type(TaskType.CLASSIFICATION)

    @classmethod
    def process_score(cls, score):
        return ( 1 + special.erf(score / np.sqrt(2)) ) / 2

    @classmethod
    def process_y(cls, y):
        return y.astype(np.float64) * 2 - 1


