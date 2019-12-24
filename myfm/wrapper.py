import numpy as np
from scipy import (special, sparse as sps)
from tqdm import tqdm
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
    
    @classmethod
    def get_score(cls, fm, X):
        sqt = (fm.V **2).sum(axis=1)
        pred = ((X.dot(fm.V) ** 2 ).sum(axis=1) - X.dot(sqt)) / 2
        pred += X.dot(fm.w)
        pred += fm.w0
        return cls.process_score(pred)

    def predict(self, X):
        if not self.fms_:
            raise RuntimeError("No available sample.")
        X = sps.csr_matrix(X)
        predictions = 0
        for sample in self.fms_:
            predictions += self.get_score(sample, X)
        return predictions / len(self.fms_)

    @classmethod
    def process_score(cls, y):
        return y

    @classmethod
    def process_y(cls, y):
        return y
    
    @classmethod
    def measure_score(cls, prediction, y):
        rmse = ( (y - prediction) ** 2 ).mean() ** 0.5
        return {'rmse': rmse}

    def fit(self, X, y, X_test=None, y_test=None, n_iter=100, n_kept_samples=10, group_index=None, callback=None):
        pbar = None 
        if (X_test is None and y_test is None):
            do_test = False
        elif (X_test is not None and y_test is not None):
            do_test = True
        else:
            raise RuntimeError("Must specify X_test and y_test.")

        if self.group_index is not None:
            assert X.shape[1] == len(self.group_index)
            self.config_builder.set_group_index(self.group_index)
        else:
            self.config_builder.set_indentical_groups(X.shape[1])

        self.config_builder.set_n_iter(n_iter).set_n_kept_samples(n_kept_samples)

        X = sps.csr_matrix(X) 
        if X.dtype != np.float64:
            X.data = X.data.astype(np.float64)
        y = self.process_y(y)
        config = self.config_builder.build()

        if callback is None:
            pbar = tqdm(total=n_iter)
            if do_test:
                X_test = sps.csr_matrix(X_test)
                X_test.data = X_test.data.astype(np.float64)
                y_test = self.process_y(y_test)
            def callback(i, fm, hyper):
                pbar.update(1)
                if i % 5:
                    return False
                log_str = "alpha = {:.2f} ".format(hyper.alpha)
                log_str += "w0 = {:.2f} ".format(fm.w0)

                if do_test:
                    pred_this = self.get_score(fm, X_test)
                    val_results = self.measure_score(pred_this, y_test)
                    for key, metric in val_results.items():
                        log_str += " {}_this: {:.2f}".format(key, metric)

                pbar.set_description(log_str)
                return False 

        try:
            self.fms_ = create_train_fm(self.rank, self.init_stdev, X, y, self.random_seed, config, callback)
            return self
        finally:
            if pbar is not None:
                pbar.close()


class MyFMClassifier(MyFMRegressor):
    def set_tasktype(self):
        self.config_builder.set_task_type(TaskType.CLASSIFICATION)

    @classmethod
    def process_score(cls, score):
        return ( 1 + special.erf(score / np.sqrt(2)) ) / 2

    @classmethod
    def process_y(cls, y):
        return y.astype(np.float64) * 2 - 1

    @classmethod
    def measure_score(cls, prediction, y):
        lp = np.log(prediction + 1e-15)
        l1mp = np.log(1 - prediction + 1e-15)
        gt = y > 0
        ll = - lp.dot(gt) - l1mp.dot(~gt)
        return {'ll': ll / prediction.shape[0]}


