from collections import OrderedDict
import numpy as np
from scipy import (special, sparse as sps)
from tqdm import tqdm
from . import _myfm as core


def std_cdf(x):
    return (1 + special.erf(x * np.sqrt(.5))) / 2


def check_data_consistency(X, X_rel):
    shape = None
    if X_rel:
        shape_rel_all = {
            rel.mapper_size for rel in X_rel
        }
        if len(shape_rel_all) > 1:
            raise ValueError('Inconsistent case size for X_rel.')
        shape = list(shape_rel_all)[0]

    if X is None:
        if not X_rel:
            raise ValueError('At lease X or X_rel must be provided.')
    else:
        if shape is not None:
            if X.shape[0] != shape:
                raise ValueError('X and X_rel have different shape.')
        shape = X.shape[0]
    return shape


REAL = np.float64


class MyFMRegressor(object):

    def __init__(
        self, rank,
        init_stdev=0.1, random_seed=42,
        alpha_0=1.0, beta_0=1.0, gamma_0=1.0, mu_0=0.0, reg_0=1.0,
    ):
        """Factorization machine with Gibbs sampling.

        Parameters
        ----------
        rank : int
            The number of factors.

        init_stdev : float, optional (defalult = 0.1)
            The standard deviation for initialization.
            The factorization machine weights are randomely sampled from
            `Normal(0, init_stdev)`.

        random_seed : integer, optional (default = 0.1)
            The random seed used inside the whole learning process.

        alpha_0 : float, optional (default = 1.0)
            The half of alpha parameter for the gamma-distribution
            prior for alpha, lambda_w and lambda_V.
            Together with beta_0, the priors for these parameters are
            alpha, lambda_w, lambda_v ~ Gamma(alpha_0 / 2, beta_0 / 2)

        beta_0 : float, optioal (default = 1.0)
            See the explanation for alpha_0 .

        gamma_0: float optional (default = 1.0)
            Inverse variance of the prior for mu_w, mu_v.
            Together with mu_0, the priors for these parameters are
            mu_w, mu_v ~ Normal(mu_0, 1 / gamma_0)

        mu_0:
            See the explanation for gamma_0.

        reg_0:
            Inverse variance of tthe prior for w0.
            w0 ~ Normal(0, 1 / reg_0)
        """
        self.rank = rank

        self.init_stdev = init_stdev
        self.random_seed = random_seed

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma_0 = gamma_0
        self.mu_0 = mu_0

        self.reg_0 = reg_0

        self.predictor_ = None
        self.hypers_ = []

        self.n_groups_ = None

        
    def __str__(self):
        return "{class_name}(init_stdev={init_stdev}, alpha_0={alpha_0}, beta_0={beta_0}, gamma_0={gamma_0}, mu_0={mu_0}, reg_0={reg_0})".format(
            class_name=self.__class__.__name__,
            init_stdev=self.init_stdev,
            alpha_0=self.alpha_0, beta_0=self.beta_0,
            gamma_0=self.gamma_0, mu_0=self.mu_0,
            reg_0=self.reg_0
        )

    def fit(self, X, y, X_rel=[],
            X_test=None, y_test=None, X_rel_test=None,
            n_iter=100, n_kept_samples=None, grouping=None, callback=None, config_builder=None):
        """Performs Gibbs sampling to fit the data.
        Parameters
        ----------
        X : 2D array-like.
            Input variable.

        y : 1D array-like.
            Target variable.

        X_rel: list of RelationBlock, optional (defalult=[])
               Relation blocks which supplement X.

        n_iter : int, optional (defalult = 100)
            Iterations to perform.

        n_kept_samples: int, optional (default = None)
            The number of samples to store.
            If `None`, the value is set to `n_iter` - 5.

        grouping: Integer array, optional (default = None)
            If not `None`, this specifies which column of X belongs to which group.
            That is, if grouping[i] is g, then, w_i and V_{i, r}
            will be distributed according to
            Normal(mu_w[g], lambda_w[g]) and Normal(mu_V[g, r], lambda_V[g,r]),
            respectively.
            If `None`, all the columns of X are assumed to belong to a single group 0.

        callback: function(int, fm, hyper) -> bool, optional(default = None)
            Called at the every end of each iteration.
        """

        if config_builder is None:
            config_builder = core.ConfigBuilder()
        train_size = check_data_consistency(X, X_rel)
        if X is None:
            X = sps.csr_matrix((train_size, 0), dtype=np.float64)

        assert X.shape[0] == y.shape[0]
        dim_all = X.shape[1] + sum([rel.feature_size for rel in X_rel])

        if n_kept_samples is None:
            n_kept_samples = n_iter - 10
        else:
            assert n_iter >= n_kept_samples


        for key in ['alpha_0', 'beta_0', 'gamma_0', 'mu_0', 'reg_0']:
            value = getattr(self, key)
            getattr(config_builder, "set_{}".format(key))(value)
        if grouping is None:
            self.n_groups_ = 1
            config_builder.set_identical_groups(dim_all)
        else:
            assert dim_all == len(grouping)
            self.n_groups_ = np.unique(grouping).shape[0]
            config_builder.set_group_index(grouping)

        pbar = None
        if (X_test is not None or X_rel_test):
            if y_test is None:
                raise RuntimeError(
                    "Must specify both (X_test or X_rel_test) and y_test.")
            test_size = check_data_consistency(X_test, X_rel_test)
            assert test_size == y_test.shape[0]
            if X_test is None:
                X_test = sps.csr_matrix((test_size, 0), dtype=np.float64)
            do_test = True
        elif y_test is not None:
            raise RuntimeError(
                "Must specify both (X_test or X_rel_test) and y_test.")
        else:
            do_test = False

        config_builder.set_n_iter(n_iter).set_n_kept_samples(n_kept_samples)

        X = sps.csr_matrix(X)
        if X.dtype != np.float64:
            X.data = X.data.astype(np.float64)
        y = self.process_y(y)
        self.set_tasktype(config_builder)

        config = config_builder.build()

        if callback is None:
            pbar = tqdm(total=n_iter)

            def callback(i, fm, hyper):
                pbar.update(1)
                if i % 5:
                    return False

                log_str = self.status_report(fm, hyper)

                if do_test:
                    pred_this = self.process_score(
                        fm.predict_score(X_test, X_rel_test))
                    val_results = self.measure_score(pred_this, y_test)
                    for key, metric in val_results.items():
                        log_str += " {}_this: {:.2f}".format(key, metric)

                pbar.set_description(log_str)
                return False

        try:
            self.predictor_, self.hypers_ = \
                core.create_train_fm(self.rank, self.init_stdev, X, X_rel,
                                     y, self.random_seed, config, callback)
            return self
        finally:
            if pbar is not None:
                pbar.close()

    def set_tasktype(self, config_builder):
        config_builder.set_task_type(core.TaskType.REGRESSION)

    def predict(self, X, X_rel=[], n_workers=None):
        shape = check_data_consistency(X, X_rel)
        if X is None:
            X = sps.csr_matrix((shape, 0), dtype=np.float64)
        if n_workers is not None:
            return self.predictor_.predict_parallel(X, X_rel, n_workers)
        else:
            return self.predictor_.predict(X, X_rel)

    @classmethod
    def status_report(cls, fm, hyper):
        log_str = "alpha = {:.2f} ".format(hyper.alpha)
        log_str += "w0 = {:.2f} ".format(fm.w0)
        return log_str

    @classmethod
    def process_score(cls, y):
        return y

    @classmethod
    def process_y(cls, y):
        return y

    @classmethod
    def measure_score(cls, prediction, y):
        result = OrderedDict()
        result['rmse'] = ((y - prediction) ** 2).mean() ** 0.5
        result['mae'] = np.abs(y - prediction).mean()
        return result

    def get_hyper_trace(self, dataframe=True):
        columns = (
            ['alpha'] +
            ['mu_w[{}]'.format(g) for g in range(self.n_groups_)] +
            ['lambda_w[{}]'.format(g) for g in range(self.n_groups_)] +
            ['mu_V[{},{}]'.format(g, r) for g in range(self.n_groups_) for r in range(self.rank)] +
            ['lambda_V[{},{}]'.format(g, r) for g in range(
                self.n_groups_) for r in range(self.rank)]
        )

        res = []
        for hyper in self.hypers_:
            res.append(np.concatenate([
                [hyper.alpha], hyper.mu_w, hyper.lambda_w,
                hyper.mu_V.ravel(), hyper.lambda_V.ravel()
            ]))
        res = np.vstack(res)
        if dataframe:
            import pandas as pd
            res = pd.DataFrame(res)
            res.columns = columns
            return res
        else:
            return [
                {key: sample[i] for i, key in enumerate(columns)}
                for sample in res
            ]


class MyFMClassifier(MyFMRegressor):
    def set_tasktype(self, config_builder):
        config_builder.set_task_type(core.TaskType.CLASSIFICATION)

    @classmethod
    def process_score(cls, score):
        return std_cdf(score)

    @classmethod
    def process_y(cls, y):
        return y.astype(np.float64) * 2 - 1

    @classmethod
    def measure_score(cls, prediction, y):
        result = OrderedDict()
        lp = np.log(prediction + 1e-15)
        l1mp = np.log(1 - prediction + 1e-15)
        gt = y > 0
        result['ll'] = - lp.dot(gt) - l1mp.dot(~gt)
        result['accuracy'] = np.mean((prediction >= 0.5) == gt)
        return result

    @classmethod
    def status_report(cls, fm, hyper):
        log_str = "w0 = {:.2f} ".format(fm.w0)
        return log_str

    def predict(self, *args, **kwargs):
        return (self.predict_proba(*args, **kwargs)) > 0.5

    def predict_proba(self, *args, **kwargs):
        return super().predict(*args, **kwargs)


class MyFMOrderedProbit(MyFMRegressor):
    def set_tasktype(self, config_builder):
        config_builder.set_task_type(core.TaskType.ORDERED)

    def fit(self, X, y, X_rel=[],
            n_iter=100, n_kept_samples=None, grouping=None, callback=None,
            cutpoint_group_configs=None
    ):
        config_builder = core.ConfigBuilder()
        y = np.asarray(y)
        if cutpoint_group_configs is None:
            n_class = y.max() + 1
            cutpoint_group_configs = [
                (int(n_class), np.arange(y.shape[0], dtype=np.int64))
            ]
        self.n_cutpoint_groups = len(cutpoint_group_configs)
        config_builder.set_cutpoint_groups(
            cutpoint_group_configs
        )
        return super().fit(
            X, y, X_rel=X_rel,
            n_iter=n_iter, n_kept_samples=n_kept_samples,
            grouping=grouping, callback=callback,
            config_builder=config_builder
        )

    @classmethod
    def process_score(cls, score):
        return (1 + special.erf(score * np.sqrt(.5))) / 2

    @classmethod
    def process_y(cls, y):
        y_as_float = y.astype(np.float64)
        assert y.min() >= 0
        return y_as_float

    @classmethod
    def measure_score(cls, prediction, y):
        raise NotImplementedError('not implemented')
        result = OrderedDict()
        lp = np.log(prediction + 1e-15)
        l1mp = np.log(1 - prediction + 1e-15)
        gt = y > 0
        result['ll'] = - lp.dot(gt) - l1mp.dot(~gt)
        result['accuracy'] = np.mean((prediction >= 0.5) == gt)
        return result

    @classmethod
    def status_report(cls, fm, hyper):
        
        log_str = "w0= {:2f}".format(fm.w0)
        if len(fm.cutpoints) == 1:
            log_str += ", cutpoint = {} ".format(
                ["{:.3f}".format(c) for c in list(fm.cutpoints[0])]
            )
        return log_str


    def predict_proba(self, X, rels=[], cutpoint_index=None, **kwargs):
        if cutpoint_index is None:
            if self.n_cutpoint_groups == 1:
                cutpoint_index = 0
        else:
            raise ValueError('specify the cutpoint index')
        X = sps.csr_matrix(X)
        if X.dtype != np.float64:
            X.data = X.data.astype(np.float64)
        p = 0

        sample_offset = len(self.hypers_) - len(self.predictor_.samples)
        for sample_index, sample in enumerate(self.predictor_.samples):
            alpha = self.hypers_[sample_index + sample_offset].alpha
            score = sample.predict_score(X, rels)
            score = std_cdf(
                np.sqrt(alpha) * (sample.cutpoints[cutpoint_index][np.newaxis, :] - score[:, np.newaxis])
            )
            score = np.hstack([
                np.zeros((score.shape[0], 1), dtype=score.dtype),
                score,
                np.ones((score.shape[0], 1), dtype=score.dtype)
            ])
            p += (score[:, 1:] - score[:, :-1])
        return p / len(self.predictor_.samples)

    def predict(self, *args, **kwargs):
        return self.predict_proba(*args, **kwargs).argmax(axis=1)
