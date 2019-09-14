"""
This file contains the main wrapper class for using the bmdcluster package.
"""

import numpy as np

from bmdcluster.optimizers.blockdiagonalBMD import run_bd_BMD
from bmdcluster.optimizers.generalBMD import run_BMD
from bmdcluster.initializers.primary_initializer import initialize_clusters


class _BMD:

    def __init__(self):
        pass

    @staticmethod
    def _get_labels(M):
        return M.argmax(axis=1)



class blockdiagonalBMD(_BMD):

    def __init__(self, n_clusters, f_clusters=None, max_iter=100, B_ident=False, use_bootstrap=False, b=None, init_ratio=1.0, seed=None):

        # TODO: add docstring

        if use_bootstrap and not b:
            raise ValueError("Must specify keyword argument 'b' when using bootstrapping.")

        if not B_ident and not f_clusters:
            raise ValueError("You must one of either 'B_ident' or 'f_clusters'")

        self.n_clusters = n_clusters
        self.B_ident = B_ident
        self.use_bootstrap = use_bootstrap
        self.b = b
        self.init_ratio = init_ratio
        self.f_clusters = f_clusters
        self.seed = seed
        self.max_iter = max_iter

        super(blockdiagonalBMD, self).__init__()


    def fit(self, W, verbose=False):

        self.W = W

        # Initialize cluster indicator matrices.
        self.A, self.B = initialize_clusters(W=self.W, 
                                             method = 'block_diagonal',
                                             n_clusters = self.n_clusters,
                                             use_bootstrap = self.use_bootstrap,
                                             B_ident = self.B_ident,
                                             b=self.b,
                                             init_ratio=self.init_ratio,
                                             seed=self.seed,
                                             f_clusters=self.f_clusters)

        self.cost, self.A, self.B = run_bd_BMD(self.A, self.W, self.max_iter, verbose)

        return self.cost, self.A, self.B

    def fit_predict(self, W, verbose=False):

        # TODO: add docstring

        cost, A, B = self.fit(W, verbose)

        return cost, self._get_labels(A), self._get_labels(B)


    def fit_transform(self, W, verbose):

        # TODO: add docstring

        cost, A, B = self.fit(W, verbose)

        return cost, A, B



class generalBMD(_BMD):

    def __init__(self, n_clusters, f_clusters=None, max_iter=100, B_ident=False, use_bootstrap=False, b=None, init_ratio=1.0, seed=None):

        # TODO: add docstring
        if use_bootstrap and not b:
            raise ValueError("Must specify keyword argument 'b' when using bootstrapping.")

        if not B_ident and not f_clusters:
            raise ValueError("You must one of either 'B_ident' or 'f_clusters'")

        self.n_clusters = n_clusters
        self.B_ident = B_ident
        self.use_bootstrap = use_bootstrap
        self.b = b
        self.init_ratio = init_ratio
        self.f_clusters = f_clusters
        self.seed = seed
        self.max_iter = max_iter


        super(generalBMD, self).__init__()

    
    def fit(self, W, verbose=False):

        # TODO: add docstring

        self.W = W

        # Initialize cluster indicator matrices.
        self.A, self.B = initialize_clusters(W=self.W, 
                                             method = 'general',
                                             n_clusters = self.n_clusters,
                                             use_bootstrap = self.use_bootstrap,
                                             B_ident = self.B_ident,
                                             b=self.b,
                                             init_ratio=self.init_ratio,
                                             seed=self.seed,
                                             f_clusters=self.f_clusters)

        self.cost, self.A, self.B = run_BMD(self.A, self.B, self.W, self.max_iter, verbose)

        return self.cost, self.A, self.B


    def fit_predict(self, W, verbose=False):

        # TODO: add docstring

        cost, A, B = self.fit(W, verbose)

        return cost, self._get_labels(A), self._get_labels(B)


    def fit_transform(self, W, verbose):

        # TODO: add docstring

        cost, A, B = self.fit(W, verbose)

        return cost, A, B