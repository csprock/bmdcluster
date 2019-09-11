"""
This file contains the main wrapper class for using the bmdcluster package.
"""

import numpy as np

from bmdcluster.optimizers.blockdiagonalBMD import run_bd_BMD
from bmdcluster.optimizers.generalBMD import run_BMD
from bmdcluster.initializers.primaryInitializer import initializeClusters
        

class bmdcluster:

    def __init__(self, n_clusters, method, B_ident, use_bootstrap = False,  **kwargs):

        """Instantiates the :code:`bmdcluster` class with given parameters.

        Parameters
        ----------
        n_clusters: int
            desired number of data clusters
        method: string
            variant of BMD clustering algorithm to use, either 'general' or 'block_diagonal'.
        B_ident: bool
            True initializes feature cluster matrix B to identity matrix
        use_bootstrap: bool
            Initialize data cluster matrix A matrix using bootstrapped subset. (If try requires 'b' to be set in kwargs)

        Keyword Arguments
        -----------------
        b: int, optional
            size of bootstrapped subset
        f_clusters: int, optional
            desired number of feature clusters (must be set of B_ident is False)
        init_ratio: int, optional
            fraction of data points to randomly initialize, must be between 0 and 1

        Tip
        ---
        If you are unsure how many feature clusters there are or you are only interested in clustering the data, we recommend setting :code:`B_ident = True`.

        Note
        -------
        Setting :code:`B_ident = True` may result in empty feature clusters. This is normal because putting each feature in its own cluster makes no assumptions on the relationship between features, leaving the algorithm free to group features as it sees fit.

        """

        # TODO: add attributes to documentation
        # TODO: more efficient handling of kwargs
        # TODO: set B_ident default to True
        # TODO: create better argument names (rename f_clusters to m_clusters)
        # TODO: create fit_transform method
        # TODO: create reverse lookup function that returns cluster given the data point

        self.n_clusters = n_clusters
        self.method = method
        self.use_bootstrap = use_bootstrap
        self.keywords = {}

        if use_bootstrap:
            assert 'b' in kwargs
            self.keywords['b'] = kwargs['b']

        if B_ident == False:
            assert 'f_clusters' in kwargs
            self.B_ident = False
            self.keywords['f_clusters'] = kwargs['f_clusters']
        else:
            self.B_ident = True

        # TODO: add assertion that is between zero and 1
        if 'init_ratio' in kwargs:
            self.keywords['init_ratio'] = kwargs['init_ratio']

        if 'seed' in kwargs:
            self.keywords['seed'] = kwargs['seed']



    def fit(self, W, verbose = 0, return_results = False):
        """Fits algorithm to the data. 
        
        Parameters
        ----------
        W : np.array
            binary data
        verbose : int, optional
            print output each iteration, by default 0
        return_results : bool, optional
            return final value of cost function and cluster membership matrices, otherwise saved in object, by default False
        
        Returns
        -------
        cost: float
            final value of objective function
        A: np.array
            data cluster assignment matrix
        B: np.array
            feature cluster assignment matrix


        Attention
        ---------
        Make sure the numpy array you are passing contains only 0s and 1s, otherwise the algorithm will not work properly.
        """

        self.W = W

        # Initialize cluster indicator matrices.
        self.A, self.B = initializeClusters(self.W, method = self.method,
                                             n_clusters = self.n_clusters,
                                             use_bootstrap = self.use_bootstrap,
                                             B_ident = self.B_ident,
                                             **self.keywords)


        # Run BMD algorithm.
        if self.method == 'general':
            self.cost, self.A, self.B = run_BMD(self.A, self.B, self.W, verbose)
        elif self.method == 'block_diagonal':
            self.cost, self.A, self.B = run_bd_BMD(self.A, self.W, verbose)

        # Optional: return final value of cost function and the indicator matrices.
        if return_results:
            return self.cost, self.A, self.B

    # TODO: chance name to get_members
    def get_indices(self, i, which):
        """Get indices of data and feature cluster members
        
        Parameters
        ----------
        i : int
            cluster
        which : str
            'data' or 'features'
        

        Returns
        -------
        indices: np.array
            data or feature indices belonging to the cluster

        Raises
        -------
        AssertionError
            'which' keyword argument must be either 'data' or 'features'
        """

        assert which in ['data', 'features']

        if which == 'data':
            return np.where(self.A.argmax(axis = 1) == i)
        elif which == 'features':
            return np.where(self.B.argmax(axis = 1) == i)