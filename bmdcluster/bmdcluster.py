"""
This file contains the main wrapper class for using the bmdcluster package.
"""
import warnings
import numpy as np

from optimizers.blockdiagonalBMD import run_bd_BMD
from optimizers.generalBMD import run_BMD
from initializers.primary_initializer import initialize_general
from initializers.primary_initializer import initialize_block_diagonal


class _BMD:

    def __init__(self):
        pass

    @staticmethod
    def _get_labels(M):

        labels = np.full(shape=(M.shape[0], ), fill_value=-1)
        outliers = M.sum(axis=1) < 1
        non_outlier_labels = M[~outliers, :].argmax(axis=1)
        labels[~outliers] = non_outlier_labels

        return labels

class blockdiagonalBMD(_BMD):

    def __init__(self, n_clusters, max_iter=100, use_bootstrap=False, b=None, init_ratio=1.0, seed=None):
        """Run the block-diagonal form of the BMD algorithm. 
        
        Parameters
        ----------
        n_clusters : int
            number of data clusters
        max_iter : int, optional
            maximum number of optimization iterations, by default 100
        use_bootstrap : bool, optional
            use bootstrap cluster initialization, by default False
        b : int, optional
            number of bootstrapped samples to use, by default None
        init_ratio : float, optional
            fraction of points to randomly initialize, by default 1.0
        seed : int, optional
            random initialization seed, by default None
        
        Raises
        ------
        ValueError
            If :code:`use_bootstrap` is set to True but and :code:`b` is not specified
        ValueError
            If both :code:`B_ident` and :code:`f_clusters` are not specified
            
        """


        if use_bootstrap and not b:
            raise ValueError("Must specify keyword argument 'b' when using bootstrapping.")

        self.n_clusters = n_clusters
        self.use_bootstrap = use_bootstrap
        self.b = b
        self.init_ratio = init_ratio
        self.seed = seed
        self.max_iter = max_iter

        super(blockdiagonalBMD, self).__init__()

    def fit(self, W, verbose=False):
        """Fit the model.
        
        Parameters
        ----------
        W : np.array
            binary data matrix
        verbose : bool, optional
            print progress during optimization, by default False
        
        """

        self.W = W

        # Initialize cluster indicator matrices.
        self.A = initialize_block_diagonal(W=self.W, 
                                             n_clusters = self.n_clusters,
                                             use_bootstrap = self.use_bootstrap,
                                             b=self.b,
                                             init_ratio=self.init_ratio,
                                             seed=self.seed)

        self.cost, self.A, self.B = run_bd_BMD(self.A, self.W, self.max_iter, verbose)


    def get_feature_labels(self):
        """Get feature cluster labels after .fit(). Outliers will be labeled -1.
        
        Returns
        -------
        np.array
            feature cluster labels
        """
        return self._get_labels(self.B)

    
    def get_data_labels(self):
        """Get data cluster labels after .fit(). Outliers will be labeled -1.
        
        Returns
        -------
        np.array
            data cluster labels
        """
        return self._get_labels(self.A)


    def fit_predict(self, W, verbose=False):
        """Fit the model and return final value of objective function and 
        cluster assignment labels for the data and features.
        
        Parameters
        ----------
        W : np.array
            binary data matrix
        verbose : bool, optional
            print progress during optimization, by default False
        
        Returns
        -------
        float
            final value of objective function
        np.array
            data cluster labels
        np.array
            feature cluster labels
        """


        self.fit(W, verbose)

        return self.cost, self._get_labels(self.A), self._get_labels(self.B)


    def fit_transform(self, W, verbose=False):
        """Fit the model and return final value of objective function
        and final values of the data and feature cluster assignment 
        matrices A and B, whose entries are cluster affinity scores.
        
        Parameters
        ----------
        W : np.array
            binary data matrix
        verbose : bool, optional
            print progress during optimization, by default False
        
        Returns
        -------
        float
            final cost of objective function
        np.array
            final value of data cluster assignment matrix A
        np.array
            final value of feature cluster assignment matrix B
            
        """


        self.fit(W, verbose)

        return self.cost, self.A, self.B



class generalBMD(_BMD):

    def __init__(self, n_clusters, f_clusters=None, B_ident=True, max_iter=100, use_bootstrap=False, b=None, init_ratio=1.0, seed=None):
        """Run the general form of the BMD algorithm.
        
        Parameters
        ----------
        n_clusters : int
            number of data clusters
        f_clusters : int, optional
            number of feature clusters, by default None
        B_ident : bool, optional
            initialize feature cluster assignment matrix to the identity, by default True
        max_iter : int, optional
            maximum number of optimization iterations, by default 100
        use_bootstrap : bool, optional
            use bootstrap cluster initialization, by default False
        b : int, optional
            number of bootstrapped samples to use, by default None
        init_ratio : float, optional
            fraction of points to randomly initialize, by default 1.0
        seed : int, optional
            random initialization seed, by default None
        
        Raises
        ------
        ValueError
            If :code:`use_bootstrap` is set to True but and :code:`b` is not specified
        ValueError
            If both :code:`B_ident` and :code:`f_clusters` are not specified
        ValueError
            If both :code:`B_ident=True` and :code:`f_clusters` is set

        Caution
        -------
        Setting both :code:`B_ident=True` and :code:`f_clusters` are mutually exclusive options and will result in 
        an error. 

        """

        if use_bootstrap and not b:
            raise ValueError("Must specify keyword argument 'b' when using bootstrapping.")

        if not B_ident and not f_clusters:
            raise ValueError("You must one of either 'B_ident' or 'f_clusters'")

        if B_ident and f_clusters is not None:
            raise ValueError("Cannot set B_ident to True and set f_clusters")

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
        """Fit the model.
        
        Parameters
        ----------
        W : np.array
            binary data matrix
        verbose : bool, optional
            print progress during optimization, by default False
        
        """
        self.W = W

        # Initialize cluster indicator matrices.
        self.A, self.B = initialize_general(W=self.W, 
                                             n_clusters = self.n_clusters,
                                             use_bootstrap = self.use_bootstrap,
                                             B_ident = self.B_ident,
                                             b=self.b,
                                             init_ratio=self.init_ratio,
                                             seed=self.seed,
                                             f_clusters=self.f_clusters)

        self.cost, self.A, self.B = run_BMD(self.A, self.B, self.W, self.max_iter, verbose)


    def get_feature_labels(self):
        """Get feature cluster labels after .fit(). Outliers will be labeled -1.
        
        Returns
        -------
        np.array
            feature cluster labels
        """
        return self._get_labels(self.B)

    
    def get_data_labels(self):
        """Get data cluster labels after .fit(). Outliers will be labeled -1.
        
        Returns
        -------
        np.array
            data cluster labels
        """
        return self._get_labels(self.A)


    def fit_predict(self, W, verbose=False):
        """Fit the model and return final value of objective function and 
        cluster assignment labels for the data and features.
        
        Parameters
        ----------
        W : np.array
            binary data matrix
        verbose : bool, optional
            print progress during optimization, by default False
        
        Returns
        -------
        float
            final value of objective function
        np.array
            data cluster labels
        np.array
            feature cluster labels
        """

        self.fit(W, verbose)

        return self.cost, self._get_labels(self.A), self._get_labels(self.B)


    def fit_transform(self, W, verbose):
        """Fit the model and return final value of objective function
        and final values of the data and feature cluster assignment 
        matrices A and B, whose entries are cluster affinity scores.
        
        Parameters
        ----------
        W : np.array
            binary data matrix
        verbose : bool, optional
            print progress during optimization, by default False
        
        Returns
        -------
        float
            final cost of objective function
        np.array
            final value of data cluster assignment matrix A
        np.array
            final value of feature cluster assignment matrix B
        """

        self.fit(W, verbose)

        return self.cost, self.A, self.B