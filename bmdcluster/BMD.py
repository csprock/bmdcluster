"""
This file contains the main wrapper class for using the BMDcluster package.
"""

import numpy as np
from optimizers.blockdiagonalBMD import run_bd_BMD
from optimizers.generalBMD import run_BMD
from initializers.primaryInitializer import initializeClusters


class BMD:
    """ Wrapper class for BMD clustering methods. """
    
    def __init__(self, data_clusters, method, B_ident, use_bootstrap = False,  **kwargs):
        """
        Instantiates configuration parameters for the BMD algorithm. 
        
        Parameters
        ----------
        data_clusters: int
            desired number of data clusters
        method: string
            variant of BMD clustering algorithm to use, either 'general' or 'block_diagonal'.
        B_ident: bool
            True initializes feature cluster matrix B to identity matrix
        use_bootstrap: bool
            Initialize data cluster matrix A matrix using bootstrapped subset. (If try requires 'b' to be set in kwargs)
            
        kwargs
        ------
        b: int
            size of bootstrapped subset
        feature_clusters: int
            desired number of feature clusters (must be set of B_ident is False)
        init_ratio: number between 0 and 1
            fraction of data points to randomly initialize
        
        """
        
        
        self.data_clusters = data_clusters
        self.method = method
        self.use_bootstrap = use_bootstrap
        self.keywords = {}
        
        if use_bootstrap:
            assert 'b' in kwargs
            self.keywords['b'] = kwargs['b']
            
        if B_ident == False:
            assert 'feature_clusters' in kwargs
            self.B_ident = False
            self.keywords['feature_clusters'] = kwargs['feature_clusters']
        else:
            self.B_ident = True
            
            
        if 'init_ratio' in kwargs:
            self.keywords['init_ratio'] = kwargs['init_ratio']
            
        
    
    def fit(self, W, verbose = 0, return_results = False):
        """ Runs BMD algorithm on data matrix W using instantiated parameters."""
        
        self.W = W
        
        # Initialize cluster indicator matrices. 
        self.A, self.B = initializeClusters(self.W, method = self.method, 
                                             data_clusters = self.data_clusters, 
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
        
        
    
    def get_indices(self, i, which):
        """ 
        Get the indices of data or features of the given cluster. 
        
        Parameters
        ----------
        i: int
            cluster
        which: specify which part to return cluster indices for.
            Either 'data' or 'features'
            
        Returns
        -------
        numpy array of indices
        
        Raises
        ------
        AssertionError: 
            'which' must be either 'data' or 'features'
        
        """
        
        
        
        assert which in ['data', 'features']
        
        if which == 'data':
            return np.where(self.A.argmax(axis = 1) == i)
        elif which == 'features':
            return np.where(self.B.argmax(axis = 1) == i)
