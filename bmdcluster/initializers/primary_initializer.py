import numpy as np

from .cluster_initializers import initialize_A, initialize_B
from .bootstrap_initializer import initialize_bootstrapped_clusters_block_diagonal
from .bootstrap_initializer import initialize_bootstrapped_clusters_general

def initialize_block_diagonal(W, n_clusters, b=None, init_ratio=1.0, use_bootstrap=False, seed=None):
    """Wrapper function for cluster initialization functions and methods to initialize 
    the data cluster matrix.
    
    Parameters
    ----------
    W : np.array
        binary data matrix
    n_clusters : int
        number od data clusters
    b : int, optional
        size of bootstrapped subset, by default None
    init_ratio : float, optional
        fraction of points in data matrix to initialize, by default 1.0
    use_bootstrap : bool, optional
        use bootstrapping to initialize data clusters, by default False
    seed : int, optional
        randomization seed, by default None
    
    Returns
    -------
    np.array
        initial cluster assignment matrix A
    """

    n, m = W.shape

    if use_bootstrap:

        boot = initialize_bootstrapped_clusters_block_diagonal(W=W, 
                                                                n_clusters=n_clusters, 
                                                                b=b,
                                                                seed=seed)

        A_init = initialize_A(n=n, 
                                n_clusters=n_clusters, 
                                bootstrap=boot, 
                                init_ratio=init_ratio)

    else:

        A_init = initialize_A(n=n, 
                                n_clusters=n_clusters, 
                                init_ratio=init_ratio, 
                                seed=seed)

    return A_init


def initialize_general(W, n_clusters, b=None, f_clusters=None, init_ratio=1.0, B_ident=False, use_bootstrap=False, seed=None):
    """Wrapper function for cluster initialization functions and methods to initialize 
    the data cluster matrix and feature cluster matrix
    
    Parameters
    ----------
    W : np.array
        binary data matrix
    n_clusters : int
        number od data clusters
    b : int, optional
        size of bootstrapped subset, by default None
    f_clusters : int, optional
        number of feature clusters if B_ident is False, by default None
    init_ratio : float, optional
        fraction of points in data matrix to initialize, by default 1.0
    B_ident : bool, optional
        initialize feature cluster matrix to the identity, by default False
    use_bootstrap : bool, optional
        use bootstrapping to initialize data clusters, by default False
    seed : int, optional
        randomization seed, by default None
    
    Returns
    -------
    tuple
        tuple of np.array (initial data clusters, initial feature clusters)
    """

    n, m = W.shape

    if use_bootstrap:

        boot = initialize_bootstrapped_clusters_general(W=W, 
                                                n_clusters=n_clusters, 
                                                B_ident=B_ident, 
                                                f_clusters=f_clusters,
                                                b=b,
                                                seed=seed)
        A_init = initialize_A(n=n, 
                            n_clusters=n_clusters, 
                            bootstrap = boot, 
                            init_ratio=init_ratio,
                            seed=seed)

        B_init = initialize_B(m=m, 
                                f_clusters=f_clusters,
                                B_ident=B_ident, 
                                seed=seed)

    else:
        A_init = initialize_A(n=n, n_clusters=n_clusters, init_ratio=init_ratio, seed=seed)
        B_init = initialize_B(m=m, B_ident=B_ident, seed=seed)

    return A_init, B_init