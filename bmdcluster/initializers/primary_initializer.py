import numpy as np

from bmdcluster.initializers.cluster_initializers import initialize_A, initialize_B
from bmdcluster.initializers.bootstrap_initializer import initialize_bootstrapped_clusters


def initialize_clusters(W, method, n_clusters, b=None, f_clusters=None, init_ratio=1.0, B_ident=False, use_bootstrap=False, seed=None):
    """Wrapper function for cluster initialization functions and methods.
    
    Parameters
    ----------
    W : np.array
        binary data matrix
    method : str
        BMD method, one of 'block_diagonal' or 'general'
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
        tuple of np.array containing (data cluster matrix, feature cluster matrix)
    
    Raises
    ------
    AssertionError
        'method' must be either 'block_diagonal' or 'general'
    """

    assert method in ['block_diagonal', 'general']

    n, m = W.shape

    if method == 'block_diagonal':
        if use_bootstrap:

            boot = initialize_bootstrapped_clusters(W=W, 
                                                  method=method, 
                                                  n_clusters=n_clusters, 
                                                  B_ident=B_ident,
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

        return A_init, None
        
    else:
        if use_bootstrap:

            boot = initialize_bootstrapped_clusters(W=W, 
                                                  method=method, 
                                                  n_clusters=n_clusters, 
                                                  B_ident=B_ident, 
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