import numpy as np

from bmdcluster.initializers.cluster_initializers import initialize_A, initialize_B
from bmdcluster.initializers.bootstrap_initializer import initialize_bootstrapped_clusters


def initialize_clusters(W, method, n_clusters, b=None, f_clusters=None, init_ratio=None, B_ident=False, use_bootstrap=False, seed=None):

    assert method in ['block_diagonal', 'general']

    if use_bootstrap and not b:
        raise ValueError("Must specify keyword argument 'b' when using bootstrapping.")

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
            assert b is not None
            boot = initialize_bootstrapped_clusters(W = W, 
                                                  method = method, 
                                                  n_clusters = n_clusters, 
                                                  B_ident = B_ident, 
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
