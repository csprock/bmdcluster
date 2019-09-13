import numpy as np

from bmdcluster.initializers.clusterInitializers import initializeA, initializeB
from bmdcluster.initializers.bootstrapInitializer import initializeBootstrappedClusters

def initializeClusters(W, method, n_clusters, b=None, f_clusters=None, init_ratio=None, B_ident=False, use_bootstrap=False, seed=None):

    n, m = W.shape

    assert method in ['block_diagonal', 'general']

    if method == 'block_diagonal':
        if use_bootstrap:
            assert b is not None
            boot = initializeBootstrappedClusters(W=W, 
                                                  method=method, 
                                                  n_clusters=n_clusters, 
                                                  B_ident=B_ident,
                                                  b=b,
                                                  seed=seed)

            A_init = initializeA(n=n, 
                                 n_clusters=n_clusters, 
                                 bootstrap=boot, 
                                 init_ratio=init_ratio)

        else:

            A_init = initializeA(n=n, 
                                 n_clusters=n_clusters, 
                                 init_ratio=init_ratio, 
                                 seed=seed)

        return A_init, None
    else:

        if use_bootstrap:
            assert b is not None
            boot = initializeBootstrappedClusters(W = W, 
                                                  method = method, 
                                                  n_clusters = n_clusters, 
                                                  B_ident = B_ident, 
                                                  b=b,
                                                  seed=seed)
            A_init = initializeA(n=n, 
                                n_clusters=n_clusters, 
                                bootstrap = boot, 
                                init_ratio=init_ratio,
                                seed=seed)

            B_init = initializeB(m=m, 
                                 f_clusters=f_clusters,
                                 B_ident=B_ident, 
                                 seed=seed)

        else:
            A_init = initializeA(n=n, n_clusters=n_clusters, init_ratio=init_ratio, seed=seed)
            B_init = initializeB(m=m, B_ident=B_ident, seed=seed)

        return A_init, B_init
