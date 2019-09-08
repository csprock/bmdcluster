import numpy as np
from .clusterInitializers import initializeA, initializeB
from .bootstrapInitializer import initializeBootstrappedClusters

def initializeClusters(W, method, n_clusters, B_ident = False, use_bootstrap = False, **kwargs):

    n, m = W.shape

    assert method in ['block_diagonal', 'general']

    if method == 'block_diagonal':
        if use_bootstrap:
            boot = initializeBootstrappedClusters(W = W, method = method, n_clusters = n_clusters, B_ident = B_ident, **kwargs)
            A_init = initializeA(n, n_clusters, bootstrap = boot, **kwargs)
        else:
            A_init = initializeA(n, n_clusters, **kwargs)
        return A_init, None
    else:

        if use_bootstrap:
            boot = initializeBootstrappedClusters(W = W, method = method, n_clusters = n_clusters, B_ident = B_ident, **kwargs)
            A_init = initializeA(n, n_clusters, bootstrap = boot, **kwargs)
            B_init = initializeB(m, B_ident = B_ident, **kwargs)

        else:
            A_init = initializeA(n, n_clusters, **kwargs)
            B_init = initializeB(m, method = method, B_ident = B_ident, **kwargs)

        return A_init, B_init
