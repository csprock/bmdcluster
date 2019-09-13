import numpy as np

from bmdcluster.initializers.clusterInitializers import initializeA, initializeB
from bmdcluster.optimizers.blockdiagonalBMD import run_bd_BMD
from bmdcluster.optimizers.generalBMD import run_BMD

###############################################################################
#####             bootstrapping cluster functions                        ######
###############################################################################


def bootstrap_data(N, b, seed=None):
    """
    This function computes sets of indices used to create a bootstrapped sample of data.
    A subset of size b is chosen randomly from a set of indices ranging from 0 to N-1. This 
    subset is used to construct a bootstrapped replicate of size N by sampling with replacement.

    Parameters
    ----------
    N: int
        size of data set
    b: int
        size of subset used to bootstrap is passed to bootstrap_data()
    seed: int
        seed for numpy's random number generator


    Returns
    -------
    x_samp: array containing the indices of the subset
    x_rep: array of length N containing replicates bootstrapped from the subset

    Raises
    ------
    AssertionError: raises assertion error if b > N

    """
    assert b <= N

    if seed:
        np.random.seed(seed)

    x_samp = np.random.choice(range(N), size=b, replace = False)
    x_rep = np.random.choice(x_samp, size =N, replace = True)

    np.random.seed(None)

    return x_samp, x_rep


def assign_bootstrapped_clusters(A_boot, x_rep, x_samp):

    """
    Parameters
    ----------
    A_boot: cluster indicator matrix
    x_rep: array containing indices of bootstrapped replicates.
    x_samp: array containing indices of the subset used to create bootstrapped replicates


    Returns
    -------
    seed_points: list of integer tuples the length of x_samp
        each tuple has the form (sample point, assigned cluster)

    """
    assignments = list()

    # Iterate over points in x_samp.
    for p in x_samp:
        # Identify rows of A_boot corresponding to sample point p.
        g = np.where(x_rep == p)[0]
        # Assign cluster number based on maximum of column sum of subsetted cluster matrix A_boot[g,:]
        assignments.append( np.argmax( A_boot[g,:].sum(axis = 0) ))

    seed_points = list(zip(list(x_samp), assignments))
    return seed_points


def initializeBootstrappedClusters(W, method, n_clusters, B_ident, b, seed=None):
    """
    This function creates initial seed clusters for the BMD algorithm by bootstrapping a subset
    of data and running the BMD algorithm on this bootstrapped subset to create data
    cluster assignments for the bootstrapped subset. These cluster assignments are used to
    seed the data clusters for use on the full dataset. This idea comes from "On Clustering
    Binary Data" by Li & Zhu (2005).

    First, the indices of the bootstrapped subset and replicated samples are generated using
    the bootstrap_data() function. These indices are uses to create a bootstrapped dataset
    from W for use with

    Second, the assign_bootstrapped_clusters() function returns the original indices of the
    subset used for bootstrapping along with their assigned cluster. These are later used as
    seed points for running the algorithm on the full dataset.


    Parameters
    ----------
    W: binary data matrix
    method: BMD clustering method either 'block_diagonal' or 'general'
    n_clusters: number of data clusters
    B_ident: bool
        initialize feature cluster matrix to identity, is passed to initializeB()
    b: int
        size of subset used to bootstrap, passed to bootstrap_data()

    kwargs
    ------
    b: size of subset used to bootstrap
        is passed to bootstrap_data()

    Returns
    -------
    seed_points: list of tuples

    """

    n, m = W.shape
    x_samp, x_rep = bootstrap_data(n, b=b, seed=seed)

    if method == 'block_diagonal':
        A_init = initializeA(n=n, n_clusters=n_clusters, seed=seed)
        _, A_boot, _ = run_bd_BMD(A_init, W[x_rep,:], verbose=0)
    else:
        A_init = initializeA(n=n, n_clusters=n_clusters, seed=seed)
        B_init = initializeB(m=m, B_ident=B_ident, seed=seed)

        _, A_boot, _ = run_BMD(A_init, B_init, W[x_rep,:], verbose=0)

    seed_points = assign_bootstrapped_clusters(A_boot, x_rep, x_samp)

    return seed_points
