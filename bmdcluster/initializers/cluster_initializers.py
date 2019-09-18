""" cluster initializers """

import numpy as np


def initialize_B(m, B_ident=False, f_clusters=None, seed=None):
    """This function initializes the feature cluster indicator matrix B. There are two initialization options.
    The first option places each each feature in its own cluster, so B is initialized to the identity matrix.
    The second option randomly assigns features to clusters uniformly.
    
    Parameters
    ----------
    m : int
        number of features
    B_ident : bool, optional
        initialize to the identity matrix, by default False
    f_clusters : int, optional
        the numberof feature clusters, must be set if B_ident is False, by default None
    seed : int, optional
        randomiazation seed, by default None
    
    Returns
    -------
    np.array
        initialized feature indicator matrix B
    
    Raises
    ------
    KeyError
        raises of 'f_cluster' not set and 'B_ident' is False
    """


    if B_ident:
        B_init = np.identity(m)
    else:

        if not f_clusters:
            raise KeyError("Missing required keyword 'f_clusters'")

        assert 1 < f_clusters <= m

        np.random.seed(seed)

        B_init = np.zeros((m, f_clusters))
        for i in range(m):
            B_init[i, np.random.randint(f_clusters)] = 1

        np.random.seed(None)

    return B_init


def initialize_A(n, n_clusters, init_ratio=1.0, bootstrap=None, seed=None):

    """Initialize data cluster indicator matrix A. There are three initialization options.
    The first option randomly assigns each point to a cluster uniformly. The second
    option selects a random subset of points and assigns those randomly to clusters
    uniformly while the rest of the points remain unassigned. The third option uses a
    user-supplied list of tuples of index-cluster pairs to initialize the matrix
    (index-cluster pairs are row-column tuples of matrix A).
    
    Parameters
    ----------
    n : int
        number of data points
    n_clusters : int
        number of data clusters
    init_ratio : float, optional
        fraction of points to initialize, by default 1.0
    bootstrap : list, optional
        list of tuples: row-column pair that corresponds to a data point and its cluster assignment, by default None
    seed : int, optional
        randomization seed, by default None
    
    Returns
    -------
    A_init: np.array
        initialized data indicator matrix
    """


    assert 1 < n_clusters < n

    A_init = np.zeros((n, n_clusters))

    if bootstrap:
        for u in bootstrap:
            A_init[u[0], u[1]] = 1
    else:
        
        if seed:
            np.random.seed(seed)
        
        assert 0 < init_ratio <= 1
        if init_ratio < 1:
            # Select a random fraction of points of size init_ratio.
            for j in np.random.choice(range(n), size = int(n*init_ratio), replace = False):
                A_init[j, np.random.randint(n_clusters)] = 1
        else:

            for j in range(n): 
                A_init[j, np.random.randint(n_clusters)] = 1

    return A_init