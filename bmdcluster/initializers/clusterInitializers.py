import numpy as np


###############################################################################
###########                 cluster initializers                  #############
###############################################################################


def initializeB(m, f_clusters=None, B_ident=False, seed=None):

    """
    This function initializes the feature cluster indicator matrix B. There are two initialization options.
    The first option places each each feature in its own cluster, so B is initialized to the identity matrix.
    The second option randomly assigns features to clusters uniformly.

    Parameters
    ----------
    m: the number of features
    B_ident: bool
        True initializes to identity matrix

    kwargs
    ------
    f_clusters: int
        the number of feature clusters

    Returns
    -------
    B_init: intialized feature indicator matrix B

    Raises
    ------
    MissingKeywordArgument: if B_ident is false, the 'feature_cluster' parameter must be specified
    AssertionError: raise if f_clusters > m

    """


    if B_ident:
        B_init = np.identity(m)
    else:

        #if 'f_clusters' not in kwargs.keys(): raise KeyError("Missing required keyword '%s'" % 'f_clusters')

        if not f_clusters:
            raise KeyError("Missing required keyword 'f_clusters'")

        assert 1 < f_clusters <= m

        np.random.seed(seed)

        #C = kwargs['f_clusters']

        B_init = np.zeros((m, f_clusters))
        for i in range(f_clusters): 
            B_init[i, np.random.randint(f_clusters)] = 1

        np.random.seed(None)

    return B_init



def initializeA(n, n_clusters, init_ratio=None, bootstrap=None, seed=None):

    """
    Initialize data cluster indicator matrix A. There are three initialization options.
    The first option randomly assigns each point to a cluster uniformly. The second
    option selects a random subset of points and assigns those randomly to clusters
    uniformly while the rest of the points remain unassigned. The third option uses a
    user-supplied list of tuples of index-cluster pairs to initialize the matrix
    (index-cluster pairs are row-column tuples of matrix A).

    Parameters
    ----------
    n: number of data points
    n_clusters: number of data clusters

    kwargs
    ------
    init_ratio: number between 0 and 1
        Share of points to initialize
    bootstrap: list of tuples
        Each tuple is a row-column pair that corresponds to a data point and its cluster assignment

    Return
    ------
    A_init: initialized data indicator matrix

    Raises
    ------
    AssertionError: if n_clusters is greater than 1 or less than 0
    AssertionError: if init_ratio is greater than 1 or less than 0

    """


    assert 1 < n_clusters < n



    A_init = np.zeros((n, n_clusters))

    if bootstrap:
        for u in bootstrap:
            A_init[u[0], u[1]] = 1
    else:
        
        if seed:
            np.random.seed(seed)
        
        if init_ratio is not None:

            assert 0 < init_ratio <= 1

            # Select a random fraction of points of size init_ratio.
            for j in np.random.choice(range(n), size = int(n*init_ratio), replace = False):
                A_init[j, np.random.randint(n_clusters)] = 1
        else:

            for j in range(n): 
                A_init[j, np.random.randint(n_clusters)] = 1

    # if 'bootstrap' in kwargs.keys():
    #     # Iterate through list of tuples, placing a 1 the corresponding position in A_init.
    #     for u in kwargs['bootstrap']: A_init[u[0], u[1]] = 1
    # else:

    #     if 'seed' in kwargs.keys(): np.random.seed(kwargs['seed'])

    #     if 'init_ratio' in kwargs.keys():
    #         assert 0 < kwargs['init_ratio'] <= 1
    #         # Select a random fraction of points of size init_ratio.
    #         for j in np.random.choice(range(n), size = int(n*kwargs['init_ratio']), replace = False):
    #             A_init[j, np.random.randint(n_clusters)] = 1
    #     else:
    #         for j in range(n): A_init[j, np.random.randint(n_clusters)] = 1


    return A_init
