import numpy as np


###############################################################################
###########                 cluster initializers                  #############
###############################################################################
    

def initializeB(m, B_ident, **kwargs):
    
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
    feature_clusters: int
        the number of feature clusters
    
    Returns
    -------
    B_init: intialized feature indicator matrix B
    
    Raises
    ------
    MissingKeywordArgument: if B_ident is false, the 'feature_cluster' parameter must be specified
    AssertionError: raise if feature_clusters > m    
    
    """
    
    
    if B_ident:
        B_init = np.identity(m)
    else:
        
        if 'feature_clusters' not in kwargs.keys(): raise KeyError("Missing required keyword '%s'" % 'feature_clusters')
        assert 1 < kwargs['feature_clusters'] <= m
        
        if 'seed' in kwargs.keys(): np.random.seed(kwargs['seed'])
        
        C = kwargs['feature_clusters']
        
        B_init = np.zeros((m, C))
        for i in range(C): B_init[i, np.random.randint(C)] = 1
        
    return B_init



def initializeA(n, data_clusters, **kwargs):
    
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
    data_clusters: number of data clusters
    
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
    AssertionError: if data_clusters is greater than 1 or less than 0
    AssertionError: if init_ratio is greater than 1 or less than 0
    
    """
    

    assert 1 < data_clusters < n


    
    A_init = np.zeros((n, data_clusters))
    
    if 'bootstrap' in kwargs.keys(): 
        # Iterate through list of tuples, placing a 1 the corresponding position in A_init. 
        for u in kwargs['bootstrap']: A_init[u[0], u[1]] = 1
    else:
        
        if 'seed' in kwargs.keys(): np.random.seed(kwargs['seed'])
        
        if 'init_ratio' in kwargs.keys():
            assert 0 < kwargs['init_ratio'] <= 1
            # Select a random fraction of points of size init_ratio. 
            for j in np.random.choice(range(n), size = int(n*kwargs['init_ratio']), replace = False):
                A_init[j, np.random.randint(data_clusters)] = 1
        else:
            for j in range(n): A_init[j, np.random.randint(data_clusters)] = 1
                    

    return A_init
    
