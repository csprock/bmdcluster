import numpy as np

"""
This module contains functions that implement the general variant of the Binary Matrix Decomposition (BMD) method 
for clustering binary data as presented by Tao Li in "A General Model for Clustering Binary Data" (2005). Two algorithms
are presented in the paper. This module implements Algorithm 1, which is a general procedure for clustering binary data. 

The BMD algorithm solves the two-sided clustering problem of clustering data points and features simultaneously. We therefore
refer to the 'data clusters' and 'feature clusters' to mean the cluster assignments of the data points and the cluster assignments 
of the features. 


General Nomenclature:

 K: the number of data clusters
 C: the number of feature clusters
 n: size of data set
 m: number of data features
 W: binary data matrix
    Is of size n x m, with data in rows and features in columns. 
 A: data cluster indicator matrix
    n x K binary indicator matrix encoding the cluster membership of the data. 
    Each point can belong to exactly one cluster, so each row consists of zeros except for a single 1. 
 B: feature cluster indicator matrix. 
    m x C binary indicator matrix encoding the cluster membership of the features. 
    Each feature can belong to exactly one cluster, so each row consists of zeros except for a single 1.  
 X: a K x C matrix that encodes the relationship between data clusters and feature clusters.

"""

ITER_MESSAGE = "Iteration: {0} ............. Cost: {1:.3f}"


def _objective(A,B,X,W):
    """ Computes the objective function for the general BMD algorithm. """
    return np.linalg.norm(W - np.dot(A, np.dot(X,B.T)))


def _is_outlier(M):
    """Determines if a point is an outlier if the affiliation scores between
    a feature/data point and a cluster are all the same. Done by checking
    if all entries in a row of the affiliation score matrix are equal. Returns
    a 1D numpy boolean array indicating if that point is an outlier
    
    Parameters
    ----------
    M : np.array
        cluster affiliation matrix for features or data points
    
    Returns
    -------
    np.array
        1D boolean array
    """
    return np.apply_along_axis(lambda x: len(x) == np.sum(x == np.min(x)), axis=1, arr=M)


def _updateX(A,B,W):
    """Updates the cluster centroid matrix X given A,B, and W according to Equation 5 in Li (2005).
    
    The kc-th entry of X is the sum of the entries of W in the kth data cluster and cth feature cluster 
    normalized by the product of the cluster sizes and can be thought of as cluster centroids.
    Entries of X that correspond to an empty cluster are set to zero. 
    
    Parameters
    ----------
    A : np.array
        old data cluster assignment matrix
    B : np.array
        old feature cluster assignment matrix
    W : np.array
        data matrix
    
    Returns
    -------
    np.array
        updated cluster correspondence matrix X
    """

    # Compute number of points in each data cluster by summing the rows of A.
    p = A.sum(axis = 0)
    # Compute number of points in each feature cluster by rumming rows of B. 
    q = B.sum(axis = 0) 
    
    # Create matrix of normalization entries as outer product of cluster size vectors. 
    denom = np.outer(p, q)
    
    # Compute updated X matrix by the formula (1/pq')*A'WB, setting nan's resulting from zero division to zero. 
    X_new = np.divide(1, denom, out = np.zeros_like(denom), where = denom!=0)*np.dot(A.T, np.dot(W,B))
    return X_new


def _m_ik(indices, W, X, B):
    """The data cluster indicator matrix A is updated using Formula 6 in Li (2005), which uses 
    an 'affiliation score' that can be thought of as a distance between the i-th point and
    the center of the k-th data cluster. The point is then assigned to the cluster with the
    lowest score. 
    
    This function computes this score for a given data point i and data cluster k by summing
    over the C feature clusters while holding k and i fixed using the following formula:
    
        m[i,k] = SUM_{c} [ (W[i,:] - X[k,c])'B[:,c] ]^2
    
    Parameters
    ----------
    indices : tuple
        (index of data point, index of data cluster)
    W : np.array
        data matrix
    X : np.array
        cluster centroid matrix
    B : np.array
        feature cluster assignment matrix
    
    Returns
    -------
    float
        affiliation score between the ith data point and the kth cluster
    """

    
    i, k = indices # get indices

    C = X.shape[1] # number of feature clusters
    m_ik = 0
    
    # W[i,:] - ith row of W
    # X[k,c] - kc-th entry of X (kc-th centriod)
    # B[:,c] - cth column of B
    
    # sum over the feature clusters C
    for c in range(C):  
        m_ik += np.dot(np.square(W[i,:] - X[k,c]), B[:,c])

    return m_ik
    

def _updateA(A,B,X,W):
    """Updates the matrix A by creating a matrix M of identical dimensions whose elements are 'affiliation scores'.
    For each row, a 1 is placed in the position of the smallest entry and the rest set to 0's. In the case
    of ties, the entire row is set to 0 following the convention set in Li and Zhu (2005) who term such 
    cases 'outliers'. 
    
    Parameters
    ----------
    A : np.array
        old data cluster assignment matrix
    B : np.array
        old data feature assignment matrix
    X : np.array
        old cluster centroid matrix
    W : np.array
        data matrix
    
    Returns
    -------
    np.array
        new data cluster assignment matrix
    """

    
    n, K = A.shape
    A_new = np.zeros((n,K))
    
    for i in range(n):                   # iterate over data (rows)
        for k in range(K):               # iterate over features (columns)
            A_new[i,k] = _m_ik((i,k), W,X,B)     
    
    
    # Compute cluster assignments by taking argmin of each row. 
    cluster_assignments = A_new.argmin(axis = 1)
    # find outliers
    is_outlier = _is_outlier(A_new)
    
    # Fill in new cluster indicator matrix A. 
    for i, q in enumerate(cluster_assignments):
        
        if is_outlier[i]:          # Set row to 0's if outlier. 
            A_new[i,:] = 0
        else:                      # Set new cluster assignment otherwise. 
            A_new[i,:], A_new[i,q] = 0,1
    
    return A_new


def _r_jc(indices, W, X, A):
    """ The feature cluster indicator matrix B is updated according to Formula 7 in Li (2005), which 
    uses an 'affiliation score' of the same form as that used to update A. The feature is assigned
    to the cluster with the lowest score. 
    
    This function computes this score for a given feature j and feature cluster c by summing
    over the K data clusters while holding c and j fixed using the following formula:
    
        r[j,c] = SUM_{k} [ A[:,k]'(W[:,j] - X[k,c]) ]^2
    
    Parameters
    ----------
    indices : tuple
        (index of feature, index of feature cluster)
    W : np.array
        data matrix
    X : np.array
        cluster centroid matrix
    A : np.array
        data cluster indicator matrix
    
    Returns
    -------
    float
        affiliation score for the jth feature in the cth cluster
    """

    j, c = indices
    
    K = X.shape[0]
    
    r_jc = 0
    
    # W[:,j] - jth column of W
    # X[k,c] - kc-th entry of X (kc-th centriod)
    # A[:,k] - kth column of A
    
    for k in range(K): 
        r_jc += np.dot(A[:,k].T, np.square(W[:,j] - X[k,c]))
        
    return r_jc

    
def _updateB(A,B,X,W):
    """Updates the matrix B by creating a matrix M of identical dimensions whose elements are 'affiliation scores'.
    For each row, a 1 is placed in the position of the smallest entry and the rest set to 0's. In the case
    of ties, the entire row is set to 0 following the convention set in Li and Zhu (2005) who term such 
    cases 'outliers'. 
    
    Parameters
    ----------
    A : np.array
        old data cluster indicator matrix
    B : np.array
        old feature cluster indicator matrix
    X : np.array
        old cluster centroid matrix
    W : np.array
        data matrix
    
    Returns
    -------
    np.array
        new feature cluster assignment matrix B
    """

    m, C = B.shape
    B_new = np.zeros((m,C))

    for j in range(m):                   # iteration over rows (features)
        for c in range(C):               # iterate over columns (clusters)
            B_new[j,c] = _r_jc((j,c), W,X,A)
            
    # Compute cluster assignments by taking argmin of each row. 
    cluster_assignments = B_new.argmin(axis = 1)
    # find outliers
    is_outlier = _is_outlier(B_new)
    
    for i, q in enumerate(cluster_assignments):
        if is_outlier[i]:                  # Set row to 0's if outlier. 
            B_new[i,:] = 0                 
        else:                              # Set new cluster assignment otherwise.
            B_new[i,:], B_new[i,q] = 0, 1
        
    return B_new


def run_BMD(A,B,W, max_iter=100, verbose = 1):
    """Executes clustering Algorithm 1 from Li (2005). 
    
    Parameters
    ----------
    A : np.array
        initial data cluster matrix
    B : np.array
        initial feature cluster matrix
    W : np.array
        binary data matrix
    max_iter : int, optional
        maximum number of algorithm iterations, by default 100
    verbose : int, optional
        print loss function and progress, by default 1
    
    Returns
    -------
    float
        final value of cost function
    np.array
        final data cluster matrix
    np.array
        final feature cluster matrix
    """

    
    X = _updateX(A,B,W)
    O_old = _objective(A, B, X, W)

    n_iter = 0

    while n_iter < max_iter:
        A = _updateA(A,B,X,W)
        B = _updateB(A,B,X,W)
        X = _updateX(A,B,W)
        O_new = _objective(A,B,X,W)
        if O_new < O_old:
            O_old = O_new
            if verbose:
                print(ITER_MESSAGE.format(n_iter, O_new))
            n_iter += 1
        else:
            break

    if verbose:
        print("Convergence reached after {0} iterations".format(n_iter))
        
    return O_new, A, B



# deprecated implementations of updateA and updateB
############ computing A given X, B #############

#def T_matrix(indices, W,X):
#    i, k = indices
#    
#    m = W.shape[1]
#    c = X.shape[1]
#    
#    wi = W[i,:]
#    w_temp = np.tile(wi, (c, 1))
#    
#    xk = X[k, :]
#    xk.shape = (xk.shape[0], 1)
#    x_temp = np.tile(xk, (1, m))
#    
#    T = w_temp - x_temp
#    return np.square(T)
#
#def m_ik(indices, W,X,B):
#    T = T_matrix(indices, W,X)
#    return np.trace(np.dot(T,B))

################ update B given A, X ##################

#def S_matrix(indices, W, X):
#    j, c = indices
#    
#    n = W.shape[0]
#    k = X.shape[0]
#    
#    wj = W[:,j]
#    wj.shape = (wj.shape[0], 1)
#    w_temp = np.tile(wj, (1, k))
#    
#    xc = X[:,c]
#    x_temp = np.tile(xc, (n,1))
#    
#    S = w_temp - x_temp
#    return np.square(S)
#    
#def r_jc(indices, W,X,A):
#    S = S_matrix(indices, W,X)
#    return np.trace(np.dot(A.T, S))   