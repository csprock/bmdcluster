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



# Computes the objective function for the general BMD algorithm.
def _objective(A,B,X,W):
    # TODO: better docstring
    return np.linalg.norm(W - np.dot(A, np.dot(X,B.T)))


# computing X given A, B #

#def updateX(A,B,W):
#    p = A.sum(axis = 0)
#    q = B.sum(axis = 0)
#    
#    X_new = (1 / np.outer(p,q))*np.dot(A.T, np.dot(W,B))
#    return X_new



# Computes updated cluster relation matrix X given A,B,W.
def _updateX(A,B,W):
    # TODO: better docstring
    """
    Updates the cluster centroid matrix X according to Equation 5 in Li (2005).
    
    The kc-th entry of X is the sum of the entries of W in the kth data cluster and cth feature cluster 
    normalized by the product of the cluster sizes and can be thought of as cluster centroids.
    Entries of X that correspond to an empty cluster are set to zero. 
    
    Parameters
    ----------
    A: data cluster assignment matrix
    B: feature cluster assignment matrix
    W: data matrix
    
    Returns
    -------
    X_new: numpy.ndarray
        NumPy array of size K x C.
    
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




def _m_ik(indices,W,X,B):
    # TODO: better docstring
    """
    The data cluster indicator matrix A is updated using Formula 6 in Li (2005), which uses 
    an 'affiliation score' that can be thought of as a distance between the i-th point and
    the center of the k-th data cluster. The point is then assigned to the cluster with the
    lowest score. 
    
    This function computes this score for a given data point i and data cluster k by summing
    over the C feature clusters while holding k and i fixed using the following formula:
    
        m[i,k] = SUM_{c} [ (W[i,:] - X[k,c])'B[:,c] ]^2
    
    Parameters
    ----------
    indices: tuple
        indices[0] index of data point
        indices[1] index of data cluster
    W: data matrix
    X: cluster centroid matrix
    B: feature cluster assignment matrix
    
    Returns
    -------
    m_ik: float
    """
    
    i, k = indices # get indices

    C = X.shape[1] # number of feature clusters
    m_ik = 0
    
    # W[i,:] - ith row of W
    # X[k,c] - kc-th entry of X (kc-th centriod)
    # B[:,c] - cth column of B
    
    # sum over the feature clusters C
    for c in range(C):  m_ik += np.dot(np.square(W[i,:] - X[k,c]), B[:,c])
    return m_ik
    

# Check if the minimum value in 1d array appears more than once. 
def _min_dupes(r):
    # TODO: better docstring
    if len(np.argwhere(r == np.argmin(r))) > 1:
        return True
    else:
        return False


# Update the data cluster indicator matrix A. 
def _updateA(A,B,X,W):
    # TODO: better docstring
    """
    Updates the matrix A by creating a matrix M of identical dimensions whose elements are 'affiliation scores'.
    For each row, a 1 is placed in the position of the smallest entry and the rest set to 0's. In the case
    of ties, the entire row is set to 0 following the convention set in Li and Zhu (2005) who term such 
    cases 'outliers'. 
    
    Parameters
    ----------
    A: data cluster indicator matrix
    B: feature cluster indicator matrix
    X: cluster centroid matrix
    W: data matrix
    
    Returns
    -------
    M: new data cluster matrix A
    """
    
    n, K = A.shape
    A_new = np.zeros((n,K))
    
    for i in range(n):                   # iterate over data (rows)
        for k in range(K):               # iterate over features (columns)
            A_new[i,k] = _m_ik((i,k), W,X,B)     
    
    
    # Compute cluster assignments by taking argmin of each row. 
    cluster_assignments = A_new.argmin(axis = 1)
    # Apply min_dupes() to the rows of A_new. 
    is_outlier = np.apply_along_axis(_min_dupes, axis = 1, arr = A_new)
    
    # Fill in new cluster indicator matrix A. 
    for i, q in enumerate(cluster_assignments):
        
        if is_outlier[i]:          # Set row to 0's if outlier. 
            A_new[i,:] = 0
        else:                      # Set new cluster assignment otherwise. 
            A_new[i,:], A_new[i,q] = 0,1
    
    return A_new


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


def _r_jc(indices, W, X, A):
    # TODO: better docstring
    """
    The feature cluster indicator matrix B is updated according to Formula 7 in Li (2005), which 
    uses an 'affiliation score' of the same form as that used to update A. The feature is assigned
    to the cluster with the lowest score. 
    
    This function computes this score for a given feature j and feature cluster c by summing
    over the K data clusters while holding c and j fixed using the following formula:
    
        r[j,c] = SUM_{k} [ A[:,k]'(W[:,j] - X[k,c]) ]^2
    
    
    Parameters
    ----------
    indices: tuple
        indices[0] index of feature
        indices[1] index of feature cluster
    W: data matrix
    X: cluster centroid matrix
    A: data cluster indicator matrix
    
    Returns
    -------
    r_jc: float
    
    """    
    j, c = indices
    
    K = X.shape[0]
    
    r_jc = 0
    
    # W[:,j] - jth column of W
    # X[k,c] - kc-th entry of X (kc-th centriod)
    # A[:,k] - kth column of A
    
    for k in range(K): r_jc += np.dot(A[:,k].T, np.square(W[:,j] - X[k,c]))
        
    return r_jc

    


def _updateB(A,B,X,W):
    # TODO: better docstring
    """
    Updates the matrix B by creating a matrix M of identical dimensions whose elements are 'affiliation scores'.
    For each row, a 1 is placed in the position of the smallest entry and the rest set to 0's. In the case
    of ties, the entire row is set to 0 following the convention set in Li and Zhu (2005) who term such 
    cases 'outliers'. 
    
    Parameters
    ----------
    A: data cluster indicator matrix
    B: feature cluster indicator matrix
    X: cluster centroid matrix
    W: data matrix
    
    Returns
    -------
    B_new: new data cluster matrix B
    
    """
    
    m, C = B.shape
    B_new = np.zeros((m,C))
    
    for j in range(m):                   # iteration over rows (features)
        for c in range(C):               # iterate over columns (clusters)
            B_new[j,c] = _r_jc((j,c), W,X,A)
            

    # Compute cluster assignments by taking argmin of each row. 
    cluster_assignments = B_new.argmin(axis = 1)
    # Apply min_dupes() to the rows of B_new.
    is_outlier = np.apply_along_axis(_min_dupes, axis = 1, arr = B_new)
    
    for i, q in enumerate(cluster_assignments):
        if is_outlier[i]:                  # Set row to 0's if outlier. 
            B_new[i,:] = 0                 
        else:                              # Set new cluster assignment otherwise.
            B_new[i,:], B_new[i,q] = 0, 1
        
    return B_new


# TODO: better docstring
def run_BMD(A,B,W, max_iter=100, verbose = 1):
    """
    Executes clustering Algorithm 1 from Li (2005). 
    
    Parameters
    ----------
    A: initial data cluster assignment matrix
    B: initial feature cluster assignment matrix
    W: binary data matrix
    verbose: logical flag to print progress to console each iteration
    
    Returns
    -------
    O_new: the minimal value of the objective function after algorithm's completion
    A: final data cluster assignment matrix
    B: final feature cluster assignment matrix
    
    """
    
    
    X = _updateX(A,B,W)
    O_old = _objective(A, B, X, W)
    
    if verbose: print(O_old)

    n_iter = 0

    while n_iter < max_iter:
        A = _updateA(A,B,X,W)
        B = _updateB(A,B,X,W)
        X = _updateX(A,B,W)
        O_new = _objective(A,B,X,W)
        if O_new < O_old:
            O_old = O_new
            if verbose and n_iter % 10 == 0: 
                print(ITER_MESSAGE.format(n_iter, O_new))
            n_iter += 1
        else:
            break
        
    return O_new, A, B



# def get_indices(A, j):
#     return np.where(A[:,j] == 1)
