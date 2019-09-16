import numpy as np

"""
This module contains a variant of the Binary Matrix Decomposition (BMD) algorithm for clustering binary data
as presented in "A General Model for Clustering Binary Data" (Tao Li, 2005) and "On Clustering Binary Data"
(Tao Li & Shenghuo Zhu, 2005). This varient of the BMD algorithm is for data whose matrix is can be 
rearranged into block-diagonal form. That is, each set of data is associated with a set of features and vice-versa. 
This module implements Algorithm 2 from Li (2005) supplemented with ideas from Li & Zhu (2005). 


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

def _bd_objective(A,B,W):
    """ Objective function for block diagonal variation of BMD."""
    return np.linalg.norm(W - np.dot(A, B.T))


def _is_bd_outlier(B):
    """Determines if a feature is an outlier if it is equally associated 
    with each cluster. This is checked by seeing if all the entries in a 
    given row of the candidate feature cluster association matrix are 1's. 
    Any rows that meet these conditions are set to 0. (see Li and Zhu)

    Parameters
    ----------
    B : np.array
        candidate feature cluster assignment matrix
    
    Returns
    -------
    np.array
        feature cluster assignment matrix
    """

    i = np.where(np.sum(B, axis=1) == B.shape[1])[0]
    B[i, :] = 0
    
    return B
    

def _d_ik(i, W, B):
    """ The data cluster matrix A is updated using formula 10 from Li (2005) which is the same as 
    formula 2.3 in Li & Zhu (2005). The formula uses the squared distance between ith point and 
    the kth cluster. The point is then assigned to the closest cluster. The squared distance
    between point i and data cluster k is computed by summing over the element-wise differences
    between the i-th row and k-th row of W and B, respectively: 
    
        d[i,k] = SUM_{j in features} (W[i,j] - B[k,j])^2j
    
    Parameters
    ----------
    i : int
        infdex of data point
    W : np.array
        binary data matrix
    B : np.array
        feature cluster assignment matrix
    
    Returns
    -------
    int
        index of assigned cluster
    """

    # Vectorized implementation to compute summations found in formula 10. 
    Di = W[i,:].reshape((W.shape[1],1)) - B           # broadcast i-th row of W across columns of B
    Di = Di*Di                                        
    Di = Di.sum(axis = 0)                             # sum over rows (features)
    assigned_cluster = Di.argmin()                    # take index of minimum quantity to be new cluster assignment
    
    return assigned_cluster
    

   
#### assign clusters ####
#def ai(B,W,i):
#    
#    q = B.T - W[i,:]
#    q = q*q
#    q = q.sum(axis = 1)
#    return q.argmin()

#########################

def _bd_updateA(A,B,W):
    """Update data cluster assignment matrix A using formula 10 in Li (2005). 
    
    Parameters
    ----------
    A : np.array
        old data cluster matrix
    B : np.array
        old feature cluster matrix
    W : np.array
        binary data matrix
    
    Returns
    -------
    np.array
        updated data cluster matrix
    """

    n, K = A.shape
    A_new = np.zeros((n,K))
    
    for i in range(n): 
        A_new[i,:], A_new[i, _d_ik(i, W, B)] = 0, 1
    
    return A_new


def _Y(A, W):
    """ The feature cluster matrix B is updated using formula 11 from Li (2005). This is done
    by computing a 'probability matrix' Y where the kj-th entry represents the probability
    feature j is in the k-th cluster. The updated matrix B is the same shape as Y and contains
    1's where the corresponding entry of Y is greater than or equal to 1/2 and 0's elsewhere. 
    (Note Li (2005) uses a strict inequality, but we have found empirically that nonstrict 
    inequality works better.) 
    
    The formula for the matrix Y is: 
    
    
        y[i,j] = (1/n_k)*SUM_{i in data} a[i,k]*w[i,j] = (1/n_k)*( a[:,k]'w[:,j] )
        n_k = number of points in cluster k
    
    Parameters
    ----------
    A : np.array
        data cluster matrix
    W : np.array
        data matrix
    
    Returns
    -------
    np.array
        probability matrix
    """

    n_k = A.sum(axis = 0)                 # Compute number of points in each cluster.
    n_k[np.where(n_k == 0)[0]] = np.inf   # Set zero entries to inf to zero out reciprocal. 
    r = 1 / n_k                           # Compute reciprocal. 
    r.shape = (A.shape[1],1)              # Reshape for broadcasting. 

    return np.dot(A.T, W)*r                  # Compute Y matrix as dot product matrix of rows of A and W. 


def _bd_updateB(A,W):
    """ Updated feature cluster matrix B. Applies the _Y() and B set to the matrix the same shape
    as Y but with 1's in the entries corresponding to where Y[>=0.5] and 0's elsewhere.
    
    Features that are associated with all clusters are 'outliers' (have a row whose entries >=0.5 in Y)
    Following Li and Zhu are not assigned to any clusters by setting all entries in B associated with those
    features to 0. 
    
    Parameters
    ----------
    A : np.array
        old data cluster matrix
    W : np.array
        data matrix
    
    Returns
    -------
    np.array
        new feature cluster matrix
    """

    
    Y = _Y(A, W)
    B_new = np.greater_equal(Y, 0.5).T    # Update B matrix. 
    
    #### setting all True rows to False ####
    # if feature has similar associate to all clusters, is an outlier (see Li and Zhu)
    # will have a row of all True by the np.greater_equal() function, reverse to make row of False
    
    # # TODO: use single outlier function and create a shared utils.py 
    # def is_outlier(d):
        
    #     if np.array_equal(d, np.array([True]*len(d))):
    #         return np.array([False]*len(d))
    #     else:
    #         return d
    
    # B_new = np.apply_along_axis(is_outlier, axis = 1, arr = B_new)

    B_new = _is_bd_outlier(B_new)
    
    return B_new
    

def run_bd_BMD(A,W, max_iter=100, verbose=False):
    """Executes clustering Algorithm 2 from Li (2005). 
    
    Parameters
    ----------
    A : np.array
        initial data cluster assignment matrix
    W : np.array
        binary data matrix
    max_iter : int, optional
        maximum number of algorithm iterations, by default 100
    verbose : bool, optional
        print progress and objective function value, by default False
    
    Returns
    -------
    float
        final value of objective function
    np.array
        final data cluster matrix
    np.array
        final feature cluster matrix
    """
    
    
    B = _bd_updateB(A,W)
    O_old = _bd_objective(A, B, W)

    n_iter = 0

    while n_iter < max_iter:
        A = _bd_updateA(A,B,W)
        B = _bd_updateB(A,W)
        O_new = _bd_objective(A,B,W)
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