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
    # TODO: better docstring
    """ Objective function for block diagonal variation of BMD."""
    return np.linalg.norm(W - np.dot(A, B.T))
    

def _d_ik(i, W, B):
    # TODO: better docstring
    """
    The data cluster matrix A is updated using formula 10 from Li (2005) which is the same as 
    formula 2.3 in Li & Zhu (2005). The formula uses the squared distance between ith point and 
    the kth cluster. The point is then assigned to the closest cluster. The squared distance
    between point i and data cluster k is computed by summing over the element-wise differences
    between the i-th row and k-th row of W and B, respectively: 
    
        d[i,k] = SUM_{j in features} (W[i,j] - B[k,j])^2j
    
    
    Parameters
    ----------
    i: index of data point
    W: binary data matrix
    B: feature cluster assignment matrix
    
    Returns
    -------
    assigned_cluster: int
        index of the assigned cluster
        
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
    # TODO: better docstring
    """
    Update data cluster assignment matrix A using formula 10 in Li (2005). 
    
    Parameters
    ----------
    A: data cluster indicator matrix
    B: feature cluster indicator matrix
    W: binary data matrix
    
    Returns
    -------
    A_new: updated data cluster matrix
    
    """
    n, K = A.shape
    A_new = np.zeros((n,K))
    
    for i in range(n): A_new[i,:], A_new[i, _d_ik(i, W, B)] = 0, 1
    
    return A_new





def _Y(A, W):
    # TODO: better docstring
    """
    The feature cluster matrix B is updated using formula 11 from Li (2005). This is done
    by computing a 'probability matrix' Y where the kj-th entry represents the probability
    feature j is in the k-th cluster. The updated matrix B is the same shape as Y and contains
    1's where the corresponding entry of Y is greater than or equal to 1/2 and 0's elsewhere. 
    (Note Li (2005) uses a strict inequality, but we have found empirically that nonstrict 
    inequality works better.) 
    
    The formula for the matrix Y is: 
    
    
        y[i,j] = (1/n_k)*SUM_{i in data} a[i,k]*w[i,j] = (1/n_k)*( a[:,k]'w[:,j] )
        n_k = number of points in cluster k
    
    """
    n_k = A.sum(axis = 0)                 # Compute number of points in each cluster.
    n_k[np.where(n_k == 0)[0]] = np.inf   # Set zero entries to inf to zero out reciprocal. 
    r = 1 / n_k                           # Compute reciprocal. 
    r.shape = (A.shape[1],1)              # Reshape for broadcasting. 

    
    return np.dot(A.T, W)*r                  # Compute Y matrix as dot product matrix of rows of A and W. 



def _bd_updateB(A,W):
    # TODO: better docstring
    """
    Updated feature cluster matrix B. Applies the _Y() and B set to the matrix the same shape
    as Y but with 1's in the entries corresponding to where Y[>=0.5] and 0's elsewhere.
    
    Features that are associated with all clusters are 'outliers' (have a row whose entries >=0.5 in Y)
    Following Li and Zhu are not assigned to any clusters by setting all entries in B associated with those
    features to 0. 
        
    
    Parameters
    ----------
    A: data cluster indicator matrix
    W: binary data matrix
    
    Returns
    -------
    B_new: updated feature cluster indicator matrix
    
    """
    
    Y = _Y(A, W)
    B_new = np.greater_equal(Y, 0.5).T    # Update B matrix. 
    
    #### setting all True rows to False ####
    # if feature has similar associate to all clusters, is an outlier (see Li and Zhu)
    # will have a row of all True by the np.greater_equal() function, reverse to make row of False
    
    def is_outlier(d):
        
        if np.array_equal(d, np.array([True]*len(d))):
            return np.array([False]*len(d))
        else:
            return d
    
    B_new = np.apply_along_axis(is_outlier, axis = 1, arr = B_new)
    
    
    
    return B_new
    


# TODO: better docstring
def run_bd_BMD(A,W, max_iter=100, verbose=1):
    
    """
    Executes clustering Algorithm 2 from Li (2005). 
    
    Parameters
    ----------
    A: initial data cluster assignment matrix
    W: binary data matrix
    verbose: logical flag to print progress to console each iteration
    
    Returns
    -------
    O_new: the minimal value of the objective function after algorithm's completion
    A: final data cluster assignment matrix
    B: final feature cluster assignment matrix
    
    """
    
    
    B = _bd_updateB(A,W)
    O_old = _bd_objective(A, B, W)
    
    if verbose: print(O_old)

    n_iter = 0

    while n_iter < max_iter:
        A = _bd_updateA(A,B,W)
        B = _bd_updateB(A,W)
        O_new = _bd_objective(A,B,W)
        if O_new < O_old:
            O_old = O_new
            if verbose and n_iter % 10 == 0: 
                print(ITER_MESSAGE.format(n_iter, O_new))
            n_iter += 1
        else:
            break
        
    return O_new, A, B
