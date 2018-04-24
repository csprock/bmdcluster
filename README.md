

<h2> Introduction </h2>

This packages contains an implementation of the Binary Matrix Decomposition (BMD) algorithm for clustering a binary matrix as presented by Li [1] and Li & Zhu [2]. BMD solves the two-sided clustering problem of clustering both the data and the features.

Two variants of the BMD algorithm are implemented. The first method assumes the data matrix is block-diagonal, i.e. that is each group of points is associated with a group of features and vice versa. The second method is a general method with no restrictions on the matrix structure. See [1] for details.

The cluster assignments of the data and features are encoded in indicator matrices that are randomly initialized at the start of the clustering procedure. [2] recommends bootstrapping a small subset of the data to get initial data cluster assignments. These are then used to seed the data indicator matrix at the beginning of the clustering procedure when used on the full dataset. `bmdcluster` supports this method of initializing the data clusters as well as random initialization. Users also have the option initialize the feature cluster indicator matrix to the identity matrix, which corresponds to putting each feature in its own cluster at initialization.

<h2> Installation </h2>

The `bmdcluster` package can be installed using pip by calling
 ````
 pip install bmdcluster
 ````

<h2> Usage </h2>

The main point of entry to the `bmdcluster` package is the `BMD` class, whose usage is modeled on that of sklearn's models. The BMD model is instantiated with a set of parameters before the .fit() method is called on the data to be clustered. The BMD model can be used by importing the `BMD` model object from the `bmdcluster` package as follows:

````
from bmdcluster import BMD
````

<h4> The BMD class </h4>

bmdcluster.BMD(n_clusters, method, B_ident, use_bootstrap = False, **kwargs)

Parameters:

* **n_clusters**: number of clusters the data is to be grouped into.
* **method**: variant of the BMD algorithm to use. The options are "block_diagonal" and "general".
* **B_ident**: logical specifying whether or not to initialize the feature cluster indicator matrix to the identity. This is equivalent to placing each feature in its own cluster at the start of the algorithm. If set to `False`, then the number of feature clusters must be set using by supplying the keyword argument `f_clusters`.
* **use_bootstrap**: logical specifying whether or not to bootstrap a subset of the data to use for initializing the data clusters. Default is `False`. If set to `True` then the keyword argument `b` must be set to the number of points to use for the bootstrapped sample.

Keyword arguments:
* **b**: optional, the size of the bootstrapped sample. Must be set when `use_bootstrap = True`.
* **f_clusters**: optional, the number of feature clusters to find. Must be set if `B_ident = False`.
* **init_ratio**: optional, fraction of data to randomly initialize when `use_bootstrap = False`. Remaining data unassigned.
* **seed**: optional, set state for random number generator


<h5> Attributes </h5>

* **A**: cluster assignment indicator matrix for the data. Is a numpy array of shape (data points, n_clusters)
* **B**: feature cluster assignment indicator matrix. Is a numpy array of shape (data features, f_clusters)
* **cost**: final value of the cost function


<h5> Methods </h5>

<h6> fit </h6>
Fit the BMD clustering model on a set of data.

Parameters:
* **W** a numpy array of data to cluster, data in rows and features in columns.
* **verbose**: logical to print the algorithm's progress, default set to 0.
* **return_results**: return the attributes as a tuple. Default is False.


<h6> get_indices </h6>
Given the number of a cluster, returns an array containing the indices of the either the data points or features in that cluster.

Parameters:
* **i**: number of the cluster
* **which**: either "data" or "features"



<h2> Examples </h2>

The following is a typical usage example.
````
from bmdcluster import BMD
model = BMD(n_clusters = 2, method = 'general', B_ident = True)
model.fit(X)

# Print members of the first data cluster and the second feature cluster
print(model.get_indices(0, 'data'))
print(model.get_indices(1, 'features'))
````

This example shows how to use bootstrapped initialization with 20 sample points. Since this only effects the initialization of the data clusters, this setting can be used with any settings that are related to the feature clusters.
````
model = BMD(data_clusters = 2, method = 'general', B_ident = True, use_bootstrap = True, b = 20)
````

Below is an example showing how to use the `init_ratio` setting to control the fraction of points randomly initialized to clusters.
````
model = BMD(data_clusters = 2, method = 'general', B_ident = True, init_ratio = 0.25)
````

To access the cluster assignment matrices, you may call them as attributes of the `BMD` object or you may assign them when fitting the model by setting `return_results = True`.
````
# assign to variables
cost, A, B = model.fit(X, return_results = True)
# access as attributes
print(model.A)
````
<h3> Notes </h3>

Unless you are using the block diagonal variant of the algorithm, there is no relationship between the data clusters and the feature clusters. That is, data cluster 1 and feature cluster 1 do not refer to the same cluster.

In the block diagonal case, it is possible that a feature will have an equally strong affiliation to each cluster. In this case, the feature is not assigned to a cluster. The authors refer to such a feature as an "outlier" [2].

Since the algorithm is heavily dependent on the choice of initial points, it may be necessary to run the algorithm several times until a sufficiently low cost or satisfactory clusters are achieved [1,2].  

If you are unsure how many feature clusters there are or you are only interested in clustering the data, we recommend setting `B_ident = True`. This may result in empty feature clusters. This is normal because putting each feature in its own cluster makes no assumptions on the relationship between features, leaving the algorithm free to group features as it sees fit, which will result in columns of 0's in the feature cluster indicator matrix.




<h2> References </h2>

[1] Li, T. (2005, August). [A general model for clustering binary data.](http://users.cs.fiu.edu/~taoli/pub/p188-li.pdf) In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining (pp. 188-197). ACM.

[2] Li, T., & Zhu, S. (2005, April). [On clustering binary data.](https://pdfs.semanticscholar.org/b3b5/c7e794df43fe89122bd39dafd9a5f504c524.pdf) In Proceedings of the 2005 SIAM International Conference on Data Mining (pp. 526-530). Society for Industrial and Applied Mathematics.
