

<h2> Introduction </h2>

This packages contains an implementation of the Binary Matrix Decomposition (BMD) algorithm for clustering a binary matrix as presented by Li [1] and Li & Zhu [2]. BMD solves the two-sided clustering problem of clustering both the data and the features. Two variants of the BMD algorithm are implemented. The first method assumes the relationship between the data clusters and feature clusters is symmetric, that is each group of points is associated with a group of features and vice versa. This implies the data matrix has a block diagonal structure. The second method is a general method. See [1] for details.

The cluster assignments of the data and features are encoded in indicator matrices that are randomly initialized at the start of the clustering procedure. [2] recommends bootstrapping a small subset of the data to get initial data cluster assignments. These are then used to seed the data indicator matrix at the beginning of the clustering procedure when used on the full dataset. ````bmdcluster```` supports this method of initializing the data clusters as well as random initialization. Users also have the option initialize the feature cluster indicator matrix to the identity matrix, which corresponds to putting each feature in its own cluster at initialization.

<h2> Usage </h2>

<h3> Initialization </h3>

The `BMD ` class contains methods for running the BMD algorithm. The algorithm is initialized upon the instantiation of a `BMD` object, whose parameters determine the number of data and feature clusters to find, the clustering method and initialization settings.

There are four required arguments.

The `data clusters` argument sets the number of clusters the data is to be grouped into.

The `method` argument determines which variant of the BMD algorithm to use. The options are "block_diagonal" and "general".

The `B_ident` argument is a logical specifying whether or not to initialize the feature cluster indicator matrix to the identity. This is equivalent to placing each feature in its own cluster at the start of the algorithm. If set to `False`, then the number of feature clusters must be set using by supplying the keyword argument `feature_clusters`.

The `use_bootstrap` argument is a logical specifying whether or not to bootstrap a subset of the data to use for initializing the data clusters. Default is `False`. If set to `True` then the keyword argument `b` must be set to the number of points to use for the bootstrapped sample.

It is also possible to specify the ratio of points to randomly initialize when `use_bootstrap` is set to `False`. The `init_ratio` is an optional keyword argument that specifies the faction of the data to randomly initialize. When this argument is set, only a random subset of the data is randomly assigned to clusters while the remaining points remain unassigned.

Note that both `use_bootstrap` and `init_ratio` only effect the initialization of the data clusters and can be used with any combination of settings that effect the initialization of the feature clusters.


<h4> Examples </h4>

Below is a minimal example of model instantiation.
````
model = BMD(data_clusters = 2, method = 'general', B_ident = True)
````

Below is an example setting the number of feature clusters manually.
````
model = BMD(data_clusters = 3, method = 'general', B_ident = False, feature_clusters = 2)
````

This example shows how to use bootstrapped initialization. Since this only effects the initialization of the data clusters, this setting can be used with any settings that are related to the feature clusters.
````
model = BMD(data_clusters = 2, method = 'general', B_ident = True, use_bootstrap = True, b = 20)
````

Below is an example showing how to use the `init_ratio` setting to control the fraction of points randomly initialized to clusters.
````
model = BMD(data_clusters = 2, method = 'general', B_ident = True, use_bootstrap = False, init_ratio = 0.25)
````

<h3> Fitting the Model </h3>

Once a `BMD` object has been instantiated, fitting the model is simply a matter of passing a data matrix (a numpy array) to the `fit` method: `BMD.fit(W)`. The array must be such that the data is contained in the rows and the features in columns.

There are two optional arguments to the `fit` method. `verbose` is a logical flag to print the progress of the algorithm which by default is set to `0`. The second optional argument `return_results` is a logical flag that returns the results of the BMD algorithm. If set to `True`, three objects are returned, the final value of the cost function, the data cluster indicator matrix A and the feature cluster indicator matrix B.

````
cost, A, B = BMD.fit(W, return_results = True)
````
Once the model has been fit, the `get_indices` method can be used to return the indices of the points or features contained in a specified cluster (numbered starting from 0). For example, to get the indices of the data points contained in the second data cluster use:

````
BMD.get_indices(1, 'data')
````
To get the indices of the features in the first feature cluster, use
````
BMD.get_indices(0, 'features')
````
Note: unless you are using the block diagonal variant of the algorithm, there is no relationship between the data clusters and the feature clusters. That is, data cluster 1 and feature cluster 1 do not refer to the same cluster.


<h2> References </h2>

[1] Li, T. (2005, August). [A general model for clustering binary data.](http://users.cs.fiu.edu/~taoli/pub/p188-li.pdf) In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining (pp. 188-197). ACM.

[2] Li, T., & Zhu, S. (2005, April). [On clustering binary data.](https://pdfs.semanticscholar.org/b3b5/c7e794df43fe89122bd39dafd9a5f504c524.pdf) In Proceedings of the 2005 SIAM International Conference on Data Mining (pp. 526-530). Society for Industrial and Applied Mathematics.
