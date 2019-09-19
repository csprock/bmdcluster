=====
Usage
=====



Block Diagonal Method
---------------------

The following example will demonstrate the basic usage of the block-diagonal variant of the BMD algorithm.
The block-diagonal method assumes there is a one-to-one correspondence between the data clusters and
the feature clusters. For example, if a group of data points has a group of attributes it is associated
with.

.. code:: python

  from bmdcluster import blockdiagonalBMD

Next we create some sample block diagonal data.

.. code:: python

  import numpy as np

  # create a 30 x 6 block diagonal matrix with each blocks
  # consisting of 10 data points and two features
  b = 2   # size of blocks
  n = 30  # number data points
  m = 3*b # number of features

  stride = b*(n // m)

  data = np.zeros((n, m))

  for i in range(m):
      W[slice(i*stride, (i+1)*stride), slice(i*b, i*(b+1))] = 1


Next we initialize the model using bootstrapping with 5 bootstrap seed points.

.. note::
  Cluster assignments are highly dependent on initialization, so bootstrapping
  is recommended.

.. code:: python

  model = blockdiagonalBMD(n_clusters=3, use_bootstrap=True, b=5)
  model.fit(data, verbose=True)

  # get feature and data cluster labels
  feature_labels = model.get_feature_labels()
  data_labels = model.get_data_labels()

You can also use the :code:`.fit_transform` method to return the cost and
cluster assignment matrices

.. code:: python

  cost, data_clusters, feature_clusters = model.fit_transform(data, verbose=True)

These are also can be accessed as attributes after calling either :code:`.fit` or
:code:`.fit_transform`.

.. code:: python

  cost = model.cost
  data_cluster_matrix = model.A
  feature_cluster_matrix = model.B

General Method
--------------

The next few examples demonstrate how to use the general variant of the BMD algorithm.
This variation doesn't make any assumptions about the number of feature clusters or
how they are associated with the data clusters. A feature cluster can be associated
with more than one data cluster etc.

There are two initialization options for the feature cluster assignment matrix. You can 
specify the number of feature clusters using the :code:`f_clusters` option, or you can 
initialize each feature to be in its own cluster by setting :code:`B_ident=True`. 

.. note::

    If you are unsure how many feature clusters there are or you are only interested in clustering the data, we recommend setting :code:`B_ident = True`. This may result in empty feature clusters.
    This is normal because putting each feature in its own cluster makes no assumptions on the relationship between features,
    leaving the algorithm free to group features as it sees fit, which will result in columns of 0's in the feature cluster indicator matrix.

.. code:: python

  from bmdcluster import generalBMD

Next we recreate the same sample data as above, except this time we'll augment
it by giving the first block of points. The data matrix will look like a
block-diagonal matrix with an additional block in the top-right corner
where we have given assigned the last two features/attributes to
the first group of points.


.. code:: python

    b = 2    # size of blocks
    n = 30   # number of points
    m = 3*b  # number of features (3 per block)

    W = np.zeros((n, m))

    stride = b*(n // m)

    for i in range(m):
      W[slice(i*stride, (i+1)*stride), slice(i*b, i*b + b)] = 1

    # augment by assigning additional features/attributes to the first cluster of
    # of data points.
    W[0:stride, (m-b):m] = W[0:stride, 0:b]


Next we initialize the model to find 3 data clusters and 3 feature clusters and to
use bootstrapped initialization with 5 points.

.. code:: python

  model = generalBMD(n_clusters=3, f_clusters=3, use_bootstrap=True, b=5)

The model fitting and label getting methods act the same as they do on the block-diagonal
model class.

Next we apply clustering to the same data only this time we initialize the each feature to be in its own cluster
by setting the cluster assignment matrix to the identity. 

.. code:: python

  model = generalBMD(n_clusters=3, B_ident=True, use_bootstrap=True, b=5)

.. caution::

    Setting both :code:`B_ident=True` and :code:`f_clusters` are mutually exclusive options and will result in 
    an error. 

.. note::
   The algorithm's runtime is :math:`O(n^3)`, so may not be suitable for large datasets.

.. note::

    Unless you are using the block diagonal variant of the algorithm, there is no relationship between the data clusters and the feature clusters.
    That is, data cluster 1 and feature cluster 1 do not refer to the same cluster. In this case, it is possible that a feature will have an equally strong affiliation to each cluster
    and the feature is not assigned to a cluster. This is normal and authors refer to such a feature as an "outlier" and will be labeled with a -1. 