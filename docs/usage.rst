=====
Usage
=====

To use :code:`bmdcluster` in a project::

    from bmdcluster import bmdcluster


The following is a typical usage example which uses the general method to find two data clusters and initializes the feature cluster membership matrix to the identity, 
so that each feature is in its own cluster initially. 

.. code:: python

   from bmdcluster import bmdcluster
   model = bmdcluster(n_clusters = 2, method = 'general', B_ident = True)
   model.fit(X)

   # Print members of the first data cluster and the second feature cluster
   print(model.get_indices(0, 'data'))
   print(model.get_indices(1, 'features'))

This example shows how to use bootstrapped initialization with 20 sample points. Since this only effects the initialization of the data clusters, this setting can be used with any settings that are related to the feature clusters.

.. code:: python

   model = bmdcluster(data_clusters = 2, method = 'general', B_ident = True, use_bootstrap = True, b = 20)


Below is an example showing how to use the init_ratio setting to control the fraction of points randomly initialized to clusters.

.. code:: python

   model = bmdcluster(data_clusters = 2, method = 'general', B_ident = True, init_ratio = 0.25)

.. note::

    Since the algorithm is heavily dependent on the choice of initial points,
    it may be necessary to run the algorithm several times until a sufficiently low cost or satisfactory clusters are achieved.

To access the cluster assignment matrices, you may call them as attributes of the :code:`bmdcluster` object or 
you may assign them when fitting the model by setting :code:`return_results = True`.

.. code:: python

   # assign to variables
   cost, A, B = model.fit(X, return_results = True)
   # access as attributes
   print(model.A)

.. note::
   The algorithm's runtime is :math:`O(n^3)`, so may not be suitable for large datasets. 

.. note::

    Unless you are using the block diagonal variant of the algorithm, there is no relationship between the data clusters and the feature clusters.
    That is, data cluster 1 and feature cluster 1 do not refer to the same cluster. In this case, it is possible that a feature will have an equally strong affiliation to each cluster
    and the feature is not assigned to a cluster. This is normal and authors refer to such a feature as an "outlier".


.. note::

    If you are unsure how many feature clusters there are or you are only interested in clustering the data, we recommend setting :code:`B_ident = True`. This may result in empty feature clusters.
    This is normal because putting each feature in its own cluster makes no assumptions on the relationship between features,
    leaving the algorithm free to group features as it sees fit, which will result in columns of 0's in the feature cluster indicator matrix.
