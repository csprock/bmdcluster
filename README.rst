=============
Introduction
=============


.. image:: https://img.shields.io/pypi/v/bmdcluster.svg
        :target: https://pypi.python.org/pypi/bmdcluster

.. image:: https://img.shields.io/travis/csprock/bmdcluster.svg
        :target: https://travis-ci.org/csprock/bmdcluster

.. image:: https://readthedocs.org/projects/bmdcluster/badge/?version=latest
        :target: https://bmdcluster.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
        :target: https://opensource.org/licenses/MIT


A Python implementation of the Binary Matrix Decomposition algorithm for clustering binary data.

* Documentation: https://bmdcluster.readthedocs.io.
* GitHub: https://github.com/csprock/bmdcluster


bmdcluster
----------

This packages contains an implementation of the Binary Matrix Decomposition (BMD) algorithm
for clustering a binary matrix as presented by Li [1]_ and Li & Zhu [2]_. BMD solves the
two-sided clustering problem of clustering both the data and the features.

The algorithm works by decomposing a binary matrix :math:`W \in \mathbb{Z}^{n \times m}` with :math:`n` points and :math:`m` features into :math:`\hat{W}=AXB^T` 
where :math:`A \in \mathbb{Z}^{n \times c}` and :math:`B \in \mathbb{Z}^{m \times k}` are cluster membership indicator matrices for the data and features with :math:`c` data clusters and :math:`k` feature clusters. :math:`X \in \mathbb{Z}^{c \times k}` is a cluster representation matrix that encodes the relationship 
between the data and feature clusters. The algorithm alternatingly solves for these matrices by minimizing

.. math::
   \begin{equation}
        \| W - AXB^T \|^2
   \end{equation}

Two variants of the algorithm are implemented.
The first method assumes the data matrix is block-diagonal, i.e. there is a one-to-one correspondence between the data and feature clusters,
which is equivalent to setting :math:`X = I`. The second method is a general method with no restrictions on the matrix structure, i.e. a group of points
can be associated with several clusters of features and vice versa. See [1]_ for details.

The cluster assignments of the data and features are encoded in indicator matrices that are randomly initialized at the start of the clustering procedure.
[2]_ recommends bootstrapping a small subset of the data to get initial data cluster assignments.
These are then used to seed the data indicator matrix at the beginning of the clustering procedure when used on the full dataset. 

:code:`bmdcluster` supports this method of initializing the data clusters as well as random initialization. Users also have the option initialize the feature cluster indicator matrix to the identity matrix, which corresponds to putting each feature in its own cluster at initialization.



Credits
-------

.. [1] Li, T. (2005, August). `A general model for clustering binary data <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.387.745&rep=rep1&type=pdf>`_. In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining (pp. 188-197). ACM.

.. [2] Li, T., & Zhu, S. (2005, April). On clustering binary data. `In Proceedings of the 2005 SIAM International Conference on Data Mining <https://pdfs.semanticscholar.org/b3b5/c7e794df43fe89122bd39dafd9a5f504c524.pdf>`_ (pp. 526-530). Society for Industrial and Applied Mathematics.


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


