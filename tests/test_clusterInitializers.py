import unittest
import numpy as np
import os
import sys

from .context import cluster_initializers

# from bmdcluster.initializers.cluster_initializers import initialize_A
# from bmdcluster.initializers.cluster_initializers import initialize_B


class Testinitialize_A(unittest.TestCase):

    def setUp(self):
        self.n = 4

    def test_initialize_A_assertions(self):

        with self.subTest('Check data_cluster size assertion'):
            # Check that assertion error raised when the number of data clusters is greater
            # than or equal to the size of the dataset.
            with self.assertRaises(AssertionError):
                cluster_initializers.initialize_A(n = self.n, n_clusters = self.n)
        
        with self.subTest('Check init_ratio assertions'):

            # Check that assertion error is raised when the init_ratio is outside of
            # the interval (0,1].

            with self.assertRaises(AssertionError):
                cluster_initializers.initialize_A(n = self.n, n_clusters = self.n - 1, init_ratio = 1.1)

            with self.assertRaises(AssertionError):
                cluster_initializers.initialize_A(n = self.n, n_clusters = self.n - 1, init_ratio = 0)



    def test_initialize_A_outputs(self):

        with self.subTest('Check sum of entries'):
            # When init_ratio not set, each point should be assigned exactly one cluster.
            # Check sum of elements of cluster assignment matrix A.
            A = cluster_initializers.initialize_A(self.n, self.n-1)
            self.assertEqual(A.sum(), self.n)



        with self.subTest('Check init_ratio'):
            # Check that when init_ratio is set, the number of assigned clusters is
            # the expected number.
            A = cluster_initializers.initialize_A(n = self.n, n_clusters = self.n - 1, init_ratio = 0.5)
            self.assertEqual(A.sum(), self.n // 2)



        with self.subTest('Check bootstrap list passing'):
            # Test passing of list of tuples containing the positions of entries
            # to be set in the returned matrix.

            A_expected = np.array([[1,0],
                                   [0,0],
                                   [0,1],
                                   [0,0]])


            A = cluster_initializers.initialize_A(n = self.n, n_clusters = 2, bootstrap = [(0,0),(2,1)])
            self.assertTrue(np.array_equal(A, A_expected))



class TestInitializeB(unittest.TestCase):

    def setUp(self):
        self.m = 3


    def test_initializeB_output(self):

        with self.subTest('Check B_ident'):
            # Check feature cluster matrix B is initialized to identity when B_ident set to True.
            B = cluster_initializers.initialize_B(self.m, B_ident = True)
            self.assertTrue(np.array_equal(np.identity(self.m), B))

    
    # def test_check_assertions(self):

    #     with self.assertRaises(AssertionError):
    #         initialize_B(self.m, B_ident = False, f_clusters = self.m + 1)

    #     with self.assertRaises(AssertionError):
    #         initialize_B(self.m, B_ident = False, f_clusters = 1)

    #@unittest.skip("No longer using keyword arguments in initialize_B")
    def test_initializeB_assertions(self):

        with self.subTest('Check missing keyword argument'):
            # Check that MissingKeywordArgument raised when B_ident set to False and
            # without additional keyword arguments.
            with self.assertRaises(KeyError):
                cluster_initializers.initialize_B(self.m, B_ident = False)

        with self.subTest('Check assertions'):
            # Check that when f_clusters is passed, that AssertionError is raised
            # f_clusters is not in the interval (1, m].

            with self.assertRaises(AssertionError):
                cluster_initializers.initialize_B(self.m, B_ident = False, f_clusters = self.m + 1)

            with self.assertRaises(AssertionError):
                cluster_initializers.initialize_B(self.m, B_ident = False, f_clusters = 1)

if __name__ == '__main__':
    unittest.main()
