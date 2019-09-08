import unittest
import numpy as np

from bmdcluster.initializers.bootstrapInitializer import bootstrap_data
from bmdcluster.initializers.bootstrapInitializer import assign_bootstrapped_clusters
from bmdcluster.initializers.bootstrapInitializer import initializeBootstrappedClusters


class TestboostrapInitializer(unittest.TestCase):

    def setUp(self):

        self.W = np.loadtxt(open('tests/data/test_set_3.csv', 'r'), delimiter = ',')
        self.K = 3
        self.C = 3
        self.seed = 123

    def test_intializeBootstrappedClusters_assertions(self):


        #with self.subTest('Test clustering method assertion'):
        #    with self.assertRaises(AssertionError):
        #        initializeBootstrappedClusters(W = self.W, n_clusters = self.K, method = 'wrong', B_ident = True)


        with self.subTest('Test output type and length using general method'):
            output = initializeBootstrappedClusters(W = self.W, n_clusters = self.K, method = 'general', B_ident = True, b = 5, seed = self.seed)
            self.assertTrue(isinstance(output, list))
            self.assertEqual(len(output), 5)


        with self.subTest('Test output type and length using block diagonal method'):
            output = initializeBootstrappedClusters(W = self.W, n_clusters = self.K, method = 'block_diagonal', B_ident = True, b = 5, seed = self.seed)
            self.assertTrue(isinstance(output, list))
            self.assertEqual(len(output), 5)


    def test_bootstrap_data_assertions(self):

        with self.subTest('Test input size assertion'):
            with self.assertRaises(AssertionError):
                bootstrap_data(N = 1, b = 2)


        with self.subTest('Test bootstrap size assertion'):
            with self.assertRaises(KeyError):
                bootstrap_data(N = 10)

    def test_bootstrap_data_outputs(self):


        with self.subTest('Test output sizes'):

            x_samp, x_rep = bootstrap_data(10, b = 5)

            self.assertEqual(5, len(x_samp))
            self.assertEqual(10, len(x_rep))


        with self.subTest('Test expected output'):

            expected_samp = np.array([3,0])
            expected_rep = np.array([3,3,3,3])

            x_samp, x_rep = bootstrap_data(4, b = 2, seed = self.seed)

            self.assertTrue(np.array_equal(expected_samp, x_samp))
            self.assertTrue(np.array_equal(expected_rep, x_rep))




    def test_assign_bootstrapped_clusters(self):

        # Check that output of bootstrapped_clusters is the same as that of known example.

        x_samp, x_rep = bootstrap_data(15, b = 5, seed = self.seed)
        actual_clusters = [1,2,0,0,1]
        expected_assignments = list(zip(list(x_samp), actual_clusters))

        A_boot = np.zeros((15,3))
        for k, i in enumerate(x_rep):
            for j in expected_assignments:
                if j[0] == i: A_boot[k,j[1]] = 1


        self.assertEqual(expected_assignments, assign_bootstrapped_clusters(A_boot, x_rep, x_samp))




if __name__ == '__main__':
    unittest.main()
