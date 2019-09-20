import unittest
import numpy as np


from .context import blockdiagonalBMD_model
from .context import generalBMD_model

class TestBMD_bd(unittest.TestCase):

    def setUp(self):

        self.W = np.loadtxt(open('tests/data/test_set_3.csv','r'), delimiter = ',')
        self.C, self.K = 3, 3
        self.seed = 123


    def test_BMD_blockdiagonal_example(self):
        # Tests the BMD module's block diagonal mode using a contrived test
        # dataset constructed for this purpose whose results are known. This test is
        # designed to test the functionality of the module, not accuracy of
        # result.

        # Construct the expected output matrices.

        # create expected A matrix
        A_expected = np.zeros((15,3))
        j, c = -1, [0,2,1]
        for i in range(0,15):
            if i % 5 == 0: 
                j+=1
            A_expected[i, c[j]] = 1

        # create expected B matrix
        B_expected = np.zeros((6,3))
        j = -1
        for i in range(0,6):
            if i % 2 == 0: 
                j+=1
            B_expected[i,c[j]] = True

        # test set
        W_test = np.zeros((4, 6))
        W_test[0, :2] = 1
        W_test[1, 2:4] = 1
        W_test[2:4, 4:6] = 1
        # expected predicted data cluster matrix
        A_test_expected = np.zeros((4, self.K))
        A_test_expected[0,0] = 1
        A_test_expected[1,2] = 1
        A_test_expected[2:4, 1] = 1
        # expected cluster assignments for test data
        A_pred_expected = np.array([0, 2, 1, 1])


        BMD_model = blockdiagonalBMD_model(n_clusters = self.C,
                                use_bootstrap = True,
                                b = 5,
                                seed = self.seed)

        BMD_model.fit(W = self.W, verbose = 0)

        with self.subTest('Test for correct cost'):
            self.assertEqual(BMD_model.cost, 0)

        with self.subTest('Test for correct data cluster matrix A'):
            self.assertTrue(np.array_equal(A_expected, BMD_model.A))

        with self.subTest('Test for correct feature cluster matrix B'):
            self.assertTrue(np.array_equal(B_expected, BMD_model.B))

        with self.subTest('Test for correct predicted cluster matrix A'):
            self.assertTrue(np.array_equal(A_test_expected, BMD_model.transform(W_test)))

        with self.subTest('Test for correct predicted cluster matrix A'):
            self.assertTrue(np.array_equal(A_pred_expected, BMD_model.predict(W_test)))
        

class TestBMD_general(unittest.TestCase):


    def setUp(self):

        self.W = np.loadtxt(open('tests/data/test_set_4.csv', 'r'), delimiter = ',')
        self.C = 3
        self.seed = 1234

    def test_BMD_general_example(self):
        # Test the BMD module's general method on a contrived test example
        # constructed for this purpose whose results are known. This test is
        # designed to test the functionality of the module, not accuracy of
        # result.

        # Construct expected output matrices

        # Construct expected A matrix
        A_expected = np.zeros((15,3))
        j, c = -1, [0,1,2]
        for i in range(0,15):
            if i % 5 == 0: j+=1
            A_expected[i, c[j]] = 1

        # Construct expected B matrix
        B_expected = np.zeros((6,6))
        j = -1
        for i in range(0,6):
            if i % 2 == 0: j+=1
            B_expected[i,c[j]*2] = True


        W_test = np.zeros((4, 6))
        W_test[0, :2] = 1
        W_test[0, 4:6] = 1
        W_test[1, 2:4] = 1
        W_test[2, 4:6] = 1
        W_test[3, 4:6] = 1

        A_pred_expected = np.array([0, 1, 2, 2])

        A_test_expected = np.zeros((4, 3))
        A_test_expected[0, 0] = 1
        A_test_expected[1, 1] = 1
        A_test_expected[2, 2] = 1
        A_test_expected[3, 2] = 1


        BMD_model = generalBMD_model(n_clusters = self.C,
                                B_ident = True,
                                use_bootstrap = False,
                                seed = self.seed)


        cost, A, B = BMD_model.fit_transform(self.W, verbose = 0)

        with self.subTest('Test for correct cost'):
            self.assertEqual(cost, 0)

        with self.subTest('Test for correct data cluster matrix A'):
            self.assertTrue(np.array_equal(A, A_expected))

        with self.subTest('Test for correct feature cluster matrix B'):
            self.assertTrue(np.array_equal(B, B_expected))

        with self.subTest('Test for correct predicted cluster matrix A'):
            self.assertTrue(np.array_equal(A_test_expected, BMD_model.transform(W_test)))

        with self.subTest('Test for correct predicted cluster matrix A'):
            self.assertTrue(np.array_equal(A_pred_expected, BMD_model.predict(W_test)))



if __name__ == '__main__':
    unittest.main()