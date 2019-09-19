import unittest
import numpy as np


from .context import blockdiagonalBMD_model, generalBMD_model

# from bmdcluster import blockdiagonalBMD
# from bmdcluster import generalBMD


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
            if i % 5 == 0: j+=1
            A_expected[i, c[j]] = 1

        # create expected B matrix
        B_expected = np.zeros((6,3))
        j = -1
        for i in range(0,6):
            if i % 2 == 0: j+=1
            B_expected[i,c[j]] = True



        BMD_model = blockdiagonalBMD_model(n_clusters = self.C,
                                use_bootstrap = True,
                                b = 5,
                                seed = self.seed)

        cost, A, B = BMD_model.fit_transform(W = self.W, verbose = 0)

        with self.subTest('Test for correct cost'):
            self.assertEqual(cost, 0)

        with self.subTest('Test for correct data cluster matrix A'):
            self.assertTrue(np.array_equal(A_expected, A))

        with self.subTest('Test for correct feature cluster matrix B'):
            self.assertTrue(np.array_equal(B_expected, B))


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



if __name__ == '__main__':
    unittest.main()