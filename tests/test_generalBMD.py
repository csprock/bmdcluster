import unittest
import numpy as np

from .context import generalBMD

# from bmdcluster.optimizers.generalBMD import run_BMD
# from bmdcluster.optimizers.generalBMD import _updateB
# from bmdcluster.optimizers.generalBMD import _updateA
# from bmdcluster.optimizers.generalBMD import _updateX


class TestExampleDataset_General(unittest.TestCase):

    def setUp(self):

        self.W = np.loadtxt(open('tests/data/test_set_2.csv', 'r'), delimiter = ',')

        self.A , self.B = np.zeros((6,3)), np.zeros((6,3))

        for i in range(0,3): 
            self.A[2*i, i], self.B[2*i,i] = 1, 1
        #self.A[0,0], self.A[2,1], self.A[4,2] = 1,1,1


        self.expected_X = np.array([[1,0,1],
                                    [0,1,0],
                                    [0,0,1]])


        self.expected_AB, j = np.zeros([6,3]), 0
        for i in range(0,3):
            self.expected_AB[j:(j+2),i] = 1
            j = j + 2


    def test_updateX(self):
        self.assertTrue(np.array_equal(self.expected_X,
                                       generalBMD._updateX(self.A, self.B, self.W)))

    def test_updateA(self):
        self.assertTrue(np.array_equal(self.expected_AB,
                                       generalBMD._updateA(self.A, self.B, self.expected_X, self.W)))

    def test_updateB(self):
        self.assertTrue(np.array_equal(self.expected_AB,
                                       generalBMD._updateB(self.A, self.B, self.expected_X, self.W)))

    def test_run_BMD(self):

        _, A, B, _ = generalBMD.run_BMD(self.A, self.B, self.W, verbose = 0)

        with self.subTest():
            self.assertTrue(np.array_equal(A, self.expected_AB))

        with self.subTest():
            self.assertTrue(np.array_equal(B, self.expected_AB))


if __name__ == '__main__':
    unittest.main()