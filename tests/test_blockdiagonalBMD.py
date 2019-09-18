import unittest
import numpy as np


from .context import blockdiagonalBMD

# from bmdcluster.optimizers.blockdiagonalBMD import run_bd_BMD
# from bmdcluster.optimizers.blockdiagonalBMD import _bd_updateB
# from bmdcluster.optimizers.blockdiagonalBMD import _bd_updateA
# from bmdcluster.optimizers.blockdiagonalBMD import _d_ik
# from bmdcluster.optimizers.blockdiagonalBMD import _Y


class TestExampleDataset_BD(unittest.TestCase):
    
    def setUp(self):
        
        # Uses the data and initialization setup from example in section 2.4 of Li & Zhu.
        
        self.W = np.loadtxt(open('tests/data/li_zhu.csv', 'r'), delimiter = ',', skiprows = 1)
        
        # Initialize data matrix A
        self.A = np.zeros((6,2))
        self.A[1,0], self.A[4,1] = 1,1
        
        # intermediate stage 
        self.step_B = np.array([[True, False],
                                [True, False],
                                [True, False],
                                [False, False],   # outlier
                                [False, True],
                                [False, True],
                                [False, False]])  # outlier
        
        
    
        self.expected_A = np.array([[1,0],
                                    [1,0],
                                    [1,0],
                                    [0,1],
                                    [0,1],
                                    [0,1]])
    
        self.expected_assignment = [0,0,0,1,1,1]
    

        self.expected_B = np.array([[True, False],
                                    [True, False],
                                    [True, False],
                                    [False, True],
                                    [False, True],
                                    [False, True],
                                    [False, False]])
            

    def test_run_bd_BMD(self):
        
        _, A, B = blockdiagonalBMD.run_bd_BMD(self.A, self.W, verbose = False)
        
        with self.subTest():
            self.assertTrue(np.array_equal(A, self.expected_A))
            
        
        with self.subTest():
            self.assertTrue(np.array_equal(B, self.expected_B))
            
        
    def test_d_ik(self):
        for i in range(0, self.A.shape[0]):
            with self.subTest(i = i):
                self.assertEqual(blockdiagonalBMD._d_ik(i, self.W, self.step_B), self.expected_assignment[i])
                

    def test_bd_updateB(self):
        self.assertTrue(np.array_equal(blockdiagonalBMD._bd_updateB(self.A, self.W), self.step_B))
            

    def test_bd_updateA(self):
        self.assertTrue(np.array_equal(self.expected_A, blockdiagonalBMD._bd_updateA(self.A, self.step_B, self.W)))

    
class IdentityTests_BD(unittest.TestCase):
    
    def setUp(self):
        self.I = np.identity(3)
        
    
    def test_bd_updateB(self):
        B = blockdiagonalBMD._bd_updateB(self.I, self.I)
        self.assertTrue(np.array_equal(B, self.I))
        

    def test_Y(self):
        self.assertTrue(np.array_equal(self.I, blockdiagonalBMD._Y(self.I, self.I)))
        
    def test_bd_updateA(self):
        A = blockdiagonalBMD._bd_updateA(self.I, self.I, self.I)
        self.assertTrue(np.array_equal(self.I, A))
        

if __name__ == '__main__':
    unittest.main()