
import numpy as np
import unittest



class TestDataset(unittest.TestCase):
    
    
    def setUp(self):
        
        self.W = np.loadtxt(open('./test_set_2.csv', 'r'), delimiter = ',')
        
        self. A , self.B = np.zeros((6,3)), np.zeros((6,3))
        for i in range(0,3): self.A[2*i, i], self.B[2*i,i] = 1, 1
        #self.A[0,0], self.A[2,1], self.A[4,2] = 1,1,1
        
        
        self.expected_X = np.array([[1,0,1],
                                    [0,1,0],
                                    [0,0,1]])
    
    
        self.expected_AB, j = np.zeros([6,3]), 0
        for i in range(0,3):
            self.expected_AB[j:(j+2),i] = 1
            j = j + 2
            
    
    
    def test_updateX(self):
        
        self.assertTrue(np.array_equal(self.expected_X, _updateX(self.A, self.B, self.W)))
        
    def test_updateA(self):
        
        self.assertTrue(np.array_equal(self.expected_AB, _updateA(self.A,self.B,self.expected_X,self.W)))
        
    def test_updateB(self):

        self.assertTrue(np.array_equal(self.expected_AB, _updateB(self.A, self.B, self.expected_X, self.W)))
        

if __name__ == '__main__':
    unittest.main()