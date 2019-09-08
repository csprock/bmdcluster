import unittest
import sys, os


if __name__ == '__main__':
    mypath = os.path.dirname(os.path.realpath('__file__'))
    sys.path.append(os.path.join(mypath, os.pardir))

import test_blockdiagonalBMD
import test_generalBMD
import test_bootstrapInitializer
import test_clusterInitializers
import test_bmdcluster



loader = unittest.TestLoader()
suite = unittest.TestSuite()


suite.addTest(loader.loadTestsFromModule(test_blockdiagonalBMD))
suite.addTest(loader.loadTestsFromModule(test_generalBMD))
suite.addTest(loader.loadTestsFromModule(test_bootstrapInitializer))
suite.addTest(loader.loadTestsFromModule(test_clusterInitializers))
suite.addTest(loader.loadTestsFromModule(test_bmdcluster))


if __name__ == '__main__':
    
    runner = unittest.TextTestRunner(verbosity = 1)
    result = runner.run(suite)
