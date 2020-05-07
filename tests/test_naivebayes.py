import unittest

import os

import numpy as np
import matplotlib.pyplot as plt

from metesatpy.production import FY4AAGRIL1FDIDISK4KM
from metesatpy.algorithms.CloudMask import NaiveBayes

data_root_dir = os.getenv('METEPY_DATA_PATH', 'data')


class TestNaiveBayes(unittest.TestCase):
    def test_something(self):
        def setUp(self) -> None:
            pass

    def test_obs_prepare(self):
        pass


if __name__ == '__main__':
    unittest.main()
