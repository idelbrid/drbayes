from unittest import TestCase
from subspace_inference.utils import *
import torch
import pandas
import numpy as np
from scipy.stats import norm, uniform

class TestRejectionSampling(TestCase):
    def test_gaussian(self):
        samples = []
        for i in range(5000):
            s = rejection_sample(lambda x: norm.logpdf(x, 0, 1) - norm.logpdf(0, 0, 1),
                                 lambda x: uniform.logpdf(x, -10, 20) - uniform.logpdf(0, -10, 20),
                                 lambda: uniform.rvs(-10, 20), using_logs=True).send(None)
            samples.append(s)
        self.assertAlmostEqual(np.mean(samples), 0, places=3)
        self.assertAlmostEqual(np.std(samples), 1, places=3)





