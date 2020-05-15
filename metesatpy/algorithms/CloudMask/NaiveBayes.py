import numexpr as ne

import h5py
import numpy as np
import xarray as xr

from metesatpy.algorithms.CloudMask.CLMClassifiers import Ref063Day, Ref138Day, Ref063Min3x3Day, RefRatioDay, NdsiDay, \
    TStd, Bt1185


class NaiveBayes(object):

    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.classifier_dict = [

        ]

    def infer(self):
        # classifier infer
        pass

    def _fusion(self):
        pass
